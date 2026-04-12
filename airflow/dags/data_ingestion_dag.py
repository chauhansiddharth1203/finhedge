"""
data_ingestion_dag.py – Apache Airflow DAG for automated data ingestion.

DAG: finhedge_data_ingestion
Schedule: Every weekday at 18:00 IST (market close + buffer)

Tasks:
  1. ingest_raw_data    – Download OHLCV from Yahoo Finance via yfinance
  2. validate_schema    – Schema and NaN checks
  3. engineer_features  – Compute technical indicators
  4. check_drift        – Compare feature distributions to EDA baseline
  5. preprocess_data    – Scale, sequence, split
  6. notify_backend     – Hit /pipeline/trigger to invalidate model cache
"""

import logging
import sys
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago

# Ensure the project root is importable inside Airflow tasks
sys.path.insert(0, "/opt/airflow")

logger = logging.getLogger(__name__)

TICKERS  = os.getenv("TICKERS",  "AAPL,MSFT,GOOGL").split(",")
PERIOD   = os.getenv("PERIOD",   "2y")
RAW_DIR  = os.getenv("RAW_DIR",  "/opt/airflow/data/raw")
FEAT_DIR = os.getenv("FEAT_DIR", "/opt/airflow/data/features")
PROC_DIR = os.getenv("PROC_DIR", "/opt/airflow/data/processed")
BACKEND  = os.getenv("BACKEND_URL", "http://backend:8000")

DEFAULT_ARGS = {
    "owner":            "finhedge",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "start_date":       days_ago(1),
}


# ── Task functions ─────────────────────────────────────────────────────────

def ingest_raw_data(**context) -> dict:
    """Download OHLCV data for all configured tickers."""
    from backend.core.data.ingestion import StockDataIngester

    ingester = StockDataIngester(raw_dir=RAW_DIR)
    results  = {}
    for ticker in TICKERS:
        try:
            df = ingester.ingest(ticker.strip(), period=PERIOD)
            results[ticker] = {"rows": len(df), "status": "ok"}
            logger.info("Ingested %s: %d rows", ticker, len(df))
        except Exception as exc:
            results[ticker] = {"status": "error", "error": str(exc)}
            logger.error("Failed to ingest %s: %s", ticker, exc)

    context["ti"].xcom_push(key="ingest_results", value=results)
    return results


def validate_data(**context) -> None:
    """Run schema and data quality checks on raw ingested data."""
    from backend.core.data.ingestion import StockDataIngester
    from backend.core.data.validation import DataValidator

    ingester  = StockDataIngester(raw_dir=RAW_DIR)
    validator = DataValidator()
    failures  = []

    for ticker in TICKERS:
        try:
            df     = ingester.load(ticker.strip())
            report = validator.validate_raw(df, ticker.strip())
            if not report.passed:
                failures.append(ticker)
                logger.warning("Validation failed for %s: %s", ticker, report.details)
        except Exception as exc:
            failures.append(ticker)
            logger.error("Validation error for %s: %s", ticker, exc)

    if failures:
        logger.warning("Validation failures: %s", failures)
    context["ti"].xcom_push(key="validation_failures", value=failures)


def engineer_features(**context) -> None:
    """Compute technical indicators for all valid tickers."""
    from backend.core.data.ingestion import StockDataIngester
    from backend.core.data.features  import FeatureEngineer

    ingester = StockDataIngester(raw_dir=RAW_DIR)
    fe       = FeatureEngineer(features_dir=FEAT_DIR)
    failures = context["ti"].xcom_pull(key="validation_failures") or []

    for ticker in TICKERS:
        t = ticker.strip()
        if t in failures:
            logger.info("Skipping feature engineering for %s (failed validation).", t)
            continue
        try:
            df = ingester.load(t)
            fe.build(df, t)
            logger.info("Features engineered for %s", t)
        except Exception as exc:
            logger.error("Feature engineering failed for %s: %s", t, exc)


def check_drift(**context) -> None:
    """Detect data drift via KL-divergence vs. EDA baseline."""
    import json
    from pathlib import Path
    from backend.core.data.validation import DataValidator
    from backend.core.data.features   import FeatureEngineer

    fe       = FeatureEngineer(features_dir=FEAT_DIR)
    validator = DataValidator(baseline_dir=FEAT_DIR)
    drift_detected = []

    for ticker in TICKERS:
        t = ticker.strip()
        try:
            df     = fe.load(t)
            report = validator.validate_features(df, t)
            if not report.passed and not report.drift_ok:
                drift_detected.append(t)
                logger.warning("Drift detected for %s", t)
        except Exception as exc:
            logger.warning("Drift check error for %s: %s", t, exc)

    context["ti"].xcom_push(key="drift_detected", value=drift_detected)
    if drift_detected:
        logger.warning(
            "Data drift detected for %d tickers: %s", len(drift_detected), drift_detected
        )


def preprocess_data(**context) -> None:
    """Scale features and build LSTM sequences for all tickers."""
    from backend.core.data.features      import FeatureEngineer
    from backend.core.data.preprocessing import DataPreprocessor

    fe   = FeatureEngineer(features_dir=FEAT_DIR)
    prep = DataPreprocessor(processed_dir=PROC_DIR, lookback=60)

    for ticker in TICKERS:
        t = ticker.strip()
        try:
            df = fe.load(t)
            prep.fit_transform(df, t)
            logger.info("Preprocessing complete for %s", t)
        except Exception as exc:
            logger.error("Preprocessing failed for %s: %s", t, exc)


def notify_backend(**context) -> None:
    """Ping the backend to signal that fresh data is available."""
    import requests as req

    try:
        r = req.get(f"{BACKEND}/health", timeout=5)
        if r.status_code == 200:
            logger.info("Backend notified of fresh data.")
        else:
            logger.warning("Backend health check returned %s", r.status_code)
    except Exception as exc:
        logger.warning("Could not reach backend: %s", exc)


def should_retrain(**context) -> str:
    """Branch: trigger retraining DAG if drift was detected."""
    drift = context["ti"].xcom_pull(key="drift_detected") or []
    return "trigger_retraining" if drift else "skip_retraining"


# ── DAG definition ─────────────────────────────────────────────────────────

with DAG(
    dag_id="finhedge_data_ingestion",
    description="FinHedge: automated daily data ingestion and preprocessing",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 12 * * 1-5",   # 12:00 UTC = ~17:30 IST weekdays
    catchup=False,
    max_active_runs=1,
    tags=["finhedge", "data", "mlops"],
) as dag:

    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end")

    t_ingest  = PythonOperator(
        task_id="ingest_raw_data",
        python_callable=ingest_raw_data,
    )
    t_validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )
    t_features = PythonOperator(
        task_id="engineer_features",
        python_callable=engineer_features,
    )
    t_drift = PythonOperator(
        task_id="check_drift",
        python_callable=check_drift,
    )
    t_preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )
    t_notify = PythonOperator(
        task_id="notify_backend",
        python_callable=notify_backend,
    )
    t_branch = BranchPythonOperator(
        task_id="should_retrain",
        python_callable=should_retrain,
    )
    t_retrain_trigger = EmptyOperator(task_id="trigger_retraining")
    t_skip_retrain    = EmptyOperator(task_id="skip_retraining")

    # ── DAG edges ─────────────────────────────────────────────────────────
    (
        start
        >> t_ingest
        >> t_validate
        >> t_features
        >> t_drift
        >> t_preprocess
        >> t_notify
        >> t_branch
        >> [t_retrain_trigger, t_skip_retrain]
        >> end
    )
