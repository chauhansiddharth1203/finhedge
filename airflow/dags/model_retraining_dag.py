"""
model_retraining_dag.py – Automated model retraining pipeline.

DAG: finhedge_model_retraining
Triggered: manually or by data_ingestion_dag when drift is detected
Schedule:  Every Sunday 02:00 UTC (weekly scheduled retraining)

Tasks:
  1. train_lstm     – Retrain LSTM predictor via MLflow run
  2. train_xgboost  – Retrain XGBoost baseline
  3. train_hedger   – Retrain deep CVaR hedging policy
  4. evaluate       – Compute test metrics and compare to production model
  5. promote_model  – Promote to Production stage in MLflow registry if better
  6. notify         – Log summary
"""

import logging
import sys
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty  import EmptyOperator
from airflow.utils.dates import days_ago

sys.path.insert(0, "/opt/airflow")

logger = logging.getLogger(__name__)

TICKERS          = os.getenv("TICKERS", "AAPL").split(",")
MLFLOW_URI       = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
BACKEND_URL      = os.getenv("BACKEND_URL", "http://backend:8000")
EPOCHS           = int(os.getenv("RETRAIN_EPOCHS", "50"))
IMPROVEMENT_THRESHOLD = 0.02   # 2% RMSE improvement to promote

DEFAULT_ARGS = {
    "owner":            "finhedge",
    "depends_on_past":  False,
    "email_on_failure": False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=10),
    "start_date":       days_ago(1),
}


# ── Task functions ─────────────────────────────────────────────────────────

def train_lstm(**context) -> None:
    """Train LSTM model for all configured tickers."""
    from backend.core.training.trainer import Trainer

    results = {}
    for ticker in TICKERS:
        t = ticker.strip()
        try:
            trainer = Trainer(
                ticker=t, model_type="lstm", epochs=EPOCHS,
            )
            result = trainer.run()
            results[t] = result
            logger.info(
                "LSTM trained for %s | run_id=%s | RMSE=%.4f",
                t, result["run_id"][:8], result["metrics"].get("rmse", 0),
            )
        except Exception as exc:
            logger.error("LSTM training failed for %s: %s", t, exc)
            results[t] = {"error": str(exc)}

    context["ti"].xcom_push(key="lstm_results", value={
        k: {kk: vv for kk, vv in v.items() if kk != "model_path"}
        for k, v in results.items()
    })


def train_xgboost(**context) -> None:
    """Train XGBoost model for all tickers."""
    from backend.core.training.trainer import Trainer

    results = {}
    for ticker in TICKERS:
        t = ticker.strip()
        try:
            trainer = Trainer(ticker=t, model_type="xgboost", epochs=EPOCHS)
            result  = trainer.run()
            results[t] = result
            logger.info("XGBoost trained for %s", t)
        except Exception as exc:
            logger.error("XGBoost training failed for %s: %s", t, exc)
            results[t] = {"error": str(exc)}

    context["ti"].xcom_push(key="xgb_results", value={
        k: {kk: vv for kk, vv in v.items() if kk != "model_path"}
        for k, v in results.items()
    })


def train_hedger(**context) -> None:
    """Train the deep CVaR hedging policy for each ticker."""
    from backend.core.models.deep_hedger import DeepHedger
    from pathlib import Path
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("finhedge")

    for ticker in TICKERS:
        t = ticker.strip()
        with mlflow.start_run(run_name=f"hedger_{t}_{datetime.utcnow():%Y%m%d}"):
            mlflow.set_tag("model_type", "deep_hedger")
            mlflow.set_tag("ticker", t)
            mlflow.log_params({"epochs": 300, "n_paths": 2048, "alpha": 0.95})

            hedger = DeepHedger()
            history = hedger.train(epochs=300)

            mlflow.log_metrics({"final_cvar_loss": history[-1]})

            path = Path(f"/opt/airflow/models/hedger_{t}.pt")
            path.parent.mkdir(exist_ok=True)
            hedger.save(path)
            mlflow.log_artifact(str(path), artifact_path="model")
            logger.info("DeepHedger trained for %s | final CVaR loss=%.4f", t, history[-1])


def evaluate_and_compare(**context) -> None:
    """
    Compare newly trained models to the current production version.
    Promotes if RMSE improved by IMPROVEMENT_THRESHOLD.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_URI)
    client    = MlflowClient()
    lstm_results = context["ti"].xcom_pull(key="lstm_results") or {}

    for ticker, result in lstm_results.items():
        if "error" in result or "metrics" not in result:
            continue

        new_rmse = result["metrics"].get("rmse", float("inf"))
        model_name = "finhedge-lstm"

        # Get production model metrics
        try:
            prod_versions = client.get_latest_versions(model_name, stages=["Production"])
            if prod_versions:
                prod_run_id = prod_versions[0].run_id
                prod_run    = client.get_run(prod_run_id)
                prod_rmse   = prod_run.data.metrics.get("rmse", float("inf"))
            else:
                prod_rmse = float("inf")   # no production model yet
        except Exception:
            prod_rmse = float("inf")

        improvement = (prod_rmse - new_rmse) / (prod_rmse + 1e-9)
        logger.info(
            "%s: prod_rmse=%.4f  new_rmse=%.4f  improvement=%.2f%%",
            ticker, prod_rmse, new_rmse, improvement * 100,
        )

        if improvement > IMPROVEMENT_THRESHOLD or prod_rmse == float("inf"):
            # Promote new version to Production
            try:
                new_version = result.get("version", 1)
                client.transition_model_version_stage(
                    name=model_name,
                    version=str(new_version),
                    stage="Production",
                    archive_existing_versions=True,
                )
                logger.info(
                    "Promoted %s v%s to Production (RMSE %.4f → %.4f)",
                    model_name, new_version, prod_rmse, new_rmse,
                )
            except Exception as exc:
                logger.error("Could not promote model: %s", exc)
        else:
            logger.info(
                "New model NOT promoted (improvement %.2f%% < threshold %.0f%%)",
                improvement * 100, IMPROVEMENT_THRESHOLD * 100,
            )


def write_summary(**context) -> None:
    """Write a training summary to the metrics directory."""
    import json
    from pathlib import Path

    lstm_results = context["ti"].xcom_pull(key="lstm_results") or {}
    xgb_results  = context["ti"].xcom_pull(key="xgb_results")  or {}

    summary = {
        "run_at": datetime.utcnow().isoformat() + "Z",
        "lstm":   lstm_results,
        "xgb":    xgb_results,
    }

    path = Path("/opt/airflow/data/retraining_summary.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Retraining summary written to %s", path)


# ── DAG definition ─────────────────────────────────────────────────────────

with DAG(
    dag_id="finhedge_model_retraining",
    description="FinHedge: weekly automated model retraining with auto-promotion",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 2 * * 0",   # Every Sunday 02:00 UTC
    catchup=False,
    max_active_runs=1,
    tags=["finhedge", "training", "mlops"],
) as dag:

    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end")

    t_lstm    = PythonOperator(task_id="train_lstm",    python_callable=train_lstm)
    t_xgb     = PythonOperator(task_id="train_xgboost", python_callable=train_xgboost)
    t_hedger  = PythonOperator(task_id="train_hedger",  python_callable=train_hedger)
    t_eval    = PythonOperator(task_id="evaluate_and_compare", python_callable=evaluate_and_compare)
    t_summary = PythonOperator(task_id="write_summary", python_callable=write_summary)

    # LSTM and XGBoost train in parallel; hedger is independent
    start >> [t_lstm, t_xgb, t_hedger] >> t_eval >> t_summary >> end
