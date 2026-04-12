"""
metrics.py – Prometheus instrumentation for the FinHedge backend.

Custom metrics exported at /metrics (via prometheus_client):

  Counters:
    finhedge_prediction_requests_total   – per ticker, model
    finhedge_prediction_errors_total     – per ticker, error_type
    finhedge_hedge_requests_total        – per ticker, action

  Histograms:
    finhedge_prediction_latency_seconds  – inference latency
    finhedge_hedge_latency_seconds       – hedge computation latency
    finhedge_data_ingestion_duration_sec – pipeline step latency

  Gauges:
    finhedge_model_rmse                  – latest test RMSE per ticker/model
    finhedge_model_direction_acc         – latest direction accuracy
    finhedge_model_sharpe                – latest Sharpe ratio
    finhedge_data_drift_score            – KL divergence score per feature
    finhedge_pipeline_last_run_ts        – Unix timestamp of last pipeline run
    finhedge_active_model_version        – currently loaded model version
"""

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CollectorRegistry,
    REGISTRY,
)

# ── Request counters ───────────────────────────────────────────────────────

PREDICTION_REQUESTS = Counter(
    "finhedge_prediction_requests_total",
    "Total number of prediction API requests",
    ["ticker", "model"],
)

PREDICTION_ERRORS = Counter(
    "finhedge_prediction_errors_total",
    "Total number of prediction errors",
    ["ticker", "error_type"],
)

HEDGE_REQUESTS = Counter(
    "finhedge_hedge_requests_total",
    "Total number of hedge recommendation requests",
    ["ticker", "action"],
)

PIPELINE_RUNS = Counter(
    "finhedge_pipeline_runs_total",
    "Total number of pipeline stage executions",
    ["stage", "status"],   # status: success | failure
)

# ── Latency histograms ────────────────────────────────────────────────────

PREDICTION_LATENCY = Histogram(
    "finhedge_prediction_latency_seconds",
    "Time to generate a price prediction",
    ["ticker", "model"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

HEDGE_LATENCY = Histogram(
    "finhedge_hedge_latency_seconds",
    "Time to generate a hedge recommendation",
    ["ticker"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

INGESTION_DURATION = Histogram(
    "finhedge_data_ingestion_duration_seconds",
    "Time taken to ingest data from Yahoo Finance",
    ["ticker"],
    buckets=[1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

TRAINING_DURATION = Histogram(
    "finhedge_model_training_duration_seconds",
    "Time taken to train a model end-to-end",
    ["ticker", "model"],
    buckets=[10.0, 30.0, 60.0, 120.0, 300.0, 600.0],
)

# ── Model performance gauges ──────────────────────────────────────────────

MODEL_RMSE = Gauge(
    "finhedge_model_rmse",
    "Latest test RMSE for the active model",
    ["ticker", "model"],
)

MODEL_DIRECTION_ACC = Gauge(
    "finhedge_model_direction_accuracy",
    "Latest directional accuracy (fraction of correct up/down calls)",
    ["ticker", "model"],
)

MODEL_SHARPE = Gauge(
    "finhedge_model_sharpe",
    "Annualised Sharpe ratio of the model strategy",
    ["ticker", "model"],
)

MODEL_VERSION = Gauge(
    "finhedge_active_model_version",
    "Currently loaded model version number",
    ["ticker", "model"],
)

# ── Data drift gauges ─────────────────────────────────────────────────────

DATA_DRIFT_SCORE = Gauge(
    "finhedge_data_drift_score",
    "KL-divergence drift score per feature (higher = more drift)",
    ["ticker", "feature"],
)

DRIFT_ALERT = Gauge(
    "finhedge_drift_alert",
    "1 if data drift is detected for ticker, 0 otherwise",
    ["ticker"],
)

# ── Pipeline timestamp ────────────────────────────────────────────────────

PIPELINE_LAST_RUN = Gauge(
    "finhedge_pipeline_last_run_timestamp",
    "Unix timestamp of the last successful pipeline run",
    ["stage"],
)

# ── Helper functions ──────────────────────────────────────────────────────

def record_prediction(ticker: str, model: str, latency: float) -> None:
    """Increment prediction counter and record latency."""
    PREDICTION_REQUESTS.labels(ticker=ticker, model=model).inc()
    PREDICTION_LATENCY.labels(ticker=ticker, model=model).observe(latency)


def record_prediction_error(ticker: str, error_type: str) -> None:
    PREDICTION_ERRORS.labels(ticker=ticker, error_type=error_type).inc()


def record_hedge(ticker: str, action: str, latency: float) -> None:
    HEDGE_REQUESTS.labels(ticker=ticker, action=action).inc()
    HEDGE_LATENCY.labels(ticker=ticker).observe(latency)


def update_model_metrics(
    ticker: str,
    model: str,
    rmse: float,
    direction_acc: float,
    sharpe: float,
    version: int = 1,
) -> None:
    MODEL_RMSE.labels(ticker=ticker, model=model).set(rmse)
    MODEL_DIRECTION_ACC.labels(ticker=ticker, model=model).set(direction_acc)
    MODEL_SHARPE.labels(ticker=ticker, model=model).set(sharpe)
    MODEL_VERSION.labels(ticker=ticker, model=model).set(version)


def update_drift(ticker: str, feature_scores: dict[str, float]) -> None:
    any_drift = False
    for feature, score in feature_scores.items():
        DATA_DRIFT_SCORE.labels(ticker=ticker, feature=feature).set(score)
        if score > 0.05:
            any_drift = True
    DRIFT_ALERT.labels(ticker=ticker).set(1 if any_drift else 0)
