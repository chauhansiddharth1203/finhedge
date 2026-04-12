"""
config.py – Central configuration for the FinHedge backend.
All settings are read from environment variables with sensible defaults
so the same image works locally and inside Docker Compose.
"""

import os
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = Path(os.getenv("DATA_DIR",    str(BASE_DIR / "data")))
MODEL_STORE     = Path(os.getenv("MODEL_STORE", str(BASE_DIR / "models")))
RAW_DATA_DIR    = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
FEATURES_DIR    = DATA_DIR / "features"
METRICS_DIR     = BASE_DIR / "metrics"
LOG_DIR         = BASE_DIR / "logs"

for _d in [RAW_DATA_DIR, PROCESSED_DIR, FEATURES_DIR, MODEL_STORE, METRICS_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── MLflow ─────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI     = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME  = os.getenv("MLFLOW_EXPERIMENT_NAME", "finhedge")
MODEL_NAME_LSTM         = "finhedge-lstm"
MODEL_NAME_XGB          = "finhedge-xgb"
MODEL_NAME_HEDGER       = "finhedge-hedger"

# ── API ────────────────────────────────────────────────────────────────────
API_HOST    = os.getenv("API_HOST", "0.0.0.0")
API_PORT    = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "1"))
LOG_LEVEL   = os.getenv("LOG_LEVEL", "INFO")

# ── Data ───────────────────────────────────────────────────────────────────
DEFAULT_TICKER  = os.getenv("DEFAULT_TICKER", "AAPL")
DEFAULT_PERIOD  = os.getenv("DEFAULT_PERIOD", "2y")
LOOKBACK        = int(os.getenv("LOOKBACK", "60"))        # sequence length
TEST_SPLIT      = float(os.getenv("TEST_SPLIT", "0.2"))
VAL_SPLIT       = float(os.getenv("VAL_SPLIT", "0.1"))

# ── Model ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = os.getenv("DEFAULT_MODEL", "lstm")
LSTM_UNITS      = [64, 32]
DROPOUT         = 0.2
EPOCHS          = int(os.getenv("EPOCHS", "50"))
BATCH_SIZE      = int(os.getenv("BATCH_SIZE", "32"))
LR              = float(os.getenv("LR", "0.001"))
PATIENCE        = int(os.getenv("PATIENCE", "10"))

# ── Hedging ────────────────────────────────────────────────────────────────
CVAR_CONFIDENCE = float(os.getenv("CVAR_CONFIDENCE", "0.95"))
COST_RATE       = float(os.getenv("COST_RATE", "0.0002"))
HEDGE_HORIZON   = int(os.getenv("HEDGE_HORIZON", "21"))   # trading days

# ── Monitoring ─────────────────────────────────────────────────────────────
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.05"))
ALERT_ERROR_RATE = float(os.getenv("ALERT_ERROR_RATE", "0.05"))
