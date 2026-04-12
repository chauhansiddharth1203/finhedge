"""
health.py – /health and /ready endpoints.

/health → simple liveness probe (always 200 if process is alive)
/ready  → readiness probe: checks MLflow reachability and model availability
"""

import time
import logging
from typing import Any

import requests
from fastapi import APIRouter

from backend.api.schemas import HealthResponse, ReadyResponse
from backend.config import MLFLOW_TRACKING_URI

logger   = APIRouter()
router   = APIRouter(prefix="", tags=["Health"])
_start_t = time.time()

APP_VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health() -> HealthResponse:
    """Returns 200 if the server process is running."""
    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        uptime_seconds=round(time.time() - _start_t, 1),
    )


@router.get("/ready", response_model=ReadyResponse, summary="Readiness probe")
def ready() -> ReadyResponse:
    """Checks all downstream dependencies before serving traffic."""
    checks: dict[str, bool] = {}

    # MLflow reachability
    try:
        r = requests.get(f"{MLFLOW_TRACKING_URI}/health", timeout=3)
        checks["mlflow"] = r.status_code == 200
    except Exception:
        checks["mlflow"] = False

    # Model files available
    from backend.config import MODEL_STORE
    model_files = list(MODEL_STORE.glob("*.pt")) + list(MODEL_STORE.glob("*.pkl"))
    checks["models_loaded"] = len(model_files) > 0

    # Data available
    from backend.config import RAW_DATA_DIR
    data_files = list(RAW_DATA_DIR.glob("*.parquet"))
    checks["data_available"] = len(data_files) > 0

    ready_flag = all(checks.values())
    return ReadyResponse(ready=ready_flag, checks=checks)
