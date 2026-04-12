"""
pipeline.py – Pipeline management endpoints.

GET  /pipeline/status/{ticker}  → current pipeline stage statuses
POST /pipeline/trigger          → kick off a pipeline stage (async)
POST /pipeline/train            → trigger full training run
GET  /pipeline/runs             → list MLflow experiment runs
"""

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException

import mlflow
from mlflow.tracking import MlflowClient

from backend.api.schemas import (
    PipelineTriggerRequest,
    PipelineTriggerResponse,
    PipelineStatusResponse,
    PipelineStage,
    PipelineStatus,
    StageInfo,
    TrainRequest,
    TrainResponse,
)
from backend.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    METRICS_DIR,
)
from backend.core.monitoring import metrics as prom

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pipeline", tags=["Pipeline"])

# Simple in-process job tracker (replace with Redis/DB in production)
_jobs: dict[str, dict[str, Any]] = {}


# ── Status ─────────────────────────────────────────────────────────────────

@router.get("/status/{ticker}", response_model=PipelineStatusResponse)
def pipeline_status(ticker: str) -> PipelineStatusResponse:
    """Return the current status of each DVC pipeline stage for a ticker."""
    ticker = ticker.upper()
    stages = []
    overall = PipelineStatus.idle

    stage_names = ["ingest", "preprocess", "train", "evaluate"]
    stage_indicators = {
        "ingest":     Path(f"data/raw/{ticker}.parquet"),
        "preprocess": Path(f"data/processed/{ticker}_feature_scaler.pkl"),
        "train":      Path(f"models/lstm_{ticker}.pt"),
        "evaluate":   METRICS_DIR / "eval_metrics.json",
    }

    any_done = False
    for name in stage_names:
        indicator = stage_indicators[name]
        if indicator.exists():
            status = PipelineStatus.success
            mtime  = datetime.fromtimestamp(indicator.stat().st_mtime).isoformat()
            any_done = True
        else:
            status = PipelineStatus.idle
            mtime  = None

        stages.append(StageInfo(
            stage=name,
            status=status,
            ended_at=mtime,
        ))

    overall = PipelineStatus.success if any_done else PipelineStatus.idle

    # Last run timestamp from metrics file
    last_run = None
    if METRICS_DIR.joinpath("eval_metrics.json").exists():
        last_run = datetime.fromtimestamp(
            METRICS_DIR.joinpath("eval_metrics.json").stat().st_mtime
        ).isoformat()

    return PipelineStatusResponse(
        ticker=ticker,
        overall=overall,
        stages=stages,
        last_run_at=last_run,
        next_run_at=None,
    )


# ── Trigger ────────────────────────────────────────────────────────────────

@router.post("/trigger", response_model=PipelineTriggerResponse)
def trigger_pipeline(
    req: PipelineTriggerRequest,
    background_tasks: BackgroundTasks,
) -> PipelineTriggerResponse:
    """Asynchronously kick off a single pipeline stage."""
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "status": "queued",
        "stage":  req.stage.value,
        "ticker": req.ticker,
        "started_at": datetime.utcnow().isoformat(),
    }
    background_tasks.add_task(_run_stage, job_id, req)
    return PipelineTriggerResponse(
        job_id=job_id,
        status="queued",
        message=f"Stage '{req.stage.value}' queued for {req.ticker}.",
    )


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """Get the status of a background pipeline job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _jobs[job_id]


# ── Training ───────────────────────────────────────────────────────────────

@router.post("/train", response_model=TrainResponse)
def train_model(req: TrainRequest) -> TrainResponse:
    """Trigger a full training run (synchronous – blocks until complete)."""
    from backend.core.training.trainer import Trainer

    t0 = time.time()
    trainer = Trainer(
        ticker=req.ticker,
        model_type=req.model_type.value,
        epochs=req.epochs,
        batch_size=req.batch_size,
        lr=req.lr,
        period=req.period,
    )
    try:
        result = trainer.run()
    except Exception as exc:
        logger.exception("Training failed for %s", req.ticker)
        raise HTTPException(status_code=500, detail=str(exc))

    duration = time.time() - t0
    prom.TRAINING_DURATION.labels(ticker=req.ticker, model=req.model_type.value).observe(duration)

    return TrainResponse(
        run_id=result["run_id"],
        model_name=result["model_name"],
        version=int(result["version"]),
        metrics=result["metrics"],
        model_path=result["model_path"],
        duration_s=round(duration, 2),
    )


# ── MLflow runs list ───────────────────────────────────────────────────────

@router.get("/runs")
def list_runs(limit: int = 20) -> list[dict]:
    """List recent MLflow experiment runs with key metrics."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        exp    = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if exp is None:
            return []
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=limit,
            order_by=["start_time DESC"],
        )
        result = []
        for r in runs:
            result.append({
                "run_id":     r.info.run_id,
                "run_name":   r.info.run_name,
                "status":     r.info.status,
                "start_time": r.info.start_time,
                "metrics":    r.data.metrics,
                "params":     r.data.params,
                "tags":       r.data.tags,
            })
        return result
    except Exception as exc:
        logger.warning("Could not fetch MLflow runs: %s", exc)
        return []


# ── Background task ────────────────────────────────────────────────────────

def _run_stage(job_id: str, req: PipelineTriggerRequest) -> None:
    """Execute a single DVC-equivalent pipeline stage in the background."""
    _jobs[job_id]["status"] = "running"
    t0 = time.time()
    try:
        if req.stage == PipelineStage.ingest:
            from backend.core.data.ingestion import StockDataIngester
            ing = StockDataIngester()
            ing.ingest(req.ticker, period=req.period)

        elif req.stage == PipelineStage.preprocess:
            from backend.core.data.ingestion     import StockDataIngester
            from backend.core.data.features      import FeatureEngineer
            from backend.core.data.preprocessing import DataPreprocessor
            raw  = StockDataIngester().load(req.ticker)
            feat = FeatureEngineer().build(raw, req.ticker)
            DataPreprocessor().fit_transform(feat, req.ticker)

        elif req.stage == PipelineStage.train:
            from backend.core.training.trainer import Trainer
            Trainer(ticker=req.ticker, model_type=req.model_type.value).run()

        elif req.stage == PipelineStage.evaluate:
            logger.info("Evaluate stage: metrics already saved by trainer.")

        prom.PIPELINE_RUNS.labels(stage=req.stage.value, status="success").inc()
        prom.PIPELINE_LAST_RUN.labels(stage=req.stage.value).set(time.time())
        _jobs[job_id].update({"status": "success", "duration_s": time.time() - t0})

    except Exception as exc:
        logger.exception("Stage %s failed for job %s", req.stage.value, job_id)
        prom.PIPELINE_RUNS.labels(stage=req.stage.value, status="failure").inc()
        _jobs[job_id].update({"status": "failed", "error": str(exc)})
