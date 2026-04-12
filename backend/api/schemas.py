"""
schemas.py – Pydantic request/response models for all API endpoints.

These schemas form the LLD contract between the frontend and backend.
Every endpoint's I/O is fully typed and validated.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ──────────────────────────────────────────────────────────────────

class ModelType(str, Enum):
    lstm    = "lstm"
    xgboost = "xgboost"

class HedgeAction(str, Enum):
    hedge_short = "HEDGE_SHORT"
    hedge_long  = "HEDGE_LONG"
    hold        = "HOLD"

class PipelineStage(str, Enum):
    ingest     = "ingest"
    preprocess = "preprocess"
    train      = "train"
    evaluate   = "evaluate"

class PipelineStatus(str, Enum):
    idle    = "idle"
    running = "running"
    success = "success"
    failed  = "failed"


# ── Health ─────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:  str = Field(..., example="ok")
    version: str = Field(..., example="1.0.0")
    uptime_seconds: float

class ReadyResponse(BaseModel):
    ready:  bool
    checks: dict[str, bool]   # mlflow, models_loaded, data_available


# ── Prediction ─────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    ticker:     str       = Field(..., example="AAPL", description="Yahoo Finance ticker symbol")
    model_type: ModelType = Field(ModelType.lstm, description="Model to use for prediction")
    horizon:    int       = Field(1, ge=1, le=5, description="Forecast horizon in trading days")

    @field_validator("ticker")
    @classmethod
    def ticker_upper(cls, v: str) -> str:
        return v.strip().upper()


class PredictionPoint(BaseModel):
    date:         str
    actual_price: Optional[float] = None
    predicted_price: float
    lower_bound:  float           # 90% confidence lower
    upper_bound:  float           # 90% confidence upper


class PredictionResponse(BaseModel):
    ticker:          str
    model_type:      str
    current_price:   float
    predictions:     list[PredictionPoint]
    direction:       str          # "UP" | "DOWN" | "FLAT"
    direction_prob:  float        # confidence of direction
    volatility_1y:   float        # annualised historical vol
    metrics:         dict[str, float]   # rmse, mae, direction_acc, sharpe
    model_version:   int
    run_id:          str


# ── Hedging ────────────────────────────────────────────────────────────────

class HedgeRequest(BaseModel):
    ticker:           str   = Field(..., example="AAPL")
    current_price:    float = Field(..., gt=0, description="Current stock price")
    position_size:    float = Field(100.0, gt=0, description="Number of shares held")
    predicted_return: float = Field(..., description="Expected 1-day return (from prediction)")
    time_fraction:    float = Field(0.0, ge=0.0, le=1.0,
                                    description="Fraction of hedge horizon elapsed")

    @field_validator("ticker")
    @classmethod
    def ticker_upper(cls, v: str) -> str:
        return v.strip().upper()


class HedgeResponse(BaseModel):
    ticker:          str
    action:          HedgeAction
    hedge_ratio:     float  = Field(..., description="Fraction of position to hedge (−1 to 1)")
    hedge_quantity:  float  = Field(..., description="Shares to short/buy as hedge")
    cvar_95:         float  = Field(..., description="Estimated 95% CVaR of unhedged position ($)")
    cost_estimate:   float  = Field(..., description="Estimated transaction cost of hedge ($)")
    rationale:       str
    delta_hedge_ref: float  = Field(..., description="Black–Scholes delta reference")


# ── Pipeline ───────────────────────────────────────────────────────────────

class PipelineTriggerRequest(BaseModel):
    ticker:     str       = Field(..., example="AAPL")
    stage:      PipelineStage = Field(PipelineStage.ingest)
    model_type: ModelType = Field(ModelType.lstm)
    period:     str       = Field("2y", description="yfinance period string")

    @field_validator("ticker")
    @classmethod
    def ticker_upper(cls, v: str) -> str:
        return v.strip().upper()


class StageInfo(BaseModel):
    stage:      str
    status:     PipelineStatus
    started_at: Optional[str] = None
    ended_at:   Optional[str] = None
    duration_s: Optional[float] = None
    message:    Optional[str] = None


class PipelineStatusResponse(BaseModel):
    ticker:      str
    overall:     PipelineStatus
    stages:      list[StageInfo]
    last_run_at: Optional[str]
    next_run_at: Optional[str]


class PipelineTriggerResponse(BaseModel):
    job_id:  str
    status:  str
    message: str


# ── Training ───────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    ticker:     str       = Field(..., example="AAPL")
    model_type: ModelType = Field(ModelType.lstm)
    epochs:     int       = Field(50, ge=1, le=500)
    batch_size: int       = Field(32, ge=8, le=256)
    lr:         float     = Field(0.001, gt=0, lt=1)
    period:     str       = Field("2y")

    @field_validator("ticker")
    @classmethod
    def ticker_upper(cls, v: str) -> str:
        return v.strip().upper()


class TrainResponse(BaseModel):
    run_id:       str
    model_name:   str
    version:      int
    metrics:      dict[str, float]
    model_path:   str
    duration_s:   float


# ── Monitoring ─────────────────────────────────────────────────────────────

class DriftReport(BaseModel):
    ticker:        str
    drift_detected: bool
    drifted_features: list[str]
    scores:        dict[str, float]   # feature → KL divergence
    checked_at:    str


class ModelMetricsResponse(BaseModel):
    ticker:        str
    model_type:    str
    version:       int
    rmse:          float
    mae:           float
    direction_acc: float
    sharpe:        float
    evaluated_at:  str
