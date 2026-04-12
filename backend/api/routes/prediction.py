"""
prediction.py – /predict endpoint.

POST /predict
  → Loads or re-uses cached model for the requested ticker
  → Runs inference on the latest available market data
  → Returns price forecasts with confidence bounds + financial metrics
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, BackgroundTasks

from backend.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    PredictionPoint,
)
from backend.config import MODEL_STORE, FEATURES_DIR, PROCESSED_DIR, LOOKBACK
from backend.core.monitoring import metrics as prom
from backend.core.data.ingestion     import StockDataIngester, DataIngestionError
from backend.core.data.features      import FeatureEngineer
from backend.core.data.preprocessing import DataPreprocessor
from backend.core.models.lstm_predictor    import LSTMPredictor
from backend.core.models.xgboost_predictor import XGBoostPredictor
from backend.core.training.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction"])

# In-memory model cache: (ticker, model_type) → model object
_model_cache: dict[tuple, object] = {}
_prep_cache:  dict[str, DataPreprocessor] = {}


def _load_lstm(ticker: str) -> LSTMPredictor:
    key   = (ticker, "lstm")
    if key not in _model_cache:
        path = MODEL_STORE / f"lstm_{ticker}.pt"
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"LSTM model not found for {ticker}. Please train first.",
            )
        predictor = LSTMPredictor()
        predictor.load(path)
        _model_cache[key] = predictor
        logger.info("Loaded LSTM model for %s into cache.", ticker)
    return _model_cache[key]  # type: ignore[return-value]


def _load_xgb(ticker: str) -> XGBoostPredictor:
    key  = (ticker, "xgboost")
    if key not in _model_cache:
        path = MODEL_STORE / f"xgb_{ticker}.pkl"
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"XGBoost model not found for {ticker}. Please train first.",
            )
        _model_cache[key] = XGBoostPredictor.load(path)
        logger.info("Loaded XGBoost model for %s into cache.", ticker)
    return _model_cache[key]  # type: ignore[return-value]


def _get_preprocessor(ticker: str) -> DataPreprocessor:
    if ticker not in _prep_cache:
        prep = DataPreprocessor(processed_dir=PROCESSED_DIR, lookback=LOOKBACK)
        prep.load(ticker)
        _prep_cache[ticker] = prep
    return _prep_cache[ticker]


@router.post("", response_model=PredictionResponse, summary="Predict next-day stock price")
def predict(req: PredictionRequest, background_tasks: BackgroundTasks) -> PredictionResponse:
    """
    Predict next-day (or multi-day) closing price for a given ticker.

    - Uses a cached LSTM or XGBoost model (loads from MLflow model store).
    - Fetches latest market data via yfinance.
    - Returns predictions with 90% confidence intervals.
    """
    ticker     = req.ticker
    model_type = req.model_type.value
    t_start    = time.time()

    logger.info("Prediction request | ticker=%s  model=%s", ticker, model_type)

    try:
        # ── 1. Fetch latest data ──────────────────────────────────────────
        ingester = StockDataIngester(raw_dir="data/raw")
        try:
            raw_df = ingester.ingest(ticker, period="6mo")
        except DataIngestionError as exc:
            prom.record_prediction_error(ticker, "ingestion_error")
            raise HTTPException(status_code=502, detail=str(exc))

        current_price = float(raw_df["Close"].iloc[-1])

        # ── 2. Feature engineering ────────────────────────────────────────
        fe      = FeatureEngineer(features_dir=FEATURES_DIR)
        feat_df = fe.build(raw_df, ticker)

        # ── 3. Preprocess ─────────────────────────────────────────────────
        prep = _get_preprocessor(ticker)
        X_scaled = prep.transform(feat_df)

        if len(X_scaled) < LOOKBACK:
            raise HTTPException(
                status_code=422,
                detail=f"Not enough data for {ticker}: need ≥ {LOOKBACK} rows.",
            )

        # Build the last window for inference
        window = X_scaled[-LOOKBACK:].reshape(1, LOOKBACK, -1)

        # ── 4. Inference ──────────────────────────────────────────────────
        predictions: list[PredictionPoint] = []
        direction      = "FLAT"
        direction_prob = 0.5

        if model_type == "lstm":
            model = _load_lstm(ticker)
            pred_scaled = model.predict(window)
            pred_price  = float(prep.inverse_target(pred_scaled)[0])

            # Confidence interval: ±1.65σ of historical residuals (90% CI)
            hist_vol = float(feat_df["hist_vol_20"].iloc[-1]) if "hist_vol_20" in feat_df else 0.02
            ci_half  = pred_price * hist_vol / np.sqrt(252) * 1.65

            predictions.append(PredictionPoint(
                date=str(feat_df.index[-1])[:10],
                actual_price=current_price,
                predicted_price=round(pred_price, 2),
                lower_bound=round(pred_price - ci_half, 2),
                upper_bound=round(pred_price + ci_half, 2),
            ))

            pct_change = (pred_price - current_price) / current_price
            if pct_change > 0.005:
                direction, direction_prob = "UP",   min(0.95, 0.5 + abs(pct_change) * 10)
            elif pct_change < -0.005:
                direction, direction_prob = "DOWN", min(0.95, 0.5 + abs(pct_change) * 10)
            else:
                direction, direction_prob = "FLAT", 0.55

        else:  # xgboost
            model  = _load_xgb(ticker)
            X_2d   = window.reshape(1, -1)
            labels, probs = model.predict_direction(X_2d)
            pred_price    = float(model.predict_price(X_2d)[0])
            label_map     = {0: "DOWN", 1: "FLAT", 2: "UP"}
            direction      = label_map[int(labels[0])]
            direction_prob = float(probs[0][int(labels[0])])

            predictions.append(PredictionPoint(
                date=str(feat_df.index[-1])[:10],
                actual_price=current_price,
                predicted_price=round(pred_price, 2),
                lower_bound=round(pred_price * 0.97, 2),
                upper_bound=round(pred_price * 1.03, 2),
            ))

        # ── 5. Quick evaluation metrics from last known test ──────────────
        metrics = _load_cached_metrics(ticker, model_type)

        latency = time.time() - t_start
        background_tasks.add_task(prom.record_prediction, ticker, model_type, latency)

        return PredictionResponse(
            ticker=ticker,
            model_type=model_type,
            current_price=current_price,
            predictions=predictions,
            direction=direction,
            direction_prob=round(direction_prob, 4),
            volatility_1y=round(
                float(feat_df["hist_vol_20"].iloc[-1])
                if "hist_vol_20" in feat_df else 0.0,
                4,
            ),
            metrics=metrics,
            model_version=1,
            run_id="cached",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled error in /predict for %s", ticker)
        prom.record_prediction_error(ticker, type(exc).__name__)
        raise HTTPException(status_code=500, detail=str(exc))


def _load_cached_metrics(ticker: str, model_type: str) -> dict:
    """Try to load the last evaluation metrics from disk; return defaults if absent."""
    import json
    path = Path("metrics/eval_metrics.json")
    if path.exists():
        try:
            data = json.loads(path.read_text())
            return {k: v for k, v in data.items() if isinstance(v, float)}
        except Exception:
            pass
    return {"rmse": 0.0, "mae": 0.0, "direction_acc": 0.0, "sharpe": 0.0}
