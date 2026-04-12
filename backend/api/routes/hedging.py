"""
hedging.py – /hedge endpoint.

POST /hedge
  → Accepts portfolio state + predicted return
  → Returns hedge ratio, recommended action, CVaR estimate, cost estimate
"""

import logging
import time
from pathlib import Path

import torch
from fastapi import APIRouter, HTTPException, BackgroundTasks

from backend.api.schemas import HedgeRequest, HedgeResponse, HedgeAction
from backend.config import MODEL_STORE, CVAR_CONFIDENCE, COST_RATE
from backend.core.models.deep_hedger import DeepHedger, black_scholes_delta
from backend.core.monitoring import metrics as prom

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/hedge", tags=["Hedging"])

# In-memory hedger cache per ticker
_hedger_cache: dict[str, DeepHedger] = {}


def _load_hedger(ticker: str) -> DeepHedger:
    if ticker not in _hedger_cache:
        path = MODEL_STORE / f"hedger_{ticker}.pt"
        hedger = DeepHedger(cost_rate=COST_RATE, alpha=CVAR_CONFIDENCE)
        if path.exists():
            hedger.load(path)
            logger.info("Loaded DeepHedger for %s from %s", ticker, path)
        else:
            # Train on simulated data with default params on-the-fly (fast)
            logger.warning(
                "No hedger found for %s. Training on default GBM simulation.", ticker
            )
            hedger.train(epochs=100)   # fast default train
            hedger.save(path)
        _hedger_cache[ticker] = hedger
    return _hedger_cache[ticker]


@router.post("", response_model=HedgeResponse, summary="Generate hedge recommendation")
def hedge(req: HedgeRequest, background_tasks: BackgroundTasks) -> HedgeResponse:
    """
    Compute an optimal hedge recommendation for the given portfolio position.

    Uses a pre-trained CVaR-minimising deep hedging policy.
    Also computes the Black-Scholes delta hedge as a reference baseline.
    """
    ticker = req.ticker
    t_start = time.time()

    logger.info(
        "Hedge request | ticker=%s  price=%.2f  pos=%.0f  pred_ret=%.4f",
        ticker, req.current_price, req.position_size, req.predicted_return,
    )

    try:
        hedger = _load_hedger(ticker)

        # Approximate S0 as today's price adjusted back by predicted return
        s0 = req.current_price / (1 + req.predicted_return + 1e-9)

        rec = hedger.recommend(
            current_price=req.current_price,
            initial_price=s0,
            time_fraction=req.time_fraction,
            predicted_return=req.predicted_return,
            current_position=req.position_size,
        )

        # Delta hedge reference (ATM call, 21-day horizon, implied vol proxy)
        hist_vol   = abs(req.predicted_return) * (252 ** 0.5) + 0.15  # rough estimate
        delta_ref  = black_scholes_delta(
            S=req.current_price,
            K=req.current_price,   # ATM
            T=21 / 252,
            sigma=hist_vol,
        )

        cost_est = (
            abs(rec["hedge_quantity"])
            * req.current_price
            * COST_RATE
        )

        action_map = {
            "HEDGE_SHORT": HedgeAction.hedge_short,
            "HEDGE_LONG":  HedgeAction.hedge_long,
            "HOLD":        HedgeAction.hold,
        }
        action = action_map.get(rec["action"], HedgeAction.hold)

        latency = time.time() - t_start
        background_tasks.add_task(prom.record_hedge, ticker, rec["action"], latency)

        return HedgeResponse(
            ticker=ticker,
            action=action,
            hedge_ratio=rec["hedge_ratio"],
            hedge_quantity=rec["hedge_quantity"],
            cvar_95=rec["cvar_estimate"],
            cost_estimate=round(cost_est, 4),
            rationale=rec["rationale"],
            delta_hedge_ref=round(delta_ref, 4),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error in /hedge for %s", ticker)
        raise HTTPException(status_code=500, detail=str(exc))
