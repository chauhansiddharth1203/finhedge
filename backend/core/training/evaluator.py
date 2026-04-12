"""
evaluator.py – Compute ML and financial performance metrics.

ML metrics:
  RMSE, MAE, MAPE, R²

Financial metrics:
  Direction Accuracy, Sharpe Ratio (annualised), Max Drawdown
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


class ModelEvaluator:
    """Evaluate regression predictions with ML and financial metrics."""

    def compute(
        self,
        y_true:   np.ndarray,
        y_pred:   np.ndarray,
        dates:    Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """
        Parameters
        ----------
        y_true : actual prices (unscaled)
        y_pred : predicted prices (unscaled)
        dates  : optional date array for time-aware metrics

        Returns
        -------
        dict of metric_name → value (all float, rounded to 6 dp)
        """
        metrics: dict[str, float] = {}

        # ── ML metrics ────────────────────────────────────────────────────
        residuals   = y_true - y_pred
        metrics["rmse"] = float(np.sqrt(np.mean(residuals ** 2)))
        metrics["mae"]  = float(np.mean(np.abs(residuals)))
        metrics["mape"] = float(
            np.mean(np.abs(residuals / (y_true + 1e-9))) * 100
        )
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        metrics["r2"] = float(1 - ss_res / (ss_tot + 1e-9))

        # ── Direction accuracy ─────────────────────────────────────────────
        actual_dir = np.diff(y_true) > 0
        pred_dir   = np.diff(y_pred) > 0
        if len(actual_dir) > 0:
            metrics["direction_acc"] = float(np.mean(actual_dir == pred_dir))

        # ── Financial metrics ──────────────────────────────────────────────
        # Simulated strategy: go long when model predicts price increase
        if len(y_pred) > 1:
            pred_returns   = np.diff(y_pred) / y_pred[:-1]
            actual_returns = np.diff(y_true) / y_true[:-1]
            # Strategy return = actual return when prediction is directionally correct
            strategy_ret = np.where(pred_dir, actual_returns, -actual_returns)

            metrics["sharpe"] = self._sharpe(strategy_ret)
            metrics["max_drawdown"] = self._max_drawdown(
                np.cumprod(1 + strategy_ret)
            )
            metrics["strategy_return"] = float(
                np.prod(1 + strategy_ret) - 1
            )
            metrics["buy_hold_return"] = float(
                (y_true[-1] - y_true[0]) / y_true[0]
            )

        self._log(metrics)
        return {k: round(v, 6) for k, v in metrics.items()}

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _sharpe(returns: np.ndarray, risk_free: float = 0.0) -> float:
        if len(returns) < 2:
            return 0.0
        excess = returns - risk_free / TRADING_DAYS
        std    = excess.std()
        if std < 1e-9:
            return 0.0
        return float(excess.mean() / std * np.sqrt(TRADING_DAYS))

    @staticmethod
    def _max_drawdown(cumulative: np.ndarray) -> float:
        peak = np.maximum.accumulate(cumulative)
        dd   = (cumulative - peak) / (peak + 1e-9)
        return float(dd.min())

    def _log(self, metrics: dict) -> None:
        for k, v in metrics.items():
            logger.info("  %-20s = %.4f", k, v)
