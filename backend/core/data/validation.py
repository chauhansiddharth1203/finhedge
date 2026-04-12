"""
validation.py – Automated data quality checks for ingested stock data.

Checks performed:
  1. Schema – required columns present and correct dtype.
  2. Missing values – fraction of NaNs per column.
  3. Outliers – prices / volumes outside ±5-sigma bounds.
  4. Data drift – KL-divergence of current feature distributions vs. baseline.
  5. Temporal gaps – unexpected missing trading days.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy

logger = logging.getLogger(__name__)

REQUIRED_OHLCV = ["Open", "High", "Low", "Close", "Volume"]
MAX_NAN_FRACTION = 0.05
OUTLIER_SIGMA    = 5.0
KL_DRIFT_THRESHOLD = 0.05   # nats


@dataclass
class ValidationReport:
    ticker: str
    passed: bool = True
    schema_ok:    bool = True
    nan_ok:       bool = True
    outlier_ok:   bool = True
    drift_ok:     bool = True
    temporal_ok:  bool = True
    details:      dict = field(default_factory=dict)

    def fail(self, check: str, detail: str) -> None:
        self.passed = False
        setattr(self, f"{check}_ok", False)
        self.details.setdefault(check, []).append(detail)
        logger.warning("[%s] Validation FAIL – %s: %s", self.ticker, check, detail)

    def warn(self, check: str, detail: str) -> None:
        self.details.setdefault(f"{check}_warning", []).append(detail)
        logger.warning("[%s] Validation WARNING – %s: %s", self.ticker, check, detail)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return f"[{self.ticker}] Validation {status} | {self.details}"


class DataValidator:
    """Runs schema, quality, and drift checks on stock DataFrames."""

    def __init__(self, baseline_dir: Optional[str | Path] = None) -> None:
        self.baseline_dir = Path(baseline_dir) if baseline_dir else None

    # ── Public API ─────────────────────────────────────────────────────────

    def validate_raw(self, df: pd.DataFrame, ticker: str) -> ValidationReport:
        """Validate a raw OHLCV DataFrame (post-ingestion)."""
        report = ValidationReport(ticker=ticker)
        self._check_schema(df, report)
        self._check_nan(df, report)
        self._check_outliers(df, report)
        self._check_temporal(df, report)
        logger.info(report.summary())
        return report

    def validate_features(
        self,
        df: pd.DataFrame,
        ticker: str,
        baseline_path: Optional[Path] = None,
    ) -> ValidationReport:
        """Validate engineered features and check for drift vs. baseline."""
        report = ValidationReport(ticker=ticker)
        self._check_nan(df, report)

        bpath = baseline_path or (
            self.baseline_dir / f"{ticker}_drift_baseline.json"
            if self.baseline_dir
            else None
        )
        if bpath and Path(bpath).exists():
            self._check_drift(df, bpath, report)
        else:
            report.warn("drift", "No baseline found – drift check skipped.")

        logger.info(report.summary())
        return report

    # ── Checks ─────────────────────────────────────────────────────────────

    def _check_schema(self, df: pd.DataFrame, report: ValidationReport) -> None:
        missing = [c for c in REQUIRED_OHLCV if c not in df.columns]
        if missing:
            report.fail("schema", f"Missing columns: {missing}")
            return
        # Numeric dtype check
        for col in REQUIRED_OHLCV:
            if not pd.api.types.is_numeric_dtype(df[col]):
                report.fail("schema", f"Column '{col}' is not numeric.")

    def _check_nan(self, df: pd.DataFrame, report: ValidationReport) -> None:
        for col in df.columns:
            frac = df[col].isna().mean()
            if frac > MAX_NAN_FRACTION:
                report.fail(
                    "nan",
                    f"Column '{col}' has {frac:.1%} NaN values (threshold {MAX_NAN_FRACTION:.0%}).",
                )

    def _check_outliers(self, df: pd.DataFrame, report: ValidationReport) -> None:
        for col in ["Close", "Volume"]:
            if col not in df.columns:
                continue
            s    = df[col].dropna()
            mean = s.mean()
            std  = s.std()
            n    = ((s - mean).abs() > OUTLIER_SIGMA * std).sum()
            if n > 0:
                report.warn(
                    "outlier",
                    f"Column '{col}' has {n} values beyond {OUTLIER_SIGMA}σ.",
                )

    def _check_temporal(self, df: pd.DataFrame, report: ValidationReport) -> None:
        if not isinstance(df.index, pd.DatetimeIndex):
            report.fail("temporal", "Index is not DatetimeIndex.")
            return
        # Check if index is monotonically increasing
        if not df.index.is_monotonic_increasing:
            report.fail("temporal", "Index is not sorted in ascending order.")
        # Detect very large gaps (> 5 calendar days, which would skip Mon–Fri)
        diffs = df.index.to_series().diff().dropna()
        large_gaps = diffs[diffs > pd.Timedelta(days=5)]
        if len(large_gaps) > 0:
            report.warn(
                "temporal",
                f"{len(large_gaps)} gaps > 5 calendar days detected.",
            )

    def _check_drift(
        self,
        df: pd.DataFrame,
        baseline_path: Path,
        report: ValidationReport,
    ) -> None:
        """Compare current feature distributions to baseline via KL-divergence."""
        with open(baseline_path) as f:
            baseline: dict = json.load(f)

        drifted: list[str] = []
        for col, stats in baseline.items():
            if col not in df.columns:
                continue
            current = df[col].dropna().values
            if len(current) < 10:
                continue

            # Build histograms over shared range
            lo = min(stats["min"], float(current.min()))
            hi = max(stats["max"], float(current.max()))
            bins = np.linspace(lo, hi + 1e-9, 20)

            cur_hist, _  = np.histogram(current, bins=bins, density=True)
            base_mean    = stats["mean"]
            base_std     = max(stats["std"], 1e-6)
            # Approximate baseline distribution as Gaussian evaluated at bin centres
            centres      = 0.5 * (bins[:-1] + bins[1:])
            base_dist    = np.exp(-0.5 * ((centres - base_mean) / base_std) ** 2)
            base_dist   /= base_dist.sum() + 1e-9
            cur_hist_n   = cur_hist / (cur_hist.sum() + 1e-9)

            # Symmetric KL divergence
            kl = float(entropy(cur_hist_n + 1e-9, base_dist + 1e-9))
            if kl > KL_DRIFT_THRESHOLD:
                drifted.append(f"{col} (KL={kl:.4f})")

        if drifted:
            report.fail(
                "drift",
                f"Data drift detected in: {', '.join(drifted)}",
            )
