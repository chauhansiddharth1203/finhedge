"""
features.py – Technical indicator feature engineering for stock data.

Features computed:
  Price-based : Returns, Log-Returns, Price-to-MA ratios
  Momentum    : RSI(14), MACD, MACD Signal, MACD Histogram
  Volatility  : Bollinger Bands (upper/lower/width), ATR(14)
  Volume      : OBV, Volume MA ratio
  Statistical : Rolling mean / std of returns (5, 10, 20 days)

Also computes and saves drift baseline statistics (mean, std, min, max,
percentiles) required for monitoring.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Computes technical indicators and packages them as a feature DataFrame."""

    def __init__(self, features_dir: str | Path = "data/features") -> None:
        self.features_dir = Path(features_dir)
        self.features_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────

    def build(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Generate all features from a raw OHLCV DataFrame.

        Parameters
        ----------
        df     : OHLCV DataFrame with DatetimeIndex.
        ticker : Ticker symbol (used for file naming).

        Returns
        -------
        pd.DataFrame with original OHLCV columns plus feature columns.
        """
        logger.info("Engineering features for %s (%d rows)", ticker, len(df))

        out = df.copy()
        out = self._price_features(out)
        out = self._momentum_features(out)
        out = self._volatility_features(out)
        out = self._volume_features(out)
        out = self._rolling_stats(out)

        # Drop rows where any feature is NaN (warm-up period)
        before = len(out)
        out.dropna(inplace=True)
        logger.info(
            "Dropped %d warm-up rows; %d rows remain after feature engineering.",
            before - len(out),
            len(out),
        )

        # Persist
        out_path = self.features_dir / f"{ticker}_features.parquet"
        out.to_parquet(out_path)
        logger.info("Features saved → %s", out_path)

        self._save_drift_baseline(out, ticker)
        return out

    def load(self, ticker: str) -> pd.DataFrame:
        """Load previously computed features from Parquet."""
        path = self.features_dir / f"{ticker}_features.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")
        return pd.read_parquet(path)

    # ── Feature groups ─────────────────────────────────────────────────────

    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["return_1d"]      = df["Close"].pct_change()
        df["log_return_1d"]  = np.log(df["Close"] / df["Close"].shift(1))
        df["ma_5"]           = df["Close"].rolling(5).mean()
        df["ma_20"]          = df["Close"].rolling(20).mean()
        df["ma_50"]          = df["Close"].rolling(50).mean()
        df["price_to_ma20"]  = df["Close"] / df["ma_20"]
        df["price_to_ma50"]  = df["Close"] / df["ma_50"]
        df["hl_ratio"]       = (df["High"] - df["Low"]) / df["Close"]   # intraday range
        return df

    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # RSI(14)
        delta  = df["Close"].diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rs     = gain / (loss + 1e-9)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # MACD (12-period EMA – 26-period EMA) + Signal (9-period EMA)
        ema12           = df["Close"].ewm(span=12, adjust=False).mean()
        ema26           = df["Close"].ewm(span=26, adjust=False).mean()
        df["macd"]      = ema12 - ema26
        df["macd_sig"]  = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_sig"]

        # Rate of change
        df["roc_5"]  = df["Close"].pct_change(5)
        df["roc_10"] = df["Close"].pct_change(10)
        return df

    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Bollinger Bands (20-day, 2-sigma)
        roll20          = df["Close"].rolling(20)
        df["bb_mid"]    = roll20.mean()
        bb_std          = roll20.std()
        df["bb_upper"]  = df["bb_mid"] + 2 * bb_std
        df["bb_lower"]  = df["bb_mid"] - 2 * bb_std
        df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_pct"]    = (df["Close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"] + 1e-9
        )

        # ATR(14) – Average True Range
        tr1 = df["High"] - df["Low"]
        tr2 = (df["High"] - df["Close"].shift(1)).abs()
        tr3 = (df["Low"]  - df["Close"].shift(1)).abs()
        df["atr_14"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

        # Historical volatility (20-day)
        df["hist_vol_20"] = df["log_return_1d"].rolling(20).std() * np.sqrt(252)
        return df

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # On-Balance Volume
        sign       = np.sign(df["Close"].diff().fillna(0))
        df["obv"]  = (sign * df["Volume"]).cumsum()

        # Volume moving average ratio
        df["vol_ma_20"]     = df["Volume"].rolling(20).mean()
        df["vol_ma_ratio"]  = df["Volume"] / (df["vol_ma_20"] + 1e-9)
        return df

    def _rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        for w in [5, 10, 20]:
            df[f"ret_mean_{w}"] = df["return_1d"].rolling(w).mean()
            df[f"ret_std_{w}"]  = df["return_1d"].rolling(w).std()
        return df

    # ── Drift baseline ─────────────────────────────────────────────────────

    def _save_drift_baseline(self, df: pd.DataFrame, ticker: str) -> None:
        """
        Persist mean / std / percentiles for every numeric feature column.
        These baselines are compared against live data to detect drift.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        baseline: dict[str, dict] = {}
        for col in numeric_cols:
            s = df[col].dropna()
            baseline[col] = {
                "mean":  float(s.mean()),
                "std":   float(s.std()),
                "min":   float(s.min()),
                "max":   float(s.max()),
                "p5":    float(s.quantile(0.05)),
                "p25":   float(s.quantile(0.25)),
                "p50":   float(s.quantile(0.50)),
                "p75":   float(s.quantile(0.75)),
                "p95":   float(s.quantile(0.95)),
            }

        path = self.features_dir / f"{ticker}_drift_baseline.json"
        path.write_text(json.dumps(baseline, indent=2))
        logger.info("Drift baseline saved → %s  (%d features)", path, len(baseline))
