"""
preprocessing.py – Scaling, sequence creation, and train/val/test splitting.

Pipeline:
  1. Scale all numeric features with MinMaxScaler (fitted on train only).
  2. Build sliding-window sequences of length `lookback` for LSTM.
  3. Flatten sequences into a 2-D feature matrix for XGBoost.
  4. Split chronologically into train / validation / test sets.
  5. Persist scalers and split indices to disk.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Transforms feature DataFrames into model-ready arrays."""

    # Columns used as input features (exclude raw OHLCV used only in features)
    FEATURE_COLS: list[str] = [
        "return_1d", "log_return_1d",
        "price_to_ma20", "price_to_ma50", "hl_ratio",
        "rsi_14",
        "macd", "macd_sig", "macd_hist",
        "roc_5", "roc_10",
        "bb_width", "bb_pct",
        "atr_14", "hist_vol_20",
        "vol_ma_ratio",
        "ret_mean_5", "ret_std_5",
        "ret_mean_10", "ret_std_10",
        "ret_mean_20", "ret_std_20",
    ]
    TARGET_COL = "Close"

    def __init__(
        self,
        processed_dir: str | Path = "data/processed",
        lookback: int = 60,
        test_split: float = 0.2,
        val_split: float = 0.1,
    ) -> None:
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.lookback   = lookback
        self.test_split = test_split
        self.val_split  = val_split

        self.feature_scaler: MinMaxScaler | None = None
        self.target_scaler:  MinMaxScaler | None = None

    # ── Public API ─────────────────────────────────────────────────────────

    def fit_transform(
        self, df: pd.DataFrame, ticker: str
    ) -> dict[str, np.ndarray]:
        """
        Full preprocessing pipeline for training.

        Returns dict with keys:
          X_train, y_train, X_val, y_val, X_test, y_test (3-D for LSTM)
          X_train_2d, X_val_2d, X_test_2d  (2-D for XGBoost)
          dates_train, dates_val, dates_test
        """
        logger.info("Preprocessing %s for training.", ticker)

        features, targets, dates = self._extract(df)
        splits = self._split_indices(len(features))

        # Fit scalers on train only
        train_idx = slice(0, splits["val_start"])
        self.feature_scaler = MinMaxScaler()
        self.target_scaler  = MinMaxScaler()

        features_scaled = self.feature_scaler.fit_transform(features)
        targets_scaled  = self.target_scaler.fit_transform(
            targets.reshape(-1, 1)
        ).ravel()

        # Build LSTM sequences
        X, y, seq_dates = self._build_sequences(
            features_scaled, targets_scaled, dates
        )
        result = self._apply_splits(X, y, seq_dates, splits)

        # 2-D version for XGBoost (flatten each window)
        result["X_train_2d"] = result["X_train"].reshape(len(result["X_train"]), -1)
        result["X_val_2d"]   = result["X_val"].reshape(len(result["X_val"]),   -1)
        result["X_test_2d"]  = result["X_test"].reshape(len(result["X_test"]),  -1)

        self._save(ticker)
        self._log_shapes(result)
        return result

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted scalers (inference path)."""
        if self.feature_scaler is None:
            raise RuntimeError("Scalers not fitted. Call fit_transform first.")
        features, _, _ = self._extract(df)
        return self.feature_scaler.transform(features)

    def inverse_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse-transform scaled target back to price."""
        return self.target_scaler.inverse_transform(
            y_scaled.reshape(-1, 1)
        ).ravel()

    def save(self, ticker: str) -> None:
        self._save(ticker)

    def load(self, ticker: str) -> None:
        """Load previously fitted scalers from disk."""
        feat_path = self.processed_dir / f"{ticker}_feature_scaler.pkl"
        tgt_path  = self.processed_dir / f"{ticker}_target_scaler.pkl"
        with open(feat_path, "rb") as f:
            self.feature_scaler = pickle.load(f)
        with open(tgt_path, "rb") as f:
            self.target_scaler = pickle.load(f)
        logger.info("Scalers loaded for %s.", ticker)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _extract(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pull feature matrix, target vector, and date array from df."""
        available = [c for c in self.FEATURE_COLS if c in df.columns]
        missing   = set(self.FEATURE_COLS) - set(available)
        if missing:
            logger.warning("Missing feature columns (will skip): %s", missing)

        features = df[available].values.astype(np.float32)
        targets  = df[self.TARGET_COL].values.astype(np.float32)
        dates    = df.index.values
        return features, targets, dates

    def _split_indices(self, n: int) -> dict[str, int]:
        """Return chronological split start indices."""
        test_start = int(n * (1 - self.test_split))
        val_start  = int(test_start * (1 - self.val_split))
        return {"val_start": val_start, "test_start": test_start}

    def _build_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sliding-window sequences of shape (N, lookback, n_features)."""
        lb = self.lookback
        Xs, ys, ds = [], [], []
        for i in range(lb, len(X)):
            Xs.append(X[i - lb : i])
            ys.append(y[i])
            ds.append(dates[i])
        return np.array(Xs), np.array(ys), np.array(ds)

    def _apply_splits(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: np.ndarray,
        splits: dict[str, int],
    ) -> dict[str, np.ndarray]:
        # Sequence indices are offset by lookback; adjust split points
        vs = splits["val_start"] - self.lookback
        ts = splits["test_start"] - self.lookback
        vs = max(0, vs)
        ts = max(vs, ts)
        return {
            "X_train":      X[:vs],       "y_train":      y[:vs],
            "X_val":        X[vs:ts],      "y_val":        y[vs:ts],
            "X_test":       X[ts:],        "y_test":       y[ts:],
            "dates_train":  dates[:vs],
            "dates_val":    dates[vs:ts],
            "dates_test":   dates[ts:],
        }

    def _save(self, ticker: str) -> None:
        feat_path = self.processed_dir / f"{ticker}_feature_scaler.pkl"
        tgt_path  = self.processed_dir / f"{ticker}_target_scaler.pkl"
        with open(feat_path, "wb") as f:
            pickle.dump(self.feature_scaler, f)
        with open(tgt_path, "wb") as f:
            pickle.dump(self.target_scaler, f)
        logger.info("Scalers saved to %s", self.processed_dir)

    def _log_shapes(self, result: dict[str, np.ndarray]) -> None:
        for split in ("train", "val", "test"):
            x = result[f"X_{split}"]
            y = result[f"y_{split}"]
            logger.info(
                "  %-6s  X=%s  y=%s", split, x.shape, y.shape
            )
