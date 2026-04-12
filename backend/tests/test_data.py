"""
test_data.py – Unit tests for data ingestion, features, preprocessing, validation.
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Minimal 200-row OHLCV DataFrame with DatetimeIndex."""
    np.random.seed(42)
    n   = 200
    idx = pd.date_range("2023-01-02", periods=n, freq="B")  # business days
    close = 150.0 + np.cumsum(np.random.randn(n) * 1.5)
    return pd.DataFrame({
        "Open":   close * (1 + np.random.uniform(-0.005, 0.005, n)),
        "High":   close * (1 + np.random.uniform(0.0, 0.01, n)),
        "Low":    close * (1 - np.random.uniform(0.0, 0.01, n)),
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 10_000_000, n).astype(float),
    }, index=idx)


# ── Ingestion ──────────────────────────────────────────────────────────────

class TestStockDataIngester:

    def test_validate_accepts_good_df(self, sample_ohlcv, tmp_path):
        from backend.core.data.ingestion import StockDataIngester
        ing = StockDataIngester(raw_dir=tmp_path)
        # Should not raise
        result = ing._validate(sample_ohlcv, "TEST")
        assert len(result) == len(sample_ohlcv)

    def test_validate_rejects_missing_column(self, sample_ohlcv, tmp_path):
        from backend.core.data.ingestion import StockDataIngester, DataIngestionError
        ing = StockDataIngester(raw_dir=tmp_path)
        bad_df = sample_ohlcv.drop(columns=["Volume"])
        with pytest.raises(DataIngestionError, match="Missing columns"):
            ing._validate(bad_df, "TEST")

    def test_validate_rejects_too_few_rows(self, tmp_path):
        from backend.core.data.ingestion import StockDataIngester, DataIngestionError
        ing = StockDataIngester(raw_dir=tmp_path)
        small_df = pd.DataFrame({
            "Open": [1], "High": [1], "Low": [1], "Close": [1], "Volume": [1],
        }, index=pd.date_range("2023-01-01", periods=1))
        with pytest.raises(DataIngestionError, match="Insufficient data"):
            ing._validate(small_df, "TEST")

    def test_clean_removes_nan_rows(self, sample_ohlcv, tmp_path):
        from backend.core.data.ingestion import StockDataIngester
        ing    = StockDataIngester(raw_dir=tmp_path)
        dirty  = sample_ohlcv.copy()
        dirty.iloc[5] = np.nan
        cleaned = ing._clean(dirty)
        assert cleaned.isna().sum().sum() == 0
        assert len(cleaned) == len(sample_ohlcv) - 1

    def test_ingest_saves_parquet(self, sample_ohlcv, tmp_path):
        from backend.core.data.ingestion import StockDataIngester
        ing = StockDataIngester(raw_dir=tmp_path)

        with patch.object(ing, "_download", return_value=sample_ohlcv):
            df = ing.ingest("TEST", period="2y")

        assert (tmp_path / "TEST.parquet").exists()
        assert len(df) == len(sample_ohlcv)


# ── Feature engineering ────────────────────────────────────────────────────

class TestFeatureEngineer:

    def test_build_produces_expected_columns(self, sample_ohlcv, tmp_path):
        from backend.core.data.features import FeatureEngineer
        fe  = FeatureEngineer(features_dir=tmp_path)
        out = fe.build(sample_ohlcv, "TEST")
        for col in ["rsi_14", "macd", "bb_width", "atr_14", "obv", "hist_vol_20"]:
            assert col in out.columns, f"Missing column: {col}"

    def test_rsi_within_bounds(self, sample_ohlcv, tmp_path):
        from backend.core.data.features import FeatureEngineer
        fe  = FeatureEngineer(features_dir=tmp_path)
        out = fe.build(sample_ohlcv, "TEST")
        rsi = out["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_no_nan_after_build(self, sample_ohlcv, tmp_path):
        from backend.core.data.features import FeatureEngineer
        fe  = FeatureEngineer(features_dir=tmp_path)
        out = fe.build(sample_ohlcv, "TEST")
        assert out.isna().sum().sum() == 0

    def test_drift_baseline_saved(self, sample_ohlcv, tmp_path):
        from backend.core.data.features import FeatureEngineer
        fe = FeatureEngineer(features_dir=tmp_path)
        fe.build(sample_ohlcv, "TEST")
        baseline_path = tmp_path / "TEST_drift_baseline.json"
        assert baseline_path.exists()
        data = json.loads(baseline_path.read_text())
        assert "Close" in data
        assert "mean" in data["Close"]


# ── Preprocessing ──────────────────────────────────────────────────────────

class TestDataPreprocessor:

    @pytest.fixture
    def feature_df(self, sample_ohlcv, tmp_path):
        from backend.core.data.features import FeatureEngineer
        return FeatureEngineer(features_dir=tmp_path).build(sample_ohlcv, "TEST")

    def test_fit_transform_shapes(self, feature_df, tmp_path):
        from backend.core.data.preprocessing import DataPreprocessor
        prep   = DataPreprocessor(processed_dir=tmp_path, lookback=30)
        splits = prep.fit_transform(feature_df, "TEST")

        # 3-D for LSTM
        assert splits["X_train"].ndim == 3
        assert splits["X_train"].shape[1] == 30   # lookback
        # 2-D for XGBoost
        assert splits["X_train_2d"].ndim == 2

    def test_no_future_leakage(self, feature_df, tmp_path):
        """Test that the last train index is before the first val index."""
        from backend.core.data.preprocessing import DataPreprocessor
        prep   = DataPreprocessor(processed_dir=tmp_path, lookback=30)
        splits = prep.fit_transform(feature_df, "TEST")
        if len(splits["dates_train"]) > 0 and len(splits["dates_val"]) > 0:
            assert splits["dates_train"][-1] < splits["dates_val"][0]

    def test_scalers_persisted(self, feature_df, tmp_path):
        from backend.core.data.preprocessing import DataPreprocessor
        prep = DataPreprocessor(processed_dir=tmp_path, lookback=30)
        prep.fit_transform(feature_df, "TEST")
        assert (tmp_path / "TEST_feature_scaler.pkl").exists()
        assert (tmp_path / "TEST_target_scaler.pkl").exists()

    def test_inverse_transform_consistent(self, feature_df, tmp_path):
        from backend.core.data.preprocessing import DataPreprocessor
        prep   = DataPreprocessor(processed_dir=tmp_path, lookback=30)
        splits = prep.fit_transform(feature_df, "TEST")
        y_test = splits["y_test"]
        y_inv  = prep.inverse_target(y_test)
        assert y_inv.shape == y_test.shape
        assert (y_inv > 0).all()   # prices should be positive


# ── Validation ─────────────────────────────────────────────────────────────

class TestDataValidator:

    def test_valid_df_passes(self, sample_ohlcv):
        from backend.core.data.validation import DataValidator
        v      = DataValidator()
        report = v.validate_raw(sample_ohlcv, "TEST")
        assert report.schema_ok
        assert report.nan_ok

    def test_missing_column_fails(self, sample_ohlcv):
        from backend.core.data.validation import DataValidator
        v    = DataValidator()
        bad  = sample_ohlcv.drop(columns=["Close"])
        rep  = v.validate_raw(bad, "TEST")
        assert not rep.schema_ok

    def test_high_nan_fraction_fails(self, sample_ohlcv):
        from backend.core.data.validation import DataValidator
        v   = DataValidator()
        bad = sample_ohlcv.copy()
        bad.loc[bad.index[:150], "Close"] = np.nan
        rep = v.validate_raw(bad, "TEST")
        assert not rep.nan_ok
