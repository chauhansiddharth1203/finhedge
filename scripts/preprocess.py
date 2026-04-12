"""
preprocess.py – DVC stage: feature engineering + preprocessing.

Usage:
    python scripts/preprocess.py [--params params.yaml]
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f) or {}

    ticker    = params.get("data", {}).get("ticker",         "AAPL")
    raw_dir   = args.input  or params.get("data", {}).get("raw_dir",   "data/raw")
    proc_dir  = args.output or params.get("data", {}).get("processed_dir", "data/processed")
    feat_dir  = params.get("data", {}).get("features_dir",  "data/features")
    lookback  = params.get("preprocessing", {}).get("lookback",   60)
    test_split = params.get("preprocessing", {}).get("test_split",  0.2)
    val_split  = params.get("preprocessing", {}).get("val_split",   0.1)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from backend.core.data.ingestion     import StockDataIngester
    from backend.core.data.features      import FeatureEngineer
    from backend.core.data.preprocessing import DataPreprocessor
    from backend.core.data.validation    import DataValidator

    # Load raw data
    ingester = StockDataIngester(raw_dir=raw_dir)
    raw_df   = ingester.load(ticker)

    # Validate
    validator = DataValidator()
    report    = validator.validate_raw(raw_df, ticker)
    if not report.passed:
        logger.warning("Data validation issues: %s", report.details)

    # Feature engineering
    fe      = FeatureEngineer(features_dir=feat_dir)
    feat_df = fe.build(raw_df, ticker)

    # Validate features + drift
    feat_report = validator.validate_features(feat_df, ticker)

    # Preprocessing (scale + split)
    prep  = DataPreprocessor(
        processed_dir=proc_dir,
        lookback=lookback,
        test_split=test_split,
        val_split=val_split,
    )
    splits = prep.fit_transform(feat_df, ticker)

    # Save drift baseline as DVC metric
    import json
    baseline_src = Path(feat_dir) / f"{ticker}_drift_baseline.json"
    baseline_dst = Path("data/drift_baseline.json")
    if baseline_src.exists():
        baseline_dst.write_text(baseline_src.read_text())

    logger.info(
        "Preprocessing done: X_train=%s  X_val=%s  X_test=%s",
        splits["X_train"].shape,
        splits["X_val"].shape,
        splits["X_test"].shape,
    )


if __name__ == "__main__":
    main()
