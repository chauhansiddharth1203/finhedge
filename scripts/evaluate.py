"""
evaluate.py – DVC stage: model evaluation + metric logging.

Loads the latest model from the MLflow registry (or local file),
runs inference on the test set, and saves metrics + predictions CSV.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_uri", default=None)
    parser.add_argument("--ticker",    default=None)
    parser.add_argument("--params",    default="params.yaml")
    args = parser.parse_args()

    with open(args.params) as f:
        p = yaml.safe_load(f) or {}

    ticker   = args.ticker or p.get("data", {}).get("ticker", "AAPL")
    lookback = p.get("preprocessing", {}).get("lookback", 60)
    proc_dir = p.get("data", {}).get("processed_dir", "data/processed")
    feat_dir = p.get("data", {}).get("features_dir",  "data/features")

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from backend.core.data.features      import FeatureEngineer
    from backend.core.data.preprocessing import DataPreprocessor
    from backend.core.models.lstm_predictor import LSTMPredictor
    from backend.core.training.evaluator    import ModelEvaluator

    # Load preprocessor
    prep = DataPreprocessor(processed_dir=proc_dir, lookback=lookback)
    prep.load(ticker)

    # Load features
    fe      = FeatureEngineer(features_dir=feat_dir)
    feat_df = fe.load(ticker)

    # Rebuild test split
    splits = prep.fit_transform(feat_df, ticker)

    # Load model
    model_path = Path(f"models/lstm_{ticker}.pt")
    if not model_path.exists():
        logger.error("No model found at %s. Run train.py first.", model_path)
        sys.exit(1)

    predictor = LSTMPredictor()
    predictor.load(model_path)

    # Inference
    y_pred_scaled = predictor.predict(splits["X_test"])
    y_pred        = prep.inverse_target(y_pred_scaled)
    y_true        = prep.inverse_target(splits["y_test"])

    # Metrics
    evaluator = ModelEvaluator()
    metrics   = evaluator.compute(y_true, y_pred, splits["dates_test"])

    # Save as DVC metric
    out_dir = Path("metrics")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics saved: %s", metrics)

    # Save predictions CSV (DVC plot)
    df = pd.DataFrame({
        "date":      [str(d)[:10] for d in splits["dates_test"]],
        "actual":    y_true.tolist(),
        "predicted": y_pred.tolist(),
    })
    df.to_csv(out_dir / "predictions.csv", index=False)
    logger.info("Predictions saved to metrics/predictions.csv")


if __name__ == "__main__":
    main()
