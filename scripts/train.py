"""
train.py – DVC stage: model training with MLflow tracking.

Usage:
    python scripts/train.py [--model lstm] [--ticker AAPL]
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
    parser.add_argument("--model",      default=None)
    parser.add_argument("--ticker",     default=None)
    parser.add_argument("--lookback",   type=int,   default=None)
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--params",     default="params.yaml")
    args = parser.parse_args()

    with open(args.params) as f:
        p = yaml.safe_load(f) or {}

    ticker     = args.ticker     or p.get("data",  {}).get("ticker",     "AAPL")
    model      = args.model      or p.get("train", {}).get("model",      "lstm")
    lookback   = args.lookback   or p.get("preprocessing", {}).get("lookback", 60)
    epochs     = args.epochs     or p.get("train", {}).get("epochs",     50)
    batch_size = args.batch_size or p.get("train", {}).get("batch_size", 32)
    lr         = args.lr         or p.get("train", {}).get("lr",         0.001)
    period     = p.get("data",   {}).get("period",     "2y")

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from backend.core.training.trainer import Trainer

    trainer = Trainer(
        ticker=ticker,
        model_type=model,
        lookback=lookback,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        period=period,
        model_dir="models",
    )

    result = trainer.run()
    logger.info(
        "Training complete | run_id=%s | metrics=%s",
        result["run_id"][:8],
        result["metrics"],
    )


if __name__ == "__main__":
    main()
