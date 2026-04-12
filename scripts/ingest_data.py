"""
ingest_data.py – DVC stage: data ingestion.

Usage (standalone):
    python scripts/ingest_data.py --ticker AAPL --period 2y

Usage (via DVC):
    dvc run -n ingest  (as defined in dvc.yaml)
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw OHLCV data from Yahoo Finance.")
    parser.add_argument("--ticker", default=None, help="Ticker symbol (overrides params.yaml)")
    parser.add_argument("--period", default=None, help="yfinance period (overrides params.yaml)")
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    # Load params
    params = {}
    if Path(args.params).exists():
        with open(args.params) as f:
            params = yaml.safe_load(f) or {}

    ticker  = args.ticker  or params.get("data", {}).get("ticker",  "AAPL")
    period  = args.period  or params.get("data", {}).get("period",  "2y")
    raw_dir = params.get("data", {}).get("raw_dir", "data/raw")

    logger.info("Ingesting %s | period=%s → %s", ticker, period, raw_dir)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from backend.core.data.ingestion import StockDataIngester, DataIngestionError

    ingester = StockDataIngester(raw_dir=raw_dir)
    try:
        df = ingester.ingest(ticker, period=period)
        logger.info("Ingestion complete: %d rows saved.", len(df))
    except DataIngestionError as exc:
        logger.error("Ingestion failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
