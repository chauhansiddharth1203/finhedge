"""
ingestion.py – Fetch OHLCV data from Yahoo Finance via yfinance.

Responsibilities:
  - Download historical price data for one or more tickers.
  - Validate schema (required columns present, no all-NaN rows).
  - Persist raw data as Parquet files, one per ticker.
  - Return a clean DataFrame ready for the preprocessing stage.
"""

import logging
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Columns we expect yfinance to return
REQUIRED_COLS = {"Open", "High", "Low", "Close", "Volume"}


class DataIngestionError(Exception):
    """Raised when data ingestion fails."""


class StockDataIngester:
    """Downloads and validates OHLCV data from Yahoo Finance."""

    def __init__(self, raw_dir: str | Path = "data/raw") -> None:
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────

    def ingest(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Download data for *ticker*, validate, and write to Parquet.

        Parameters
        ----------
        ticker   : Yahoo Finance ticker symbol (e.g. "AAPL").
        period   : Data period string accepted by yfinance ("1y", "2y", …).
        interval : Bar interval ("1d", "1wk", …).

        Returns
        -------
        pd.DataFrame with DatetimeIndex and OHLCV columns.

        Raises
        ------
        DataIngestionError if download fails or schema is invalid.
        """
        logger.info("Ingesting %s | period=%s interval=%s", ticker, period, interval)

        df = self._download(ticker, period, interval)
        df = self._validate(df, ticker)
        df = self._clean(df)

        out_path = self.raw_dir / f"{ticker}.parquet"
        df.to_parquet(out_path)
        logger.info("Saved raw data → %s  (%d rows)", out_path, len(df))

        self._write_manifest(ticker, period, interval, len(df))
        return df

    def ingest_many(
        self,
        tickers: list[str],
        period: str = "2y",
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Ingest multiple tickers; returns {ticker: DataFrame}."""
        results: dict[str, pd.DataFrame] = {}
        for t in tickers:
            try:
                results[t] = self.ingest(t, period, interval)
            except DataIngestionError as exc:
                logger.error("Skipping %s: %s", t, exc)
        return results

    def load(self, ticker: str) -> pd.DataFrame:
        """Load previously ingested raw data from Parquet."""
        path = self.raw_dir / f"{ticker}.parquet"
        if not path.exists():
            raise DataIngestionError(f"No raw data found for {ticker} at {path}")
        return pd.read_parquet(path)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _download(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        """Call yfinance and return a raw DataFrame."""
        try:
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
            )
        except Exception as exc:
            raise DataIngestionError(f"yfinance download failed for {ticker}: {exc}") from exc

        if data is None or data.empty:
            raise DataIngestionError(f"yfinance returned empty data for {ticker}")

        # yfinance sometimes returns MultiIndex columns; flatten them
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data

    def _validate(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Check required columns and sufficient row count."""
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise DataIngestionError(
                f"Missing columns for {ticker}: {missing}. Got: {list(df.columns)}"
            )

        all_nan_rows = df[list(REQUIRED_COLS)].isnull().all(axis=1).sum()
        if all_nan_rows > len(df) * 0.5:
            raise DataIngestionError(
                f"More than 50% rows are all-NaN for {ticker}; data may be unavailable."
            )

        if len(df) < 60:
            raise DataIngestionError(
                f"Insufficient data for {ticker}: only {len(df)} rows (need ≥ 60)."
            )

        logger.info("%s validated: %d rows, %d all-NaN rows", ticker, len(df), all_nan_rows)
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep OHLCV columns, sort by date, drop NaN rows."""
        df = df[list(REQUIRED_COLS)].copy()
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        before = len(df)
        df.dropna(inplace=True)
        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %d NaN rows during cleaning.", dropped)
        return df

    def _write_manifest(
        self,
        ticker: str,
        period: str,
        interval: str,
        n_rows: int,
    ) -> None:
        """Write a JSON manifest so downstream stages know provenance."""
        manifest = {
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "n_rows": n_rows,
            "ingested_at": datetime.utcnow().isoformat() + "Z",
        }
        manifest_path = self.raw_dir / f"{ticker}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        logger.debug("Manifest written → %s", manifest_path)
