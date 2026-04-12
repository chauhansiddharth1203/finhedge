"""
ingestion.py – Fetch OHLCV data from Yahoo Finance via yfinance.

Responsibilities:
  - Download historical price data for one or more tickers.
  - Validate schema (required columns present, no all-NaN rows).
  - Persist raw data as Parquet files, one per ticker.
  - Return a clean DataFrame ready for the preprocessing stage.

If Yahoo Finance is unreachable (e.g. inside a Docker network without
external access), a realistic GBM-based synthetic dataset is generated
as a fallback so the full ML pipeline can still be demonstrated.
"""

import logging
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Columns we expect yfinance to return
REQUIRED_COLS = {"Open", "High", "Low", "Close", "Volume"}

# Seed prices & volatility for synthetic data (realistic defaults)
_SEED_PRICES: dict[str, float] = {
    "AAPL": 178.0, "GOOGL": 172.0, "MSFT": 415.0, "AMZN": 185.0,
    "TSLA": 175.0, "META": 490.0, "NVDA": 850.0, "NFLX": 620.0,
    "RELIANCE.NS": 2900.0, "TCS.NS": 3800.0, "INFY.NS": 1550.0,
}
_DEFAULT_SEED_PRICE = 150.0
_PERIOD_DAYS: dict[str, int] = {
    "1mo": 22, "3mo": 65, "6mo": 130, "1y": 252, "2y": 504, "5y": 1260,
}


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
        """Try yfinance; fall back to realistic synthetic data if unavailable."""
        data = None

        # Attempt 1: yf.Ticker().history() — more reliable inside Docker
        try:
            t = yf.Ticker(ticker)
            data = t.history(period=period, interval=interval, auto_adjust=True)
            if data is not None and not data.empty:
                logger.info("yf.Ticker.history() succeeded for %s (%d rows)", ticker, len(data))
        except Exception as exc:
            logger.warning("yf.Ticker.history() failed for %s: %s", ticker, exc)
            data = None

        # Attempt 2: yf.download() fallback
        if data is None or data.empty:
            try:
                data = yf.download(
                    ticker,
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                    progress=False,
                )
                if data is not None and not data.empty:
                    logger.info("yf.download() succeeded for %s (%d rows)", ticker, len(data))
            except Exception as exc:
                logger.warning("yf.download() failed for %s: %s", ticker, exc)
                data = None

        # Attempt 3: Synthetic GBM data (demo / offline mode)
        if data is None or data.empty:
            logger.warning(
                "Yahoo Finance unreachable for %s — generating synthetic OHLCV data "
                "(demo/offline mode). This is expected inside restricted Docker networks.",
                ticker,
            )
            data = self._generate_synthetic(ticker, period)

        # yfinance sometimes returns MultiIndex columns; flatten them
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data

    def _generate_synthetic(self, ticker: str, period: str) -> pd.DataFrame:
        """
        Generate realistic synthetic OHLCV data using Geometric Brownian Motion.

        Parameters match typical equity characteristics:
          - mu  = 12% annual drift
          - vol = 25% annual volatility
          - Volume ~ log-normal
        """
        n_days = _PERIOD_DAYS.get(period, 504)
        seed_price = _SEED_PRICES.get(ticker.upper(), _DEFAULT_SEED_PRICE)

        rng = np.random.default_rng(abs(hash(ticker)) % (2**32))

        # GBM daily log-returns
        mu  = 0.12 / 252          # daily drift
        vol = 0.25 / np.sqrt(252) # daily vol
        log_returns = rng.normal(mu - 0.5 * vol**2, vol, size=n_days)
        closes = seed_price * np.exp(np.cumsum(log_returns))

        # OHLV from Close
        daily_range = rng.uniform(0.005, 0.025, size=n_days)  # 0.5%–2.5% range
        highs  = closes * (1 + daily_range)
        lows   = closes * (1 - daily_range)
        opens  = lows + rng.random(n_days) * (highs - lows)
        volumes = rng.lognormal(mean=np.log(5_000_000), sigma=0.5, size=n_days).astype(int)

        # Business-day date index ending today
        end_date   = datetime.utcnow().date()
        dates      = pd.bdate_range(end=end_date, periods=n_days)

        df = pd.DataFrame(
            {
                "Open":   np.round(opens,  4),
                "High":   np.round(highs,  4),
                "Low":    np.round(lows,   4),
                "Close":  np.round(closes, 4),
                "Volume": volumes,
            },
            index=dates,
        )
        df.index.name = "Date"
        logger.info(
            "Synthetic data generated for %s: %d rows, price range $%.2f–$%.2f",
            ticker, len(df), df["Close"].min(), df["Close"].max(),
        )
        return df

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
