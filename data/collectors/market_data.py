"""
Data Collector Module - V1
Efficient data collection from yfinance with caching and failover.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from functools import wraps
import time
import pickle
from pathlib import Path


class RateLimiter:
    """Token bucket rate limiter."""

    __slots__ = ("interval", "last_call")

    def __init__(self, calls_per_minute: int = 60):
        self.interval = 60.0 / calls_per_minute
        self.last_call = 0.0

    def wait(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_call = time.time()


def retry(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed calls with exponential backoff."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay * (2**attempt))
            return None

        return wrapper

    return decorator


class DataCollector:
    """
    Unified data collector with caching.

    Features:
    - YFinance as primary source (free, unlimited)
    - Local file caching for efficiency
    - Automatic retry with exponential backoff
    """

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.rate_limiter = RateLimiter(120)
        self._yf = None

    @property
    def yf(self):
        """Lazy load yfinance."""
        if self._yf is None:
            import yfinance

            self._yf = yfinance
        return self._yf

    def _cache_path(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> Path:
        """Generate cache file path."""
        key = f"{symbol}_{start.date()}_{end.date()}_{interval}"
        return self.cache_dir / f"{key}.pkl"

    def _load_cache(
        self, path: Path, max_age_hours: int = 24
    ) -> Optional[pd.DataFrame]:
        """Load cached data if fresh."""
        if not path.exists():
            return None

        mod_time = datetime.fromtimestamp(path.stat().st_mtime)
        if datetime.now() - mod_time > timedelta(hours=max_age_hours):
            return None

        with open(path, "rb") as f:
            return pickle.load(f)

    def _save_cache(self, path: Path, data: pd.DataFrame):
        """Save data to cache."""
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @retry(max_retries=3)
    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Stock ticker
            start: Start date
            end: End date
            interval: Data interval ('1d', '1h', '5m', etc.)
            use_cache: Whether to use file cache

        Returns:
            DataFrame with columns: open, high, low, close, volume, returns
        """
        cache_path = self._cache_path(symbol, start, end, interval)

        if use_cache:
            cached = self._load_cache(cache_path)
            if cached is not None:
                return cached

        self.rate_limiter.wait()

        ticker = self.yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval)

        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        # Standardize columns
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        # Keep only required columns
        df = df[["open", "high", "low", "close", "volume"]]

        # Calculate returns
        df["returns"] = df["close"].pct_change()

        if use_cache:
            self._save_cache(cache_path, df)

        return df

    def fetch_batch(
        self, symbols: List[str], start: datetime, end: datetime, interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.

        Args:
            symbols: List of tickers
            start: Start date
            end: End date
            interval: Data interval

        Returns:
            Dict mapping symbol to DataFrame
        """
        results = {}
        failed = []

        for symbol in symbols:
            try:
                df = self.fetch_ohlcv(symbol, start, end, interval)
                results[symbol] = df
            except Exception as e:
                failed.append((symbol, str(e)))

        if failed:
            print(f"Failed symbols: {failed}")

        return results

    def fetch_fundamentals(self, symbol: str) -> Dict:
        """
        Fetch fundamental data for a symbol.

        Args:
            symbol: Stock ticker

        Returns:
            Dict with fundamental metrics
        """
        self.rate_limiter.wait()

        ticker = self.yf.Ticker(symbol)
        info = ticker.info

        return {
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "profit_margins": info.get("profitMargins"),
            "return_on_equity": info.get("returnOnEquity"),
            "debt_to_equity": info.get("debtToEquity"),
            "beta": info.get("beta"),
            "dividend_yield": info.get("dividendYield"),
            "avg_volume": info.get("averageVolume"),
            "short_ratio": info.get("shortRatio"),
        }

    def get_sp500_symbols(self) -> List[str]:
        """
        Get S&P 500 component symbols.

        Returns:
            List of ticker symbols
        """
        try:
            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )
            return tables[0]["Symbol"].str.replace(".", "-").tolist()
        except Exception:
            # Fallback to major symbols
            return [
                "AAPL",
                "MSFT",
                "AMZN",
                "GOOGL",
                "META",
                "NVDA",
                "TSLA",
                "JPM",
                "V",
                "JNJ",
                "UNH",
                "XOM",
                "PG",
                "MA",
                "HD",
            ]
