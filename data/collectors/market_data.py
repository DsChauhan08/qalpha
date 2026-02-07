"""
Data Collector Module - V1
Efficient data collection from yfinance with caching and failover.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from functools import wraps
import time
import pickle
from pathlib import Path

from quantum_alpha.data.storage.sqlite_cache import SQLiteCache
from quantum_alpha.data.storage.data_quality import DataQualityChecker
from quantum_alpha.data.storage.parquet_manager import ParquetManager

logger = logging.getLogger(__name__)

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

    def __init__(
        self,
        cache_dir: str = ".cache",
        use_sqlite_cache: bool = True,
        use_parquet_cache: bool = True,
        cache_ttl_hours: int = 24,
        parquet_min_rows: int = 1000,
        use_stooq_fallback: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.rate_limiter = RateLimiter(120)
        self._yf = None
        self.cache_ttl_hours = cache_ttl_hours
        self.parquet_min_rows = parquet_min_rows
        self.use_stooq_fallback = use_stooq_fallback
        self.sqlite_cache = None
        if use_sqlite_cache:
            db_path = self.cache_dir / "market_data.db"
            self.sqlite_cache = SQLiteCache(str(db_path), ttl_hours=cache_ttl_hours)
        self.parquet_manager = None
        if use_parquet_cache:
            parquet_dir = self.cache_dir / "parquet"
            self.parquet_manager = ParquetManager(
                str(parquet_dir), ttl_hours=cache_ttl_hours
            )
        self.quality_checker = DataQualityChecker()

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

    def _cache_key(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> str:
        return f"{symbol}|{start.date()}|{end.date()}|{interval}"

    def _fetch_stooq(self, symbol: str) -> pd.DataFrame:
        stooq_symbol = f"{symbol.lower()}.us"
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
        df = pd.read_csv(url)
        if df.empty:
            return df
        df = df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df

    def _estimate_bars(self, start: datetime, end: datetime, interval: str) -> int:
        days = max((end - start).days, 1)
        interval = interval.lower()
        if interval.endswith("d"):
            bars_per_day = 1
        elif interval.endswith("h"):
            bars_per_day = 7
        elif interval.endswith("m"):
            try:
                minutes = int(interval[:-1])
            except ValueError:
                minutes = 1
            bars_per_day = max(1, 390 // minutes)
        else:
            bars_per_day = 1
        return int(days * bars_per_day)

    def _intraday_period(self, start: datetime, end: datetime) -> str:
        days = max((end - start).days, 1)
        days = min(days, 60)
        return f"{days}d"

    def _use_parquet(self, start: datetime, end: datetime, interval: str) -> bool:
        if not self.parquet_manager:
            return False
        return self._estimate_bars(start, end, interval) >= self.parquet_min_rows

    def _load_cache(
        self, path: Path, max_age_hours: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """Load cached data if fresh."""
        if not path.exists():
            return None

        if max_age_hours is None:
            max_age_hours = self.cache_ttl_hours

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
        cache_key = self._cache_key(symbol, start, end, interval)

        if use_cache:
            if self._use_parquet(start, end, interval):
                try:
                    cached = self.parquet_manager.load(symbol, start, end, interval)
                    if cached is not None:
                        return cached
                except ImportError as exc:
                    logger.warning("Parquet cache unavailable: %s", exc)

            if self.sqlite_cache:
                cached = self.sqlite_cache.get(cache_key)
                if cached is not None:
                    return cached
            cached = self._load_cache(cache_path)
            if cached is not None:
                return cached

        self.rate_limiter.wait()

        ticker = self.yf.Ticker(symbol)
        try:
            df = ticker.history(start=start, end=end, interval=interval)
        except Exception:
            df = self.yf.download(
                symbol, start=start, end=end, interval=interval, progress=False
            )

        if isinstance(df.columns, pd.MultiIndex):
            if symbol in df.columns.get_level_values(-1):
                df = df.xs(symbol, level=-1, axis=1)
            else:
                df.columns = df.columns.droplevel(0)

        if df.empty and interval != "1d":
            period = self._intraday_period(start, end)
            try:
                df = self.yf.download(
                    symbol, period=period, interval=interval, progress=False
                )
            except Exception:
                df = pd.DataFrame()

        if df.empty:
            if self.use_stooq_fallback and interval == "1d":
                df = self._fetch_stooq(symbol)
                df = df.loc[(df.index >= start) & (df.index <= end)]
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

        quality = self.quality_checker.validate_ohlcv(df)
        if not quality["is_valid"]:
            logger.warning(
                "Data quality issues for %s: %s", symbol, ", ".join(quality["issues"])
            )

        if use_cache:
            self._save_cache(cache_path, df)
            if self.sqlite_cache:
                self.sqlite_cache.set(cache_key, df)
            if self._use_parquet(start, end, interval):
                try:
                    self.parquet_manager.save(df, symbol, start, end, interval)
                except ImportError as exc:
                    logger.warning("Parquet save unavailable: %s", exc)

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
            "enterprise_to_ebitda": info.get("enterpriseToEbitda"),
            "enterprise_to_revenue": info.get("enterpriseToRevenue"),
            "peg_ratio": info.get("pegRatio"),
            "profit_margins": info.get("profitMargins"),
            "gross_margins": info.get("grossMargins"),
            "operating_margins": info.get("operatingMargins"),
            "ebitda_margins": info.get("ebitdaMargins"),
            "return_on_equity": info.get("returnOnEquity"),
            "return_on_assets": info.get("returnOnAssets"),
            "return_on_investment": info.get("returnOnInvestment"),
            "debt_to_equity": info.get("debtToEquity"),
            "total_debt": info.get("totalDebt"),
            "total_cash": info.get("totalCash"),
            "ebitda": info.get("ebitda"),
            "operating_cashflow": info.get("operatingCashflow"),
            "free_cashflow": info.get("freeCashflow"),
            "payout_ratio": info.get("payoutRatio"),
            "dividend_rate": info.get("dividendRate"),
            "five_year_avg_dividend_yield": info.get("fiveYearAvgDividendYield"),
            "earnings_growth": info.get("earningsGrowth"),
            "revenue_growth": info.get("revenueGrowth"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
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
                "BRK-B",
                "JPM",
                "V",
                "UNH",
                "XOM",
                "PG",
                "MA",
                "HD",
                "AVGO",
                "LLY",
                "COST",
                "ABBV",
                "WMT",
                "DIS",
                "KO",
                "PEP",
                "BAC",
                "ADBE",
                "CRM",
                "NFLX",
                "ORCL",
                "CSCO",
                "INTC",
                "QCOM",
                "TXN",
                "TMO",
                "LIN",
                "ACN",
                "MCD",
                "NKE",
                "UPS",
                "HON",
                "UNP",
                "CAT",
                "MMM",
                "GE",
                "IBM",
                "AMD",
                "AMAT",
                "GS",
                "MS",
                "C",
                "BA",
                "RTX",
                "LMT",
                "GM",
                "F",
                "CVX",
                "COP",
                "SLB",
                "SPGI",
                "BKNG",
                "SBUX",
                "T",
                "VZ",
                "PFE",
                "MRK",
                "ABT",
                "CVS",
                "DHR",
                "LOW",
                "ISRG",
                "GILD",
                "MDT",
                "INTU",
                "NOW",
                "PYPL",
                "SNPS",
                "VRTX",
                "ADP",
                "BLK",
                "DE",
                "MO",
                "SO",
                "DUK",
                "NEE",
                "PLD",
                "AMT",
                "CCI",
                "SCHW",
                "USB",
                "AXP",
                "TGT",
                "CME",
                "CB",
                "ZTS",
                "MMM",
                "FDX",
                "PNC",
                "APD",
                "EOG",
                "LRCX",
                "KLAC",
                "REGN",
                "ETN",
            ]
