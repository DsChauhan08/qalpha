"""Unified data collector with pluggable provider routing and local caching."""

from __future__ import annotations

import io
import json
import logging
import pickle
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from quantum_alpha.data.routing import (
    DeterministicQualityRouter,
    build_provider_registry,
)
from quantum_alpha.data.storage.data_quality import DataQualityChecker
from quantum_alpha.data.storage.parquet_manager import ParquetManager
from quantum_alpha.data.storage.sqlite_cache import SQLiteCache

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
                except Exception:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(delay * (2**attempt))
            return None

        return wrapper

    return decorator


class DataCollector:
    """Unified data collector with provider routing and cache support."""

    def __init__(
        self,
        cache_dir: str = ".cache",
        use_sqlite_cache: bool = True,
        use_parquet_cache: bool = True,
        cache_ttl_hours: int = 24,
        parquet_min_rows: int = 1000,
        use_stooq_fallback: bool = True,
        runtime_mode: str = "backtest",
        config_path: Optional[str] = None,
        enable_openbb: Optional[bool] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.rate_limiter = RateLimiter(120)
        self._yf = None
        self.cache_ttl_hours = int(cache_ttl_hours)
        self.parquet_min_rows = int(parquet_min_rows)
        self.use_stooq_fallback = bool(use_stooq_fallback)
        self.runtime_mode = str(runtime_mode).lower().strip()
        self.config_path = config_path

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

        self.data_cfg = self._load_data_config(config_path=config_path)
        self.provider_registry = build_provider_registry(
            runtime_mode=self.runtime_mode,
            data_cfg=self.data_cfg,
            enable_openbb=enable_openbb,
        )
        routing_cfg = self.data_cfg.get("routing", {}) if isinstance(self.data_cfg, dict) else {}
        self.quality_router = DeterministicQualityRouter(config=routing_cfg)

        self._provider_events: Dict[str, Dict[str, Dict[str, object]]] = {}
        self._last_provider_selection: Dict[str, Dict[str, object]] = {}

    @property
    def yf(self):
        """Backward-compatible lazy yfinance accessor."""
        if self._yf is None:
            import yfinance

            self._yf = yfinance
        return self._yf

    def _quiet_yf_call(self, fn, *args, **kwargs):
        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            return fn(*args, **kwargs)

    def _default_settings_path(self) -> Path:
        return Path(__file__).resolve().parents[2] / "config" / "settings.yaml"

    def _resolve_settings_path(self, config_path: Optional[str]) -> Path:
        if not config_path:
            return self._default_settings_path()
        p = Path(config_path)
        if p.is_dir():
            return p / "settings.yaml"
        return p

    def _load_data_config(self, config_path: Optional[str]) -> Dict:
        settings_path = self._resolve_settings_path(config_path)
        if not settings_path.exists():
            return {}
        try:
            settings = yaml.safe_load(settings_path.read_text(encoding="utf-8")) or {}
            return settings.get("data", {}) if isinstance(settings, dict) else {}
        except Exception as exc:
            logger.warning("Failed to load data config from %s: %s", settings_path, exc)
            return {}

    def _cache_path(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> Path:
        key = f"{symbol}_{start.date()}_{end.date()}_{interval}"
        return self.cache_dir / f"{key}.pkl"

    def _cache_meta_path(self, cache_path: Path) -> Path:
        return cache_path.with_suffix(cache_path.suffix + ".meta.json")

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

    def _intraday_period(self, start: datetime, end: datetime, interval: str) -> str:
        days = max((end - start).days, 1)
        iv = str(interval).strip().lower()
        if iv.endswith("m"):
            try:
                minutes = int(iv[:-1])
            except ValueError:
                minutes = 5
            if minutes <= 1:
                days = min(days, 8)
            else:
                days = min(days, 60)
        else:
            days = min(days, 730)
        return f"{days}d"

    def _use_parquet(self, start: datetime, end: datetime, interval: str) -> bool:
        if not self.parquet_manager:
            return False
        return self._estimate_bars(start, end, interval) >= self.parquet_min_rows

    def _load_cache(
        self, path: Path, max_age_hours: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
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
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def _save_cache_metadata(self, cache_path: Path, meta: Dict[str, object]) -> None:
        try:
            self._cache_meta_path(cache_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_cache_metadata(self, cache_path: Path) -> Dict[str, object]:
        p = self._cache_meta_path(cache_path)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _record_provider_event(
        self,
        domain: str,
        provider: str,
        latency_ms: float,
        error: Optional[str] = None,
    ) -> None:
        by_domain = self._provider_events.setdefault(domain, {})
        stats = by_domain.setdefault(
            provider,
            {
                "successes": 0,
                "failures": 0,
                "avg_latency_ms": 0.0,
                "last_error": None,
                "last_seen_utc": None,
            },
        )

        total_calls = int(stats["successes"]) + int(stats["failures"]) + 1
        prev_avg = float(stats["avg_latency_ms"])
        stats["avg_latency_ms"] = ((prev_avg * (total_calls - 1)) + float(latency_ms)) / total_calls
        stats["last_seen_utc"] = datetime.utcnow().isoformat()

        if error:
            stats["failures"] = int(stats["failures"]) + 1
            stats["last_error"] = str(error)
        else:
            stats["successes"] = int(stats["successes"]) + 1

    def _set_last_selection(self, domain: str, payload: Dict[str, object]) -> None:
        p = dict(payload)
        p.setdefault("timestamp_utc", datetime.utcnow().isoformat())
        self._last_provider_selection[domain] = p

    def get_provider_diagnostics(self) -> Dict[str, object]:
        return {
            "runtime_mode": self.runtime_mode,
            "last_selection": self._last_provider_selection,
            "events": self._provider_events,
            "openbb_enabled": bool((self.data_cfg or {}).get("openbb", {}).get("enabled", False)),
        }

    def _select_provider_result(self, domain: str, candidates: List[object]):
        decision = self.quality_router.select(domain=domain, candidates=candidates)
        selected = None
        for result in candidates:
            if result.provider == decision.selected_provider and result.error is None:
                selected = result
                break
        if selected is None and candidates:
            selected = candidates[0]

        self._set_last_selection(
            domain,
            {
                "selected_provider": decision.selected_provider,
                "score": float(decision.selected_score),
                "degraded": bool(decision.degraded),
                "tried": decision.tried,
                "error": decision.error,
            },
        )
        return selected, decision

    def _fetch_ohlcv_from_providers(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> pd.DataFrame:
        candidates = []
        errors = []
        for provider in self.provider_registry.get("market_data"):
            name = getattr(provider, "name", provider.__class__.__name__.lower())
            try:
                result = provider.fetch_ohlcv(symbol=symbol, start=start, end=end, interval=interval)
            except Exception as exc:
                self._record_provider_event("market_data", name, latency_ms=0.0, error=str(exc))
                errors.append(f"{name}:{exc}")
                continue

            self._record_provider_event(
                "market_data",
                result.provider,
                latency_ms=float(result.latency_ms),
                error=result.error,
            )
            if result.ok and isinstance(result.data, pd.DataFrame) and not result.data.empty:
                candidates.append(result)
            elif result.error:
                errors.append(f"{result.provider}:{result.error}")

        if not candidates:
            raise ValueError(f"No provider returned market data for {symbol} ({'; '.join(errors)})")

        selected, _ = self._select_provider_result("market_data", candidates)
        if selected is None or selected.data is None:
            raise ValueError(f"Provider routing failed for {symbol}")
        return selected.data.copy()

    def _fetch_fundamentals_from_providers(self, symbol: str) -> Dict:
        candidates = []
        errors = []
        for provider in self.provider_registry.get("fundamentals"):
            name = getattr(provider, "name", provider.__class__.__name__.lower())
            try:
                result = provider.fetch_fundamentals(symbol=symbol)
            except Exception as exc:
                self._record_provider_event("fundamentals", name, latency_ms=0.0, error=str(exc))
                errors.append(f"{name}:{exc}")
                continue

            self._record_provider_event(
                "fundamentals",
                result.provider,
                latency_ms=float(result.latency_ms),
                error=result.error,
            )
            if result.ok and isinstance(result.data, dict) and result.data:
                candidates.append(result)
            elif result.error:
                errors.append(f"{result.provider}:{result.error}")

        if not candidates:
            raise ValueError(f"No provider returned fundamentals for {symbol} ({'; '.join(errors)})")

        selected, _ = self._select_provider_result("fundamentals", candidates)
        if selected is None or selected.data is None:
            raise ValueError(f"Provider routing failed for fundamentals {symbol}")
        return dict(selected.data)

    @retry(max_retries=3)
    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        cache_path = self._cache_path(symbol, start, end, interval)
        cache_key = self._cache_key(symbol, start, end, interval)

        if use_cache:
            if self._use_parquet(start, end, interval):
                try:
                    cached = self.parquet_manager.load(symbol, start, end, interval)
                    if cached is not None:
                        self._set_last_selection(
                            "market_data",
                            {
                                "selected_provider": "cache_parquet",
                                "score": 1.0,
                                "degraded": False,
                                "tried": [],
                            },
                        )
                        return cached
                except ImportError as exc:
                    logger.warning("Parquet cache unavailable: %s", exc)

            if self.sqlite_cache:
                cached = self.sqlite_cache.get(cache_key)
                if cached is not None:
                    self._set_last_selection(
                        "market_data",
                        {
                            "selected_provider": "cache_sqlite",
                            "score": 1.0,
                            "degraded": False,
                            "tried": [],
                        },
                    )
                    return cached

            cached = self._load_cache(cache_path)
            if cached is not None:
                meta = self._load_cache_metadata(cache_path)
                self._set_last_selection(
                    "market_data",
                    {
                        "selected_provider": "cache_file",
                        "source_provider": meta.get("provider"),
                        "fetched_at": meta.get("fetched_at"),
                        "score": 1.0,
                        "degraded": False,
                        "tried": [],
                    },
                )
                return cached

        self.rate_limiter.wait()
        df = self._fetch_ohlcv_from_providers(
            symbol=symbol,
            start=start,
            end=end,
            interval=interval,
        )

        if isinstance(df.columns, pd.MultiIndex):
            if symbol in df.columns.get_level_values(-1):
                df = df.xs(symbol, level=-1, axis=1)
            else:
                df.columns = df.columns.droplevel(0)

        if df.empty and self.use_stooq_fallback and interval == "1d":
            stooq = self._fetch_stooq(symbol)
            stooq = stooq.loc[(stooq.index >= start) & (stooq.index <= end)]
            if not stooq.empty:
                df = stooq

        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required OHLCV columns for {symbol}: {missing}")

        df = df[required].copy()
        df["returns"] = df["close"].pct_change()

        quality = self.quality_checker.validate_ohlcv(df)
        if not quality["is_valid"]:
            logger.warning(
                "Data quality issues for %s: %s", symbol, ", ".join(quality["issues"])
            )

        if use_cache:
            self._save_cache(cache_path, df)
            meta = self._last_provider_selection.get("market_data", {})
            self._save_cache_metadata(
                cache_path,
                {
                    "provider": meta.get("selected_provider"),
                    "fetched_at": datetime.utcnow().isoformat(),
                    "domain": "market_data",
                },
            )
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
        self.rate_limiter.wait()
        return self._fetch_fundamentals_from_providers(symbol=symbol)

    def get_sp500_symbols(self) -> List[str]:
        try:
            from quantum_alpha.universe import get_sp500

            return get_sp500()
        except ImportError:
            pass

        try:
            tables = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )
            return tables[0]["Symbol"].str.replace(".", "-").tolist()
        except Exception:
            pass

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
            "AMD",
        ]
