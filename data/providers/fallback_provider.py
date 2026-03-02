"""Fallback provider utilities (stooq/local baseline)."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict

import pandas as pd

from .base import ProviderResult

logger = logging.getLogger(__name__)


class FallbackProvider:
    """Simple fallback provider with stooq daily price support."""

    name = "fallback"

    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> ProviderResult:
        t0 = time.perf_counter()
        if interval != "1d":
            return ProviderResult(
                data=None,
                provider=self.name,
                domain="market_data",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                completeness=0.0,
                reliability=0.0,
                error="fallback_only_supports_daily",
            )

        stooq_symbol = f"{symbol.lower()}.us"
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
        try:
            df = pd.read_csv(url)
            if df.empty:
                raise ValueError("empty_stooq")
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
            df = df.loc[(df.index >= start) & (df.index <= end)]
            if df.empty:
                raise ValueError("no_rows_after_filter")
            df = df[["open", "high", "low", "close", "volume"]]
            df["returns"] = df["close"].pct_change()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return ProviderResult(
                data=df,
                provider=self.name,
                domain="market_data",
                latency_ms=latency_ms,
                completeness=1.0,
                metadata={"source": "stooq", "rows": int(len(df))},
            )
        except Exception as exc:
            return ProviderResult(
                data=None,
                provider=self.name,
                domain="market_data",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                completeness=0.0,
                reliability=0.0,
                error=str(exc),
            )

    def fetch_fundamentals(self, symbol: str) -> ProviderResult:
        return ProviderResult(
            data={},
            provider=self.name,
            domain="fundamentals",
            latency_ms=0.0,
            completeness=0.0,
            degraded=True,
            metadata={"reason": "no_fundamental_fallback"},
        )

    def fetch_news(self, symbol: str, max_items: int = 50) -> ProviderResult:
        return ProviderResult(
            data=[],
            provider=self.name,
            domain="news",
            completeness=0.0,
            degraded=True,
            metadata={"reason": "no_news_fallback"},
        )

    def fetch_macro(self, series: str, start: datetime | None = None) -> ProviderResult:
        return ProviderResult(
            data=[],
            provider=self.name,
            domain="macro",
            completeness=0.0,
            degraded=True,
            metadata={"reason": "no_macro_fallback"},
        )

    def fetch_options(self, symbol: str) -> ProviderResult:
        return ProviderResult(
            data=[],
            provider=self.name,
            domain="options",
            completeness=0.0,
            degraded=True,
            metadata={"reason": "no_options_fallback"},
        )

    def fetch_insider(self, symbol: str) -> ProviderResult:
        return ProviderResult(
            data=[],
            provider=self.name,
            domain="insider",
            completeness=0.0,
            degraded=True,
            metadata={"reason": "no_insider_fallback"},
        )

    def fetch_congress(self, symbol: str) -> ProviderResult:
        return ProviderResult(
            data=[],
            provider=self.name,
            domain="congress",
            completeness=0.0,
            degraded=True,
            metadata={"reason": "no_congress_fallback"},
        )

    def fetch_earnings(self, symbol: str) -> ProviderResult:
        return ProviderResult(
            data=[],
            provider=self.name,
            domain="earnings",
            completeness=0.0,
            degraded=True,
            metadata={"reason": "no_earnings_fallback"},
        )


__all__ = ["FallbackProvider"]
