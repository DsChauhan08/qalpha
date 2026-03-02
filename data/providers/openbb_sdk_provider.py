"""OpenBB SDK-backed provider for local in-process usage."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from .base import ProviderResult

logger = logging.getLogger(__name__)


def _to_df(payload: Any) -> pd.DataFrame:
    if payload is None:
        return pd.DataFrame()
    if isinstance(payload, pd.DataFrame):
        return payload
    if hasattr(payload, "to_df"):
        try:
            return payload.to_df()
        except Exception:
            pass
    if isinstance(payload, dict):
        try:
            return pd.DataFrame([payload])
        except Exception:
            return pd.DataFrame()
    if isinstance(payload, list):
        try:
            return pd.DataFrame(payload)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _pick_first(dct: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in dct:
            return dct.get(key)
    lower = {str(k).lower(): k for k in dct.keys()}
    for key in keys:
        lk = str(key).lower()
        if lk in lower:
            return dct.get(lower[lk])
    return None


class OpenBBSDKProvider:
    name = "openbb_sdk"

    def __init__(self, provider: Optional[str] = None, free_only: bool = True) -> None:
        self.provider = provider
        self.free_only = bool(free_only)
        self._obb = None

    @property
    def obb(self):
        if self._obb is not None:
            return self._obb

        # OpenBB has exposed both `from openbb import obb` and package variants.
        try:
            from openbb import obb as _obb  # type: ignore

            self._obb = _obb
            return self._obb
        except Exception:
            pass

        try:
            import openbb

            self._obb = getattr(openbb, "obb", None)
            if self._obb is not None:
                return self._obb
        except Exception:
            pass

        raise ImportError("openbb_sdk_not_available")

    def _historical_attempts(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str,
    ) -> pd.DataFrame:
        obb = self.obb
        kwargs = {
            "symbol": symbol,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "interval": interval,
        }
        if self.provider:
            kwargs["provider"] = self.provider

        attempts = [
            lambda: obb.equity.price.historical(**kwargs),
            lambda: obb.equity.price.historical(
                symbol=symbol,
                start_date=kwargs["start_date"],
                end_date=kwargs["end_date"],
                provider=kwargs.get("provider"),
            ),
            lambda: obb.equity.price.historical(symbol, kwargs["start_date"], kwargs["end_date"]),
        ]

        last_err = None
        for call in attempts:
            try:
                payload = call()
                df = _to_df(payload)
                if not df.empty:
                    return df
            except Exception as exc:
                last_err = exc
                continue
        if last_err:
            raise last_err
        return pd.DataFrame()

    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> ProviderResult:
        t0 = time.perf_counter()
        try:
            df = self._historical_attempts(symbol=symbol, start=start, end=end, interval=interval)
            if df.empty:
                raise ValueError("empty_openbb_ohlcv")

            rename = {
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "adj_close": "close",
                "Adj Close": "close",
            }
            df = df.rename(columns=rename)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"]).set_index("date")
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                df = df.dropna(subset=["datetime"]).set_index("datetime")
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index, errors="coerce")
                    df = df[~df.index.isna()]
                except Exception:
                    pass

            required = ["open", "high", "low", "close", "volume"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                raise ValueError(f"openbb_missing_ohlcv_columns={missing}")

            df = df[required].copy().sort_index()
            df["returns"] = df["close"].pct_change()

            latency_ms = (time.perf_counter() - t0) * 1000.0
            return ProviderResult(
                data=df,
                provider=self.name,
                domain="market_data",
                latency_ms=latency_ms,
                completeness=1.0,
                metadata={"rows": int(len(df)), "provider": self.provider or "auto"},
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return ProviderResult(
                data=None,
                provider=self.name,
                domain="market_data",
                latency_ms=latency_ms,
                completeness=0.0,
                reliability=0.0,
                error=str(exc),
            )

    def _fundamental_attempts(self, symbol: str) -> Dict[str, Any]:
        obb = self.obb

        calls = []
        kwargs = {"symbol": symbol}
        if self.provider:
            kwargs["provider"] = self.provider

        calls.append(lambda: obb.equity.fundamental.metrics(**kwargs))
        calls.append(lambda: obb.equity.fundamental.overview(**kwargs))
        calls.append(lambda: obb.equity.fundamental.company(**kwargs))

        last_err = None
        for call in calls:
            try:
                payload = call()
                df = _to_df(payload)
                if not df.empty:
                    row = df.iloc[-1].to_dict()
                    return {str(k): v for k, v in row.items()}
                if isinstance(payload, dict) and payload:
                    return payload
            except Exception as exc:
                last_err = exc
                continue
        if last_err:
            raise last_err
        return {}

    def fetch_fundamentals(self, symbol: str) -> ProviderResult:
        t0 = time.perf_counter()
        try:
            raw = self._fundamental_attempts(symbol=symbol)
            mapped = {
                "market_cap": _pick_first(raw, ["market_cap", "marketCap"]),
                "pe_ratio": _pick_first(raw, ["pe_ratio", "trailing_pe", "trailingPE", "pe"]),
                "forward_pe": _pick_first(raw, ["forward_pe", "forwardPE"]),
                "price_to_book": _pick_first(raw, ["price_to_book", "priceToBook", "pb_ratio"]),
                "enterprise_to_ebitda": _pick_first(raw, ["enterprise_to_ebitda", "enterpriseToEbitda", "ev_to_ebitda"]),
                "enterprise_to_revenue": _pick_first(raw, ["enterprise_to_revenue", "enterpriseToRevenue", "ev_to_revenue", "ev_to_sales"]),
                "peg_ratio": _pick_first(raw, ["peg_ratio", "pegRatio"]),
                "profit_margins": _pick_first(raw, ["profit_margins", "profitMargins", "net_margin"]),
                "gross_margins": _pick_first(raw, ["gross_margins", "grossMargins", "gross_margin"]),
                "operating_margins": _pick_first(raw, ["operating_margins", "operatingMargins", "operating_margin"]),
                "ebitda_margins": _pick_first(raw, ["ebitda_margins", "ebitdaMargins", "ebitda_margin"]),
                "return_on_equity": _pick_first(raw, ["return_on_equity", "returnOnEquity", "roe"]),
                "return_on_assets": _pick_first(raw, ["return_on_assets", "returnOnAssets", "roa"]),
                "return_on_investment": _pick_first(raw, ["return_on_investment", "roic", "return_on_capital"]),
                "debt_to_equity": _pick_first(raw, ["debt_to_equity", "debtToEquity"]),
                "total_debt": _pick_first(raw, ["total_debt", "totalDebt"]),
                "total_cash": _pick_first(raw, ["total_cash", "totalCash"]),
                "ebitda": _pick_first(raw, ["ebitda"]),
                "operating_cashflow": _pick_first(raw, ["operating_cashflow", "operatingCashflow"]),
                "free_cashflow": _pick_first(raw, ["free_cashflow", "freeCashflow"]),
                "payout_ratio": _pick_first(raw, ["payout_ratio", "payoutRatio"]),
                "dividend_rate": _pick_first(raw, ["dividend_rate", "dividendRate"]),
                "five_year_avg_dividend_yield": _pick_first(raw, ["five_year_avg_dividend_yield", "fiveYearAvgDividendYield"]),
                "earnings_growth": _pick_first(raw, ["earnings_growth", "earningsGrowth"]),
                "revenue_growth": _pick_first(raw, ["revenue_growth", "revenueGrowth"]),
                "price_to_sales": _pick_first(raw, ["price_to_sales", "priceToSalesTrailing12Months", "ps_ratio"]),
                "beta": _pick_first(raw, ["beta"]),
                "dividend_yield": _pick_first(raw, ["dividend_yield", "dividendYield"]),
                "avg_volume": _pick_first(raw, ["avg_volume", "averageVolume"]),
                "short_ratio": _pick_first(raw, ["short_ratio", "shortRatio"]),
            }
            filled = sum(v is not None for v in mapped.values())
            completeness = float(filled / max(len(mapped), 1))
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return ProviderResult(
                data=mapped,
                provider=self.name,
                domain="fundamentals",
                latency_ms=latency_ms,
                completeness=completeness,
                metadata={"fields": len(mapped), "filled": filled, "provider": self.provider or "auto"},
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return ProviderResult(
                data=None,
                provider=self.name,
                domain="fundamentals",
                latency_ms=latency_ms,
                completeness=0.0,
                reliability=0.0,
                error=str(exc),
            )

    def fetch_news(self, symbol: str, max_items: int = 50) -> ProviderResult:
        t0 = time.perf_counter()
        try:
            obb = self.obb
            kwargs = {"symbol": symbol, "limit": int(max_items)}
            if self.provider:
                kwargs["provider"] = self.provider
            payload = obb.news.company(**kwargs)
            df = _to_df(payload)
            rows = df.to_dict(orient="records") if not df.empty else []
            return ProviderResult(
                data=rows,
                provider=self.name,
                domain="news",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                completeness=1.0 if rows else 0.0,
                metadata={"items": len(rows)},
            )
        except Exception as exc:
            return ProviderResult(
                data=[],
                provider=self.name,
                domain="news",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                completeness=0.0,
                reliability=0.0,
                error=str(exc),
            )

    def fetch_macro(self, series: str, start: datetime | None = None) -> ProviderResult:
        return ProviderResult(
            data=[],
            provider=self.name,
            domain="macro",
            completeness=0.0,
            degraded=True,
            metadata={"reason": "macro_not_wired"},
        )

    def fetch_options(self, symbol: str) -> ProviderResult:
        return ProviderResult(
            data=[],
            provider=self.name,
            domain="options",
            completeness=0.0,
            degraded=True,
            metadata={"reason": "options_not_wired"},
        )

    def fetch_insider(self, symbol: str) -> ProviderResult:
        return ProviderResult(
            data=[],
            provider=self.name,
            domain="insider",
            completeness=0.0,
            degraded=True,
            metadata={"reason": "insider_not_wired"},
        )

    def fetch_congress(self, symbol: str) -> ProviderResult:
        return ProviderResult(
            data=[],
            provider=self.name,
            domain="congress",
            completeness=0.0,
            degraded=True,
            metadata={"reason": "congress_not_wired"},
        )

    def fetch_earnings(self, symbol: str) -> ProviderResult:
        return ProviderResult(
            data=[],
            provider=self.name,
            domain="earnings",
            completeness=0.0,
            degraded=True,
            metadata={"reason": "earnings_not_wired"},
        )


__all__ = ["OpenBBSDKProvider"]
