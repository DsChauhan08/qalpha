"""OpenBB local API provider for realtime/live pathways."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import requests

from .base import ProviderResult

logger = logging.getLogger(__name__)


class OpenBBAPIProvider:
    name = "openbb_api"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:6900",
        provider: Optional[str] = None,
        timeout: float = 1.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.provider = provider
        self.timeout = float(timeout)

    def _request(self, path: str, params: Dict[str, Any]) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _extract_df(payload: Any) -> pd.DataFrame:
        if payload is None:
            return pd.DataFrame()
        if isinstance(payload, dict):
            for key in ("results", "data", "items"):
                if key in payload:
                    payload = payload[key]
                    break
        if isinstance(payload, dict):
            return pd.DataFrame([payload])
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        return pd.DataFrame()

    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> ProviderResult:
        t0 = time.perf_counter()
        params: Dict[str, Any] = {
            "symbol": symbol,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "interval": interval,
        }
        if self.provider:
            params["provider"] = self.provider

        endpoints = [
            "api/v1/equity/price/historical",
            "api/v1/equity/price/history",
        ]

        last_err = None
        df = pd.DataFrame()
        for ep in endpoints:
            try:
                payload = self._request(ep, params)
                df = self._extract_df(payload)
                if not df.empty:
                    break
            except Exception as exc:
                last_err = exc
                continue

        if df.empty:
            return ProviderResult(
                data=None,
                provider=self.name,
                domain="market_data",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                completeness=0.0,
                reliability=0.0,
                error=str(last_err) if last_err else "empty_openbb_api_ohlcv",
            )

        rename = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "adj_close": "close",
        }
        df = df.rename(columns=rename)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).set_index("date")
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.dropna(subset=["datetime"]).set_index("datetime")
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()]
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return ProviderResult(
                data=None,
                provider=self.name,
                domain="market_data",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                completeness=0.0,
                reliability=0.0,
                error=f"openbb_api_missing_columns={missing}",
            )

        df = df[required].copy().sort_index()
        df["returns"] = df["close"].pct_change()
        return ProviderResult(
            data=df,
            provider=self.name,
            domain="market_data",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            completeness=1.0,
            metadata={"rows": int(len(df)), "base_url": self.base_url},
        )

    def fetch_fundamentals(self, symbol: str) -> ProviderResult:
        t0 = time.perf_counter()
        params: Dict[str, Any] = {"symbol": symbol}
        if self.provider:
            params["provider"] = self.provider

        endpoints = [
            "api/v1/equity/fundamental/metrics",
            "api/v1/equity/fundamental/overview",
        ]
        raw: Dict[str, Any] = {}
        last_err = None
        for ep in endpoints:
            try:
                payload = self._request(ep, params)
                df = self._extract_df(payload)
                if not df.empty:
                    raw = {str(k): v for k, v in df.iloc[-1].to_dict().items()}
                    break
                if isinstance(payload, dict):
                    raw = payload
                    break
            except Exception as exc:
                last_err = exc
                continue

        if not raw:
            return ProviderResult(
                data=None,
                provider=self.name,
                domain="fundamentals",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                completeness=0.0,
                reliability=0.0,
                error=str(last_err) if last_err else "empty_openbb_api_fundamentals",
            )

        mapped = {
            "market_cap": raw.get("market_cap", raw.get("marketCap")),
            "pe_ratio": raw.get("pe_ratio", raw.get("trailingPE")),
            "forward_pe": raw.get("forward_pe", raw.get("forwardPE")),
            "price_to_book": raw.get("price_to_book", raw.get("priceToBook")),
            "enterprise_to_ebitda": raw.get("enterprise_to_ebitda", raw.get("enterpriseToEbitda")),
            "enterprise_to_revenue": raw.get("enterprise_to_revenue", raw.get("enterpriseToRevenue")),
            "peg_ratio": raw.get("peg_ratio", raw.get("pegRatio")),
            "profit_margins": raw.get("profit_margins", raw.get("profitMargins")),
            "gross_margins": raw.get("gross_margins", raw.get("grossMargins")),
            "operating_margins": raw.get("operating_margins", raw.get("operatingMargins")),
            "ebitda_margins": raw.get("ebitda_margins", raw.get("ebitdaMargins")),
            "return_on_equity": raw.get("return_on_equity", raw.get("returnOnEquity")),
            "return_on_assets": raw.get("return_on_assets", raw.get("returnOnAssets")),
            "return_on_investment": raw.get("return_on_investment", raw.get("roic")),
            "debt_to_equity": raw.get("debt_to_equity", raw.get("debtToEquity")),
            "total_debt": raw.get("total_debt", raw.get("totalDebt")),
            "total_cash": raw.get("total_cash", raw.get("totalCash")),
            "ebitda": raw.get("ebitda"),
            "operating_cashflow": raw.get("operating_cashflow", raw.get("operatingCashflow")),
            "free_cashflow": raw.get("free_cashflow", raw.get("freeCashflow")),
            "payout_ratio": raw.get("payout_ratio", raw.get("payoutRatio")),
            "dividend_rate": raw.get("dividend_rate", raw.get("dividendRate")),
            "five_year_avg_dividend_yield": raw.get("five_year_avg_dividend_yield", raw.get("fiveYearAvgDividendYield")),
            "earnings_growth": raw.get("earnings_growth", raw.get("earningsGrowth")),
            "revenue_growth": raw.get("revenue_growth", raw.get("revenueGrowth")),
            "price_to_sales": raw.get("price_to_sales", raw.get("priceToSalesTrailing12Months")),
            "beta": raw.get("beta"),
            "dividend_yield": raw.get("dividend_yield", raw.get("dividendYield")),
            "avg_volume": raw.get("avg_volume", raw.get("averageVolume")),
            "short_ratio": raw.get("short_ratio", raw.get("shortRatio")),
        }

        filled = sum(v is not None for v in mapped.values())
        completeness = float(filled / max(len(mapped), 1))
        return ProviderResult(
            data=mapped,
            provider=self.name,
            domain="fundamentals",
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            completeness=completeness,
            metadata={"fields": len(mapped), "filled": filled, "base_url": self.base_url},
        )

    def healthcheck(self) -> bool:
        for path in ("health", "api/v1/health", "api/v1/system/health"):
            try:
                url = f"{self.base_url}/{path}"
                r = requests.get(url, timeout=min(self.timeout, 1.0))
                if r.status_code == 200:
                    return True
            except Exception:
                continue
        return False

    def _fetch_domain_records(self, path: str, params: Dict[str, Any], domain: str) -> ProviderResult:
        t0 = time.perf_counter()
        try:
            payload = self._request(path, params)
            df = self._extract_df(payload)
            rows = df.to_dict(orient="records") if not df.empty else []
            return ProviderResult(
                data=rows,
                provider=self.name,
                domain=domain,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                completeness=1.0 if rows else 0.0,
                metadata={"items": len(rows)},
            )
        except Exception as exc:
            return ProviderResult(
                data=[],
                provider=self.name,
                domain=domain,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                completeness=0.0,
                reliability=0.0,
                error=str(exc),
            )

    def fetch_news(self, symbol: str, max_items: int = 50) -> ProviderResult:
        params: Dict[str, Any] = {"symbol": symbol, "limit": int(max_items)}
        if self.provider:
            params["provider"] = self.provider
        return self._fetch_domain_records("api/v1/news/company", params=params, domain="news")

    def fetch_macro(self, series: str, start: datetime | None = None) -> ProviderResult:
        params: Dict[str, Any] = {"series": series}
        if start is not None:
            params["start_date"] = start.strftime("%Y-%m-%d")
        if self.provider:
            params["provider"] = self.provider
        return self._fetch_domain_records("api/v1/economy/series", params=params, domain="macro")

    def fetch_options(self, symbol: str) -> ProviderResult:
        params: Dict[str, Any] = {"symbol": symbol}
        if self.provider:
            params["provider"] = self.provider
        return self._fetch_domain_records("api/v1/equity/options/chains", params=params, domain="options")

    def fetch_insider(self, symbol: str) -> ProviderResult:
        params: Dict[str, Any] = {"symbol": symbol}
        if self.provider:
            params["provider"] = self.provider
        return self._fetch_domain_records("api/v1/equity/ownership/insider", params=params, domain="insider")

    def fetch_congress(self, symbol: str) -> ProviderResult:
        params: Dict[str, Any] = {"symbol": symbol}
        if self.provider:
            params["provider"] = self.provider
        return self._fetch_domain_records("api/v1/equity/ownership/congress", params=params, domain="congress")

    def fetch_earnings(self, symbol: str) -> ProviderResult:
        params: Dict[str, Any] = {"symbol": symbol}
        if self.provider:
            params["provider"] = self.provider
        return self._fetch_domain_records("api/v1/equity/calendar/earnings", params=params, domain="earnings")


__all__ = ["OpenBBAPIProvider"]
