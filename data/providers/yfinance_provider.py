"""YFinance-backed provider used as baseline source."""

from __future__ import annotations

import io
import logging
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from typing import Dict

import pandas as pd

from .base import ProviderResult

logger = logging.getLogger(__name__)


class YFinanceProvider:
    name = "yfinance"

    def __init__(self) -> None:
        self._yf = None

    @property
    def yf(self):
        if self._yf is None:
            import yfinance

            self._yf = yfinance
        return self._yf

    @staticmethod
    def _quiet_call(fn, *args, **kwargs):
        sink = io.StringIO()
        with redirect_stdout(sink), redirect_stderr(sink):
            return fn(*args, **kwargs)

    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> ProviderResult:
        t0 = time.perf_counter()
        try:
            ticker = self.yf.Ticker(symbol)
            try:
                df = self._quiet_call(ticker.history, start=start, end=end, interval=interval)
            except Exception:
                df = self._quiet_call(
                    self.yf.download,
                    symbol,
                    start=start,
                    end=end,
                    interval=interval,
                    progress=False,
                )

            if isinstance(df.columns, pd.MultiIndex):
                if symbol in df.columns.get_level_values(-1):
                    df = df.xs(symbol, level=-1, axis=1)
                else:
                    df.columns = df.columns.droplevel(0)

            if (df is None or df.empty) and interval != "1d":
                days = max((end - start).days, 1)
                iv = str(interval).strip().lower()
                if iv.endswith("m"):
                    try:
                        minutes = int(iv[:-1])
                    except ValueError:
                        minutes = 5
                    days = min(days, 8 if minutes <= 1 else 60)
                else:
                    days = min(days, 730)
                period = f"{days}d"
                try:
                    df = self._quiet_call(
                        self.yf.download,
                        symbol,
                        period=period,
                        interval=interval,
                        progress=False,
                    )
                except Exception:
                    df = pd.DataFrame()

            if df is None or df.empty:
                raise ValueError(f"No yfinance data for {symbol}")

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
                raise ValueError(f"yfinance missing columns: {missing}")
            df = df[required].copy()
            df["returns"] = df["close"].pct_change()

            latency_ms = (time.perf_counter() - t0) * 1000.0
            return ProviderResult(
                data=df,
                provider=self.name,
                domain="market_data",
                latency_ms=latency_ms,
                completeness=1.0,
                metadata={"rows": int(len(df))},
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

    def fetch_fundamentals(self, symbol: str) -> ProviderResult:
        t0 = time.perf_counter()
        try:
            ticker = self.yf.Ticker(symbol)
            info: Dict = ticker.info or {}
            data = {
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
            filled = sum(v is not None for v in data.values())
            completeness = float(filled / max(len(data), 1))
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return ProviderResult(
                data=data,
                provider=self.name,
                domain="fundamentals",
                latency_ms=latency_ms,
                completeness=completeness,
                metadata={"fields": len(data), "filled": filled},
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

    def fetch_news(self, symbol: str, max_items: int = 20) -> ProviderResult:
        t0 = time.perf_counter()
        try:
            ticker = self.yf.Ticker(symbol)
            items = ticker.news or []
            rows = []
            for item in items[: max(1, int(max_items))]:
                rows.append(
                    {
                        "title": item.get("title", ""),
                        "summary": item.get("summary", item.get("title", "")),
                        "source": item.get("publisher", "Yahoo Finance"),
                        "published_at": item.get("providerPublishTime"),
                        "url": item.get("link", ""),
                    }
                )
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
                data=None,
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
            metadata={"reason": "unsupported_domain"},
        )

    def fetch_options(self, symbol: str) -> ProviderResult:
        t0 = time.perf_counter()
        try:
            ticker = self.yf.Ticker(symbol)
            expiries = ticker.options
            if not expiries:
                return ProviderResult(
                    data=[],
                    provider=self.name,
                    domain="options",
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    completeness=0.0,
                )
            chain = ticker.option_chain(expiries[0])
            calls = chain.calls.assign(side="call")
            puts = chain.puts.assign(side="put")
            df = pd.concat([calls, puts], ignore_index=True, sort=False)
            return ProviderResult(
                data=df.to_dict(orient="records"),
                provider=self.name,
                domain="options",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                completeness=1.0 if not df.empty else 0.0,
                metadata={"rows": int(len(df))},
            )
        except Exception as exc:
            return ProviderResult(
                data=None,
                provider=self.name,
                domain="options",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                completeness=0.0,
                reliability=0.0,
                error=str(exc),
            )

    def fetch_insider(self, symbol: str) -> ProviderResult:
        return ProviderResult(
            data=[],
            provider=self.name,
            domain="insider",
            completeness=0.0,
            degraded=True,
            metadata={"reason": "unsupported_domain"},
        )

    def fetch_congress(self, symbol: str) -> ProviderResult:
        return ProviderResult(
            data=[],
            provider=self.name,
            domain="congress",
            completeness=0.0,
            degraded=True,
            metadata={"reason": "unsupported_domain"},
        )

    def fetch_earnings(self, symbol: str) -> ProviderResult:
        t0 = time.perf_counter()
        try:
            ticker = self.yf.Ticker(symbol)
            ed = ticker.earnings_dates
            if ed is None or len(ed) == 0:
                rows = []
            else:
                df = ed.reset_index()
                df.columns = ["earnings_datetime", "eps_estimate", "reported_eps", "surprise_pct"]
                rows = df.to_dict(orient="records")
            return ProviderResult(
                data=rows,
                provider=self.name,
                domain="earnings",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                completeness=1.0 if rows else 0.0,
                metadata={"items": len(rows)},
            )
        except Exception as exc:
            return ProviderResult(
                data=None,
                provider=self.name,
                domain="earnings",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                completeness=0.0,
                reliability=0.0,
                error=str(exc),
            )


__all__ = ["YFinanceProvider"]
