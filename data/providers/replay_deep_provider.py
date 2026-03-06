"""Replay-backed deep market data provider for smoke and offline research."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from quantum_alpha.data.collectors.intraday_replay import IntradayReplayStore
from quantum_alpha.data.providers.base import ProviderResult


class ReplayDeepMarketProvider:
    name = "replay_deep"

    def __init__(self, replay_root: str | Path):
        self.store = IntradayReplayStore(replay_root)

    @staticmethod
    def _utc_timestamp(value: datetime) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def _load_slice(
        self,
        date: str,
        symbol: str,
        domain: str,
        start: datetime,
        end: datetime,
    ) -> ProviderResult:
        start_ts = self._utc_timestamp(start)
        end_ts = self._utc_timestamp(end)
        try:
            df = self.store.load_domain(date, symbol, domain)
        except FileNotFoundError as exc:
            return ProviderResult(
                data=None,
                provider=self.name,
                domain=domain,
                error=str(exc),
                degraded=True,
                completeness=0.0,
                reliability=0.0,
            )
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], utc=True)
            df = df.loc[(ts >= start_ts) & (ts <= end_ts)].copy()
        quality = self.store.summarize_quality(dates=[date], symbols=[symbol]).to_dict()
        return ProviderResult(
            data=df,
            provider=self.name,
            domain=domain,
            completeness=float(quality.get("completeness", 1.0)),
            reliability=1.0 - float(quality.get("crossed_market_rate", 0.0)),
            degraded=False,
            metadata=quality,
        )

    def fetch_trades(self, symbol: str, start: datetime, end: datetime) -> ProviderResult:
        start_ts = self._utc_timestamp(start)
        end_ts = self._utc_timestamp(end)
        return self._load_slice(start_ts.strftime("%Y-%m-%d"), symbol, "trades", start_ts.to_pydatetime(), end_ts.to_pydatetime())

    def fetch_quotes(self, symbol: str, start: datetime, end: datetime) -> ProviderResult:
        start_ts = self._utc_timestamp(start)
        end_ts = self._utc_timestamp(end)
        return self._load_slice(start_ts.strftime("%Y-%m-%d"), symbol, "quotes", start_ts.to_pydatetime(), end_ts.to_pydatetime())

    def fetch_order_book(
        self,
        symbol: str,
        at: Optional[datetime] = None,
        levels: int = 10,
    ) -> ProviderResult:
        ts = self._utc_timestamp(at or datetime.utcnow())
        date = ts.strftime("%Y-%m-%d")
        try:
            df = self.store.load_domain(date, symbol, "depth")
        except FileNotFoundError as exc:
            return ProviderResult(
                data=None,
                provider=self.name,
                domain="depth",
                error=str(exc),
                degraded=True,
                completeness=0.0,
                reliability=0.0,
            )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        snap = df.loc[df["timestamp"] <= ts].copy()
        if snap.empty:
            snap = df.copy()
        latest_ts = snap["timestamp"].max()
        out = snap.loc[snap["timestamp"] == latest_ts].sort_values("level").head(max(1, int(levels)))
        quality = self.store.summarize_quality(dates=[date], symbols=[symbol]).to_dict()
        return ProviderResult(
            data=out.reset_index(drop=True),
            provider=self.name,
            domain="depth",
            completeness=float(quality.get("depth_completeness", 1.0)),
            reliability=1.0 - float(quality.get("crossed_market_rate", 0.0)),
            metadata=quality,
        )
