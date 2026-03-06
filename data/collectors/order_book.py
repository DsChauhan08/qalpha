"""
Order book collector with replay and provider-backed support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from quantum_alpha.data.collectors.intraday_replay import IntradayReplayStore
from quantum_alpha.data.providers.replay_deep_provider import ReplayDeepMarketProvider


class OrderBookCollector:
    """
    Fetch level-2 style order book snapshots.

    Args:
        provider: Optional provider name (currently only 'dummy').
        levels: Number of bid/ask levels to return.
        seed: Optional RNG seed for reproducibility in tests.
    """

    def __init__(
        self,
        provider: str = "dummy",
        levels: int = 10,
        seed: Optional[int] = None,
        replay_root: Optional[str] = None,
    ):
        self.provider = provider
        self.levels = levels
        self.rng = np.random.default_rng(seed)
        self.replay_root = replay_root
        self._replay_store = IntradayReplayStore(replay_root) if replay_root else None

    @staticmethod
    def _utc_timestamp(value: datetime) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    def fetch_order_book(
        self,
        symbol: str,
        price_hint: float = 100.0,
        at: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch an order book snapshot.

        Returns:
            DataFrame with columns [level, bid_price, bid_size, ask_price, ask_size, timestamp].
        """
        if self.provider == "replay":
            if not self.replay_root:
                raise ValueError("replay_root is required when provider='replay'")
            provider = ReplayDeepMarketProvider(self.replay_root)
            result = provider.fetch_order_book(symbol, at=at, levels=self.levels)
            if not result.ok:
                raise FileNotFoundError(result.error or f"Replay depth unavailable for {symbol}")
            return result.data.reset_index(drop=True)

        if self.provider == "fixture":
            if self._replay_store is None:
                raise ValueError("replay_root is required when provider='fixture'")
            ts = self._utc_timestamp(at or datetime.utcnow())
            date = ts.strftime("%Y-%m-%d")
            df = self._replay_store.load_domain(date, symbol, "depth")
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            snap = df.loc[df["timestamp"] <= ts].copy()
            if snap.empty:
                snap = df.copy()
            latest_ts = snap["timestamp"].max()
            return (
                snap.loc[snap["timestamp"] == latest_ts]
                .sort_values("level")
                .head(max(1, int(self.levels)))
                .reset_index(drop=True)
            )

        if self.provider != "dummy":
            raise NotImplementedError(f"Provider '{self.provider}' not integrated yet.")

        mid = price_hint
        # Generate monotonic levels around mid with small random spreads
        spreads = self.rng.normal(0.01, 0.002, size=self.levels)
        bid_prices = mid - np.cumsum(np.abs(spreads))
        ask_prices = mid + np.cumsum(np.abs(spreads))
        bid_sizes = self.rng.integers(100, 500, size=self.levels)
        ask_sizes = self.rng.integers(100, 500, size=self.levels)

        ts = datetime.utcnow()
        df = pd.DataFrame(
            {
                "level": np.arange(1, self.levels + 1),
                "bid_price": bid_prices,
                "bid_size": bid_sizes,
                "ask_price": ask_prices,
                "ask_size": ask_sizes,
                "timestamp": ts,
            }
        )
        return df


__all__ = ["OrderBookCollector"]
