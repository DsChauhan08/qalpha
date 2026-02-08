"""
Order book collector.

Provides a lightweight, dependency-free placeholder implementation that
returns synthetic depth levels while keeping the interface extendable to
real providers (Polygon, Tradier, etc.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional


class OrderBookCollector:
    """
    Fetch level-2 style order book snapshots.

    Args:
        provider: Optional provider name (currently only 'dummy').
        levels: Number of bid/ask levels to return.
        seed: Optional RNG seed for reproducibility in tests.
    """

    def __init__(self, provider: str = "dummy", levels: int = 10, seed: Optional[int] = None):
        self.provider = provider
        self.levels = levels
        self.rng = np.random.default_rng(seed)

    def fetch_order_book(self, symbol: str, price_hint: float = 100.0) -> pd.DataFrame:
        """
        Fetch an order book snapshot.

        Returns:
            DataFrame with columns [level, bid_price, bid_size, ask_price, ask_size, timestamp].
        """
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

