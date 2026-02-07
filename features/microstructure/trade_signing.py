"""
Trade Signing Classifier.

Classifies individual trades as buyer- or seller-initiated using
established algorithms from market microstructure literature.

Algorithms:
- Lee-Ready (1991): Compare trade price to quote midpoint, with
  tick test fallback.
- Tick Rule: Sign based on price change from previous trade.
- Bulk Volume Classification (BVC): Probabilistic classification
  from bar data using the normal CDF.

When tick-level data is unavailable (our yfinance case), the BVC
method operates on OHLCV bars.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TradeSigningClassifier:
    """
    Classify trades as buyer- or seller-initiated.

    Args:
        method: Classification algorithm.
            'lee_ready' : Quote-based with tick fallback (needs L2)
            'tick'       : Pure tick rule (needs tick data)
            'bvc'        : Bulk Volume Classification (OHLCV bars)
    """

    def __init__(self, method: str = "bvc") -> None:
        if method not in ("lee_ready", "tick", "bvc"):
            raise ValueError(f"Unknown method: {method!r}")
        self.method = method

    # ------------------------------------------------------------------
    # Lee-Ready (requires tick + quote data)
    # ------------------------------------------------------------------

    def lee_ready(
        self,
        trades: pd.DataFrame,
        quotes: pd.DataFrame,
    ) -> pd.Series:
        """
        Lee-Ready (1991) trade classification.

        Step 1: Compare trade price to quote midpoint.
            price > mid => buy (+1)
            price < mid => sell (-1)
            price == mid => use tick test

        Step 2 (tick test fallback):
            price > prev_price => buy
            price < prev_price => sell
            price == prev_price => use previous classification

        Args:
            trades: DataFrame with 'price' column, DatetimeIndex.
            quotes: DataFrame with 'bid', 'ask' columns, DatetimeIndex.

        Returns:
            Series of trade signs (+1 buy, -1 sell).
        """
        # Merge trades with most recent quote
        merged = pd.merge_asof(
            trades.sort_index(),
            quotes[["bid", "ask"]].sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
        )

        merged["mid"] = (merged["bid"] + merged["ask"]) / 2

        # Quote test
        signs = np.where(
            merged["price"] > merged["mid"],
            1,
            np.where(merged["price"] < merged["mid"], -1, 0),
        )

        # Tick test for ties
        price_changes = merged["price"].diff()
        tick_signs = np.sign(price_changes).fillna(0).astype(int)

        # Replace zeros with tick rule
        result = np.where(signs != 0, signs, tick_signs.values)

        # Forward-fill remaining zeros
        result_series = pd.Series(result, index=merged.index)
        result_series = result_series.replace(0, np.nan).ffill().fillna(0).astype(int)

        return result_series.rename("trade_sign")

    # ------------------------------------------------------------------
    # Pure tick rule
    # ------------------------------------------------------------------

    def tick_rule(self, trades: pd.DataFrame) -> pd.Series:
        """
        Tick rule classification.

        sign(trade) = sign(price_t - price_{t-1})

        If price unchanged, carry forward previous sign.

        Args:
            trades: DataFrame with 'price' column.

        Returns:
            Series of trade signs.
        """
        changes = trades["price"].diff()
        signs = np.sign(changes)
        signs = signs.replace(0, np.nan).ffill().fillna(0).astype(int)
        return signs.rename("trade_sign")

    # ------------------------------------------------------------------
    # Bulk Volume Classification (OHLCV bars)
    # ------------------------------------------------------------------

    def bulk_volume_classification(
        self,
        bars: pd.DataFrame,
        sigma_window: int = 20,
    ) -> pd.DataFrame:
        """
        Bulk Volume Classification (BVC).

        Uses the normal CDF to probabilistically assign volume
        to buy/sell sides based on the standardised price change.

        buy_pct = Phi(z), where z = delta_price / sigma
        sell_pct = 1 - buy_pct

        Reference: Easley, Lopez de Prado, O'Hara (2012)

        Args:
            bars: OHLCV DataFrame.
            sigma_window: Window for volatility estimation.

        Returns:
            DataFrame with [buy_volume, sell_volume, buy_pct,
            order_imbalance].
        """
        from scipy.stats import norm

        df = bars.copy()
        returns = df["close"].pct_change()
        sigma = returns.rolling(sigma_window, min_periods=1).std()

        # Standardised return
        z = np.where(sigma > 0, returns / sigma, 0.0)

        # CDF gives probability that trade is buyer-initiated
        buy_pct = norm.cdf(z)

        df["buy_pct"] = buy_pct
        df["buy_volume"] = df["volume"] * buy_pct
        df["sell_volume"] = df["volume"] * (1 - buy_pct)
        df["order_imbalance"] = (df["buy_volume"] - df["sell_volume"]) / df[
            "volume"
        ].replace(0, np.nan)

        return df

    # ------------------------------------------------------------------
    # Unified interface
    # ------------------------------------------------------------------

    def classify(
        self,
        data: pd.DataFrame,
        quotes: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Classify trades using the configured method.

        For 'bvc': pass OHLCV bars as ``data``.
        For 'lee_ready': pass tick trades as ``data`` and quotes.
        For 'tick': pass tick trades as ``data``.

        Returns:
            DataFrame with classification columns.
        """
        if self.method == "bvc":
            return self.bulk_volume_classification(data)
        elif self.method == "lee_ready":
            if quotes is None:
                raise ValueError("Lee-Ready requires quote data")
            signs = self.lee_ready(data, quotes)
            data = data.copy()
            data["trade_sign"] = signs
            return data
        else:  # tick
            signs = self.tick_rule(data)
            data = data.copy()
            data["trade_sign"] = signs
            return data

    def generate_signals(
        self,
        bars: pd.DataFrame,
        imbalance_threshold: float = 0.3,
        window: int = 5,
    ) -> pd.DataFrame:
        """
        Generate signals from trade classification.

        Persistent buy imbalance => bullish.
        Persistent sell imbalance => bearish.

        Args:
            bars: OHLCV DataFrame.
            imbalance_threshold: Minimum absolute imbalance for signal.
            window: Smoothing window.

        Returns:
            DataFrame with [order_imbalance, imbalance_ma, signal].
        """
        df = self.bulk_volume_classification(bars)
        df["imbalance_ma"] = df["order_imbalance"].rolling(window, min_periods=1).mean()

        df["signal"] = 0
        df.loc[df["imbalance_ma"] > imbalance_threshold, "signal"] = 1
        df.loc[df["imbalance_ma"] < -imbalance_threshold, "signal"] = -1

        return df
