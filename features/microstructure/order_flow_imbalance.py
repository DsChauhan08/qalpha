"""
Order Flow Imbalance Analyzer.

Measures the imbalance between buying and selling pressure.
Order flow imbalance (OFI) is a strong predictor of short-term
price movements and is widely used in HFT and intraday strategies.

Key metrics:
- Volume-weighted OFI
- Dollar-weighted OFI
- Cumulative OFI
- OFI momentum (change in imbalance)
- VPIN (Volume-synchronized Probability of Informed Trading)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OrderFlowImbalanceAnalyzer:
    """
    Compute order flow imbalance features from trade/quote data.

    Can operate on:
    1. Tick-level data with signed trades (ideal)
    2. Bar-level data with volume & close (approximation using
       close-location-in-range proxy)

    Args:
        window: Rolling window for smoothed OFI.
        vpin_n_buckets: Number of volume buckets for VPIN calculation.
    """

    def __init__(
        self,
        window: int = 20,
        vpin_n_buckets: int = 50,
    ) -> None:
        self.window = window
        self.vpin_n_buckets = vpin_n_buckets

    # ------------------------------------------------------------------
    # Tick-level OFI (ideal case)
    # ------------------------------------------------------------------

    def compute_ofi_from_trades(
        self,
        trades: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute OFI from signed trade data.

        Expected columns:
            timestamp, price, volume, side (+1 buy, -1 sell)

        Returns:
            DataFrame with [timestamp, ofi, cum_ofi, ofi_ma].
        """
        df = trades.copy()
        df["signed_volume"] = df["volume"] * df["side"]

        # Group by time period (e.g., 1-minute bars)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        ofi = df["signed_volume"].resample("1min").sum().rename("ofi")
        result = ofi.to_frame()
        result["cum_ofi"] = result["ofi"].cumsum()
        result["ofi_ma"] = result["ofi"].rolling(self.window, min_periods=1).mean()

        return result.reset_index()

    # ------------------------------------------------------------------
    # Bar-level OFI (approximation)
    # ------------------------------------------------------------------

    def compute_ofi_from_bars(
        self,
        bars: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Approximate OFI from OHLCV bars using the close-location
        value (CLV) method.

        CLV = (close - low) - (high - close)
              ----------------------------
                     high - low

        OFI = CLV * volume

        This approximates buy/sell pressure: if close is near the
        high, most volume was buyer-initiated.

        Args:
            bars: OHLCV DataFrame with columns [open, high, low, close, volume].

        Returns:
            DataFrame augmented with [clv, ofi, cum_ofi, ofi_ma,
            ofi_momentum].
        """
        df = bars.copy()

        hl_range = df["high"] - df["low"]
        # Avoid division by zero
        hl_range = hl_range.replace(0, np.nan)

        df["clv"] = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl_range
        df["clv"] = df["clv"].fillna(0.0)

        df["ofi"] = df["clv"] * df["volume"]
        df["cum_ofi"] = df["ofi"].cumsum()
        df["ofi_ma"] = df["ofi"].rolling(self.window, min_periods=1).mean()
        df["ofi_momentum"] = df["ofi_ma"].diff(self.window)

        return df

    # ------------------------------------------------------------------
    # VPIN
    # ------------------------------------------------------------------

    def compute_vpin(
        self,
        bars: pd.DataFrame,
        bucket_volume: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Volume-Synchronized Probability of Informed Trading (VPIN).

        VPIN aggregates buy/sell volume into equal-volume buckets
        and measures the imbalance. High VPIN indicates potential
        informed trading (toxic flow).

        Reference: Easley, Lopez de Prado & O'Hara (2012)

        Args:
            bars: OHLCV DataFrame.
            bucket_volume: Volume per bucket. If None, auto-computed
                as total_volume / n_buckets.

        Returns:
            DataFrame with [bucket_id, buy_volume, sell_volume, vpin].
        """
        df = bars.copy()

        # CLV-based buy/sell split
        hl_range = (df["high"] - df["low"]).replace(0, np.nan)
        clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl_range
        clv = clv.fillna(0.0)

        buy_pct = (1 + clv) / 2  # [0, 1]
        df["buy_vol"] = df["volume"] * buy_pct
        df["sell_vol"] = df["volume"] * (1 - buy_pct)

        total_vol = df["volume"].sum()
        if bucket_volume is None:
            bucket_volume = total_vol / self.vpin_n_buckets

        if bucket_volume <= 0:
            return pd.DataFrame(
                columns=["bucket_id", "buy_volume", "sell_volume", "vpin"]
            )

        # Bucket aggregation
        df["cum_vol"] = df["volume"].cumsum()
        df["bucket_id"] = (df["cum_vol"] / bucket_volume).astype(int)

        buckets = (
            df.groupby("bucket_id")
            .agg(
                buy_volume=("buy_vol", "sum"),
                sell_volume=("sell_vol", "sum"),
            )
            .reset_index()
        )

        total_bucket_vol = buckets["buy_volume"] + buckets["sell_volume"]
        total_bucket_vol = total_bucket_vol.replace(0, np.nan)
        buckets["vpin"] = (
            buckets["buy_volume"] - buckets["sell_volume"]
        ).abs() / total_bucket_vol

        return buckets

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        bars: pd.DataFrame,
        ofi_z_threshold: float = 2.0,
    ) -> pd.DataFrame:
        """
        Generate signals from OFI.

        Strong positive OFI (z > threshold) => buy pressure => +1
        Strong negative OFI                 => sell pressure => -1

        Returns:
            DataFrame augmented with [ofi_zscore, signal].
        """
        df = self.compute_ofi_from_bars(bars)

        ofi_mean = df["ofi"].rolling(self.window * 5, min_periods=self.window).mean()
        ofi_std = df["ofi"].rolling(self.window * 5, min_periods=self.window).std()

        df["ofi_zscore"] = np.where(
            ofi_std > 0,
            (df["ofi"] - ofi_mean) / ofi_std,
            0.0,
        )

        df["signal"] = 0
        df.loc[df["ofi_zscore"] > ofi_z_threshold, "signal"] = 1
        df.loc[df["ofi_zscore"] < -ofi_z_threshold, "signal"] = -1

        return df
