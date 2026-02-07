"""
Bid-Ask Spread Analyzer.

Analyzes bid-ask spread dynamics as a proxy for:
- Liquidity conditions
- Transaction costs
- Information asymmetry

When actual L2 data is unavailable, estimates effective spreads
from OHLC data using established estimators:
- Roll (1984) estimator
- Corwin-Schultz (2012) high-low estimator
- Abdi-Ranaldo (2017) close-high-low estimator
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BidAskSpreadAnalyzer:
    """
    Estimate and analyze bid-ask spreads.

    Supports both direct spread data and OHLC-based estimation
    when Level 2 data is unavailable (our typical case with yfinance).

    Args:
        window: Rolling window for smoothed spread metrics.
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window

    # ------------------------------------------------------------------
    # Direct spread (if L2 data available)
    # ------------------------------------------------------------------

    def compute_from_quotes(
        self,
        quotes: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute spread metrics from bid/ask quotes.

        Expected columns: timestamp, bid, ask, mid (optional)

        Returns:
            DataFrame with [spread, relative_spread, spread_ma].
        """
        df = quotes.copy()
        df["mid"] = (df["bid"] + df["ask"]) / 2
        df["spread"] = df["ask"] - df["bid"]
        df["relative_spread"] = df["spread"] / df["mid"]
        df["spread_ma"] = df["spread"].rolling(self.window, min_periods=1).mean()
        df["relative_spread_ma"] = (
            df["relative_spread"].rolling(self.window, min_periods=1).mean()
        )

        return df

    # ------------------------------------------------------------------
    # OHLC-based estimators
    # ------------------------------------------------------------------

    def roll_estimator(self, bars: pd.DataFrame) -> pd.Series:
        """
        Roll (1984) spread estimator.

        S = 2 * sqrt(-Cov(r_t, r_{t-1}))

        Based on the serial covariance of price changes. Negative
        serial covariance indicates bid-ask bounce.

        Args:
            bars: OHLCV DataFrame with 'close' column.

        Returns:
            Series of estimated spreads (NaN where covariance is positive).
        """
        returns = bars["close"].pct_change()

        cov = returns.rolling(self.window).apply(
            lambda x: np.cov(x[:-1], x[1:])[0, 1] if len(x) > 1 else np.nan,
            raw=True,
        )

        # Spread is only defined when covariance is negative
        spread = np.where(cov < 0, 2.0 * np.sqrt(-cov), np.nan)
        return pd.Series(spread, index=bars.index, name="roll_spread")

    def corwin_schultz_estimator(self, bars: pd.DataFrame) -> pd.Series:
        """
        Corwin-Schultz (2012) high-low spread estimator.

        Uses the ratio of high-low range across 1-day and 2-day
        windows to decompose spread from volatility.

        S = 2 * (e^alpha - 1) / (1 + e^alpha)

        where alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2))
              - sqrt(gamma / (3 - 2*sqrt(2)))
        beta  = sum of squared log(H/L) over 2 consecutive days
        gamma = log(H_2d / L_2d)^2

        Args:
            bars: OHLCV DataFrame with 'high' and 'low' columns.

        Returns:
            Series of estimated spreads.
        """
        high = bars["high"].values
        low = bars["low"].values

        n = len(high)
        spread = np.full(n, np.nan)

        for i in range(1, n):
            # Single-day log range
            h1 = np.log(high[i] / low[i])
            h0 = np.log(high[i - 1] / low[i - 1])

            # Beta: sum of squared single-day ranges
            beta = h1**2 + h0**2

            # 2-day high-low
            h2d = max(high[i], high[i - 1])
            l2d = min(low[i], low[i - 1])
            gamma = np.log(h2d / l2d) ** 2 if l2d > 0 else 0

            denom = 3 - 2 * np.sqrt(2)
            if denom == 0:
                continue

            alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)

            if alpha > 0:
                s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
                spread[i] = max(s, 0)

        return pd.Series(spread, index=bars.index, name="cs_spread")

    def abdi_ranaldo_estimator(self, bars: pd.DataFrame) -> pd.Series:
        """
        Abdi-Ranaldo (2017) close-high-low spread estimator.

        A simplified estimator using:
        S^2 = 4 * (close_t - mid_HL_t) * (close_t - mid_HL_{t+1})

        where mid_HL = (high + low) / 2

        This estimator is more robust than Roll and simpler than CS.

        Returns:
            Series of estimated spreads.
        """
        close = bars["close"].values
        mid_hl = ((bars["high"] + bars["low"]) / 2).values

        n = len(close)
        sq_spread = np.full(n, np.nan)

        for i in range(n - 1):
            sq_spread[i] = 4 * (close[i] - mid_hl[i]) * (close[i] - mid_hl[i + 1])

        # Take sqrt where positive, NaN otherwise
        spread = np.where(sq_spread > 0, np.sqrt(sq_spread), np.nan)
        return pd.Series(spread, index=bars.index, name="ar_spread")

    # ------------------------------------------------------------------
    # Composite
    # ------------------------------------------------------------------

    def estimate_spread(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all three OHLC spread estimators and a consensus.

        Returns:
            DataFrame with [roll_spread, cs_spread, ar_spread,
            consensus_spread, liquidity_score].
        """
        df = bars.copy()
        df["roll_spread"] = self.roll_estimator(bars)
        df["cs_spread"] = self.corwin_schultz_estimator(bars)
        df["ar_spread"] = self.abdi_ranaldo_estimator(bars)

        # Consensus: median of available estimators
        spread_cols = df[["roll_spread", "cs_spread", "ar_spread"]]
        df["consensus_spread"] = spread_cols.median(axis=1)

        # Liquidity score: inverse of spread percentile
        df["liquidity_score"] = 1 - df["consensus_spread"].rank(pct=True)

        return df

    def generate_signals(
        self,
        bars: pd.DataFrame,
        illiquidity_z_threshold: float = 2.0,
    ) -> pd.DataFrame:
        """
        Generate liquidity-based signals.

        Sudden widening of spreads (illiquidity spike) can signal:
        - Risk-off / reduce positions
        - Potential dislocations to exploit

        Returns:
            DataFrame with [consensus_spread, spread_zscore, signal].
        """
        df = self.estimate_spread(bars)

        spread_ma = (
            df["consensus_spread"]
            .rolling(self.window * 5, min_periods=self.window)
            .mean()
        )
        spread_std = (
            df["consensus_spread"]
            .rolling(self.window * 5, min_periods=self.window)
            .std()
        )

        df["spread_zscore"] = np.where(
            spread_std > 0,
            (df["consensus_spread"] - spread_ma) / spread_std,
            0.0,
        )

        # High spread = reduce exposure (-1), normal = neutral (0)
        df["signal"] = 0
        df.loc[df["spread_zscore"] > illiquidity_z_threshold, "signal"] = -1

        return df
