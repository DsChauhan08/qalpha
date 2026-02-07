"""
Options Market Sentiment Analyzer.

Extracts sentiment signals from options market data.
The options market is often a leading indicator because
informed traders use options for leverage and anonymity.

Key indicators:
- Put / Call ratio (contrarian)
- Implied Volatility skew (OTM put IV - ATM call IV)
- Unusual options volume (z-score of total volume)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OptionsSentimentAnalyzer:
    """
    Analyze options market data to produce sentiment signals.

    Uses a *contrarian* interpretation:
    - High put/call ratio => retail is too bearish => bullish signal
    - High IV skew       => fear is overpriced     => bullish signal
    - Unusual call volume => informed buying        => bullish signal

    Args:
        lookback: Rolling window for moving averages.
        pc_ratio_upper: Put/call ratio multiple above MA to trigger
            contrarian long.
        pc_ratio_lower: Put/call ratio multiple below MA to trigger
            contrarian short.
        volume_z_threshold: Z-score threshold for unusual volume.
    """

    def __init__(
        self,
        lookback: int = 20,
        pc_ratio_upper: float = 1.5,
        pc_ratio_lower: float = 0.5,
        volume_z_threshold: float = 2.0,
    ) -> None:
        self.lookback = lookback
        self.pc_upper = pc_ratio_upper
        self.pc_lower = pc_ratio_lower
        self.vol_z = volume_z_threshold

    # ------------------------------------------------------------------
    # Indicator calculations
    # ------------------------------------------------------------------

    def calculate_put_call_ratio(self, options_data: pd.DataFrame) -> pd.Series:
        """
        Put / Call volume ratio.

        Args:
            options_data: Must contain 'put_volume' and 'call_volume'.

        Returns:
            Series of put/call ratios.
        """
        call_vol = options_data["call_volume"].replace(0, np.nan)
        return options_data["put_volume"] / call_vol

    def calculate_iv_skew(self, options_data: pd.DataFrame) -> pd.Series:
        """
        Implied volatility skew: OTM put IV minus ATM call IV.

        Positive skew => market pricing crash protection more expensively
        (fear premium).

        Args:
            options_data: Must contain 'otm_put_iv' and 'atm_call_iv'.

        Returns:
            IV skew series.
        """
        return options_data["otm_put_iv"] - options_data["atm_call_iv"]

    def detect_unusual_volume(
        self,
        options_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Flag dates with unusual total options volume via z-score.

        Returns:
            Input DataFrame augmented with [volume_ma, volume_std,
            volume_zscore, unusual_volume].
        """
        df = options_data.copy()
        df["volume_ma"] = (
            df["total_volume"].rolling(self.lookback, min_periods=1).mean()
        )
        df["volume_std"] = (
            df["total_volume"].rolling(self.lookback, min_periods=1).std()
        )

        df["volume_zscore"] = np.where(
            df["volume_std"] > 0,
            (df["total_volume"] - df["volume_ma"]) / df["volume_std"],
            0.0,
        )
        df["unusual_volume"] = df["volume_zscore"] > self.vol_z
        return df

    # ------------------------------------------------------------------
    # Composite signal
    # ------------------------------------------------------------------

    def generate_signals(self, options_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate composite sentiment signals from options data.

        Sub-signals (all contrarian):
            1. Put/Call ratio vs. its MA -> bullish when PC high
            2. IV skew vs. its MA -> bullish when fear premium high
            3. Unusual call volume -> bullish
            4. Unusual put volume -> bearish

        Final signal is the sign of the equally-weighted sum.

        Returns:
            DataFrame with [signal, pc_ratio, iv_skew, volume_zscore,
            pc_signal, iv_signal, vol_signal].
        """
        df = options_data.copy()

        # Put / Call
        pc = self.calculate_put_call_ratio(df)
        pc_ma = pc.rolling(self.lookback, min_periods=1).mean()

        df["pc_ratio"] = pc
        df["pc_signal"] = 0
        df.loc[pc > pc_ma * self.pc_upper, "pc_signal"] = 1  # contrarian long
        df.loc[pc < pc_ma * self.pc_lower, "pc_signal"] = -1  # contrarian short

        # IV skew
        if {"otm_put_iv", "atm_call_iv"}.issubset(df.columns):
            skew = self.calculate_iv_skew(df)
            skew_ma = skew.rolling(self.lookback, min_periods=1).mean()
            skew_std = skew.rolling(self.lookback, min_periods=1).std()
            df["iv_skew"] = skew
            df["iv_signal"] = 0
            df.loc[skew > skew_ma + skew_std, "iv_signal"] = (
                1  # fear premium -> bullish
            )
        else:
            df["iv_skew"] = np.nan
            df["iv_signal"] = 0

        # Unusual volume
        if "total_volume" in df.columns:
            df = self.detect_unusual_volume(df)
            df["vol_signal"] = 0
            call_unusual = df["unusual_volume"] & (df["call_volume"] > df["put_volume"])
            put_unusual = df["unusual_volume"] & (df["put_volume"] > df["call_volume"])
            df.loc[call_unusual, "vol_signal"] = 1
            df.loc[put_unusual, "vol_signal"] = -1
        else:
            df["volume_zscore"] = 0.0
            df["vol_signal"] = 0

        # Composite
        df["signal"] = np.sign(
            df["pc_signal"] + df["iv_signal"] + df["vol_signal"]
        ).astype(int)

        return df[
            [
                "signal",
                "pc_ratio",
                "iv_skew",
                "volume_zscore",
                "pc_signal",
                "iv_signal",
                "vol_signal",
            ]
        ].copy()

    def calculate_fear_greed_index(
        self,
        options_data: pd.DataFrame,
    ) -> pd.Series:
        """
        Simple fear/greed index from options data.

        Composite of normalised PC ratio, IV skew, and volume z-score,
        scaled to [0, 100] where 0 = extreme fear, 100 = extreme greed.

        Returns:
            Series of fear/greed index values.
        """
        pc = self.calculate_put_call_ratio(options_data)

        # Percentile rank (inverted: high PC = fear)
        pc_rank = pc.rank(pct=True)
        fear_from_pc = 1 - pc_rank  # high PC -> low greed

        # IV skew rank (inverted)
        if {"otm_put_iv", "atm_call_iv"}.issubset(options_data.columns):
            skew = self.calculate_iv_skew(options_data)
            skew_rank = skew.rank(pct=True)
            fear_from_skew = 1 - skew_rank
        else:
            fear_from_skew = pd.Series(0.5, index=options_data.index)

        index = (fear_from_pc * 0.6 + fear_from_skew * 0.4) * 100
        return index.clip(0, 100)
