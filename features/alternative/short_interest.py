"""
Short Interest Analyzer.

Analyses short interest data to detect short squeeze potential
and bearish sentiment extremes.

High short interest can indicate:
- Potential short squeeze (contrarian long)
- Fundamental problems (follow the shorts)

Key metrics:
- Short interest as % of float
- Days to cover (short interest / avg daily volume)
- Short ratio trend changes
"""

import logging
from typing import List, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ShortInterestAnalyzer:
    """
    Analyze short interest data and generate squeeze / bearish signals.

    Args:
        short_interest_threshold: Short interest (% of float) above
            which squeeze potential is elevated.
        days_to_cover_threshold: Days-to-cover above which covering
            pressure intensifies.
        squeeze_score_threshold: Combined squeeze score above which
            a contrarian long signal is generated.
    """

    def __init__(
        self,
        short_interest_threshold: float = 0.20,
        days_to_cover_threshold: float = 5.0,
        squeeze_score_threshold: float = 1.5,
    ) -> None:
        self.si_threshold = short_interest_threshold
        self.dtc_threshold = days_to_cover_threshold
        self.squeeze_threshold = squeeze_score_threshold

    # ------------------------------------------------------------------
    # Squeeze scoring
    # ------------------------------------------------------------------

    def calculate_squeeze_potential(
        self,
        short_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute a composite short-squeeze potential score.

        Scoring components (normalised to threshold = 1.0):
            1. SI score   = short_interest / si_threshold        (weight 0.5)
            2. DTC score  = days_to_cover  / dtc_threshold       (weight 0.3)
            3. Ratio score = short_ratio (raw, fill 0)           (weight 0.2)

        squeeze_score = 0.5 * si_score + 0.3 * dtc_score + 0.2 * ratio_score

        Args:
            short_data: Must contain 'short_interest' (fraction of float)
                and 'days_to_cover'. Optionally 'short_ratio'.

        Returns:
            Input DataFrame augmented with [si_score, dtc_score,
            squeeze_score].
        """
        df = short_data.copy()
        df["si_score"] = df["short_interest"] / self.si_threshold
        df["dtc_score"] = df["days_to_cover"] / self.dtc_threshold
        ratio = df["short_ratio"].fillna(0) if "short_ratio" in df.columns else 0.0

        df["squeeze_score"] = df["si_score"] * 0.5 + df["dtc_score"] * 0.3 + ratio * 0.2
        return df

    def detect_squeeze_buildup(
        self,
        short_data: pd.DataFrame,
        lookback: int = 5,
    ) -> pd.DataFrame:
        """
        Detect *increasing* short interest over recent reporting periods.

        A rising short interest with stable/rising price is the classic
        squeeze setup.

        Args:
            short_data: Must contain 'date', 'symbol', 'short_interest'.
            lookback: Number of reporting periods to check for uptrend.

        Returns:
            DataFrame augmented with [si_change, si_increasing].
        """
        df = short_data.copy()
        df = df.sort_values(["symbol", "date"])

        df["si_change"] = df.groupby("symbol")["short_interest"].diff()
        df["si_increasing"] = (
            df.groupby("symbol")["si_change"].transform(
                lambda x: (x > 0).rolling(lookback, min_periods=1).mean()
            )
            > 0.6
        )
        return df

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, short_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate contrarian long signals from short interest data.

        High squeeze score => contrarian long (potential squeeze).

        Returns:
            DataFrame with [symbol, squeeze_score, signal].
        """
        scored = self.calculate_squeeze_potential(short_data)
        scored["signal"] = 0
        scored.loc[scored["squeeze_score"] > self.squeeze_threshold, "signal"] = 1

        cols = ["symbol", "squeeze_score", "signal"]
        out_cols = [c for c in cols if c in scored.columns]
        return scored[out_cols].copy()

    def generate_bearish_signals(
        self,
        short_data: pd.DataFrame,
        min_si: float = 0.05,
        max_dtc: float = 2.0,
    ) -> pd.DataFrame:
        """
        Generate *follow-the-shorts* bearish signals.

        Logic: moderate short interest + low days-to-cover
        (easy to cover quickly) indicates informed shorting
        rather than squeeze risk.

        Returns:
            DataFrame with [symbol, signal] where signal = -1.
        """
        df = short_data.copy()
        bearish_mask = (df["short_interest"] >= min_si) & (
            df["days_to_cover"] <= max_dtc
        )
        df["signal"] = 0
        df.loc[bearish_mask, "signal"] = -1

        cols = ["symbol", "signal"]
        out_cols = [c for c in cols if c in df.columns]
        return df.loc[bearish_mask, out_cols].copy()
