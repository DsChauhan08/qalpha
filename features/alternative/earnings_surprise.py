"""
Earnings Surprise Detector.

Detects and scores earnings surprises for PEAD
(Post-Earnings Announcement Drift) trading.

Academic research documents a significant drift in the direction
of the surprise for 60-90 days after the announcement. This is
one of the most robust anomalies in finance.

Key metrics:
- Surprise % = (actual EPS - estimate EPS) / |estimate EPS|
- Standardised Unexpected Earnings (SUE)
- Consecutive surprise streaks
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EarningsSurpriseDetector:
    """
    Detect and score earnings surprises for PEAD signals.

    Args:
        surprise_threshold: Minimum absolute surprise % to trigger
            a signal.
        holding_period: Days to hold a PEAD position.
        sue_lookback: Quarters of EPS history used to compute
            Standardised Unexpected Earnings.
    """

    def __init__(
        self,
        surprise_threshold: float = 0.10,
        holding_period: int = 5,
        sue_lookback: int = 8,
    ) -> None:
        self.surprise_threshold = surprise_threshold
        self.holding_period = holding_period
        self.sue_lookback = sue_lookback

    # ------------------------------------------------------------------
    # Surprise calculation
    # ------------------------------------------------------------------

    def calculate_surprise(self, earnings_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute earnings surprise and surprise percentage.

        Expected columns:
            symbol, announcement_date, actual_eps, estimate_eps

        Returns:
            Input DataFrame augmented with [surprise, surprise_pct].
        """
        df = earnings_data.copy()
        df["surprise"] = df["actual_eps"] - df["estimate_eps"]
        # Guard against zero estimate
        denom = df["estimate_eps"].abs().replace(0, np.nan)
        df["surprise_pct"] = df["surprise"] / denom
        return df

    def calculate_sue(self, earnings_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Standardised Unexpected Earnings (SUE).

        SUE = (actual_eps - E[eps]) / std(eps)

        where E[eps] and std(eps) are computed from the trailing
        ``sue_lookback`` quarters.

        Returns:
            DataFrame augmented with [sue].
        """
        df = earnings_data.copy()
        df = df.sort_values(["symbol", "announcement_date"])

        sue_values: List[float] = []
        for _, grp in df.groupby("symbol"):
            actuals = grp["actual_eps"].values
            sues = np.full(len(actuals), np.nan)
            for i in range(self.sue_lookback, len(actuals)):
                window = actuals[i - self.sue_lookback : i]
                mu = window.mean()
                sigma = window.std()
                if sigma > 0:
                    sues[i] = (actuals[i] - mu) / sigma
            sue_values.extend(sues.tolist())

        df["sue"] = sue_values
        return df

    def detect_streak(self, earnings_data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect consecutive beat/miss streaks per symbol.

        Streaks of 3+ consecutive beats (or misses) amplify the PEAD effect.

        Returns:
            DataFrame augmented with [streak] (positive = consecutive beats,
            negative = consecutive misses).
        """
        df = self.calculate_surprise(earnings_data)
        df = df.sort_values(["symbol", "announcement_date"])

        streaks: List[int] = []
        for _, grp in df.groupby("symbol"):
            surp = grp["surprise"].values
            streak = 0
            for val in surp:
                if val > 0:
                    streak = max(streak, 0) + 1
                elif val < 0:
                    streak = min(streak, 0) - 1
                else:
                    streak = 0
                streaks.append(streak)

        df["streak"] = streaks
        return df

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, earnings_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate PEAD signals from earnings surprises.

        Signal logic:
            surprise_pct >  threshold => +1 (long, drift up)
            surprise_pct < -threshold => -1 (short, drift down)

        Returns:
            DataFrame with [symbol, announcement_date, signal,
            surprise_pct].
        """
        df = self.calculate_surprise(earnings_data)

        significant = df["surprise_pct"].abs() > self.surprise_threshold
        signals = df.loc[significant].copy()
        signals["signal"] = np.sign(signals["surprise_pct"]).astype(int)

        out_cols = ["symbol", "announcement_date", "signal", "surprise_pct"]
        return signals[[c for c in out_cols if c in signals.columns]]

    def generate_sue_signals(
        self,
        earnings_data: pd.DataFrame,
        sue_threshold: float = 2.0,
    ) -> pd.DataFrame:
        """
        Generate signals using SUE instead of raw surprise %.

        SUE is more robust because it normalises by historical EPS
        volatility.

        Returns:
            DataFrame with [symbol, announcement_date, signal, sue].
        """
        df = self.calculate_sue(earnings_data)
        df = df.dropna(subset=["sue"])

        significant = df["sue"].abs() > sue_threshold
        signals = df.loc[significant].copy()
        signals["signal"] = np.sign(signals["sue"]).astype(int)

        out_cols = ["symbol", "announcement_date", "signal", "sue"]
        return signals[[c for c in out_cols if c in signals.columns]]

    def score_upcoming_earnings(
        self,
        historical_earnings: pd.DataFrame,
        upcoming_symbols: List[str],
    ) -> pd.DataFrame:
        """
        Score symbols with upcoming earnings based on historical
        surprise patterns.

        Symbols with consistent positive surprises (streaks) are more
        likely to beat again.

        Returns:
            DataFrame with [symbol, avg_surprise, streak, beat_rate,
            predictability_score].
        """
        df = self.detect_streak(historical_earnings)

        results: List[Dict] = []
        for symbol in upcoming_symbols:
            sym_data = df[df["symbol"] == symbol]
            if sym_data.empty:
                continue

            n_reports = len(sym_data)
            n_beats = int((sym_data["surprise"] > 0).sum())
            avg_surp = float(sym_data["surprise_pct"].mean())
            latest_streak = int(sym_data["streak"].iloc[-1])
            beat_rate = n_beats / n_reports if n_reports > 0 else 0.0

            # Predictability: high beat rate + low surprise variance
            surp_std = float(sym_data["surprise_pct"].std()) if n_reports > 1 else 1.0
            predictability = beat_rate / (1 + surp_std)

            results.append(
                {
                    "symbol": symbol,
                    "avg_surprise": avg_surp,
                    "streak": latest_streak,
                    "beat_rate": beat_rate,
                    "predictability_score": predictability,
                }
            )

        return (
            pd.DataFrame(results)
            if results
            else pd.DataFrame(
                columns=[
                    "symbol",
                    "avg_surprise",
                    "streak",
                    "beat_rate",
                    "predictability_score",
                ]
            )
        )
