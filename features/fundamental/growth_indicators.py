"""
Growth Indicator Analyzer.

Measures revenue, earnings, and cash-flow growth rates to identify
companies in accelerating or decelerating growth phases.

Metrics:
- Revenue growth rate (YoY, QoQ)
- EPS growth rate
- EBITDA growth
- Book value growth
- Dividend growth rate (CAGR)
- Growth acceleration / deceleration
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GrowthIndicatorAnalyzer:
    """
    Compute growth indicators from fundamental data.

    Args:
        min_periods: Minimum data points for growth calculations.
    """

    def __init__(self, min_periods: int = 2) -> None:
        self.min_periods = min_periods

    # ------------------------------------------------------------------
    # Snapshot growth from ticker.info
    # ------------------------------------------------------------------

    def extract_growth_snapshot(self, info: Dict) -> Dict[str, Optional[float]]:
        """
        Extract point-in-time growth metrics from yfinance info.

        Returns:
            Dict of growth metric -> value.
        """
        return {
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
            "revenue_per_share_growth": self._compute_rps_growth(info),
            "peg_ratio": info.get("pegRatio"),
        }

    @staticmethod
    def _compute_rps_growth(info: Dict) -> Optional[float]:
        """Revenue-per-share growth proxy."""
        rps = info.get("revenuePerShare")
        if rps is None:
            return None
        # Without historical RPS, return None
        return None

    # ------------------------------------------------------------------
    # Time-series growth from financials
    # ------------------------------------------------------------------

    def compute_growth_rates(
        self,
        financials: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute YoY growth rates from quarterly/annual financials.

        Args:
            financials: DataFrame with DatetimeIndex (period dates)
                and financial line items as columns.
            columns: Specific columns to compute growth for.
                Defaults to all numeric columns.

        Returns:
            DataFrame of YoY percentage changes.
        """
        if columns is None:
            columns = financials.select_dtypes(include=[np.number]).columns.tolist()

        df = financials[columns].copy()
        # Sort chronologically
        df = df.sort_index()

        growth = df.pct_change()
        growth.columns = [f"{c}_growth" for c in growth.columns]

        return growth

    def compute_growth_acceleration(
        self,
        growth_rates: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute the *change* in growth rates (second derivative).

        Accelerating growth (positive acceleration) is a strong
        bullish signal.

        Returns:
            DataFrame of growth accelerations.
        """
        accel = growth_rates.diff()
        accel.columns = [c.replace("_growth", "_acceleration") for c in accel.columns]
        return accel

    def compute_cagr(
        self,
        series: pd.Series,
        periods_per_year: int = 4,
    ) -> float:
        """
        Compound Annual Growth Rate.

        CAGR = (end / begin) ^ (1 / n_years) - 1

        Args:
            series: Time series of values (e.g., revenue per quarter).
            periods_per_year: 4 for quarterly, 1 for annual.

        Returns:
            CAGR as a float.
        """
        vals = series.dropna()
        if len(vals) < self.min_periods:
            return 0.0

        begin = float(vals.iloc[0])
        end = float(vals.iloc[-1])

        if begin <= 0 or end <= 0:
            return 0.0

        n_years = len(vals) / periods_per_year
        if n_years <= 0:
            return 0.0

        return (end / begin) ** (1.0 / n_years) - 1.0

    # ------------------------------------------------------------------
    # Universe scoring
    # ------------------------------------------------------------------

    def extract_universe(
        self,
        info_dict: Dict[str, Dict],
    ) -> pd.DataFrame:
        """
        Extract growth snapshots for a universe.

        Returns:
            DataFrame indexed by symbol.
        """
        rows: List[Dict] = []
        for symbol, info in info_dict.items():
            row = {"symbol": symbol}
            row.update(self.extract_growth_snapshot(info))
            rows.append(row)
        return pd.DataFrame(rows).set_index("symbol")

    def composite_growth_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Composite growth score from percentile ranks.

        Higher growth => higher score.

        Returns:
            Series in [0, 1].
        """
        growth_cols = [
            c for c in df.columns if df[c].dtype in [np.float64, np.float32, float]
        ]
        if not growth_cols:
            return pd.Series(0.5, index=df.index, name="growth_score")

        ranked = df[growth_cols].rank(pct=True, na_option="bottom")
        return ranked.mean(axis=1).rename("growth_score")

    def classify_growth_phase(
        self,
        revenue_growth: float,
        earnings_growth: float,
    ) -> str:
        """
        Classify a company's growth phase.

        Categories:
            'hyper_growth'     : rev > 30% and earnings > 30%
            'growth'           : rev > 10% or earnings > 15%
            'stable'           : rev in [-5%, 10%] and earnings in [-10%, 15%]
            'decelerating'     : positive but declining growth
            'contracting'      : negative growth

        Returns:
            Phase label string.
        """
        if revenue_growth is None or earnings_growth is None:
            return "unknown"

        if revenue_growth > 0.30 and earnings_growth > 0.30:
            return "hyper_growth"
        if revenue_growth > 0.10 or earnings_growth > 0.15:
            return "growth"
        if -0.05 <= revenue_growth <= 0.10 and -0.10 <= earnings_growth <= 0.15:
            return "stable"
        if revenue_growth < 0 or earnings_growth < 0:
            return "contracting"
        return "decelerating"
