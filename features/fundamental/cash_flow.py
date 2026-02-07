"""
Cash Flow Analyzer.

Analyzes cash flow statements to evaluate financial health
and intrinsic value. Cash flow is harder to manipulate than
earnings and provides a truer picture of financial strength.

Metrics:
- Free Cash Flow (FCF) = OCF - CapEx
- Operating Cash Flow (OCF)
- FCF yield = FCF / Market Cap
- Cash conversion ratio = OCF / Net Income
- CapEx intensity = CapEx / Revenue
- Cash flow stability (coefficient of variation)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CashFlowAnalyzer:
    """
    Analyze cash flow data for quality and valuation signals.

    Args:
        min_periods: Minimum quarters of data for trend analysis.
    """

    def __init__(self, min_periods: int = 4) -> None:
        self.min_periods = min_periods

    # ------------------------------------------------------------------
    # Snapshot from ticker.info
    # ------------------------------------------------------------------

    def extract_snapshot(self, info: Dict) -> Dict[str, Optional[float]]:
        """
        Extract cash-flow metrics from yfinance ticker.info.

        Returns:
            Dict of metric_name -> value.
        """
        ocf = info.get("operatingCashflow")
        fcf = info.get("freeCashflow")
        mcap = info.get("marketCap")
        net_income = info.get("netIncomeToCommon")
        revenue = info.get("totalRevenue")
        capex = None

        # Derive capex = OCF - FCF
        if ocf is not None and fcf is not None:
            capex = ocf - fcf

        return {
            "ocf": ocf,
            "fcf": fcf,
            "capex": capex,
            "fcf_yield": fcf / mcap
            if (fcf is not None and mcap and mcap > 0)
            else None,
            "cash_conversion": (
                ocf / net_income
                if (ocf is not None and net_income and net_income != 0)
                else None
            ),
            "capex_intensity": (
                abs(capex) / revenue
                if (capex is not None and revenue and revenue > 0)
                else None
            ),
        }

    # ------------------------------------------------------------------
    # Time-series analysis
    # ------------------------------------------------------------------

    def compute_fcf_from_statements(
        self,
        cash_flow_stmt: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute FCF from cash flow statement DataFrame.

        FCF = Operating Cash Flow - Capital Expenditures

        Args:
            cash_flow_stmt: DataFrame with columns containing
                'Operating Cash Flow' or similar, and 'Capital Expenditure'.

        Returns:
            Series of FCF values.
        """
        ocf_col = self._find_column(
            cash_flow_stmt,
            ["Operating Cash Flow", "Total Cash From Operating Activities"],
        )
        capex_col = self._find_column(
            cash_flow_stmt, ["Capital Expenditure", "Capital Expenditures"]
        )

        if ocf_col is None:
            return pd.Series(dtype=float, name="fcf")

        ocf = pd.to_numeric(cash_flow_stmt[ocf_col], errors="coerce")
        capex = (
            pd.to_numeric(cash_flow_stmt[capex_col], errors="coerce")
            if capex_col
            else 0
        )

        return (ocf - capex.abs()).rename("fcf")

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find the first matching column name."""
        for c in candidates:
            if c in df.columns:
                return c
            # Case-insensitive
            matches = [col for col in df.columns if c.lower() in col.lower()]
            if matches:
                return matches[0]
        return None

    def compute_stability(self, series: pd.Series) -> float:
        """
        Cash flow stability = 1 / (1 + CV).

        CV = coefficient of variation = std / |mean|.
        Stability near 1.0 = very stable cash flows.
        Stability near 0.0 = highly volatile cash flows.
        """
        vals = series.dropna()
        if len(vals) < self.min_periods:
            return 0.5

        mean = vals.mean()
        std = vals.std()

        if abs(mean) < 1e-10:
            return 0.0

        cv = abs(std / mean)
        return 1.0 / (1.0 + cv)

    def compute_fcf_growth(
        self,
        fcf_series: pd.Series,
        periods_per_year: int = 4,
    ) -> float:
        """FCF CAGR over the available history."""
        vals = fcf_series.dropna()
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
    # Universe
    # ------------------------------------------------------------------

    def extract_universe(
        self,
        info_dict: Dict[str, Dict],
    ) -> pd.DataFrame:
        """
        Extract cash flow snapshots for a universe.

        Returns:
            DataFrame indexed by symbol.
        """
        rows: List[Dict] = []
        for symbol, info in info_dict.items():
            row = {"symbol": symbol}
            row.update(self.extract_snapshot(info))
            rows.append(row)
        return pd.DataFrame(rows).set_index("symbol")

    def composite_cashflow_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Composite cash flow quality score.

        Weights: FCF yield (40%), cash conversion (30%),
        low capex intensity (30%).

        Returns:
            Series in [0, 1].
        """
        scores = pd.DataFrame(index=df.index)

        if "fcf_yield" in df.columns:
            scores["fcf"] = df["fcf_yield"].rank(pct=True, na_option="bottom")
        if "cash_conversion" in df.columns:
            # Higher is better, but cap at 1.5 (>1 is fine)
            capped = df["cash_conversion"].clip(upper=2.0)
            scores["conv"] = capped.rank(pct=True, na_option="bottom")
        if "capex_intensity" in df.columns:
            # Lower is better
            scores["capex"] = 1 - df["capex_intensity"].rank(
                pct=True, na_option="bottom"
            )

        if scores.empty:
            return pd.Series(0.5, index=df.index, name="cashflow_score")

        return scores.mean(axis=1).rename("cashflow_score")
