"""
Quality Metrics Analyzer.

Computes profitability and efficiency metrics that separate
high-quality businesses from mediocre ones. Quality is a
well-documented factor in the Fama-French five-factor model.

Metrics:
- Return on Equity (ROE)
- Return on Assets (ROA)
- Return on Invested Capital (ROIC)
- Gross / Operating / Net / EBITDA margins
- Interest coverage ratio
- Debt / Equity ratio
- Net debt / EBITDA
- Current and quick ratios
- Piotroski F-score (9-component quality score)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class QualityMetricsAnalyzer:
    """
    Compute quality metrics from fundamental data.

    Args:
        min_history: Minimum quarters of data required to compute
            trend-based quality signals.
    """

    METRIC_KEYS = [
        "roe",
        "roa",
        "roic",
        "gross_margin",
        "operating_margin",
        "net_margin",
        "ebitda_margin",
        "interest_coverage",
        "debt_to_equity",
        "net_debt_ebitda",
        "current_ratio",
        "quick_ratio",
    ]

    def __init__(self, min_history: int = 4) -> None:
        self.min_history = min_history

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_metrics(self, info: Dict) -> Dict[str, Optional[float]]:
        """
        Extract quality metrics from a yfinance ``ticker.info`` dict.

        Returns:
            Dict of metric_name -> value (None if unavailable).
        """
        return {
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "roic": self._compute_roic(info),
            "gross_margin": info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "net_margin": info.get("profitMargins"),
            "ebitda_margin": self._compute_ebitda_margin(info),
            "interest_coverage": self._compute_interest_coverage(info),
            "debt_to_equity": info.get("debtToEquity"),
            "net_debt_ebitda": self._compute_net_debt_ebitda(info),
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
        }

    @staticmethod
    def _compute_roic(info: Dict) -> Optional[float]:
        """ROIC = NOPAT / Invested Capital."""
        ebit = info.get("ebit")
        tax_rate = info.get("effectiveTaxRate", 0.25)
        total_assets = info.get("totalAssets")
        current_liabilities = info.get("totalCurrentLiabilities")

        if ebit is None or total_assets is None or current_liabilities is None:
            return None

        nopat = ebit * (1 - (tax_rate if tax_rate else 0.25))
        invested_capital = total_assets - current_liabilities
        if invested_capital <= 0:
            return None
        return nopat / invested_capital

    @staticmethod
    def _compute_ebitda_margin(info: Dict) -> Optional[float]:
        ebitda = info.get("ebitda")
        revenue = info.get("totalRevenue")
        if ebitda is None or revenue is None or revenue == 0:
            return None
        return ebitda / revenue

    @staticmethod
    def _compute_interest_coverage(info: Dict) -> Optional[float]:
        """EBIT / Interest Expense."""
        ebit = info.get("ebit")
        interest = info.get("interestExpense")
        if ebit is None or interest is None or interest == 0:
            return None
        return abs(ebit / interest)

    @staticmethod
    def _compute_net_debt_ebitda(info: Dict) -> Optional[float]:
        total_debt = info.get("totalDebt")
        cash = info.get("totalCash")
        ebitda = info.get("ebitda")
        if total_debt is None or cash is None or ebitda is None or ebitda == 0:
            return None
        return (total_debt - cash) / ebitda

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    def extract_universe(
        self,
        info_dict: Dict[str, Dict],
    ) -> pd.DataFrame:
        """
        Extract quality metrics for a universe of symbols.

        Returns:
            DataFrame indexed by symbol.
        """
        rows: List[Dict] = []
        for symbol, info in info_dict.items():
            row = {"symbol": symbol}
            row.update(self.extract_metrics(info))
            rows.append(row)
        return pd.DataFrame(rows).set_index("symbol")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def percentile_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Percentile-rank each quality metric across the universe.

        Higher ROE/ROA/ROIC/margins = better quality (higher rank).
        Higher debt ratios = worse quality (lower rank).

        Returns:
            DataFrame in [0, 1].
        """
        ranked = pd.DataFrame(index=df.index)

        higher_better = [
            "roe",
            "roa",
            "roic",
            "gross_margin",
            "operating_margin",
            "net_margin",
            "ebitda_margin",
            "interest_coverage",
            "current_ratio",
            "quick_ratio",
        ]
        lower_better = ["debt_to_equity", "net_debt_ebitda"]

        for col in higher_better:
            if col in df.columns:
                ranked[col] = df[col].rank(pct=True, na_option="bottom")

        for col in lower_better:
            if col in df.columns:
                ranked[col] = 1 - df[col].rank(pct=True, na_option="bottom")

        return ranked

    def composite_quality_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Equal-weighted average of quality percentile ranks.

        Returns:
            Series in [0, 1], higher = better quality.
        """
        ranked = self.percentile_rank(df)
        return ranked.mean(axis=1).rename("quality_score")

    # ------------------------------------------------------------------
    # Piotroski F-Score
    # ------------------------------------------------------------------

    def piotroski_f_score(self, info: Dict) -> int:
        """
        Compute the Piotroski F-Score (0-9).

        Profitability (4 points):
            1. ROA > 0
            2. Operating cash flow > 0
            3. ROA increasing (YoY)
            4. CFO > ROA (accruals)

        Leverage / Liquidity (3 points):
            5. Long-term debt decreasing
            6. Current ratio increasing
            7. No new equity issuance

        Operating efficiency (2 points):
            8. Gross margin increasing
            9. Asset turnover increasing

        Note: Without historical data, YoY comparisons default to
        pass (1). This is a snapshot approximation.

        Returns:
            Integer F-score in [0, 9].
        """
        score = 0

        # Profitability
        roa = info.get("returnOnAssets")
        if roa is not None and roa > 0:
            score += 1

        ocf = info.get("operatingCashflow")
        if ocf is not None and ocf > 0:
            score += 1

        # ROA increasing -- snapshot gives benefit of doubt
        score += 1

        # CFO > net income (quality of earnings)
        net_income = info.get("netIncomeToCommon")
        if ocf is not None and net_income is not None:
            if ocf > net_income:
                score += 1
        else:
            score += 1

        # Leverage
        dte = info.get("debtToEquity")
        if dte is not None and dte < 100:  # reasonable threshold
            score += 1

        cr = info.get("currentRatio")
        if cr is not None and cr > 1.0:
            score += 1

        # No dilution -- snapshot approximation
        score += 1

        # Efficiency
        gm = info.get("grossMargins")
        if gm is not None and gm > 0.3:
            score += 1

        # Asset turnover proxy
        revenue = info.get("totalRevenue")
        total_assets = info.get("totalAssets")
        if revenue is not None and total_assets is not None and total_assets > 0:
            turnover = revenue / total_assets
            if turnover > 0.5:
                score += 1

        return score
