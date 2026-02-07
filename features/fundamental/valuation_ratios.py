"""
Valuation Ratio Analyzer.

Computes and analyses key valuation ratios that indicate whether
a security is cheap or expensive relative to fundamentals.

Ratios implemented:
- P/E (trailing and forward)
- P/B (price to book)
- P/S (price to sales)
- EV/EBITDA
- EV/Sales
- PEG ratio
- P/CF (price to cash flow)
- FCF yield (free cash flow / market cap)
- Dividend yield

Data source: yfinance ticker.info dictionary.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ValuationRatioAnalyzer:
    """
    Compute valuation ratios and percentile rankings from fundamental data.

    Args:
        sector_relative: If True, rank ratios vs. sector peers.
        history_lookback: Quarters of historical data for
            time-series valuation context.
    """

    RATIO_KEYS = [
        "pe_ratio",
        "forward_pe",
        "price_to_book",
        "price_to_sales",
        "ev_ebitda",
        "ev_sales",
        "peg_ratio",
        "price_to_cashflow",
        "fcf_yield",
        "dividend_yield",
    ]

    def __init__(
        self,
        sector_relative: bool = False,
        history_lookback: int = 20,
    ) -> None:
        self.sector_relative = sector_relative
        self.history_lookback = history_lookback

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_ratios(self, info: Dict) -> Dict[str, Optional[float]]:
        """
        Extract valuation ratios from a yfinance ``ticker.info`` dict.

        Args:
            info: Dictionary returned by ``yfinance.Ticker(sym).info``.

        Returns:
            Dict of ratio_name -> value (None if unavailable).
        """
        ev = info.get("enterpriseValue")
        ebitda = info.get("ebitda")
        revenue = info.get("totalRevenue")
        mcap = info.get("marketCap")
        fcf = info.get("freeCashflow")
        ocf = info.get("operatingCashflow")

        ratios: Dict[str, Optional[float]] = {
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "peg_ratio": info.get("pegRatio"),
            "dividend_yield": info.get("dividendYield"),
        }

        # Derived ratios
        if ev is not None and ebitda is not None and ebitda > 0:
            ratios["ev_ebitda"] = ev / ebitda
        else:
            ratios["ev_ebitda"] = None

        if ev is not None and revenue is not None and revenue > 0:
            ratios["ev_sales"] = ev / revenue
        else:
            ratios["ev_sales"] = None

        if mcap is not None and ocf is not None and mcap > 0:
            ratios["price_to_cashflow"] = mcap / ocf if ocf else None
        else:
            ratios["price_to_cashflow"] = None

        if mcap is not None and fcf is not None and mcap > 0:
            ratios["fcf_yield"] = fcf / mcap
        else:
            ratios["fcf_yield"] = None

        return ratios

    def extract_universe(
        self,
        info_dict: Dict[str, Dict],
    ) -> pd.DataFrame:
        """
        Extract ratios for a universe of symbols.

        Args:
            info_dict: ``{symbol: ticker.info}``.

        Returns:
            DataFrame indexed by symbol with ratio columns.
        """
        rows: List[Dict] = []
        for symbol, info in info_dict.items():
            row = {"symbol": symbol}
            row.update(self.extract_ratios(info))
            rows.append(row)

        return pd.DataFrame(rows).set_index("symbol")

    # ------------------------------------------------------------------
    # Ranking / scoring
    # ------------------------------------------------------------------

    def percentile_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute percentile ranks across the universe for each ratio.

        Lower P/E, P/B, EV/EBITDA => higher rank (more "value").
        Higher FCF yield, dividend yield => higher rank.

        Returns:
            DataFrame with same shape, values in [0, 1].
        """
        ranked = pd.DataFrame(index=df.index)

        # Ratios where lower is better (more value)
        lower_better = [
            "pe_ratio",
            "forward_pe",
            "price_to_book",
            "price_to_sales",
            "ev_ebitda",
            "ev_sales",
            "peg_ratio",
            "price_to_cashflow",
        ]
        # Ratios where higher is better
        higher_better = ["fcf_yield", "dividend_yield"]

        for col in lower_better:
            if col in df.columns:
                ranked[col] = 1 - df[col].rank(pct=True, na_option="bottom")

        for col in higher_better:
            if col in df.columns:
                ranked[col] = df[col].rank(pct=True, na_option="bottom")

        return ranked

    def composite_value_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute a composite value score from all available ratios.

        Equal-weighted average of percentile ranks.

        Returns:
            Series of composite scores in [0, 1], higher = cheaper.
        """
        ranked = self.percentile_rank(df)
        return ranked.mean(axis=1).rename("value_score")

    def detect_value_traps(
        self,
        ratios_df: pd.DataFrame,
        quality_scores: Optional[pd.Series] = None,
        value_threshold: float = 0.8,
        quality_threshold: float = 0.3,
    ) -> pd.Series:
        """
        Identify stocks that appear cheap but have poor quality.

        A value trap is high value score + low quality score.

        Returns:
            Boolean Series flagging potential value traps.
        """
        value = self.composite_value_score(ratios_df)
        if quality_scores is None:
            return pd.Series(False, index=ratios_df.index, name="value_trap")

        return (
            (value > value_threshold) & (quality_scores < quality_threshold)
        ).rename("value_trap")
