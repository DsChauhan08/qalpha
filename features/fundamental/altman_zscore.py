"""
Altman Z-Score Calculator.

Predicts probability of corporate bankruptcy using the Altman Z-Score
model (Altman, 1968). Still one of the most widely used credit-risk
models 50+ years later.

Formula (original manufacturing model):
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

Where:
    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Cap / Total Liabilities
    X5 = Revenue / Total Assets

Interpretation:
    Z > 2.99   : Safe zone (low bankruptcy risk)
    1.81 < Z < 2.99 : Grey zone (moderate risk)
    Z < 1.81   : Distress zone (high bankruptcy risk)

Also implements the Altman Z''-Score for non-manufacturing / emerging markets.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Zone boundaries
SAFE_THRESHOLD = 2.99
GREY_THRESHOLD = 1.81


class AltmanZScoreCalculator:
    """
    Compute Altman Z-Score and related distress indicators.

    Args:
        model: Which Z-score variant to use.
            'original'  : Classic 5-factor (for public manufacturing)
            'revised'   : Z'-score (for private firms)
            'emerging'  : Z''-score (for non-manufacturing / EM)
    """

    def __init__(self, model: str = "original") -> None:
        if model not in ("original", "revised", "emerging"):
            raise ValueError(f"Unknown model: {model!r}")
        self.model = model

    # ------------------------------------------------------------------
    # Component extraction
    # ------------------------------------------------------------------

    def extract_components(self, info: Dict) -> Dict[str, Optional[float]]:
        """
        Extract the five Z-score input ratios from yfinance info.

        Returns:
            Dict with keys x1..x5 and their values (None if missing).
        """
        total_assets = info.get("totalAssets")
        if not total_assets or total_assets <= 0:
            return {f"x{i}": None for i in range(1, 6)}

        current_assets = info.get("totalCurrentAssets", 0) or 0
        current_liabilities = info.get("totalCurrentLiabilities", 0) or 0
        retained_earnings = info.get("retainedEarnings")
        ebit = info.get("ebit")
        market_cap = info.get("marketCap")
        total_liabilities = info.get("totalDebt")  # approximation
        revenue = info.get("totalRevenue")

        # X1: Working Capital / Total Assets
        working_capital = current_assets - current_liabilities
        x1 = working_capital / total_assets

        # X2: Retained Earnings / Total Assets
        x2 = retained_earnings / total_assets if retained_earnings is not None else None

        # X3: EBIT / Total Assets
        x3 = ebit / total_assets if ebit is not None else None

        # X4: Market Cap / Total Liabilities
        if (
            market_cap is not None
            and total_liabilities is not None
            and total_liabilities > 0
        ):
            x4 = market_cap / total_liabilities
        else:
            x4 = None

        # X5: Revenue / Total Assets (asset turnover)
        x5 = revenue / total_assets if revenue is not None else None

        return {"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5}

    # ------------------------------------------------------------------
    # Z-score computation
    # ------------------------------------------------------------------

    def compute_zscore(self, info: Dict) -> Optional[float]:
        """
        Compute the Altman Z-Score.

        Returns:
            Z-score float, or None if insufficient data.
        """
        components = self.extract_components(info)
        return self._compute_from_components(components)

    def _compute_from_components(
        self,
        components: Dict[str, Optional[float]],
    ) -> Optional[float]:
        """Compute Z-score from pre-extracted components."""
        vals = {k: v for k, v in components.items() if v is not None}

        if self.model == "original":
            # Need at least x1, x3, x4
            required = {"x1", "x3"}
            if not required.issubset(vals.keys()):
                return None

            x1 = vals.get("x1", 0.0)
            x2 = vals.get("x2", 0.0)
            x3 = vals.get("x3", 0.0)
            x4 = vals.get("x4", 0.0)
            x5 = vals.get("x5", 0.0)

            return 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

        elif self.model == "revised":
            # Z' = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4 + 0.998*X5
            x1 = vals.get("x1", 0.0)
            x2 = vals.get("x2", 0.0)
            x3 = vals.get("x3", 0.0)
            x4 = vals.get("x4", 0.0)
            x5 = vals.get("x5", 0.0)
            return 0.717 * x1 + 0.847 * x2 + 3.107 * x3 + 0.420 * x4 + 0.998 * x5

        else:  # emerging
            # Z'' = 3.25 + 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
            x1 = vals.get("x1", 0.0)
            x2 = vals.get("x2", 0.0)
            x3 = vals.get("x3", 0.0)
            x4 = vals.get("x4", 0.0)
            return 3.25 + 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4

    def classify_zone(self, z_score: Optional[float]) -> str:
        """
        Classify Z-score into risk zones.

        Returns:
            'safe', 'grey', 'distress', or 'unknown'.
        """
        if z_score is None:
            return "unknown"
        if z_score > SAFE_THRESHOLD:
            return "safe"
        if z_score > GREY_THRESHOLD:
            return "grey"
        return "distress"

    def compute_bankruptcy_probability(self, z_score: Optional[float]) -> float:
        """
        Approximate bankruptcy probability using logistic mapping.

        P(default) = 1 / (1 + exp(z_score - 1.81))

        This is a rough approximation; the original Altman model
        doesn't produce calibrated probabilities.

        Returns:
            Probability in [0, 1].
        """
        if z_score is None:
            return 0.5

        return 1.0 / (1.0 + np.exp(z_score - GREY_THRESHOLD))

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------

    def score_universe(
        self,
        info_dict: Dict[str, Dict],
    ) -> pd.DataFrame:
        """
        Compute Z-scores and zones for a universe of symbols.

        Returns:
            DataFrame with [symbol, z_score, zone, bankruptcy_prob,
            x1, x2, x3, x4, x5].
        """
        rows: List[Dict] = []
        for symbol, info in info_dict.items():
            components = self.extract_components(info)
            z = self._compute_from_components(components)
            zone = self.classify_zone(z)
            prob = self.compute_bankruptcy_probability(z)

            row = {
                "symbol": symbol,
                "z_score": z,
                "zone": zone,
                "bankruptcy_prob": prob,
            }
            row.update(components)
            rows.append(row)

        return pd.DataFrame(rows).set_index("symbol")

    def generate_signals(
        self,
        info_dict: Dict[str, Dict],
    ) -> pd.DataFrame:
        """
        Generate trading signals from Z-scores.

        Signal logic:
            distress zone => -1 (avoid / short)
            grey zone     =>  0 (neutral)
            safe zone     =>  1 (safe to hold)

        Returns:
            DataFrame with [symbol, z_score, zone, signal].
        """
        scored = self.score_universe(info_dict)
        scored["signal"] = scored["zone"].map(
            {"safe": 1, "grey": 0, "distress": -1, "unknown": 0}
        )
        return scored[["z_score", "zone", "signal"]].copy()
