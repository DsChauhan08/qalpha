"""
Volatility Signature Analyzer.

Analyzes the volatility signature plot -- realised volatility as a
function of sampling frequency. This reveals microstructure noise,
optimal sampling frequency, and market efficiency.

Key concepts:
- At very high frequencies, realised vol is inflated by bid-ask bounce
  (microstructure noise).
- At very low frequencies, there are insufficient observations.
- The optimal sampling frequency minimises the MSE of the RV estimator.
- The signature plot shape reveals information about market quality.

Also implements realised volatility estimators:
- Close-to-close
- Parkinson (1980) high-low
- Garman-Klass (1980) OHLC
- Rogers-Satchell (1991)
- Yang-Zhang (2000)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VolatilitySignatureAnalyzer:
    """
    Compute volatility signature plots and realized volatility estimators.

    Args:
        annualization_factor: Trading days per year for annualisation.
    """

    def __init__(self, annualization_factor: int = 252) -> None:
        self.ann_factor = annualization_factor

    # ------------------------------------------------------------------
    # Realized volatility estimators
    # ------------------------------------------------------------------

    def close_to_close_rv(
        self,
        close: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """
        Standard close-to-close realised volatility.

        RV = sqrt(252) * std(log_returns)

        Args:
            close: Closing prices.
            window: Rolling window.

        Returns:
            Annualised RV series.
        """
        log_returns = np.log(close / close.shift(1))
        return (
            log_returns.rolling(window, min_periods=2).std() * np.sqrt(self.ann_factor)
        ).rename("cc_rv")

    def parkinson_rv(
        self,
        high: pd.Series,
        low: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """
        Parkinson (1980) high-low range estimator.

        sigma^2 = (1 / 4*ln(2)) * E[(ln(H/L))^2]

        ~5x more efficient than close-to-close.

        Returns:
            Annualised Parkinson RV.
        """
        log_hl = np.log(high / low)
        sq_log_hl = log_hl**2

        variance = sq_log_hl.rolling(window, min_periods=2).mean() / (4 * np.log(2))
        return (np.sqrt(variance * self.ann_factor)).rename("parkinson_rv")

    def garman_klass_rv(
        self,
        bars: pd.DataFrame,
        window: int = 20,
    ) -> pd.Series:
        """
        Garman-Klass (1980) OHLC estimator.

        sigma^2 = 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2

        ~8x more efficient than close-to-close.

        Returns:
            Annualised GK RV.
        """
        log_hl = np.log(bars["high"] / bars["low"])
        log_co = np.log(bars["close"] / bars["open"])

        variance = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        rv = variance.rolling(window, min_periods=2).mean()
        rv = rv.clip(lower=0)
        return (np.sqrt(rv * self.ann_factor)).rename("gk_rv")

    def rogers_satchell_rv(
        self,
        bars: pd.DataFrame,
        window: int = 20,
    ) -> pd.Series:
        """
        Rogers-Satchell (1991) estimator.

        sigma^2 = ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)

        Handles non-zero drift (unlike Parkinson and GK).

        Returns:
            Annualised RS RV.
        """
        log_hc = np.log(bars["high"] / bars["close"])
        log_ho = np.log(bars["high"] / bars["open"])
        log_lc = np.log(bars["low"] / bars["close"])
        log_lo = np.log(bars["low"] / bars["open"])

        variance = log_hc * log_ho + log_lc * log_lo
        rv = variance.rolling(window, min_periods=2).mean()
        rv = rv.clip(lower=0)
        return (np.sqrt(rv * self.ann_factor)).rename("rs_rv")

    def yang_zhang_rv(
        self,
        bars: pd.DataFrame,
        window: int = 20,
    ) -> pd.Series:
        """
        Yang-Zhang (2000) estimator.

        Combines overnight (close-to-open) and Rogers-Satchell
        estimators. The most complete OHLC estimator; handles
        both drift and opening jumps.

        sigma^2 = sigma_o^2 + k * sigma_c^2 + (1-k) * sigma_rs^2

        where k = 0.34 / (1.34 + (n+1)/(n-1))

        Returns:
            Annualised YZ RV.
        """
        n = window

        # Overnight return variance
        log_oc = np.log(bars["open"] / bars["close"].shift(1))
        sigma_o_sq = log_oc.rolling(n, min_periods=2).var()

        # Close-to-close variance
        log_cc = np.log(bars["close"] / bars["close"].shift(1))
        sigma_c_sq = log_cc.rolling(n, min_periods=2).var()

        # Rogers-Satchell
        log_hc = np.log(bars["high"] / bars["close"])
        log_ho = np.log(bars["high"] / bars["open"])
        log_lc = np.log(bars["low"] / bars["close"])
        log_lo = np.log(bars["low"] / bars["open"])
        rs = (log_hc * log_ho + log_lc * log_lo).rolling(n, min_periods=2).mean()
        rs = rs.clip(lower=0)

        k = 0.34 / (1.34 + (n + 1) / (n - 1)) if n > 1 else 0.34 / 2.34

        variance = sigma_o_sq + k * sigma_c_sq + (1 - k) * rs
        variance = variance.clip(lower=0)
        return (np.sqrt(variance * self.ann_factor)).rename("yz_rv")

    # ------------------------------------------------------------------
    # Signature plot
    # ------------------------------------------------------------------

    def compute_signature(
        self,
        close: pd.Series,
        frequencies: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Compute the volatility signature plot.

        For each sampling frequency (in bars), compute the annualised
        realised volatility.

        Args:
            close: Full-resolution closing prices.
            frequencies: List of sampling intervals (in bars).
                Defaults to [1, 2, 5, 10, 15, 20, 30, 60, 120].

        Returns:
            DataFrame with [frequency, realised_vol].
        """
        if frequencies is None:
            max_freq = max(len(close) // 10, 1)
            frequencies = [
                f for f in [1, 2, 5, 10, 15, 20, 30, 60, 120] if f < max_freq
            ]
            if not frequencies:
                frequencies = [1]

        results: List[Dict] = []
        for freq in frequencies:
            sampled = close.iloc[::freq]
            log_ret = np.log(sampled / sampled.shift(1)).dropna()

            if len(log_ret) < 2:
                continue

            rv = float(log_ret.std() * np.sqrt(self.ann_factor / freq))
            results.append({"frequency": freq, "realised_vol": rv})

        return pd.DataFrame(results)

    def find_optimal_frequency(
        self,
        close: pd.Series,
        frequencies: Optional[List[int]] = None,
    ) -> int:
        """
        Find the optimal sampling frequency (where the signature
        plot flattens).

        Uses the first difference of the signature plot: the optimal
        frequency is where the rate of change is minimised.

        Returns:
            Optimal sampling interval (in bars).
        """
        sig = self.compute_signature(close, frequencies)
        if len(sig) < 3:
            return 1

        # Rate of change of RV with respect to frequency
        sig["rv_change"] = sig["realised_vol"].diff().abs()

        # Find the frequency with minimum change (stabilisation)
        # Skip the first point (highest frequency has most noise)
        stable = sig.iloc[1:]
        if stable.empty:
            return int(sig["frequency"].iloc[0])

        idx = stable["rv_change"].idxmin()
        return int(sig.loc[idx, "frequency"])

    # ------------------------------------------------------------------
    # Composite
    # ------------------------------------------------------------------

    def compute_all_estimators(
        self,
        bars: pd.DataFrame,
        window: int = 20,
    ) -> pd.DataFrame:
        """
        Compute all RV estimators on a single OHLCV DataFrame.

        Returns:
            DataFrame with all RV columns.
        """
        df = bars.copy()
        df["cc_rv"] = self.close_to_close_rv(bars["close"], window)
        df["parkinson_rv"] = self.parkinson_rv(bars["high"], bars["low"], window)
        df["gk_rv"] = self.garman_klass_rv(bars, window)
        df["rs_rv"] = self.rogers_satchell_rv(bars, window)
        df["yz_rv"] = self.yang_zhang_rv(bars, window)

        # Consensus (median)
        rv_cols = ["cc_rv", "parkinson_rv", "gk_rv", "rs_rv", "yz_rv"]
        df["consensus_rv"] = df[rv_cols].median(axis=1)

        return df
