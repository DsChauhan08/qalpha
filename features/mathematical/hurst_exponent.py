"""
Hurst Exponent Calculator for Mean-Reversion/Trend Detection

The Hurst exponent H determines the long-term memory of a time series:
- H < 0.5: Mean-reverting (anti-persistent)
- H = 0.5: Random walk (Brownian motion)
- H > 0.5: Trending (persistent)

Formula: H = log(R/S) / log(n)
"""

import numpy as np
from typing import Dict, Optional, Tuple


class HurstExponentCalculator:
    """
    Computes Hurst exponent using multiple methods for robustness.

    Methods:
    1. Rescaled Range (R/S) Analysis - classical method
    2. Detrended Fluctuation Analysis (DFA) - robust to trends
    3. Variance-Time Method - quick estimation
    """

    def __init__(self, min_window: int = 10, max_window: Optional[int] = None):
        """
        Args:
            min_window: Minimum window size for R/S analysis
            max_window: Maximum window size (default: len(data) // 4)
        """
        self.min_window = min_window
        self.max_window = max_window

    def compute_hurst_rs(self, data: np.ndarray) -> Dict:
        """
        Compute Hurst exponent using Rescaled Range (R/S) analysis.

        Args:
            data: Time series (prices or returns)

        Returns:
            dict: Hurst exponent and statistics
        """
        data = np.asarray(data, dtype=float)
        n = len(data)

        max_window = self.max_window or n // 4

        if n < self.min_window * 4:
            return {
                "hurst": 0.5,
                "method": "rs",
                "confidence": 0.0,
                "interpretation": "INSUFFICIENT_DATA",
            }

        # Calculate R/S for different window sizes
        window_sizes = []
        rs_values = []

        for window in range(self.min_window, max_window + 1):
            # Number of non-overlapping windows
            num_windows = n // window
            if num_windows < 2:
                continue

            rs_list = []
            for i in range(num_windows):
                start = i * window
                end = start + window
                window_data = data[start:end]

                # Mean and deviations
                mean = np.mean(window_data)
                deviations = window_data - mean

                # Cumulative deviations
                cumulative = np.cumsum(deviations)

                # Range
                R = np.max(cumulative) - np.min(cumulative)

                # Standard deviation
                S = np.std(window_data, ddof=1)

                if S > 0:
                    rs_list.append(R / S)

            if rs_list:
                window_sizes.append(window)
                rs_values.append(np.mean(rs_list))

        if len(window_sizes) < 3:
            return {
                "hurst": 0.5,
                "method": "rs",
                "confidence": 0.0,
                "interpretation": "INSUFFICIENT_WINDOWS",
            }

        # Linear regression: log(R/S) = H * log(n) + c
        log_sizes = np.log(window_sizes)
        log_rs = np.log(rs_values)

        hurst, intercept, r_squared = self._linear_regression(log_sizes, log_rs)

        # Interpret
        interpretation = self._interpret_hurst(hurst)

        return {
            "hurst": float(hurst),
            "method": "rs",
            "confidence": float(r_squared),
            "interpretation": interpretation,
            "intercept": float(intercept),
        }

    def compute_hurst_dfa(self, data: np.ndarray, order: int = 1) -> Dict:
        """
        Compute Hurst exponent using Detrended Fluctuation Analysis.

        DFA is more robust to non-stationarity and trends.

        Args:
            data: Time series
            order: Polynomial order for detrending (1=linear)

        Returns:
            dict: Hurst exponent and statistics
        """
        data = np.asarray(data, dtype=float)
        n = len(data)

        if n < self.min_window * 4:
            return {
                "hurst": 0.5,
                "method": "dfa",
                "confidence": 0.0,
                "interpretation": "INSUFFICIENT_DATA",
            }

        # Integrate the series (cumulative sum of deviations from mean)
        mean = np.mean(data)
        cumulative = np.cumsum(data - mean)

        max_window = self.max_window or n // 4

        window_sizes = []
        fluctuations = []

        for window in range(self.min_window, max_window + 1):
            num_windows = n // window
            if num_windows < 2:
                continue

            local_fluct = []
            for i in range(num_windows):
                start = i * window
                end = start + window
                segment = cumulative[start:end]

                # Fit polynomial trend
                x = np.arange(window)
                coeffs = np.polyfit(x, segment, order)
                trend = np.polyval(coeffs, x)

                # Fluctuation (RMS of detrended)
                detrended = segment - trend
                fluct = np.sqrt(np.mean(detrended**2))
                local_fluct.append(fluct)

            if local_fluct:
                window_sizes.append(window)
                fluctuations.append(np.mean(local_fluct))

        if len(window_sizes) < 3:
            return {
                "hurst": 0.5,
                "method": "dfa",
                "confidence": 0.0,
                "interpretation": "INSUFFICIENT_WINDOWS",
            }

        # Linear regression: log(F) = H * log(n) + c
        log_sizes = np.log(window_sizes)
        log_fluct = np.log(fluctuations)

        hurst, intercept, r_squared = self._linear_regression(log_sizes, log_fluct)

        interpretation = self._interpret_hurst(hurst)

        return {
            "hurst": float(hurst),
            "method": "dfa",
            "confidence": float(r_squared),
            "interpretation": interpretation,
            "intercept": float(intercept),
        }

    def compute_hurst_variance_time(self, data: np.ndarray) -> Dict:
        """
        Quick Hurst estimation using variance-time method.

        Based on the relationship: Var(X_m) ∝ m^(2H-2)
        where X_m is the aggregated series at scale m.

        Args:
            data: Time series

        Returns:
            dict: Hurst exponent
        """
        data = np.asarray(data, dtype=float)
        n = len(data)

        if n < 20:
            return {
                "hurst": 0.5,
                "method": "variance_time",
                "confidence": 0.0,
                "interpretation": "INSUFFICIENT_DATA",
            }

        # Aggregate at different scales
        scales = [1, 2, 4, 8, 16, 32]
        scales = [s for s in scales if n // s >= 4]

        if len(scales) < 3:
            return {
                "hurst": 0.5,
                "method": "variance_time",
                "confidence": 0.0,
                "interpretation": "INSUFFICIENT_SCALES",
            }

        variances = []
        for m in scales:
            # Aggregate series
            agg_len = n // m
            aggregated = np.array(
                [np.mean(data[i * m : (i + 1) * m]) for i in range(agg_len)]
            )
            variances.append(np.var(aggregated))

        # Linear regression: log(Var) = (2H-2) * log(m) + c
        log_scales = np.log(scales)
        log_vars = np.log(variances)

        slope, intercept, r_squared = self._linear_regression(log_scales, log_vars)

        # H = (slope + 2) / 2
        hurst = (slope + 2) / 2
        hurst = np.clip(hurst, 0, 1)

        interpretation = self._interpret_hurst(hurst)

        return {
            "hurst": float(hurst),
            "method": "variance_time",
            "confidence": float(r_squared),
            "interpretation": interpretation,
        }

    def compute_ensemble(self, data: np.ndarray) -> Dict:
        """
        Compute Hurst exponent using ensemble of methods.

        Weights methods by their confidence (R²).

        Args:
            data: Time series

        Returns:
            dict: Ensemble Hurst exponent
        """
        results = {
            "rs": self.compute_hurst_rs(data),
            "dfa": self.compute_hurst_dfa(data),
            "variance_time": self.compute_hurst_variance_time(data),
        }

        # Weighted average by confidence
        total_weight = 0
        weighted_hurst = 0

        for method, result in results.items():
            weight = max(result["confidence"], 0.1)  # Minimum weight
            weighted_hurst += result["hurst"] * weight
            total_weight += weight

        ensemble_hurst = weighted_hurst / total_weight if total_weight > 0 else 0.5
        ensemble_confidence = np.mean([r["confidence"] for r in results.values()])

        return {
            "hurst": float(ensemble_hurst),
            "method": "ensemble",
            "confidence": float(ensemble_confidence),
            "interpretation": self._interpret_hurst(ensemble_hurst),
            "individual_results": results,
        }

    def _linear_regression(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, float, float]:
        """Simple linear regression returning slope, intercept, R²."""
        x = np.asarray(x)
        y = np.asarray(y)

        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_xx = np.sum(x * x)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x**2 + 1e-10)
        intercept = (sum_y - slope * sum_x) / n

        # R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10)

        return slope, intercept, max(0, r_squared)

    def _interpret_hurst(self, hurst: float) -> str:
        """Interpret Hurst exponent value."""
        if hurst < 0.35:
            return "STRONGLY_MEAN_REVERTING"
        elif hurst < 0.45:
            return "MEAN_REVERTING"
        elif hurst < 0.55:
            return "RANDOM_WALK"
        elif hurst < 0.65:
            return "TRENDING"
        else:
            return "STRONGLY_TRENDING"

    def generate_trading_signal(self, data: np.ndarray) -> Dict:
        """
        Generate trading signal based on Hurst exponent.

        Args:
            data: Price series

        Returns:
            dict: Trading signal and metadata
        """
        result = self.compute_ensemble(data)
        hurst = result["hurst"]

        # Signal logic:
        # - Mean reverting: favor contrarian strategies (-1 to 0)
        # - Trending: favor momentum strategies (0 to 1)

        if hurst < 0.5:
            # Mean reverting - signal strength increases as H decreases
            signal = -(0.5 - hurst) * 2  # Maps H=0 to -1, H=0.5 to 0
            strategy_type = "MEAN_REVERSION"
        else:
            # Trending - signal strength increases as H increases
            signal = (hurst - 0.5) * 2  # Maps H=0.5 to 0, H=1 to 1
            strategy_type = "MOMENTUM"

        # Adjust by confidence
        signal *= result["confidence"]

        return {
            "signal": float(signal),
            "hurst": float(hurst),
            "confidence": result["confidence"],
            "interpretation": result["interpretation"],
            "strategy_type": strategy_type,
        }

    def compute_rolling(
        self, data: np.ndarray, window: int = 100, step: int = 10
    ) -> np.ndarray:
        """
        Compute rolling Hurst exponent.

        Args:
            data: Full time series
            window: Rolling window size
            step: Step between calculations

        Returns:
            Array of Hurst values (NaN padded at start)
        """
        n = len(data)
        hurst_values = np.full(n, np.nan)

        for i in range(window, n, step):
            window_data = data[i - window : i]
            result = self.compute_hurst_rs(window_data)
            hurst_values[i] = result["hurst"]

        # Forward fill
        last_valid = np.nan
        for i in range(n):
            if not np.isnan(hurst_values[i]):
                last_valid = hurst_values[i]
            elif not np.isnan(last_valid):
                hurst_values[i] = last_valid

        return hurst_values
