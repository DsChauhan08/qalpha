"""
Fractal Dimension Calculator for Market Analysis

Fractal dimension measures the complexity/roughness of a time series.
Financial markets exhibit fractal behavior - self-similarity across scales.

Key Measures:
1. Box-counting dimension - classic measure of space-filling
2. Higuchi fractal dimension - optimized for time series
3. Correlation dimension - captures attractor complexity
4. Katz fractal dimension - quick estimation

Interpretation:
- D ≈ 1.0: Smooth, trending
- D ≈ 1.5: Random walk (Brownian motion)
- D ≈ 2.0: Highly irregular, space-filling
"""

import numpy as np
from typing import Dict, Optional, Tuple, List


class FractalDimensionCalculator:
    """
    Computes fractal dimension of financial time series.

    Applications:
    - Market complexity assessment
    - Trend vs mean-reversion detection
    - Risk regime identification
    """

    def __init__(self, min_k: int = 2, max_k: Optional[int] = None):
        """
        Args:
            min_k: Minimum scale for analysis
            max_k: Maximum scale (default: len(data) // 4)
        """
        self.min_k = min_k
        self.max_k = max_k

    def higuchi_fd(self, data: np.ndarray, k_max: Optional[int] = None) -> Dict:
        """
        Compute Higuchi Fractal Dimension.

        Most suitable for time series analysis. Efficient and robust.

        Args:
            data: Time series
            k_max: Maximum interval (default: len(data) // 4)

        Returns:
            dict: Fractal dimension and statistics
        """
        data = np.asarray(data, dtype=float)
        N = len(data)

        if k_max is None:
            k_max = self.max_k or min(N // 4, 64)

        k_max = min(k_max, N // 4)

        if N < self.min_k * 4 or k_max < self.min_k:
            return {
                "dimension": 1.5,
                "method": "higuchi",
                "confidence": 0.0,
                "interpretation": "INSUFFICIENT_DATA",
            }

        L = []

        for k in range(self.min_k, k_max + 1):
            Lk = []

            for m in range(1, k + 1):
                # Create subseries X_m^k
                indices = np.arange(m - 1, N, k)
                if len(indices) < 2:
                    continue

                # Compute length for this subseries
                L_mk = 0
                for i in range(1, len(indices)):
                    L_mk += abs(data[indices[i]] - data[indices[i - 1]])

                # Normalize
                norm_factor = (N - 1) / (k * ((N - m) // k) * k)
                L_mk *= norm_factor

                Lk.append(L_mk)

            if Lk:
                L.append(np.mean(Lk))

        if len(L) < 3:
            return {
                "dimension": 1.5,
                "method": "higuchi",
                "confidence": 0.0,
                "interpretation": "INSUFFICIENT_SCALES",
            }

        # Linear regression: log(L) = -D * log(k) + c
        k_values = np.arange(self.min_k, self.min_k + len(L))
        log_k = np.log(k_values)
        log_L = np.log(L)

        slope, intercept, r_squared = self._linear_regression(log_k, log_L)

        # Fractal dimension is negative of slope
        fd = -slope
        fd = np.clip(fd, 1.0, 2.0)  # Valid range for 1D signal

        return {
            "dimension": float(fd),
            "method": "higuchi",
            "confidence": float(r_squared),
            "interpretation": self._interpret_fd(fd),
            "k_max": k_max,
        }

    def katz_fd(self, data: np.ndarray) -> Dict:
        """
        Compute Katz Fractal Dimension.

        Fast computation, useful for real-time analysis.

        Formula: D = log(L) / log(d)
        where L = total length, d = diameter

        Args:
            data: Time series

        Returns:
            dict: Fractal dimension
        """
        data = np.asarray(data, dtype=float)
        N = len(data)

        if N < 10:
            return {
                "dimension": 1.5,
                "method": "katz",
                "confidence": 0.0,
                "interpretation": "INSUFFICIENT_DATA",
            }

        # Total length of the path
        L = np.sum(np.abs(np.diff(data)))

        # Diameter (max distance from first point)
        d = np.max(np.abs(data - data[0]))

        # Average step
        a = L / (N - 1)

        # Katz formula
        if d > 0 and a > 0:
            n = L / a
            fd = np.log10(n) / (np.log10(n) + np.log10(d / L))
        else:
            fd = 1.5

        fd = np.clip(fd, 1.0, 2.0)

        return {
            "dimension": float(fd),
            "method": "katz",
            "confidence": 0.8,  # Katz is less accurate
            "interpretation": self._interpret_fd(fd),
            "path_length": float(L),
            "diameter": float(d),
        }

    def box_counting_fd(self, data: np.ndarray, n_sizes: int = 10) -> Dict:
        """
        Compute Box-Counting Fractal Dimension.

        Classic method - counts how many boxes needed to cover the signal.

        Args:
            data: Time series
            n_sizes: Number of different box sizes

        Returns:
            dict: Fractal dimension
        """
        data = np.asarray(data, dtype=float)
        N = len(data)

        if N < 20:
            return {
                "dimension": 1.5,
                "method": "box_counting",
                "confidence": 0.0,
                "interpretation": "INSUFFICIENT_DATA",
            }

        # Normalize data to [0, 1]
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)

        # Create 2D representation (index, value)
        points = np.column_stack(
            [
                np.arange(N) / (N - 1),  # x: normalized time
                data_norm,  # y: normalized value
            ]
        )

        # Box sizes (powers of 2)
        max_size = min(N, 256)
        sizes = np.logspace(0, np.log2(max_size), n_sizes, base=2).astype(int)
        sizes = np.unique(sizes[sizes >= 1])

        counts = []

        for size in sizes:
            # Grid resolution
            eps = 1.0 / size

            # Count occupied boxes
            grid_x = (points[:, 0] / eps).astype(int)
            grid_y = (points[:, 1] / eps).astype(int)

            # Unique boxes
            boxes = set(zip(grid_x, grid_y))
            counts.append(len(boxes))

        if len(counts) < 3:
            return {
                "dimension": 1.5,
                "method": "box_counting",
                "confidence": 0.0,
                "interpretation": "INSUFFICIENT_SCALES",
            }

        # Linear regression: log(N) = D * log(1/eps) + c
        log_eps_inv = np.log(sizes)
        log_counts = np.log(counts)

        slope, intercept, r_squared = self._linear_regression(log_eps_inv, log_counts)

        fd = np.clip(slope, 1.0, 2.0)

        return {
            "dimension": float(fd),
            "method": "box_counting",
            "confidence": float(r_squared),
            "interpretation": self._interpret_fd(fd),
        }

    def petrosian_fd(self, data: np.ndarray) -> Dict:
        """
        Compute Petrosian Fractal Dimension.

        Very fast, based on zero-crossings of first derivative.

        Args:
            data: Time series

        Returns:
            dict: Fractal dimension
        """
        data = np.asarray(data, dtype=float)
        N = len(data)

        if N < 10:
            return {
                "dimension": 1.5,
                "method": "petrosian",
                "confidence": 0.0,
                "interpretation": "INSUFFICIENT_DATA",
            }

        # Binary sequence
        diff = np.diff(data)
        N_delta = len(diff)

        # Count sign changes (zero-crossings)
        sign_changes = np.sum(diff[:-1] * diff[1:] < 0)

        # Petrosian formula
        fd = np.log10(N_delta) / (
            np.log10(N_delta) + np.log10(N_delta / (N_delta + 0.4 * sign_changes))
        )

        fd = np.clip(fd, 1.0, 2.0)

        return {
            "dimension": float(fd),
            "method": "petrosian",
            "confidence": 0.7,  # Less accurate method
            "interpretation": self._interpret_fd(fd),
            "sign_changes": int(sign_changes),
        }

    def compute_ensemble(self, data: np.ndarray) -> Dict:
        """
        Compute fractal dimension using ensemble of methods.

        Weights by confidence for robust estimate.

        Args:
            data: Time series

        Returns:
            dict: Ensemble fractal dimension
        """
        results = {
            "higuchi": self.higuchi_fd(data),
            "katz": self.katz_fd(data),
            "box_counting": self.box_counting_fd(data),
            "petrosian": self.petrosian_fd(data),
        }

        # Weighted average by confidence
        total_weight = 0
        weighted_fd = 0

        for method, result in results.items():
            weight = result["confidence"]
            weighted_fd += result["dimension"] * weight
            total_weight += weight

        ensemble_fd = weighted_fd / total_weight if total_weight > 0 else 1.5
        ensemble_confidence = np.mean([r["confidence"] for r in results.values()])

        return {
            "dimension": float(ensemble_fd),
            "method": "ensemble",
            "confidence": float(ensemble_confidence),
            "interpretation": self._interpret_fd(ensemble_fd),
            "individual_results": results,
        }

    def _linear_regression(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[float, float, float]:
        """Simple linear regression returning slope, intercept, R²."""
        x = np.asarray(x)
        y = np.asarray(y)

        # Remove any infinities
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if len(x) < 2:
            return 1.5, 0, 0

        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_xx = np.sum(x * x)

        denom = n * sum_xx - sum_x**2
        if abs(denom) < 1e-10:
            return 1.5, 0, 0

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n

        # R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-10) if ss_tot > 0 else 0

        return slope, intercept, max(0, r_squared)

    def _interpret_fd(self, fd: float) -> str:
        """Interpret fractal dimension value."""
        if fd < 1.2:
            return "VERY_SMOOTH_TRENDING"
        elif fd < 1.4:
            return "SMOOTH_TRENDING"
        elif fd < 1.6:
            return "RANDOM_WALK"
        elif fd < 1.8:
            return "ROUGH_IRREGULAR"
        else:
            return "VERY_ROUGH_COMPLEX"

    def generate_trading_signal(self, data: np.ndarray) -> Dict:
        """
        Generate trading signal based on fractal analysis.

        Lower FD (smoother) -> Better for trend following
        Higher FD (rougher) -> Better for mean reversion / reduce exposure

        Args:
            data: Price series

        Returns:
            dict: Trading signal and metadata
        """
        result = self.compute_ensemble(data)
        fd = result["dimension"]

        # Signal logic based on FD
        # D < 1.5: Smoother, trending - favor momentum
        # D > 1.5: Rougher, irregular - favor mean reversion or caution

        if fd < 1.4:
            # Smooth/trending - momentum likely to work
            signal = 0.5
            strategy_type = "MOMENTUM"
            confidence = (1.5 - fd) * 2  # Higher confidence as FD decreases
        elif fd < 1.6:
            # Near random walk
            signal = 0.0
            strategy_type = "NEUTRAL"
            confidence = 0.5
        else:
            # Rough/complex - mean reversion or reduce exposure
            signal = -0.3
            strategy_type = "CAUTION"
            confidence = (fd - 1.5) * 2  # Higher confidence as FD increases

        confidence = min(confidence * result["confidence"], 1.0)

        return {
            "signal": float(signal),
            "fractal_dimension": float(fd),
            "confidence": float(confidence),
            "interpretation": result["interpretation"],
            "strategy_type": strategy_type,
        }

    def compute_rolling(
        self,
        data: np.ndarray,
        window: int = 100,
        step: int = 10,
        method: str = "higuchi",
    ) -> np.ndarray:
        """
        Compute rolling fractal dimension.

        Args:
            data: Full time series
            window: Rolling window size
            step: Step between calculations
            method: 'higuchi', 'katz', or 'petrosian'

        Returns:
            Array of fractal dimension values
        """
        n = len(data)
        fd_values = np.full(n, np.nan)

        method_func = {
            "higuchi": self.higuchi_fd,
            "katz": self.katz_fd,
            "petrosian": self.petrosian_fd,
        }.get(method, self.higuchi_fd)

        for i in range(window, n, step):
            window_data = data[i - window : i]
            result = method_func(window_data)
            fd_values[i] = result["dimension"]

        # Forward fill
        last_valid = np.nan
        for i in range(n):
            if not np.isnan(fd_values[i]):
                last_valid = fd_values[i]
            elif not np.isnan(last_valid):
                fd_values[i] = last_valid

        return fd_values
