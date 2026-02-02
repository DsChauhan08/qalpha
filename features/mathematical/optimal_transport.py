"""
Optimal Transport (Wasserstein Distance) for Distribution Analysis

Optimal Transport measures the "cost" of transforming one probability distribution
into another. In finance, this quantifies how much a return distribution has changed.

Wasserstein Distance Formula:
W_p(μ, ν) = (inf ∫ d(x,y)^p dπ(x,y))^(1/p)

Applications:
1. Detect anomalous return distributions
2. Measure portfolio drift from target allocation
3. Quantify market regime changes
"""

import numpy as np
from typing import Dict, Optional, Tuple, List


class OptimalTransportAnalyzer:
    """
    Uses Optimal Transport theory to measure distribution shifts.

    Applications:
    1. Detect anomalous return distributions
    2. Measure portfolio drift from target allocation
    3. Quantify market regime changes
    """

    def __init__(self, n_bins: int = 50, epsilon: float = 0.01):
        """
        Args:
            n_bins: Number of bins for histogram discretization
            epsilon: Entropic regularization parameter
        """
        self.n_bins = n_bins
        self.epsilon = epsilon
        self._has_ot = self._check_ot()

    def _check_ot(self) -> bool:
        """Check if POT (Python Optimal Transport) is available."""
        try:
            import ot

            return True
        except ImportError:
            return False

    def distribution_to_histogram(
        self,
        data: np.ndarray,
        bins: Optional[int] = None,
        range_val: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert data samples to normalized histogram.

        Args:
            data: Array of samples
            bins: Number of bins (default: self.n_bins)
            range_val: (min, max) tuple

        Returns:
            Tuple of (histogram, bin_edges)
        """
        if bins is None:
            bins = self.n_bins

        data = np.asarray(data, dtype=float)
        data = data[~np.isnan(data)]

        hist, edges = np.histogram(data, bins=bins, range=range_val, density=True)

        # Normalize to sum to 1
        hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist

        return hist, edges

    def wasserstein_distance_1d(
        self, dist1: np.ndarray, dist2: np.ndarray, p: int = 1
    ) -> float:
        """
        Compute 1D Wasserstein distance between two distributions.

        Uses the analytical formula for 1D case:
        W_p = (∫|F_1^{-1}(t) - F_2^{-1}(t)|^p dt)^{1/p}

        Args:
            dist1: First distribution (samples or histogram)
            dist2: Second distribution (samples or histogram)
            p: Order of Wasserstein distance (1 or 2)

        Returns:
            float: Wasserstein distance
        """
        dist1 = np.asarray(dist1, dtype=float).flatten()
        dist2 = np.asarray(dist2, dtype=float).flatten()

        # Remove NaN
        dist1 = dist1[~np.isnan(dist1)]
        dist2 = dist2[~np.isnan(dist2)]

        if len(dist1) == 0 or len(dist2) == 0:
            return 0.0

        # Sort samples
        dist1_sorted = np.sort(dist1)
        dist2_sorted = np.sort(dist2)

        # Interpolate to common support
        n = max(len(dist1), len(dist2))
        quantiles = np.linspace(0, 1, n)

        q1 = np.quantile(dist1_sorted, quantiles)
        q2 = np.quantile(dist2_sorted, quantiles)

        # Wasserstein distance
        if p == 1:
            return float(np.mean(np.abs(q1 - q2)))
        else:
            return float(np.power(np.mean(np.abs(q1 - q2) ** p), 1 / p))

    def wasserstein_distance_2d(
        self, samples1: np.ndarray, samples2: np.ndarray, metric: str = "euclidean"
    ) -> float:
        """
        Compute 2D/multi-dimensional Wasserstein distance.

        Args:
            samples1: First set of samples (n1, d)
            samples2: Second set of samples (n2, d)
            metric: Distance metric ('euclidean', 'sqeuclidean')

        Returns:
            float: Wasserstein distance
        """
        samples1 = np.asarray(samples1, dtype=float)
        samples2 = np.asarray(samples2, dtype=float)

        if samples1.ndim == 1:
            samples1 = samples1.reshape(-1, 1)
        if samples2.ndim == 1:
            samples2 = samples2.reshape(-1, 1)

        n1, n2 = len(samples1), len(samples2)

        if n1 == 0 or n2 == 0:
            return 0.0

        if self._has_ot:
            import ot

            # Compute cost matrix
            M = ot.dist(samples1, samples2, metric=metric)
            # Uniform weights
            a = np.ones(n1) / n1
            b = np.ones(n2) / n2
            # Optimal transport
            W = ot.emd2(a, b, M)
            return float(W)
        else:
            # Fallback: sliced Wasserstein
            return self._sliced_wasserstein(samples1, samples2)

    def _sliced_wasserstein(
        self, samples1: np.ndarray, samples2: np.ndarray, n_projections: int = 50
    ) -> float:
        """
        Compute Sliced Wasserstein distance (approximation for high dimensions).

        Projects to 1D and averages Wasserstein distances.
        """
        d = samples1.shape[1]
        distances = []

        for _ in range(n_projections):
            # Random projection direction
            theta = np.random.randn(d)
            theta = theta / np.linalg.norm(theta)

            # Project
            proj1 = samples1 @ theta
            proj2 = samples2 @ theta

            # 1D Wasserstein
            w = self.wasserstein_distance_1d(proj1, proj2)
            distances.append(w)

        return float(np.mean(distances))

    def sinkhorn_distance(
        self,
        a: np.ndarray,
        b: np.ndarray,
        M: np.ndarray,
        reg: Optional[float] = None,
        n_iter: int = 100,
    ) -> float:
        """
        Compute Sinkhorn distance (entropic regularized OT).

        Faster than exact OT for large problems.

        Args:
            a: Source distribution (weights)
            b: Target distribution (weights)
            M: Cost matrix
            reg: Regularization parameter
            n_iter: Number of iterations

        Returns:
            float: Sinkhorn distance
        """
        if reg is None:
            reg = self.epsilon

        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        M = np.asarray(M, dtype=float)

        # Normalize
        a = a / np.sum(a)
        b = b / np.sum(b)

        # Kernel
        K = np.exp(-M / reg)

        # Sinkhorn iterations
        u = np.ones_like(a)

        for _ in range(n_iter):
            v = b / (K.T @ u + 1e-10)
            u = a / (K @ v + 1e-10)

        # Transport plan
        P = np.diag(u) @ K @ np.diag(v)

        # Distance
        return float(np.sum(P * M))

    def detect_distribution_shift(
        self, historical: np.ndarray, current: np.ndarray, threshold: float = 2.0
    ) -> Dict:
        """
        Detect if current distribution has shifted from historical baseline.

        Args:
            historical: Historical samples (baseline)
            current: Current samples
            threshold: Z-score threshold for shift detection

        Returns:
            dict: Shift detection results
        """
        historical = np.asarray(historical, dtype=float).flatten()
        current = np.asarray(current, dtype=float).flatten()

        # Remove NaN
        historical = historical[~np.isnan(historical)]
        current = current[~np.isnan(current)]

        if len(historical) < 20 or len(current) < 10:
            return {
                "shift_detected": False,
                "wasserstein_distance": 0.0,
                "z_score": 0.0,
                "interpretation": "INSUFFICIENT_DATA",
            }

        # Compute Wasserstein distance
        w_distance = self.wasserstein_distance_1d(historical, current)

        # Bootstrap to get baseline distribution of distances
        n_bootstrap = 100
        bootstrap_distances = []

        for _ in range(n_bootstrap):
            # Split historical in two
            idx = np.random.permutation(len(historical))
            half = len(historical) // 2
            h1 = historical[idx[:half]]
            h2 = historical[idx[half:]]

            d = self.wasserstein_distance_1d(h1, h2)
            bootstrap_distances.append(d)

        mean_dist = np.mean(bootstrap_distances)
        std_dist = np.std(bootstrap_distances)

        if std_dist < 1e-10:
            z_score = 0.0
        else:
            z_score = (w_distance - mean_dist) / std_dist

        shift_detected = z_score > threshold

        # Interpretation
        if z_score > 3:
            interpretation = "SEVERE_SHIFT"
        elif z_score > 2:
            interpretation = "SIGNIFICANT_SHIFT"
        elif z_score > 1:
            interpretation = "MILD_SHIFT"
        else:
            interpretation = "NORMAL"

        return {
            "shift_detected": shift_detected,
            "wasserstein_distance": float(w_distance),
            "z_score": float(z_score),
            "baseline_mean": float(mean_dist),
            "baseline_std": float(std_dist),
            "interpretation": interpretation,
        }

    def compute_rolling_distance(
        self,
        data: np.ndarray,
        window: int = 50,
        baseline_window: int = 200,
        step: int = 10,
    ) -> np.ndarray:
        """
        Compute rolling Wasserstein distance from baseline.

        Args:
            data: Full time series
            window: Current window size
            baseline_window: Historical baseline window
            step: Step between calculations

        Returns:
            Array of distances
        """
        n = len(data)
        distances = np.full(n, np.nan)

        for i in range(baseline_window + window, n, step):
            baseline = data[i - baseline_window - window : i - window]
            current = data[i - window : i]

            w = self.wasserstein_distance_1d(baseline, current)
            distances[i] = w

        # Forward fill
        last_valid = np.nan
        for i in range(n):
            if not np.isnan(distances[i]):
                last_valid = distances[i]
            elif not np.isnan(last_valid):
                distances[i] = last_valid

        return distances

    def generate_trading_signal(
        self, returns: np.ndarray, baseline_window: int = 200, current_window: int = 20
    ) -> Dict:
        """
        Generate trading signal based on distribution shift detection.

        Large shifts suggest regime change -> reduce exposure.
        Small shifts suggest stable regime -> normal trading.

        Args:
            returns: Return series
            baseline_window: Historical window for baseline
            current_window: Recent window to compare

        Returns:
            dict: Trading signal and metadata
        """
        returns = np.asarray(returns, dtype=float)

        if len(returns) < baseline_window + current_window:
            return {
                "signal": 0.0,
                "confidence": 0.0,
                "shift_detected": False,
                "interpretation": "INSUFFICIENT_DATA",
            }

        baseline = returns[-baseline_window - current_window : -current_window]
        current = returns[-current_window:]

        result = self.detect_distribution_shift(baseline, current)

        # Signal logic:
        # Large shift = reduce confidence, smaller positions
        # Normal = maintain confidence

        z_score = result["z_score"]

        if result["shift_detected"]:
            # Regime change - be cautious
            signal = 0.0  # Neutral
            confidence = max(0, 1 - z_score / 5)  # Decreasing confidence
            strategy_hint = "REDUCE_EXPOSURE"
        else:
            # Stable regime - maintain normal operations
            signal = 0.0  # No directional signal from OT
            confidence = min(1, 1 - z_score / 3)
            strategy_hint = "NORMAL"

        return {
            "signal": float(signal),
            "confidence": float(confidence),
            "shift_detected": result["shift_detected"],
            "wasserstein_distance": result["wasserstein_distance"],
            "z_score": float(z_score),
            "strategy_hint": strategy_hint,
            "interpretation": result["interpretation"],
        }

    def compare_distributions(
        self,
        dist_a: np.ndarray,
        dist_b: np.ndarray,
        name_a: str = "A",
        name_b: str = "B",
    ) -> Dict:
        """
        Comprehensive comparison of two distributions.

        Args:
            dist_a: First distribution samples
            dist_b: Second distribution samples
            name_a: Name for first distribution
            name_b: Name for second distribution

        Returns:
            dict: Comparison metrics
        """
        dist_a = np.asarray(dist_a, dtype=float).flatten()
        dist_b = np.asarray(dist_b, dtype=float).flatten()

        dist_a = dist_a[~np.isnan(dist_a)]
        dist_b = dist_b[~np.isnan(dist_b)]

        return {
            "wasserstein_1": self.wasserstein_distance_1d(dist_a, dist_b, p=1),
            "wasserstein_2": self.wasserstein_distance_1d(dist_a, dist_b, p=2),
            f"{name_a}_mean": float(np.mean(dist_a)),
            f"{name_b}_mean": float(np.mean(dist_b)),
            f"{name_a}_std": float(np.std(dist_a)),
            f"{name_b}_std": float(np.std(dist_b)),
            "mean_difference": float(np.mean(dist_a) - np.mean(dist_b)),
            "std_ratio": float(np.std(dist_a) / (np.std(dist_b) + 1e-10)),
        }
