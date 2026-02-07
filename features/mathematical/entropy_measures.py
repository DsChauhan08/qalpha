"""
Entropy Measures for Market Analysis

Information-theoretic measures that quantify:
- Market uncertainty and predictability
- Complexity and randomness of price movements
- Structural changes in market dynamics

Types:
1. Shannon Entropy - classic information measure
2. Sample Entropy - regularity/complexity without self-matching
3. Permutation Entropy - ordinal pattern complexity
4. Approximate Entropy - pattern regularity
"""

import numpy as np
from math import factorial
from typing import Dict, Optional, Tuple, List
from collections import Counter
from itertools import permutations


class EntropyCalculator:
    """
    Computes various entropy measures for financial time series.

    Applications:
    - Market regime detection (high entropy = random/uncertain)
    - Predictability assessment (low entropy = more predictable)
    - Complexity analysis for strategy selection
    """

    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: Whether to normalize entropy to [0, 1]
        """
        self.normalize = normalize

    def shannon_entropy(
        self, data: np.ndarray, n_bins: int = 10, method: str = "histogram"
    ) -> Dict:
        """
        Compute Shannon entropy of the distribution.

        H(X) = -sum(p(x) * log(p(x)))

        Args:
            data: Time series
            n_bins: Number of bins for discretization
            method: 'histogram' or 'kde'

        Returns:
            dict: Entropy value and metadata
        """
        data = np.asarray(data, dtype=float)
        data = data[~np.isnan(data)]

        if len(data) < 10:
            return {
                "entropy": 0.0,
                "normalized": 0.0,
                "method": "shannon",
                "interpretation": "INSUFFICIENT_DATA",
            }

        if method == "histogram":
            # Histogram-based estimation
            hist, _ = np.histogram(data, bins=n_bins, density=True)
            hist = hist[hist > 0]  # Remove zeros

            # Normalize to get probabilities
            probs = hist / np.sum(hist)
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            max_entropy = np.log2(n_bins)

        else:
            # KDE-based estimation
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(data)
            x_grid = np.linspace(data.min(), data.max(), 100)
            density = kde(x_grid)
            density = density / np.sum(density)
            entropy = -np.sum(density * np.log2(density + 1e-10))
            max_entropy = np.log2(len(x_grid))

        normalized = entropy / max_entropy if max_entropy > 0 else 0.0

        return {
            "entropy": float(entropy),
            "normalized": float(normalized),
            "method": "shannon",
            "max_entropy": float(max_entropy),
            "interpretation": self._interpret_entropy(normalized),
        }

    def sample_entropy(
        self, data: np.ndarray, m: int = 2, r: Optional[float] = None
    ) -> Dict:
        """
        Compute Sample Entropy (SampEn).

        Measures complexity/regularity without self-matching.
        Lower values indicate more self-similarity (predictable).

        Args:
            data: Time series
            m: Embedding dimension
            r: Tolerance (default: 0.2 * std)

        Returns:
            dict: Sample entropy value
        """
        data = np.asarray(data, dtype=float)
        data = data[~np.isnan(data)]
        n = len(data)

        if n < m + 1:
            return {
                "entropy": 0.0,
                "method": "sample",
                "interpretation": "INSUFFICIENT_DATA",
            }

        if r is None:
            r = float(0.2 * np.std(data))

        # Count template matches
        def count_matches(template_len: int) -> int:
            count = 0
            templates = []

            for i in range(n - template_len):
                templates.append(data[i : i + template_len])

            for i in range(len(templates)):
                for j in range(i + 1, len(templates)):
                    # Chebyshev distance (max absolute difference)
                    dist = np.max(np.abs(templates[i] - templates[j]))
                    if dist < r:
                        count += 1

            return count

        # Count B (m-length matches) and A (m+1-length matches)
        B = count_matches(m)
        A = count_matches(m + 1)

        if B == 0:
            return {
                "entropy": float("inf"),
                "method": "sample",
                "interpretation": "HIGHLY_RANDOM",
            }

        # Sample entropy
        sample_ent = float(-np.log(A / B)) if A > 0 else float("inf")

        # Approximate normalization (typical range 0-3)
        normalized = min(sample_ent / 3.0, 1.0) if np.isfinite(sample_ent) else 1.0

        return {
            "entropy": sample_ent if np.isfinite(sample_ent) else 3.0,
            "normalized": float(normalized),
            "method": "sample",
            "m": m,
            "r": r,
            "interpretation": self._interpret_sample_entropy(sample_ent),
        }

    def permutation_entropy(
        self, data: np.ndarray, order: int = 3, delay: int = 1
    ) -> Dict:
        """
        Compute Permutation Entropy.

        Based on ordinal patterns - robust to noise and nonlinearity.

        Args:
            data: Time series
            order: Order of permutation (embedding dimension)
            delay: Time delay

        Returns:
            dict: Permutation entropy
        """
        data = np.asarray(data, dtype=float)
        data = data[~np.isnan(data)]
        n = len(data)

        if n < order * delay:
            return {
                "entropy": 0.0,
                "normalized": 0.0,
                "method": "permutation",
                "interpretation": "INSUFFICIENT_DATA",
            }

        # Extract ordinal patterns
        pattern_counts = Counter()

        for i in range(n - (order - 1) * delay):
            # Extract embedding
            indices = range(i, i + order * delay, delay)
            values = [data[j] for j in indices]

            # Get ordinal pattern (rank order)
            pattern = tuple(np.argsort(values))
            pattern_counts[pattern] += 1

        # Calculate entropy
        total = sum(pattern_counts.values())
        probs = np.array([count / total for count in pattern_counts.values()])

        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(float(factorial(order)))  # Max possible patterns

        normalized = entropy / max_entropy if max_entropy > 0 else 0.0

        return {
            "entropy": float(entropy),
            "normalized": float(normalized),
            "method": "permutation",
            "order": order,
            "delay": delay,
            "n_patterns": len(pattern_counts),
            "max_patterns": factorial(order),
            "interpretation": self._interpret_entropy(normalized),
        }

    def approximate_entropy(
        self, data: np.ndarray, m: int = 2, r: Optional[float] = None
    ) -> Dict:
        """
        Compute Approximate Entropy (ApEn).

        Similar to Sample Entropy but includes self-matches.

        Args:
            data: Time series
            m: Embedding dimension
            r: Tolerance (default: 0.2 * std)

        Returns:
            dict: Approximate entropy
        """
        data = np.asarray(data, dtype=float)
        data = data[~np.isnan(data)]
        n = len(data)

        if n < m + 1:
            return {
                "entropy": 0.0,
                "method": "approximate",
                "interpretation": "INSUFFICIENT_DATA",
            }

        if r is None:
            r = float(0.2 * np.std(data))

        def phi(template_len: int) -> float:
            templates = []
            for i in range(n - template_len + 1):
                templates.append(data[i : i + template_len])

            C = np.zeros(len(templates))
            for i in range(len(templates)):
                count = 0
                for j in range(len(templates)):
                    dist = np.max(np.abs(templates[i] - templates[j]))
                    if dist <= r:
                        count += 1
                C[i] = count / len(templates)

            return float(np.sum(np.log(C + 1e-10)) / len(templates))

        ap_en = phi(m) - phi(m + 1)

        # Normalize
        normalized = min(ap_en / 2.0, 1.0) if ap_en > 0 else 0.0

        return {
            "entropy": float(ap_en),
            "normalized": float(normalized),
            "method": "approximate",
            "m": m,
            "r": r,
            "interpretation": self._interpret_sample_entropy(ap_en),
        }

    def multiscale_entropy(
        self,
        data: np.ndarray,
        scales: Optional[List[int]] = None,
        m: int = 2,
        r: Optional[float] = None,
    ) -> Dict:
        """
        Compute Multiscale Entropy (MSE).

        Sample entropy at multiple time scales - captures complexity
        across different time horizons.

        Args:
            data: Time series
            scales: List of scale factors
            m: Embedding dimension
            r: Tolerance

        Returns:
            dict: MSE values at each scale
        """
        data = np.asarray(data, dtype=float)
        data = data[~np.isnan(data)]
        n = len(data)

        if scales is None:
            scales = [1, 2, 4, 8, 16]

        scales = [s for s in scales if n // s >= m + 10]

        if not scales:
            return {
                "mse": {},
                "complexity_index": 0.0,
                "method": "multiscale",
                "interpretation": "INSUFFICIENT_DATA",
            }

        mse_values = {}

        for scale in scales:
            # Coarse-grain the series
            coarse_len = n // scale
            coarse_data = np.array(
                [np.mean(data[i * scale : (i + 1) * scale]) for i in range(coarse_len)]
            )

            # Compute sample entropy at this scale
            result = self.sample_entropy(coarse_data, m=m, r=r)
            mse_values[scale] = result["entropy"]

        # Complexity index (area under MSE curve)
        valid_mse = [v for v in mse_values.values() if np.isfinite(v)]
        complexity_index = np.mean(valid_mse) if valid_mse else 0.0

        return {
            "mse": mse_values,
            "complexity_index": float(complexity_index),
            "method": "multiscale",
            "scales": scales,
            "interpretation": self._interpret_complexity(float(complexity_index)),
        }

    def compute_all(self, data: np.ndarray, include_multiscale: bool = False) -> Dict:
        """
        Compute all entropy measures.

        Args:
            data: Time series
            include_multiscale: Whether to include MSE (slower)

        Returns:
            dict: All entropy measures
        """
        results = {
            "shannon": self.shannon_entropy(data),
            "sample": self.sample_entropy(data),
            "permutation": self.permutation_entropy(data),
            "approximate": self.approximate_entropy(data),
        }

        if include_multiscale:
            results["multiscale"] = self.multiscale_entropy(data)

        # Composite score (average of normalized entropies)
        normalized_values = []
        for key, val in results.items():
            if isinstance(val, dict) and "normalized" in val:
                normalized_values.append(val["normalized"])

        composite = np.mean(normalized_values) if normalized_values else 0.5

        results["composite"] = {
            "entropy": float(composite),
            "interpretation": self._interpret_entropy(float(composite)),
        }

        return results

    def _interpret_entropy(self, normalized_entropy: float) -> str:
        """Interpret normalized entropy (0-1)."""
        if normalized_entropy < 0.3:
            return "LOW_ENTROPY_PREDICTABLE"
        elif normalized_entropy < 0.5:
            return "MODERATE_LOW_ENTROPY"
        elif normalized_entropy < 0.7:
            return "MODERATE_HIGH_ENTROPY"
        else:
            return "HIGH_ENTROPY_RANDOM"

    def _interpret_sample_entropy(self, sample_ent: float) -> str:
        """Interpret sample entropy (typically 0-3+)."""
        if not np.isfinite(sample_ent):
            return "HIGHLY_RANDOM"
        elif sample_ent < 0.5:
            return "HIGHLY_REGULAR"
        elif sample_ent < 1.0:
            return "MODERATELY_REGULAR"
        elif sample_ent < 2.0:
            return "MODERATELY_COMPLEX"
        else:
            return "HIGHLY_COMPLEX"

    def _interpret_complexity(self, complexity_index: float) -> str:
        """Interpret multiscale complexity index."""
        if complexity_index < 0.5:
            return "LOW_COMPLEXITY"
        elif complexity_index < 1.0:
            return "MODERATE_COMPLEXITY"
        elif complexity_index < 1.5:
            return "HIGH_COMPLEXITY"
        else:
            return "VERY_HIGH_COMPLEXITY"

    def generate_trading_signal(self, data: np.ndarray) -> Dict:
        """
        Generate trading signal based on entropy analysis.

        Low entropy -> More predictable -> Higher confidence in signals
        High entropy -> More random -> Reduce position sizes

        Args:
            data: Price or returns series

        Returns:
            dict: Signal adjustments and metadata
        """
        results = self.compute_all(data)

        composite = results["composite"]["entropy"]
        perm_normalized = results["permutation"].get("normalized", 0.5)
        sample_normalized = results["sample"].get("normalized", 0.5)

        # Signal adjustment factor:
        # Low entropy = high factor (more confident)
        # High entropy = low factor (less confident)
        confidence_factor = 1.0 - composite

        # Regime detection based on entropy
        if composite < 0.3:
            regime = "ORDERED"
            strategy_hint = "TREND_FOLLOWING"
        elif composite > 0.7:
            regime = "CHAOTIC"
            strategy_hint = "REDUCE_EXPOSURE"
        else:
            regime = "NORMAL"
            strategy_hint = "BALANCED"

        return {
            "confidence_factor": float(confidence_factor),
            "entropy_composite": float(composite),
            "regime": regime,
            "strategy_hint": strategy_hint,
            "shannon": results["shannon"]["normalized"],
            "permutation": float(perm_normalized),
            "sample": float(sample_normalized),
            "interpretation": results["composite"]["interpretation"],
        }

    def compute_rolling(
        self,
        data: np.ndarray,
        window: int = 100,
        step: int = 10,
        entropy_type: str = "permutation",
    ) -> np.ndarray:
        """
        Compute rolling entropy.

        Args:
            data: Full time series
            window: Rolling window size
            step: Step between calculations
            entropy_type: 'shannon', 'sample', or 'permutation'

        Returns:
            Array of entropy values (NaN padded at start)
        """
        n = len(data)
        entropy_values = np.full(n, np.nan)

        for i in range(window, n, step):
            window_data = data[i - window : i]

            if entropy_type == "shannon":
                result = self.shannon_entropy(window_data)
            elif entropy_type == "sample":
                result = self.sample_entropy(window_data)
            else:
                result = self.permutation_entropy(window_data)

            entropy_values[i] = result.get("normalized", result.get("entropy", 0.5))

        # Forward fill
        last_valid = np.nan
        for i in range(n):
            if not np.isnan(entropy_values[i]):
                last_valid = entropy_values[i]
            elif not np.isnan(last_valid):
                entropy_values[i] = last_valid

        return entropy_values
