"""
Mathematical Features Module

Advanced quantitative features for algorithmic trading based on:
- Topological Data Analysis (TDA)
- Information Theory (Entropy)
- Fractal Geometry
- Optimal Transport Theory
- Wavelet Analysis

These features capture market structure beyond traditional technical indicators.
"""

from .persistent_homology import PersistentHomologyAnalyzer
from .hurst_exponent import HurstExponentCalculator
from .entropy_measures import EntropyCalculator
from .wavelet_analysis import WaveletAnalyzer
from .optimal_transport import OptimalTransportAnalyzer
from .fractal_dimension import FractalDimensionCalculator

__all__ = [
    "PersistentHomologyAnalyzer",
    "HurstExponentCalculator",
    "EntropyCalculator",
    "WaveletAnalyzer",
    "OptimalTransportAnalyzer",
    "FractalDimensionCalculator",
]


class MathematicalFeatureGenerator:
    """
    Unified interface for generating all mathematical features.

    Combines signals from multiple advanced analysis methods.
    """

    def __init__(self):
        self.tda = PersistentHomologyAnalyzer()
        self.hurst = HurstExponentCalculator()
        self.entropy = EntropyCalculator()
        self.wavelet = WaveletAnalyzer()
        self.ot = OptimalTransportAnalyzer()
        self.fractal = FractalDimensionCalculator()

    def generate_all(self, price_data: "pd.DataFrame") -> dict:
        """
        Generate all mathematical features.

        Args:
            price_data: DataFrame with OHLCV columns

        Returns:
            dict: All mathematical features
        """
        import numpy as np

        close = price_data["close"].values
        returns = np.diff(np.log(close))

        features = {}

        # Hurst exponent
        try:
            hurst_result = self.hurst.compute_ensemble(returns)
            features["hurst"] = hurst_result["hurst"]
            features["hurst_interpretation"] = hurst_result["interpretation"]
        except Exception:
            features["hurst"] = 0.5
            features["hurst_interpretation"] = "ERROR"

        # Entropy
        try:
            entropy_result = self.entropy.compute_all(returns)
            features["entropy_composite"] = entropy_result["composite"]["entropy"]
            features["entropy_permutation"] = entropy_result["permutation"][
                "normalized"
            ]
        except Exception:
            features["entropy_composite"] = 0.5
            features["entropy_permutation"] = 0.5

        # Fractal dimension
        try:
            fd_result = self.fractal.compute_ensemble(close)
            features["fractal_dimension"] = fd_result["dimension"]
            features["fractal_interpretation"] = fd_result["interpretation"]
        except Exception:
            features["fractal_dimension"] = 1.5
            features["fractal_interpretation"] = "ERROR"

        # Wavelet features
        try:
            wavelet_features = self.wavelet.generate_features(close)
            features.update({f"wavelet_{k}": v for k, v in wavelet_features.items()})
        except Exception:
            features["wavelet_trend_energy"] = 0.5

        return features

    def generate_signals(self, price_data: "pd.DataFrame") -> dict:
        """
        Generate trading signals from mathematical analysis.

        Args:
            price_data: DataFrame with OHLCV columns

        Returns:
            dict: Aggregated trading signals
        """
        import numpy as np

        close = price_data["close"].values
        returns = np.diff(np.log(close))

        signals = {}
        weights = []
        weighted_signal = 0

        # Hurst signal
        try:
            hurst_signal = self.hurst.generate_trading_signal(returns)
            signals["hurst"] = hurst_signal
            weights.append((hurst_signal["signal"], hurst_signal["confidence"]))
        except Exception:
            signals["hurst"] = {"signal": 0, "confidence": 0}

        # Entropy signal
        try:
            entropy_signal = self.entropy.generate_trading_signal(returns)
            signals["entropy"] = entropy_signal
            weights.append(
                (0, entropy_signal["confidence_factor"])
            )  # Entropy modifies confidence
        except Exception:
            signals["entropy"] = {"confidence_factor": 1.0}

        # Fractal signal
        try:
            fractal_signal = self.fractal.generate_trading_signal(close)
            signals["fractal"] = fractal_signal
            weights.append((fractal_signal["signal"], fractal_signal["confidence"]))
        except Exception:
            signals["fractal"] = {"signal": 0, "confidence": 0}

        # Wavelet signal
        try:
            wavelet_signal = self.wavelet.generate_trading_signal(close)
            signals["wavelet"] = wavelet_signal
            weights.append((wavelet_signal["signal"], wavelet_signal["confidence"]))
        except Exception:
            signals["wavelet"] = {"signal": 0, "confidence": 0}

        # Aggregate signals
        total_weight = sum(w[1] for w in weights)
        if total_weight > 0:
            weighted_signal = sum(w[0] * w[1] for w in weights) / total_weight

        # Adjust by entropy confidence
        entropy_conf = signals.get("entropy", {}).get("confidence_factor", 1.0)
        final_signal = weighted_signal * entropy_conf

        return {
            "composite_signal": float(final_signal),
            "individual_signals": signals,
            "confidence": float(
                entropy_conf * (total_weight / len(weights) if weights else 0)
            ),
        }
