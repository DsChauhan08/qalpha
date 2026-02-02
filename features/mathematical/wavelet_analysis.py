"""
Wavelet Analysis for Multi-Resolution Decomposition

Wavelet transforms decompose time series into different frequency components:
- Noise reduction (denoising)
- Trend extraction
- Cycle detection
- Multi-scale feature extraction

Supports both DWT (Discrete Wavelet Transform) and CWT (Continuous Wavelet Transform).
"""

import numpy as np
from typing import Dict, Optional, Tuple, List


class WaveletAnalyzer:
    """
    Multi-resolution analysis using wavelet transforms.

    Applications:
    1. Time series denoising
    2. Trend extraction
    3. Volatility decomposition
    4. Feature extraction for ML
    """

    def __init__(self, wavelet: str = "db4", mode: str = "symmetric"):
        """
        Args:
            wavelet: Wavelet family ('db4', 'haar', 'coif1', etc.)
            mode: Signal extension mode
        """
        self.wavelet = wavelet
        self.mode = mode
        self._has_pywt = self._check_pywt()

    def _check_pywt(self) -> bool:
        """Check if PyWavelets is available."""
        try:
            import pywt

            return True
        except ImportError:
            return False

    def decompose(
        self, signal: np.ndarray, level: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Perform wavelet decomposition.

        Args:
            signal: Input time series
            level: Decomposition level (auto if None)

        Returns:
            list: [cA_n, cD_n, cD_{n-1}, ..., cD_1]
                  Approximation and detail coefficients
        """
        signal = np.asarray(signal, dtype=float)

        if level is None:
            level = min(int(np.log2(len(signal))), 6)

        if self._has_pywt:
            import pywt

            coeffs = pywt.wavedec(signal, self.wavelet, level=level, mode=self.mode)
            return coeffs
        else:
            # Fallback: simple Haar-like decomposition
            return self._simple_decompose(signal, level)

    def _simple_decompose(self, signal: np.ndarray, level: int) -> List[np.ndarray]:
        """Simple Haar-like wavelet decomposition without pywt."""
        coeffs = []
        current = signal.copy()

        for _ in range(level):
            n = len(current)
            if n < 2:
                break

            # Pad if odd length
            if n % 2:
                current = np.append(current, current[-1])
                n += 1

            # Haar transform
            evens = current[::2]
            odds = current[1::2]

            # Approximation (low-pass)
            approx = (evens + odds) / np.sqrt(2)
            # Detail (high-pass)
            detail = (evens - odds) / np.sqrt(2)

            coeffs.insert(0, detail)
            current = approx

        coeffs.insert(0, current)  # Final approximation
        return coeffs

    def reconstruct(self, coeffs: List[np.ndarray]) -> np.ndarray:
        """
        Reconstruct signal from wavelet coefficients.

        Args:
            coeffs: List of coefficients from decompose()

        Returns:
            np.array: Reconstructed signal
        """
        if self._has_pywt:
            import pywt

            return pywt.waverec(coeffs, self.wavelet, mode=self.mode)
        else:
            return self._simple_reconstruct(coeffs)

    def _simple_reconstruct(self, coeffs: List[np.ndarray]) -> np.ndarray:
        """Simple reconstruction without pywt."""
        current = coeffs[0]

        for detail in coeffs[1:]:
            n = len(detail)

            # Inverse Haar transform
            result = np.zeros(n * 2)
            result[::2] = (current[:n] + detail) / np.sqrt(2)
            result[1::2] = (current[:n] - detail) / np.sqrt(2)
            current = result

        return current

    def denoise(
        self,
        signal: np.ndarray,
        threshold_mode: str = "soft",
        threshold_factor: float = 1.0,
    ) -> np.ndarray:
        """
        Denoise signal using wavelet thresholding.

        Args:
            signal: Noisy input
            threshold_mode: 'soft' or 'hard' thresholding
            threshold_factor: Multiplier for universal threshold

        Returns:
            np.array: Denoised signal
        """
        signal = np.asarray(signal, dtype=float)
        original_len = len(signal)

        # Decompose
        coeffs = self.decompose(signal)

        # Threshold detail coefficients
        denoised_coeffs = [coeffs[0]]  # Keep approximation

        for detail in coeffs[1:]:
            # Universal threshold (Donoho-Johnstone)
            sigma = np.median(np.abs(detail)) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(signal))) * threshold_factor

            if threshold_mode == "soft":
                denoised_detail = self._soft_threshold(detail, threshold)
            else:
                denoised_detail = self._hard_threshold(detail, threshold)

            denoised_coeffs.append(denoised_detail)

        # Reconstruct
        denoised_signal = self.reconstruct(denoised_coeffs)

        # Trim to original length
        return denoised_signal[:original_len]

    def _soft_threshold(self, data: np.ndarray, threshold: float) -> np.ndarray:
        """Soft thresholding."""
        return np.sign(data) * np.maximum(np.abs(data) - threshold, 0)

    def _hard_threshold(self, data: np.ndarray, threshold: float) -> np.ndarray:
        """Hard thresholding."""
        result = data.copy()
        result[np.abs(data) < threshold] = 0
        return result

    def extract_trend(self, signal: np.ndarray, level: int = 3) -> np.ndarray:
        """
        Extract long-term trend from signal.

        Args:
            signal: Input time series
            level: Decomposition level for trend

        Returns:
            np.array: Trend component
        """
        signal = np.asarray(signal, dtype=float)
        original_len = len(signal)

        coeffs = self.decompose(signal, level=level)

        # Keep only approximation (low frequencies)
        trend_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]

        trend = self.reconstruct(trend_coeffs)
        return trend[:original_len]

    def extract_cycles(
        self, signal: np.ndarray, min_period: int = 5, max_period: int = 50
    ) -> np.ndarray:
        """
        Extract cyclical components within period range.

        Args:
            signal: Input time series
            min_period: Minimum cycle period
            max_period: Maximum cycle period

        Returns:
            np.array: Cyclical component
        """
        signal = np.asarray(signal, dtype=float)
        original_len = len(signal)

        coeffs = self.decompose(signal)

        # Zero out approximation
        cycle_coeffs = [np.zeros_like(coeffs[0])]

        for i, detail in enumerate(coeffs[1:], 1):
            # Period range for this level (approximate)
            period_min = 2**i
            period_max = 2 ** (i + 1)

            if period_min >= min_period and period_max <= max_period * 2:
                cycle_coeffs.append(detail)
            else:
                cycle_coeffs.append(np.zeros_like(detail))

        cycles = self.reconstruct(cycle_coeffs)
        return cycles[:original_len]

    def extract_noise(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract high-frequency noise component.

        Args:
            signal: Input time series

        Returns:
            np.array: Noise component
        """
        denoised = self.denoise(signal)
        return signal[: len(denoised)] - denoised

    def compute_energy_distribution(self, signal: np.ndarray) -> Dict:
        """
        Compute energy distribution across frequency bands.

        Args:
            signal: Input time series

        Returns:
            dict: Energy by frequency band
        """
        coeffs = self.decompose(signal)

        energies = {}
        total_energy = sum(np.sum(c**2) for c in coeffs)

        if total_energy == 0:
            return {"trend": 1.0}

        # Approximation energy (trend)
        energies["trend"] = float(np.sum(coeffs[0] ** 2) / total_energy)

        # Detail energies (different frequencies)
        for i, detail in enumerate(coeffs[1:], 1):
            band_name = f"detail_{i}"
            energies[band_name] = float(np.sum(detail**2) / total_energy)

        return energies

    def compute_wavelet_variance(self, signal: np.ndarray) -> Dict:
        """
        Compute variance at each wavelet scale.

        Useful for identifying dominant frequencies.

        Args:
            signal: Input time series

        Returns:
            dict: Variance by scale
        """
        coeffs = self.decompose(signal)

        variances = {}
        variances["approximation"] = float(np.var(coeffs[0]))

        for i, detail in enumerate(coeffs[1:], 1):
            scale = 2**i
            variances[f"scale_{scale}"] = float(np.var(detail))

        # Dominant scale
        detail_vars = {k: v for k, v in variances.items() if k.startswith("scale_")}
        if detail_vars:
            dominant = max(detail_vars, key=detail_vars.get)
            variances["dominant_scale"] = dominant

        return variances

    def generate_features(self, signal: np.ndarray) -> Dict:
        """
        Generate wavelet-based features for ML.

        Args:
            signal: Input time series

        Returns:
            dict: Wavelet features
        """
        coeffs = self.decompose(signal)
        energy = self.compute_energy_distribution(signal)
        variance = self.compute_wavelet_variance(signal)

        features = {}

        # Energy features
        features["trend_energy"] = energy.get("trend", 0)
        features["high_freq_energy"] = sum(
            v for k, v in energy.items() if k.startswith("detail")
        )

        # Coefficient statistics
        for i, detail in enumerate(coeffs[1:], 1):
            if len(detail) > 0:
                features[f"d{i}_mean"] = float(np.mean(np.abs(detail)))
                features[f"d{i}_std"] = float(np.std(detail))
                features[f"d{i}_max"] = float(np.max(np.abs(detail)))

        # Trend strength (approximation energy ratio)
        features["trend_strength"] = features["trend_energy"]

        # Noise level (highest detail energy)
        if len(coeffs) > 1:
            features["noise_level"] = float(np.std(coeffs[-1]))

        return features

    def generate_trading_signal(self, signal: np.ndarray) -> Dict:
        """
        Generate trading signal from wavelet analysis.

        Args:
            signal: Price series

        Returns:
            dict: Trading signal and metadata
        """
        # Extract components
        trend = self.extract_trend(signal)
        denoised = self.denoise(signal)

        # Trend direction
        if len(trend) > 1:
            trend_slope = (trend[-1] - trend[-5]) / (5 * np.std(trend) + 1e-10)
        else:
            trend_slope = 0

        # Position relative to trend
        if len(signal) > 0 and len(trend) > 0:
            deviation = (signal[-1] - trend[-1]) / (np.std(signal) + 1e-10)
        else:
            deviation = 0

        # Energy distribution
        energy = self.compute_energy_distribution(signal)
        trend_energy = energy.get("trend", 0)

        # Signal logic
        # Strong trend + following trend = momentum signal
        # Weak trend + deviation = mean reversion signal

        if trend_energy > 0.7:
            # Strong trend - follow it
            signal_value = np.clip(trend_slope, -1, 1)
            strategy = "MOMENTUM"
        elif trend_energy < 0.3:
            # Weak trend - mean revert
            signal_value = np.clip(-deviation * 0.5, -1, 1)
            strategy = "MEAN_REVERSION"
        else:
            # Mixed - blend signals
            signal_value = np.clip(trend_slope * 0.5 - deviation * 0.3, -1, 1)
            strategy = "BLENDED"

        return {
            "signal": float(signal_value),
            "trend_slope": float(trend_slope),
            "deviation_from_trend": float(deviation),
            "trend_energy": float(trend_energy),
            "strategy": strategy,
            "confidence": float(abs(trend_energy - 0.5) * 2),  # Higher at extremes
        }

    def compute_rolling_trend(
        self, signal: np.ndarray, window: int = 100, step: int = 10
    ) -> np.ndarray:
        """
        Compute rolling wavelet trend.

        Args:
            signal: Full time series
            window: Rolling window size
            step: Step between calculations

        Returns:
            Array of trend values
        """
        n = len(signal)
        trends = np.full(n, np.nan)

        for i in range(window, n, step):
            window_data = signal[i - window : i]
            trend = self.extract_trend(window_data)
            if len(trend) > 0:
                trends[i] = trend[-1]

        # Interpolate gaps
        mask = ~np.isnan(trends)
        if np.any(mask):
            indices = np.arange(n)
            trends = np.interp(indices, indices[mask], trends[mask])

        return trends
