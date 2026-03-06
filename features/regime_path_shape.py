"""
Regime path-shape feature engine.

Builds past-only regime features from OHLCV history using a small set of
mathematical descriptors that are already present in the codebase.

Feature families:
- Hurst persistence / mean-reversion pressure
- Entropy / randomness
- Wavelet trend-vs-detail energy
- Fractal roughness
- Tail and jump diagnostics

All emitted columns use the ``ps_`` prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from quantum_alpha.features.mathematical.entropy_measures import EntropyCalculator
from quantum_alpha.features.mathematical.fractal_dimension import (
    FractalDimensionCalculator,
)
from quantum_alpha.features.mathematical.hurst_exponent import HurstExponentCalculator
from quantum_alpha.features.mathematical.wavelet_analysis import WaveletAnalyzer


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(v)


def _clip_series(series: pd.Series, lower: float, upper: float, fill: float) -> pd.Series:
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .clip(lower=lower, upper=upper)
        .fillna(fill)
    )


@dataclass(frozen=True)
class _WindowSpec:
    window: int
    step: int


class RegimePathShapeFeatureGenerator:
    """
    Generate rolling path-shape features from OHLCV history.

    Expensive mathematical descriptors are computed on sparse checkpoints and
    forward-filled so the output remains fully past-only while still being fast
    enough for multi-symbol daily research.
    """

    WINDOW_SPECS: tuple[_WindowSpec, ...] = (
        _WindowSpec(window=21, step=5),
        _WindowSpec(window=63, step=10),
        _WindowSpec(window=126, step=15),
    )

    def __init__(self) -> None:
        self._hurst = HurstExponentCalculator()
        self._entropy = EntropyCalculator()
        self._wavelet = WaveletAnalyzer()
        self._fractal = FractalDimensionCalculator()

    @classmethod
    def get_feature_names(cls) -> List[str]:
        names: List[str] = []
        for spec in cls.WINDOW_SPECS:
            w = spec.window
            names.extend(
                [
                    f"ps_hurst_{w}",
                    f"ps_entropy_{w}",
                    f"ps_wavelet_trend_energy_{w}",
                    f"ps_wavelet_detail_energy_{w}",
                    f"ps_wavelet_noise_level_{w}",
                    f"ps_fractal_dimension_{w}",
                    f"ps_realized_skew_{w}",
                    f"ps_realized_kurtosis_{w}",
                    f"ps_downside_semivariance_{w}",
                    f"ps_tail_ratio_{w}",
                    f"ps_jumpiness_{w}",
                ]
            )
        names.extend(
            [
                "ps_hurst_delta_21_63",
                "ps_hurst_delta_63_126",
                "ps_entropy_slope_21_63",
                "ps_entropy_slope_63_126",
                "ps_fractal_delta_21_63",
                "ps_fractal_delta_63_126",
                "ps_transition_score_21_63",
                "ps_transition_score_63_126",
                "ps_trend_persistence",
                "ps_mean_reversion_pressure",
                "ps_compression_breakout",
                "ps_crash_instability",
            ]
        )
        return names

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or len(df) == 0:
            return df.copy()

        out = df.copy()
        close = pd.to_numeric(out.get("close"), errors="coerce").ffill().bfill()
        if close.isna().all():
            for col in self.get_feature_names():
                out[col] = 0.0
            return out

        log_returns = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
        log_returns = log_returns.fillna(0.0)
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        returns = returns.fillna(0.0)

        for spec in self.WINDOW_SPECS:
            w = spec.window
            step = spec.step

            out[f"ps_hurst_{w}"] = self._rolling_hurst(
                log_returns.to_numpy(dtype=float), out.index, window=w, step=step
            )
            out[f"ps_entropy_{w}"] = self._rolling_entropy(
                log_returns.to_numpy(dtype=float), out.index, window=w, step=step
            )
            (
                out[f"ps_wavelet_trend_energy_{w}"],
                out[f"ps_wavelet_detail_energy_{w}"],
                out[f"ps_wavelet_noise_level_{w}"],
            ) = self._rolling_wavelet(close.to_numpy(dtype=float), out.index, window=w, step=step)
            out[f"ps_fractal_dimension_{w}"] = self._rolling_fractal(
                close.to_numpy(dtype=float), out.index, window=w, step=step
            )

            min_periods = max(8, w // 2)
            out[f"ps_realized_skew_{w}"] = _clip_series(
                returns.rolling(w, min_periods=min_periods).skew(),
                lower=-10.0,
                upper=10.0,
                fill=0.0,
            )
            out[f"ps_realized_kurtosis_{w}"] = _clip_series(
                returns.rolling(w, min_periods=min_periods).kurt(),
                lower=-10.0,
                upper=30.0,
                fill=0.0,
            )

            downside = returns.clip(upper=0.0)
            out[f"ps_downside_semivariance_{w}"] = _clip_series(
                downside.pow(2).rolling(w, min_periods=min_periods).mean(),
                lower=0.0,
                upper=1.0,
                fill=0.0,
            )

            q95 = returns.rolling(w, min_periods=min_periods).quantile(0.95).abs()
            q05 = returns.rolling(w, min_periods=min_periods).quantile(0.05).abs()
            tail_ratio = q95 / (q05 + 1e-8)
            out[f"ps_tail_ratio_{w}"] = _clip_series(
                tail_ratio,
                lower=0.0,
                upper=10.0,
                fill=1.0,
            )

            jumpiness = returns.abs().rolling(w, min_periods=min_periods).max() / (
                returns.rolling(w, min_periods=min_periods).std() + 1e-8
            )
            out[f"ps_jumpiness_{w}"] = _clip_series(
                jumpiness,
                lower=0.0,
                upper=25.0,
                fill=0.0,
            )

        out["ps_hurst_delta_21_63"] = _clip_series(
            out["ps_hurst_21"] - out["ps_hurst_63"], lower=-2.0, upper=2.0, fill=0.0
        )
        out["ps_hurst_delta_63_126"] = _clip_series(
            out["ps_hurst_63"] - out["ps_hurst_126"], lower=-2.0, upper=2.0, fill=0.0
        )
        out["ps_entropy_slope_21_63"] = _clip_series(
            out["ps_entropy_21"] - out["ps_entropy_63"],
            lower=-2.0,
            upper=2.0,
            fill=0.0,
        )
        out["ps_entropy_slope_63_126"] = _clip_series(
            out["ps_entropy_63"] - out["ps_entropy_126"],
            lower=-2.0,
            upper=2.0,
            fill=0.0,
        )
        out["ps_fractal_delta_21_63"] = _clip_series(
            out["ps_fractal_dimension_21"] - out["ps_fractal_dimension_63"],
            lower=-2.0,
            upper=2.0,
            fill=0.0,
        )
        out["ps_fractal_delta_63_126"] = _clip_series(
            out["ps_fractal_dimension_63"] - out["ps_fractal_dimension_126"],
            lower=-2.0,
            upper=2.0,
            fill=0.0,
        )

        out["ps_transition_score_21_63"] = _clip_series(
            (
                (out["ps_hurst_delta_21_63"]).abs()
                + (out["ps_entropy_slope_21_63"]).abs()
                + (out["ps_fractal_delta_21_63"]).abs()
                + (
                    out["ps_wavelet_trend_energy_21"]
                    - out["ps_wavelet_trend_energy_63"]
                ).abs()
            )
            / 4.0,
            lower=0.0,
            upper=5.0,
            fill=0.0,
        )
        out["ps_transition_score_63_126"] = _clip_series(
            (
                (out["ps_hurst_delta_63_126"]).abs()
                + (out["ps_entropy_slope_63_126"]).abs()
                + (out["ps_fractal_delta_63_126"]).abs()
                + (
                    out["ps_wavelet_trend_energy_63"]
                    - out["ps_wavelet_trend_energy_126"]
                ).abs()
            )
            / 4.0,
            lower=0.0,
            upper=5.0,
            fill=0.0,
        )

        vol_21 = returns.rolling(21, min_periods=10).std()
        vol_126 = returns.rolling(126, min_periods=63).std()
        compression_ratio = 1.0 - (vol_21 / (vol_126 + 1e-8))
        downside_share_63 = out["ps_downside_semivariance_63"] / (
            returns.pow(2).rolling(63, min_periods=32).mean().replace(0, np.nan) + 1e-8
        )

        out["ps_trend_persistence"] = np.tanh(
            (
                2.2 * (out["ps_hurst_63"] - 0.5)
                + 1.3 * (1.5 - out["ps_fractal_dimension_63"])
                + 1.1
                * (
                    out["ps_wavelet_trend_energy_63"]
                    - out["ps_wavelet_detail_energy_63"]
                )
                + 0.8 * (1.0 - out["ps_entropy_63"])
            ).fillna(0.0)
        )
        out["ps_mean_reversion_pressure"] = np.tanh(
            (
                2.0 * (0.5 - out["ps_hurst_21"])
                + 1.2 * (out["ps_fractal_dimension_21"] - 1.5)
                + 0.9
                * (
                    out["ps_wavelet_detail_energy_21"]
                    - out["ps_wavelet_trend_energy_21"]
                )
                + 0.7 * out["ps_entropy_21"]
            ).fillna(0.0)
        )
        out["ps_compression_breakout"] = np.tanh(
            (
                1.6 * _clip_series(compression_ratio, -5.0, 5.0, 0.0)
                + 0.8 * (1.0 - out["ps_entropy_21"])
                + 0.8 * out["ps_transition_score_21_63"]
                + 0.6
                * (
                    out["ps_wavelet_trend_energy_21"]
                    - out["ps_wavelet_detail_energy_21"]
                )
            ).fillna(0.0)
        )
        out["ps_crash_instability"] = np.tanh(
            (
                1.4 * np.maximum(-out["ps_realized_skew_63"], 0.0)
                + 0.6 * np.maximum(out["ps_realized_kurtosis_63"], 0.0)
                + 1.2 * _clip_series(downside_share_63, 0.0, 5.0, 0.0)
                + 0.9 * out["ps_jumpiness_21"]
                + 0.7 * out["ps_transition_score_21_63"]
                + 0.5 * np.maximum(1.0 - out["ps_tail_ratio_63"], 0.0)
            ).fillna(0.0)
        )

        for col in self.get_feature_names():
            out[col] = _clip_series(out.get(col, 0.0), lower=-50.0, upper=50.0, fill=0.0)

        return out

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.generate(df)

    def _rolling_hurst(
        self, values: np.ndarray, index: pd.Index, window: int, step: int
    ) -> pd.Series:
        arr = self._hurst.compute_rolling(values, window=window, step=step)
        return pd.Series(arr, index=index).ffill().fillna(0.5)

    def _rolling_entropy(
        self, values: np.ndarray, index: pd.Index, window: int, step: int
    ) -> pd.Series:
        arr = self._entropy.compute_rolling(
            values, window=window, step=step, entropy_type="permutation"
        )
        return pd.Series(arr, index=index).ffill().fillna(0.5)

    def _rolling_fractal(
        self, values: np.ndarray, index: pd.Index, window: int, step: int
    ) -> pd.Series:
        arr = self._fractal.compute_rolling(values, window=window, step=step, method="katz")
        return pd.Series(arr, index=index).ffill().fillna(1.5)

    def _rolling_wavelet(
        self, values: np.ndarray, index: pd.Index, window: int, step: int
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        trend = np.full(len(values), np.nan, dtype=float)
        detail = np.full(len(values), np.nan, dtype=float)
        noise = np.full(len(values), np.nan, dtype=float)

        for i in range(window, len(values), step):
            window_values = values[i - window : i]
            try:
                features = self._wavelet.generate_features(window_values)
            except Exception:
                features = {}
            trend[i] = _safe_float(features.get("trend_energy"), 0.5)
            detail[i] = _safe_float(features.get("high_freq_energy"), 0.5)
            noise[i] = _safe_float(features.get("noise_level"), 0.0)

        trend_s = pd.Series(trend, index=index).ffill().fillna(0.5)
        detail_s = pd.Series(detail, index=index).ffill().fillna(0.5)
        noise_s = pd.Series(noise, index=index).ffill().fillna(0.0)
        return trend_s, detail_s, noise_s
