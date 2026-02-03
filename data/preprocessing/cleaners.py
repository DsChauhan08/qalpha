"""
Data cleaning utilities for OHLCV time series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, Tuple


class DataCleaner:
    """
    Clean and validate OHLCV data.

    Actions:
    - Sort index and remove duplicates
    - Remove invalid prices/volume
    - Robust outlier clipping
    - Enforce OHLC consistency
    """

    def __init__(
        self,
        price_cols: Tuple[str, str, str, str] = ("open", "high", "low", "close"),
        volume_col: str = "volume",
        max_z: float = 6.0,
        winsor_limits: Tuple[float, float] = (0.001, 0.999),
    ) -> None:
        self.price_cols = price_cols
        self.volume_col = volume_col
        self.max_z = max_z
        self.winsor_limits = winsor_limits

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        if not result.index.is_monotonic_increasing:
            result = result.sort_index()

        if result.index.has_duplicates:
            result = result[~result.index.duplicated(keep="last")]

        for col in self.price_cols:
            if col in result.columns:
                result.loc[result[col] <= 0, col] = np.nan

        if self.volume_col in result.columns:
            result.loc[result[self.volume_col] < 0, self.volume_col] = np.nan

        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result = self._robust_clip(result, numeric_cols)
        result = self._winsorize(result, numeric_cols)
        result = self._enforce_ohlc(result)

        return result

    def _robust_clip(self, df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        result = df.copy()

        for col in columns:
            series = result[col].astype(float)
            median = series.median(skipna=True)
            mad = np.median(np.abs(series - median))
            if not np.isfinite(mad) or mad == 0:
                continue

            scale = 1.4826 * mad
            z = (series - median) / scale
            if z.abs().max(skipna=True) <= self.max_z:
                continue

            clipped = median + np.clip(z, -self.max_z, self.max_z) * scale
            result[col] = clipped

        return result

    def _winsorize(self, df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        result = df.copy()
        low_q, high_q = self.winsor_limits

        if not (0 < low_q < high_q < 1):
            return result

        for col in columns:
            q_low = result[col].quantile(low_q)
            q_high = result[col].quantile(high_q)
            result[col] = result[col].clip(q_low, q_high)

        return result

    def _enforce_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        open_col, high_col, low_col, close_col = self.price_cols

        if not all(col in df.columns for col in self.price_cols):
            return df

        result = df.copy()
        open_v = result[open_col]
        high_v = result[high_col]
        low_v = result[low_col]
        close_v = result[close_col]

        max_oc = np.maximum(open_v, close_v)
        min_oc = np.minimum(open_v, close_v)

        result[high_col] = np.maximum(high_v, max_oc)
        result[low_col] = np.minimum(low_v, min_oc)

        return result
