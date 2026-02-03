"""
Missing data imputation for OHLCV time series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple


class MissingValueImputer:
    """
    Impute missing values with conservative, market-safe defaults.
    """

    def __init__(
        self,
        price_cols: Tuple[str, str, str, str] = ("open", "high", "low", "close"),
        volume_col: str = "volume",
        max_ffill: int = 5,
        volume_method: str = "median",
    ) -> None:
        self.price_cols = price_cols
        self.volume_col = volume_col
        self.max_ffill = max_ffill
        self.volume_method = volume_method

    def impute(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        price_cols = [c for c in self.price_cols if c in result.columns]
        if price_cols:
            result[price_cols] = result[price_cols].ffill(limit=self.max_ffill)
            result[price_cols] = result[price_cols].bfill(limit=self.max_ffill)

        if self.volume_col in result.columns:
            result[self.volume_col] = self._impute_volume(result[self.volume_col])

        if "close" in result.columns:
            returns = result["close"].pct_change()
            returns = returns.replace([np.inf, -np.inf], np.nan)
            result["returns"] = returns

        return result

    def _impute_volume(self, series: pd.Series) -> pd.Series:
        if self.volume_method == "ffill":
            return series.ffill(limit=self.max_ffill).fillna(0.0)

        if self.volume_method == "zero":
            return series.fillna(0.0)

        rolling_median = series.rolling(20, min_periods=1).median()
        return series.fillna(rolling_median)
