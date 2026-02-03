"""
Feature normalization utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, Dict, Optional


class ZScoreNormalizer:
    """
    Z-score normalization with optional clipping.
    """

    def __init__(self, columns: Iterable[str], clip: Optional[float] = 5.0) -> None:
        self.columns = list(columns)
        self.clip = clip
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> "ZScoreNormalizer":
        for col in self.columns:
            if col not in df.columns:
                continue
            series = df[col].astype(float)
            mean = series.mean()
            std = series.std()
            if not np.isfinite(std) or std == 0:
                std = 1.0
            self.means[col] = float(mean)
            self.stds[col] = float(std)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for col in self.columns:
            if col not in result.columns or col not in self.means:
                continue
            result[col] = (result[col] - self.means[col]) / self.stds[col]
            if self.clip is not None:
                result[col] = result[col].clip(-self.clip, self.clip)
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
