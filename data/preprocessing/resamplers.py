"""
Resampling utilities for OHLCV time series.
"""

from __future__ import annotations

import pandas as pd
from typing import Dict


def resample_ohlcv(df: pd.DataFrame, rule: str = "1D") -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex for resampling")

    agg: Dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    available = {k: v for k, v in agg.items() if k in df.columns}
    resampled = df.resample(rule).agg(available)

    if "close" in resampled.columns:
        resampled = resampled.dropna(subset=["close"])
        resampled["returns"] = resampled["close"].pct_change()

    return resampled
