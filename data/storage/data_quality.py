"""
Data quality checks for OHLCV time series.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class DataQualityChecker:
    """
    Validate OHLCV data for common issues.
    """

    def __init__(
        self,
        required_cols: Tuple[str, str, str, str, str] = (
            "open",
            "high",
            "low",
            "close",
            "volume",
        ),
        max_missing_ratio: float = 0.05,
        max_abs_return: float = 0.5,
    ) -> None:
        self.required_cols = required_cols
        self.max_missing_ratio = max_missing_ratio
        self.max_abs_return = max_abs_return

    def validate_ohlcv(self, df: pd.DataFrame) -> Dict[str, object]:
        issues: List[str] = []
        metrics: Dict[str, float] = {}

        missing_cols = [c for c in self.required_cols if c not in df.columns]
        if missing_cols:
            issues.append(f"missing_columns={missing_cols}")

        if df.empty:
            issues.append("empty_dataframe")
            return {
                "is_valid": False,
                "issues": issues,
                "metrics": metrics,
            }

        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append("index_not_datetime")

        missing_ratio = df.isna().mean().mean()
        metrics["missing_ratio"] = float(missing_ratio)
        if missing_ratio > self.max_missing_ratio:
            issues.append(f"missing_ratio>{self.max_missing_ratio}")

        if all(col in df.columns for col in self.required_cols[:4]):
            price_df = df[list(self.required_cols[:4])]
            non_positive = (price_df <= 0).sum().sum()
            metrics["non_positive_prices"] = float(non_positive)
            if non_positive > 0:
                issues.append("non_positive_prices")

            if "high" in df.columns and "low" in df.columns:
                invalid_hl = (df["high"] < df["low"]).sum()
                metrics["invalid_high_low"] = float(invalid_hl)
                if invalid_hl > 0:
                    issues.append("high_lt_low")

        if "volume" in df.columns:
            negative_vol = (df["volume"] < 0).sum()
            metrics["negative_volume"] = float(negative_vol)
            if negative_vol > 0:
                issues.append("negative_volume")

        if "close" in df.columns:
            returns = df["close"].pct_change().replace([np.inf, -np.inf], np.nan)
            outliers = (returns.abs() > self.max_abs_return).sum()
            metrics["return_outliers"] = float(outliers)
            if outliers > 0:
                issues.append("return_outliers")

        if df.index.has_duplicates:
            issues.append("duplicate_index")
        if not df.index.is_monotonic_increasing:
            issues.append("index_not_sorted")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "metrics": metrics,
        }
