"""Volatility estimators used by the portfolio construction engine."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd


ANNUALIZATION = 252.0


def _safe_std(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna()
    if len(arr) < 2:
        return 0.0
    return float(arr.std(ddof=1))


def close_to_close_vol(close: pd.Series, annualization: float = ANNUALIZATION) -> float:
    rets = pd.to_numeric(close, errors="coerce").pct_change().dropna()
    return float(_safe_std(rets) * math.sqrt(float(annualization)))


def parkinson_vol(high: pd.Series, low: pd.Series, annualization: float = ANNUALIZATION) -> float:
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    ratio = np.log((high / low).replace([np.inf, -np.inf], np.nan)).dropna()
    if len(ratio) < 2:
        return 0.0
    var = (1.0 / (4.0 * math.log(2.0))) * float((ratio**2).mean())
    return float(math.sqrt(max(var, 0.0) * annualization))


def garman_klass_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    annualization: float = ANNUALIZATION,
) -> float:
    o = pd.to_numeric(open_, errors="coerce")
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    log_hl = np.log((h / l).replace([np.inf, -np.inf], np.nan))
    log_co = np.log((c / o).replace([np.inf, -np.inf], np.nan))
    series = 0.5 * (log_hl**2) - (2.0 * math.log(2.0) - 1.0) * (log_co**2)
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 2:
        return 0.0
    return float(math.sqrt(max(float(series.mean()), 0.0) * annualization))


def rogers_satchell_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    annualization: float = ANNUALIZATION,
) -> float:
    o = pd.to_numeric(open_, errors="coerce")
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    ho = np.log((h / o).replace([np.inf, -np.inf], np.nan))
    lo = np.log((l / o).replace([np.inf, -np.inf], np.nan))
    hc = np.log((h / c).replace([np.inf, -np.inf], np.nan))
    lc = np.log((l / c).replace([np.inf, -np.inf], np.nan))
    series = (ho * hc + lo * lc).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 2:
        return 0.0
    return float(math.sqrt(max(float(series.mean()), 0.0) * annualization))


def yang_zhang_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    annualization: float = ANNUALIZATION,
) -> float:
    """Yang-Zhang volatility estimator with finite-sample k adjustment."""

    o = pd.to_numeric(open_, errors="coerce")
    h = pd.to_numeric(high, errors="coerce")
    l = pd.to_numeric(low, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")

    prev_c = c.shift(1)
    log_oc = np.log((o / prev_c).replace([np.inf, -np.inf], np.nan))
    log_co = np.log((c / o).replace([np.inf, -np.inf], np.nan))

    rs = (
        np.log((h / c).replace([np.inf, -np.inf], np.nan))
        * np.log((h / o).replace([np.inf, -np.inf], np.nan))
        + np.log((l / c).replace([np.inf, -np.inf], np.nan))
        * np.log((l / o).replace([np.inf, -np.inf], np.nan))
    )

    frame = pd.DataFrame({"oc": log_oc, "co": log_co, "rs": rs}).replace(
        [np.inf, -np.inf], np.nan
    )
    frame = frame.dropna()
    n = len(frame)
    if n < 3:
        return 0.0

    sigma_o = float(frame["oc"].var(ddof=1))
    sigma_c = float(frame["co"].var(ddof=1))
    sigma_rs = float(frame["rs"].mean())
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var = sigma_o + k * sigma_c + (1.0 - k) * sigma_rs
    return float(math.sqrt(max(var, 0.0) * annualization))


def estimate_all_vols(df: pd.DataFrame) -> Dict[str, float]:
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return {
            "close_to_close": 0.0,
            "parkinson": 0.0,
            "garman_klass": 0.0,
            "rogers_satchell": 0.0,
            "yang_zhang": 0.0,
        }

    return {
        "close_to_close": close_to_close_vol(df["close"]),
        "parkinson": parkinson_vol(df["high"], df["low"]),
        "garman_klass": garman_klass_vol(df["open"], df["high"], df["low"], df["close"]),
        "rogers_satchell": rogers_satchell_vol(df["open"], df["high"], df["low"], df["close"]),
        "yang_zhang": yang_zhang_vol(df["open"], df["high"], df["low"], df["close"]),
    }


__all__ = [
    "close_to_close_vol",
    "parkinson_vol",
    "garman_klass_vol",
    "rogers_satchell_vol",
    "yang_zhang_vol",
    "estimate_all_vols",
]
