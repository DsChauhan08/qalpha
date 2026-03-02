"""Risk metrics for allocation diagnostics and guards."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def drawdown_series(returns: pd.Series) -> pd.Series:
    r = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    equity = (1.0 + r).cumprod()
    peak = equity.cummax().replace(0.0, np.nan)
    dd = (equity / peak) - 1.0
    return dd.fillna(0.0)


def max_drawdown(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    if dd.empty:
        return 0.0
    return float(dd.min())


def historical_var(returns: pd.Series, alpha: float = 0.95) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return 0.0
    q = np.quantile(r.values, 1.0 - float(alpha))
    return float(q)


def historical_cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if r.empty:
        return 0.0
    var = historical_var(r, alpha=alpha)
    tail = r[r <= var]
    if tail.empty:
        return float(var)
    return float(tail.mean())


def ulcer_index(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    if dd.empty:
        return 0.0
    return float(np.sqrt(np.mean(np.square(dd.values))))


def sharpe_ratio(returns: pd.Series, annualization: float = 252.0) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 2:
        return 0.0
    vol = float(r.std(ddof=1))
    if vol <= 1e-12:
        return 0.0
    return float((r.mean() / vol) * np.sqrt(annualization))


def risk_snapshot(returns: pd.Series, alpha: float = 0.95) -> Dict[str, float]:
    return {
        "var": historical_var(returns, alpha=alpha),
        "cvar": historical_cvar(returns, alpha=alpha),
        "drawdown": max_drawdown(returns),
        "ulcer_index": ulcer_index(returns),
        "sharpe": sharpe_ratio(returns),
    }


__all__ = [
    "drawdown_series",
    "max_drawdown",
    "historical_var",
    "historical_cvar",
    "ulcer_index",
    "sharpe_ratio",
    "risk_snapshot",
]
