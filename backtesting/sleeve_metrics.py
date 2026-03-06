"""Shared performance and validation helpers for sleeve-level research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from quantum_alpha.backtesting.validation import MCPT


def annualize_periods(period: str) -> float:
    mapping = {
        "1min": 252.0 * 390.0,
        "5min": 252.0 * 78.0,
        "30min": 252.0 * 13.0,
        "1h": 252.0 * 6.5,
        "1d": 252.0,
    }
    return float(mapping.get(period, 252.0))


def compute_basic_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    *,
    periods_per_year: float = 252.0,
) -> Dict[str, float]:
    ret = pd.Series(returns, copy=True).fillna(0.0)
    equity = (1.0 + ret).cumprod()
    total_return = float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0

    if len(ret) > 0:
        annual_return = float((1.0 + total_return) ** (periods_per_year / max(len(ret), 1)) - 1.0)
        vol = float(ret.std(ddof=0))
        sharpe = float((ret.mean() / vol) * np.sqrt(periods_per_year)) if vol > 0 else 0.0
    else:
        annual_return = 0.0
        sharpe = 0.0

    peak = equity.cummax() if not equity.empty else pd.Series(dtype=float)
    drawdown = (equity / peak - 1.0) if not equity.empty else pd.Series(dtype=float)
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    beta = 0.0
    if benchmark_returns is not None:
        bench = pd.Series(benchmark_returns, copy=True).reindex(ret.index).fillna(0.0)
        if len(ret) > 1 and float(bench.var(ddof=0)) > 0:
            cov = float(np.cov(ret.values, bench.values, ddof=0)[0, 1])
            beta = cov / float(bench.var(ddof=0))

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "beta": float(beta),
    }


def rolling_beat_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    window: int = 63,
) -> float:
    strat = pd.Series(strategy_returns, copy=True).fillna(0.0)
    bench = pd.Series(benchmark_returns, copy=True).reindex(strat.index).fillna(0.0)
    if len(strat) < max(3, window):
        return float((strat.sum() > bench.sum()))
    strat_roll = strat.rolling(window, min_periods=max(3, window // 3)).sum()
    bench_roll = bench.rolling(window, min_periods=max(3, window // 3)).sum()
    valid = strat_roll.notna() & bench_roll.notna()
    if not valid.any():
        return 0.0
    return float((strat_roll[valid] > bench_roll[valid]).mean())


def run_mcpt_on_returns(
    returns: pd.Series,
    *,
    n_permutations: int = 200,
    method: str = "sign_flip",
) -> Dict[str, float | bool]:
    clean = pd.Series(returns, copy=True).fillna(0.0)
    if len(clean) < 10:
        return {"p_value": 1.0, "is_significant": False, "n_permutations": 0}
    mcpt = MCPT(n_permutations=n_permutations, test_statistic="sharpe")
    out = mcpt.run_on_returns(clean.to_numpy(dtype=float), show_progress=False, block_size=5, method=method)
    return {
        "p_value": float(out.get("p_value", 1.0)),
        "is_significant": bool(out.get("is_significant", False)),
        "n_permutations": int(out.get("n_permutations", 0)),
    }


def summarize_quality_gates(metrics: Dict[str, float], quality: Dict[str, float]) -> Dict[str, bool]:
    return {
        "depth_ok": float(quality.get("depth_completeness", 0.0)) >= 0.95,
        "staleness_ok": float(quality.get("median_quote_staleness_ms", 1e9)) <= 2000.0,
        "crossed_ok": float(quality.get("crossed_market_rate", 1.0)) <= 0.001,
        "negative_ok": float(quality.get("negative_spread_rate", 1.0)) <= 0.001,
        "drawdown_ok": abs(float(metrics.get("max_drawdown", 1.0))) <= 0.20,
    }


__all__ = [
    "annualize_periods",
    "compute_basic_metrics",
    "rolling_beat_ratio",
    "run_mcpt_on_returns",
    "summarize_quality_gates",
]
