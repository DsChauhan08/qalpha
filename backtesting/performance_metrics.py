"""
Extended performance metrics for strategy evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DrawdownStats:
    max_drawdown: float
    ulcer_index: float
    recovery_time: int


def _annualize_return(
    total_return: float, n_periods: int, periods_per_year: int = 252
) -> float:
    if n_periods <= 0:
        return 0.0
    years = n_periods / periods_per_year
    return (1 + total_return) ** (1 / max(years, 1e-6)) - 1


def _drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (equity - peak) / peak


def _drawdown_stats(equity: pd.Series) -> DrawdownStats:
    dd = _drawdown_series(equity)
    max_dd = float(dd.min()) if not dd.empty else 0.0
    ulcer = float(np.sqrt(np.mean(np.square(dd.values)))) if len(dd) else 0.0

    recovery_time = 0
    if len(equity) > 1:
        peak_idx = equity.idxmax()
        trough_idx = dd.idxmin() if not dd.empty else peak_idx
        if trough_idx >= peak_idx:
            recovery_slice = equity.loc[trough_idx:]
            recovered = recovery_slice[recovery_slice >= equity.loc[peak_idx]]
            if not recovered.empty:
                recovery_time = int((recovered.index[0] - trough_idx).days)
            else:
                recovery_time = int((equity.index[-1] - trough_idx).days)
    return DrawdownStats(
        max_drawdown=max_dd, ulcer_index=ulcer, recovery_time=recovery_time
    )


def _max_consecutive_runs(series: pd.Series) -> Tuple[int, int]:
    # Filter to only days with actual non-zero returns (i.e. trading activity)
    active = series[series != 0]
    if active.empty:
        return 0, 0
    wins = (active > 0).astype(int)
    losses = (active < 0).astype(int)

    def max_run(arr: np.ndarray) -> int:
        max_len = 0
        current = 0
        for val in arr:
            if val == 1:
                current += 1
                max_len = max(max_len, current)
            else:
                current = 0
        return int(max_len)

    return max_run(wins.values), max_run(losses.values)


def _omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    if returns.empty:
        return 0.0
    gains = (returns[returns > threshold] - threshold).sum()
    losses = (threshold - returns[returns < threshold]).sum()
    return float(gains / losses) if losses > 0 else float("inf")


def _capture_ratio(
    returns: pd.Series, benchmark: pd.Series
) -> Tuple[float, float, float]:
    aligned = returns.align(benchmark, join="inner")
    if aligned[0].empty:
        return 0.0, 0.0, 0.0

    strat, bench = aligned
    upside = bench > 0
    downside = bench < 0

    upside_capture = (
        strat[upside].mean() / bench[upside].mean() if upside.any() else 0.0
    )
    downside_capture = (
        strat[downside].mean() / bench[downside].mean() if downside.any() else 0.0
    )
    capture = (
        upside_capture / downside_capture
        if downside_capture not in (0, np.nan)
        else 0.0
    )

    return float(upside_capture), float(downside_capture), float(capture)


def compute_metrics(
    equity_curve: List[Dict],
    trades: Optional[List[Dict]] = None,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free: float = 0.02,
) -> Dict[str, float]:
    if not equity_curve:
        return {}

    equity = pd.DataFrame(equity_curve).set_index("timestamp")["equity"].dropna()
    returns = equity.pct_change().dropna()
    if returns.empty:
        return {}

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    annual_return = _annualize_return(total_return, len(returns))
    annual_vol = returns.std() * np.sqrt(252)
    downside = returns[returns < 0]
    downside_dev = downside.std() * np.sqrt(252) if len(downside) else 0.0

    sharpe = (annual_return - risk_free) / annual_vol if annual_vol > 0 else 0.0
    sortino = (annual_return - risk_free) / downside_dev if downside_dev > 0 else 0.0

    dd_stats = _drawdown_stats(equity)
    calmar = (
        annual_return / abs(dd_stats.max_drawdown)
        if dd_stats.max_drawdown != 0
        else 0.0
    )

    var_level = 0.95
    var = float(np.percentile(returns, (1 - var_level) * 100))
    cvar = (
        float(returns[returns <= var].mean()) if len(returns[returns <= var]) else var
    )

    omega = _omega_ratio(returns)

    win_rate = float((returns > 0).mean())
    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0.0
    avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0.0
    payoff_ratio = float(avg_win / abs(avg_loss)) if avg_loss != 0 else float("inf")
    profit_factor = (
        float(returns[returns > 0].sum() / abs(returns[returns < 0].sum()))
        if (returns < 0).any()
        else float("inf")
    )
    max_consec_wins, max_consec_losses = _max_consecutive_runs(returns)

    metrics = {
        "total_return": float(total_return),
        "cagr": float(annual_return),
        "annual_volatility": float(annual_vol),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": float(dd_stats.max_drawdown),
        "calmar_ratio": float(calmar),
        "ulcer_index": float(dd_stats.ulcer_index),
        "downside_deviation": float(downside_dev),
        "var": float(abs(var)),
        "cvar": float(abs(cvar)),
        "omega_ratio": float(omega),
        "recovery_time": float(dd_stats.recovery_time),
        "win_rate": float(win_rate),
        "payoff_ratio": float(payoff_ratio),
        "profit_factor": float(profit_factor),
        "max_consecutive_wins": float(max_consec_wins),
        "max_consecutive_losses": float(max_consec_losses),
    }

    if trades:
        trade_pnls = np.array([t.get("pnl", 0.0) for t in trades], dtype=float)
        if trade_pnls.size:
            metrics["trade_win_rate"] = float((trade_pnls > 0).mean())
            trade_wins = trade_pnls[trade_pnls > 0]
            trade_losses = trade_pnls[trade_pnls < 0]
            metrics["trade_payoff_ratio"] = (
                float(trade_wins.mean() / abs(trade_losses.mean()))
                if trade_losses.size
                else float("inf")
            )

    if benchmark_returns is not None and not benchmark_returns.empty:
        aligned = returns.align(benchmark_returns, join="inner")
        if not aligned[0].empty:
            strat, bench = aligned
            cov = np.cov(strat.values, bench.values)[0, 1]
            var_bench = np.var(bench.values)
            beta = cov / var_bench if var_bench > 0 else 0.0
            alpha = annual_return - (
                risk_free + beta * (bench.mean() * 252 - risk_free)
            )
            tracking_error = np.std((strat - bench).values) * np.sqrt(252)
            # Use geometric benchmark return for consistency with strategy's
            # geometric CAGR (annual_return).  Arithmetic mean * 252
            # systematically over-estimates the benchmark and penalises active
            # strategies.
            bench_equity = (1 + bench).cumprod()
            bench_total = (
                float(bench_equity.iloc[-1] / bench_equity.iloc[0] - 1)
                if len(bench_equity) > 1
                else 0.0
            )
            bench_ann = _annualize_return(bench_total, len(bench))
            info_ratio = (
                (annual_return - bench_ann) / tracking_error
                if tracking_error > 1e-10
                else 0.0
            )
            r2 = (
                np.corrcoef(strat.values, bench.values)[0, 1] ** 2
                if len(strat) > 1
                else 0.0
            )
            treynor = (annual_return - risk_free) / beta if beta != 0 else 0.0

            upside_cap, downside_cap, capture = _capture_ratio(
                returns, benchmark_returns
            )

            m2 = risk_free + sharpe * (bench.std() * np.sqrt(252))
            sterling = annual_return / (abs(dd_stats.max_drawdown) + 1e-6)

            metrics.update(
                {
                    "beta": float(beta),
                    "jensen_alpha": float(alpha),
                    "information_ratio": float(info_ratio),
                    "tracking_error": float(tracking_error),
                    "r_squared": float(r2),
                    "treynor_ratio": float(treynor),
                    "upside_capture": float(upside_cap),
                    "downside_capture": float(downside_cap),
                    "capture_ratio": float(capture),
                    "m2": float(m2),
                    "sterling_ratio": float(sterling),
                }
            )

    return metrics


def compute_metrics_from_returns(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free: float = 0.02,
) -> Dict[str, float]:
    if returns.empty:
        return {}

    equity = (1 + returns).cumprod()
    equity_curve = [
        {"timestamp": idx, "equity": float(val)} for idx, val in equity.items()
    ]
    return compute_metrics(
        equity_curve,
        trades=None,
        benchmark_returns=benchmark_returns,
        risk_free=risk_free,
    )
