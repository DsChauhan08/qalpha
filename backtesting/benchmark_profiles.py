"""
Quant-firm proxy benchmark profiles and comparison utilities.

These profiles are ETF-based proxies for broad quant styles, not direct
representations of any firm's proprietary portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from quantum_alpha.backtesting.performance_metrics import compute_metrics_from_returns
from quantum_alpha.data.collectors.market_data import DataCollector


@dataclass(frozen=True)
class BenchmarkProfile:
    key: str
    label: str
    description: str
    tickers: Tuple[str, ...]
    weights: Optional[Tuple[float, ...]] = None


DEFAULT_PROFILES: Dict[str, BenchmarkProfile] = {
    "aqr_style_factors": BenchmarkProfile(
        key="aqr_style_factors",
        label="AQR-Style Factors",
        description=(
            "US multi-factor proxy basket (momentum, quality, value, low-vol, size)."
        ),
        tickers=("MTUM", "QUAL", "VLUE", "USMV", "IJR"),
    ),
    "citadel_style_multistrat": BenchmarkProfile(
        key="citadel_style_multistrat",
        label="Citadel-Style Multi-Strategy",
        description="Cross-asset liquid proxy with equities, duration, trend, and gold.",
        tickers=("SPY", "QQQ", "TLT", "DBMF", "GLD"),
    ),
    "two_sigma_style_data": BenchmarkProfile(
        key="two_sigma_style_data",
        label="Two Sigma-Style Data/Tech",
        description=(
            "US data/tech-heavy liquid proxy with broad beta and growth sectors."
        ),
        tickers=("QQQ", "SOXX", "IGV", "XBI", "IWM"),
    ),
    "de_shaw_style_macro": BenchmarkProfile(
        key="de_shaw_style_macro",
        label="D.E. Shaw-Style Macro",
        description=(
            "Macro-tilted liquid proxy with equities, rates, commodities, and gold."
        ),
        tickers=("SPY", "IEF", "TLT", "DBC", "GLD"),
    ),
    "man_ahl_style_trend": BenchmarkProfile(
        key="man_ahl_style_trend",
        label="Man AHL-Style Trend",
        description=(
            "Managed-futures trend proxy basket with CTA/alt trend plus diversifiers."
        ),
        tickers=("DBMF", "KMLM", "CTA", "GLD", "TLT"),
    ),
}


def available_profiles() -> Dict[str, BenchmarkProfile]:
    return dict(DEFAULT_PROFILES)


def resolve_profiles(profile_names: Optional[Iterable[str]]) -> List[BenchmarkProfile]:
    if not profile_names:
        return list(DEFAULT_PROFILES.values())

    resolved: List[BenchmarkProfile] = []
    unknown: List[str] = []
    for name in profile_names:
        key = str(name).strip()
        if not key:
            continue
        profile = DEFAULT_PROFILES.get(key)
        if profile is None:
            unknown.append(key)
            continue
        resolved.append(profile)

    if unknown:
        valid = ", ".join(sorted(DEFAULT_PROFILES.keys()))
        missing = ", ".join(sorted(set(unknown)))
        raise ValueError(
            f"Unknown quant benchmark profile(s): {missing}. Valid options: {valid}"
        )
    return resolved


def _to_returns(df: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(df.get("close"), errors="coerce")
    if close is None:
        return pd.Series(dtype=float)
    returns = close.pct_change(fill_method=None).fillna(0.0)
    idx = pd.to_datetime(df.index)
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        idx = idx.tz_localize(None)
    returns.index = idx
    return returns.sort_index()


def _normalised_weights(
    tickers: Sequence[str],
    available: Sequence[str],
    weights: Optional[Sequence[float]],
) -> np.ndarray:
    if not available:
        return np.array([], dtype=float)

    if weights is not None and len(weights) == len(tickers):
        base = np.array(weights, dtype=float)
    else:
        base = np.ones(len(tickers), dtype=float)

    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
    if base.sum() <= 0:
        base = np.ones(len(tickers), dtype=float)
    base = base / base.sum()

    ticker_to_w = {t: float(w) for t, w in zip(tickers, base)}
    active = np.array([ticker_to_w[t] for t in available], dtype=float)
    if active.sum() <= 0:
        active = np.ones(len(available), dtype=float)
    return active / active.sum()


def build_profile_returns(
    collector: DataCollector,
    profile: BenchmarkProfile,
    start_date: datetime,
    end_date: datetime,
    interval: str = "1d",
) -> Tuple[pd.Series, List[str], List[str]]:
    series_map: Dict[str, pd.Series] = {}
    failed: List[str] = []

    for ticker in profile.tickers:
        try:
            df = collector.fetch_ohlcv(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
            )
        except Exception:
            failed.append(ticker)
            continue
        if df is None or df.empty or "close" not in df.columns:
            failed.append(ticker)
            continue
        ret = _to_returns(df)
        if ret.empty:
            failed.append(ticker)
            continue
        series_map[ticker] = ret

    if not series_map:
        return pd.Series(dtype=float), [], failed

    available = list(series_map.keys())
    aligned = pd.concat([series_map[t] for t in available], axis=1)
    aligned.columns = available
    aligned = aligned.sort_index().ffill().bfill().fillna(0.0)

    weights = _normalised_weights(profile.tickers, available, profile.weights)
    weighted = aligned.mul(weights, axis=1).sum(axis=1)
    weighted.name = profile.key
    return weighted, available, failed


def _total_return(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    eq = (1.0 + returns).cumprod()
    if len(eq) <= 1:
        return 0.0
    return float(eq.iloc[-1] / eq.iloc[0] - 1.0)


def evaluate_quant_firm_benchmarks(
    strategy_returns: pd.Series,
    collector: DataCollector,
    start_date: datetime,
    end_date: datetime,
    profile_names: Optional[Iterable[str]] = None,
    interval: str = "1d",
    min_assets: int = 2,
) -> Dict[str, Dict[str, object]]:
    strategy = (
        pd.Series(strategy_returns)
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_index()
    )
    if strategy.empty:
        return {}

    out: Dict[str, Dict[str, object]] = {}
    for profile in resolve_profiles(profile_names):
        profile_returns, used, failed = build_profile_returns(
            collector=collector,
            profile=profile,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
        )
        if len(used) < max(1, int(min_assets)):
            out[profile.key] = {
                "status": "insufficient_assets",
                "label": profile.label,
                "description": profile.description,
                "used_tickers": used,
                "failed_tickers": failed,
            }
            continue

        strat_aligned, bench_aligned = strategy.align(profile_returns, join="inner")
        if strat_aligned.empty:
            out[profile.key] = {
                "status": "no_overlap",
                "label": profile.label,
                "description": profile.description,
                "used_tickers": used,
                "failed_tickers": failed,
            }
            continue

        strategy_vs_profile = compute_metrics_from_returns(
            strat_aligned,
            benchmark_returns=bench_aligned,
        )
        profile_metrics = compute_metrics_from_returns(
            bench_aligned,
            benchmark_returns=bench_aligned,
        )

        strat_total = _total_return(strat_aligned)
        profile_total = _total_return(bench_aligned)
        excess_total = float(strat_total - profile_total)
        hit_rate = float((strat_aligned > bench_aligned).mean())

        down_mask = bench_aligned < 0
        if down_mask.any():
            downside_outperf = float(
                (strat_aligned[down_mask] > bench_aligned[down_mask]).mean()
            )
        else:
            downside_outperf = 0.0

        out[profile.key] = {
            "status": "ok",
            "label": profile.label,
            "description": profile.description,
            "used_tickers": used,
            "failed_tickers": failed,
            "n_days": int(len(strat_aligned)),
            "strategy_total_return": strat_total,
            "profile_total_return": profile_total,
            "excess_total_return": excess_total,
            "hit_rate_vs_profile": hit_rate,
            "downside_outperformance_rate": downside_outperf,
            "strategy_vs_profile_metrics": strategy_vs_profile,
            "profile_metrics": profile_metrics,
        }

    return out


def benchmark_rows(results: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for key, blob in results.items():
        if blob.get("status") != "ok":
            continue
        metrics = blob.get("strategy_vs_profile_metrics", {}) or {}
        rows.append(
            {
                "profile_key": key,
                "profile_label": blob.get("label"),
                "n_days": int(blob.get("n_days", 0)),
                "excess_total_return": float(blob.get("excess_total_return", 0.0)),
                "information_ratio": float(metrics.get("information_ratio", 0.0)),
                "downside_capture": float(metrics.get("downside_capture", 0.0)),
                "hit_rate_vs_profile": float(blob.get("hit_rate_vs_profile", 0.0)),
            }
        )
    rows.sort(key=lambda x: x["excess_total_return"], reverse=True)
    return rows

