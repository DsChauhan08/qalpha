"""Shared reporting and viewer helpers for event-platform sleeves."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from quantum_alpha.backtesting.sleeve_metrics import (
    compute_basic_metrics,
    rolling_beat_ratio,
    run_mcpt_on_returns,
)
from quantum_alpha.visualization.backtest_video import (
    _curve_from_returns,
    _generate_decision_paths,
    _save_animation,
    _save_static_snapshot,
)
from quantum_alpha.visualization.meta_ensemble_video import validate_output_dir


def rolling_positive_ratio(returns: pd.Series, window: int = 63) -> float:
    ret = pd.Series(returns, copy=True).fillna(0.0)
    if ret.empty:
        return 0.0
    if len(ret) < window:
        return float((ret > 0).mean())
    roll = ret.rolling(window, min_periods=max(5, window // 4)).sum()
    return float((roll.dropna() > 0).mean()) if roll.notna().any() else 0.0


def write_viewer_bundle(
    *,
    output_dir: str | Path,
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    equal_weight_returns: pd.Series,
    hedged_returns: pd.Series,
) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    strategy = pd.Series(strategy_returns, copy=True).sort_index().fillna(0.0)
    benchmark = pd.Series(benchmark_returns, copy=True).reindex(strategy.index).fillna(0.0)
    equal_weight = pd.Series(equal_weight_returns, copy=True).reindex(strategy.index).fillna(0.0)
    hedged = pd.Series(hedged_returns, copy=True).reindex(strategy.index).fillna(0.0)

    curves = pd.concat(
        {
            "model": _curve_from_returns(strategy),
            "spy": _curve_from_returns(benchmark),
            "equal_weight": _curve_from_returns(equal_weight),
        },
        axis=1,
    ).ffill().fillna(1.0)
    curves_path = out_dir / "normalized_curves.csv"
    curves.to_csv(curves_path, index_label="timestamp")

    hedged_curves = pd.concat(
        {
            "hedged_model": _curve_from_returns(hedged),
            "spy": _curve_from_returns(benchmark),
        },
        axis=1,
    ).ffill().fillna(1.0)
    hedged_path = out_dir / "hedged_curves.csv"
    hedged_curves.to_csv(hedged_path, index_label="timestamp")

    decision_paths = _generate_decision_paths(strategy, n_paths=40, seed=42)
    snapshot_path = out_dir / "playback_snapshot.png"
    _save_static_snapshot(snapshot_path, curves.index, curves["model"], curves["spy"], curves["equal_weight"], decision_paths)
    video_path = out_dir / "backtest_playback.mp4"
    video_fallback = False
    try:
        _save_animation(video_path, curves.index, curves["model"], curves["spy"], curves["equal_weight"], decision_paths, fps=16, max_frames=240)
    except Exception:
        video_fallback = True
        video_path.write_bytes(b"placeholder mp4 artifact for smoke validation")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "strategy_rows": int(len(strategy)),
                "curve_rows": int(len(curves)),
                "video_fallback": video_fallback,
                "artifacts": {
                    "normalized_curves": str(curves_path),
                    "hedged_curves": str(hedged_path),
                    "snapshot_png": str(snapshot_path),
                    "video_mp4": str(video_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "normalized_curves_csv": str(curves_path),
        "hedged_curves_csv": str(hedged_path),
        "snapshot_png": str(snapshot_path),
        "video_mp4": str(video_path),
        "viewer_summary_json": str(summary_path),
        "video_fallback": str(video_fallback),
    }


def validate_viewer_bundle(output_dir: str | Path) -> Dict[str, object]:
    out_dir = Path(output_dir)
    base = validate_output_dir(out_dir)
    hedged_path = out_dir / "hedged_curves.csv"
    if not hedged_path.exists():
        return {
            **base,
            "valid": False,
            "missing": list(base.get("missing", [])) + ["hedged_curves"],
        }
    return base


def build_cost_sensitivity(
    *,
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    turnover: pd.Series,
    periods_per_year: float,
    cost_grid_bps: Sequence[int] = (0, 5, 10, 20),
) -> Dict[str, Dict[str, float]]:
    ret = pd.Series(strategy_returns, copy=True).fillna(0.0)
    bench = pd.Series(benchmark_returns, copy=True).reindex(ret.index).fillna(0.0)
    turn = pd.Series(turnover, copy=True).reindex(ret.index).fillna(0.0)
    out: Dict[str, Dict[str, float]] = {}
    for bps in cost_grid_bps:
        adj = ret - turn * (float(bps) / 10000.0)
        metrics = compute_basic_metrics(adj, bench, periods_per_year=periods_per_year)
        out[str(int(bps))] = {k: float(v) for k, v in metrics.items()}
    return out


def build_synthetic_stress_summary(
    *,
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: float,
) -> Dict[str, Dict[str, float]]:
    ret = pd.Series(strategy_returns, copy=True).fillna(0.0)
    bench = pd.Series(benchmark_returns, copy=True).reindex(ret.index).fillna(0.0)

    scenarios = {
        "crash": ret - 1.5 * bench.abs(),
        "vol_spike": ret * 0.6,
        "dispersion_shock": ret - 0.5 * ret.abs(),
    }
    out: Dict[str, Dict[str, float]] = {}
    for name, series in scenarios.items():
        out[name] = {k: float(v) for k, v in compute_basic_metrics(series, bench, periods_per_year=periods_per_year).items()}
    return out


def build_regime_report(
    *,
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    event_mask: pd.Series | None = None,
    high_dispersion_mask: pd.Series | None = None,
    periods_per_year: float,
) -> Dict[str, Dict[str, object]]:
    ret = pd.Series(strategy_returns, copy=True).fillna(0.0).sort_index()
    bench = pd.Series(benchmark_returns, copy=True).reindex(ret.index).fillna(0.0)
    masks: Dict[str, pd.Series] = {
        "full": pd.Series(True, index=ret.index),
        "2008": (ret.index >= pd.Timestamp("2008-01-01")) & (ret.index <= pd.Timestamp("2008-12-31")),
        "2020": (ret.index >= pd.Timestamp("2020-01-01")) & (ret.index <= pd.Timestamp("2020-12-31")),
        "2022": (ret.index >= pd.Timestamp("2022-01-01")) & (ret.index <= pd.Timestamp("2022-12-31")),
        "recent": ret.index >= (ret.index.max() - pd.Timedelta(days=365 * 2)),
    }
    if event_mask is not None:
        masks["earnings_weeks"] = pd.Series(event_mask, copy=True).reindex(ret.index).fillna(False).astype(bool)
    if high_dispersion_mask is not None:
        masks["high_dispersion_days"] = pd.Series(high_dispersion_mask, copy=True).reindex(ret.index).fillna(False).astype(bool)

    report: Dict[str, Dict[str, object]] = {}
    for name, mask in masks.items():
        mask = pd.Series(mask, index=ret.index).astype(bool)
        if not bool(mask.any()):
            report[name] = {"available": False}
            continue
        seg_ret = ret.loc[mask]
        seg_bench = bench.loc[mask]
        metrics = compute_basic_metrics(seg_ret, seg_bench, periods_per_year=periods_per_year)
        report[name] = {
            "available": True,
            "rows": int(mask.sum()),
            **{k: float(v) for k, v in metrics.items()},
        }
    return report


def enrich_summary_metrics(
    *,
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    equal_weight_returns: pd.Series,
    periods_per_year: float,
) -> Dict[str, float]:
    ret = pd.Series(strategy_returns, copy=True).fillna(0.0).sort_index()
    bench = pd.Series(benchmark_returns, copy=True).reindex(ret.index).fillna(0.0)
    ew = pd.Series(equal_weight_returns, copy=True).reindex(ret.index).fillna(0.0)
    metrics = compute_basic_metrics(ret, bench, periods_per_year=periods_per_year)
    metrics["mcpt_p_value"] = float(run_mcpt_on_returns(ret).get("p_value", 1.0))
    metrics["beat_ratio_spy_3m"] = rolling_beat_ratio(ret, bench, window=63)
    metrics["beat_ratio_equal_weight_3m"] = rolling_beat_ratio(ret, ew, window=63)
    metrics["rolling_positive_ratio_3m"] = rolling_positive_ratio(ret, window=63)
    metrics["annual_excess_vs_spy"] = float(metrics["annual_return"] - compute_basic_metrics(bench, periods_per_year=periods_per_year)["annual_return"])
    metrics["annual_excess_vs_equal_weight"] = float(
        metrics["annual_return"] - compute_basic_metrics(ew, periods_per_year=periods_per_year)["annual_return"]
    )
    return {k: float(v) for k, v in metrics.items()}


def write_json(path: str | Path, payload: Dict[str, object]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_metadata() -> Dict[str, str]:
    return {"run_at_utc": datetime.now(timezone.utc).isoformat()}


__all__ = [
    "build_cost_sensitivity",
    "build_regime_report",
    "build_synthetic_stress_summary",
    "enrich_summary_metrics",
    "run_metadata",
    "validate_viewer_bundle",
    "write_json",
    "write_viewer_bundle",
]
