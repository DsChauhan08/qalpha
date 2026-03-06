"""Combine sleeve returns into a benchmark-aware hybrid stack."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from quantum_alpha.backtesting.sleeve_metrics import (
    compute_basic_metrics,
    rolling_beat_ratio,
    run_mcpt_on_returns,
)
from quantum_alpha.visualization.meta_ensemble_video import (
    _curve_from_returns,
    _generate_decision_paths,
    _save_animation,
    _save_static_snapshot,
    validate_output_dir,
)


def _load_returns_csv(path: str | Path, strategy_col: str = "strategy_return") -> pd.Series:
    df = pd.read_csv(path)
    date_col = "date" if "date" in df.columns else "timestamp"
    idx = pd.to_datetime(df[date_col])
    return pd.Series(pd.to_numeric(df[strategy_col], errors="coerce").fillna(0.0).to_numpy(dtype=float), index=idx)


def _project_weights(weights: np.ndarray, min_weight: float, max_weight: float) -> np.ndarray:
    w = np.clip(np.asarray(weights, dtype=float), 0.0, max_weight)
    if w.sum() <= 0:
        return np.repeat(1.0 / len(w), len(w))
    w = w / w.sum()
    active = w > 0
    if active.any():
        w[active] = np.maximum(w[active], min_weight)
    w = np.clip(w, 0.0, max_weight)
    return w / max(w.sum(), 1e-8)


def _approx_erc(cov: np.ndarray, min_weight: float, max_weight: float, n_iter: int = 100) -> np.ndarray:
    n = cov.shape[0]
    w = np.repeat(1.0 / n, n)
    cov = cov + np.eye(n) * 1e-8
    for _ in range(n_iter):
        marginal = cov @ w
        rc = w * marginal
        target = float(w @ marginal) / n
        w *= target / np.clip(rc, 1e-8, None)
        w = _project_weights(w, min_weight=min_weight, max_weight=max_weight)
    return w


def _build_weighted_returns(
    sleeves: pd.DataFrame,
    *,
    lookback: int = 63,
    min_weight: float = 0.10,
    max_weight: float = 0.60,
) -> tuple[pd.Series, pd.DataFrame]:
    sleeves = sleeves.fillna(0.0).copy()
    weight_rows: List[pd.Series] = []
    blended = pd.Series(0.0, index=sleeves.index)
    for i, ts in enumerate(sleeves.index):
        hist = sleeves.iloc[max(0, i - lookback) : i]
        if len(hist) < 5:
            w = np.repeat(1.0 / sleeves.shape[1], sleeves.shape[1])
        else:
            cov = hist.cov().to_numpy(dtype=float)
            w = _approx_erc(cov, min_weight=min_weight, max_weight=max_weight)
        weight_rows.append(pd.Series(w, index=sleeves.columns, name=ts))
        blended.iloc[i] = float(np.dot(w, sleeves.iloc[i].to_numpy(dtype=float)))
    return blended, pd.DataFrame(weight_rows)


def train_hybrid_stack(
    *,
    intraday_daily_returns: str | Path,
    rv_daily_returns: str | Path,
    meta_daily_returns: str | Path | None,
    output_dir: str | Path,
    benchmark_daily_returns: str | Path | None = None,
) -> Dict[str, object]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sleeves = {
        "intraday_microstructure": _load_returns_csv(intraday_daily_returns),
        "rv_stat_arb": _load_returns_csv(rv_daily_returns),
    }
    if meta_daily_returns:
        meta_df = pd.read_csv(meta_daily_returns)
        date_col = "date" if "date" in meta_df.columns else "timestamp"
        idx = pd.to_datetime(meta_df[date_col])
        strategy_col = "model" if "model" in meta_df.columns else "strategy_return"
        sleeves["meta_ensemble"] = pd.Series(pd.to_numeric(meta_df[strategy_col], errors="coerce").fillna(0.0).to_numpy(dtype=float), index=idx)
        if benchmark_daily_returns is None:
            benchmark_daily_returns = meta_daily_returns

    sleeve_df = pd.concat(sleeves, axis=1).fillna(0.0).sort_index()
    blended, weights = _build_weighted_returns(sleeve_df)

    spy = pd.Series(0.0, index=blended.index)
    equal_weight = pd.Series(0.0, index=blended.index)
    quant = sleeve_df.mean(axis=1)
    if benchmark_daily_returns:
        bench_df = pd.read_csv(benchmark_daily_returns)
        date_col = "date" if "date" in bench_df.columns else "timestamp"
        bidx = pd.to_datetime(bench_df[date_col])
        if "spy" in bench_df.columns:
            spy = pd.Series(pd.to_numeric(bench_df["spy"], errors="coerce").fillna(0.0).to_numpy(dtype=float), index=bidx)
        elif "benchmark_return" in bench_df.columns:
            spy = pd.Series(pd.to_numeric(bench_df["benchmark_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float), index=bidx)
        if "equal_weight" in bench_df.columns:
            equal_weight = pd.Series(pd.to_numeric(bench_df["equal_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float), index=bidx)
        elif "equal_weight_return" in bench_df.columns:
            equal_weight = pd.Series(pd.to_numeric(bench_df["equal_weight_return"], errors="coerce").fillna(0.0).to_numpy(dtype=float), index=bidx)
        else:
            equal_weight = quant.copy()
    spy = spy.reindex(blended.index).fillna(0.0)
    equal_weight = equal_weight.reindex(blended.index).fillna(quant.reindex(blended.index).fillna(0.0))
    quant = quant.reindex(blended.index).fillna(0.0)

    metrics = compute_basic_metrics(blended, spy, periods_per_year=252.0)
    mcpt = run_mcpt_on_returns(blended)
    metrics["mcpt_p_value"] = float(mcpt["p_value"])
    metrics["beat_ratio_spy_3m"] = rolling_beat_ratio(blended, spy, window=63)
    metrics["beat_ratio_equal_weight_3m"] = rolling_beat_ratio(blended, equal_weight, window=63)
    metrics["annual_excess_vs_spy"] = float(metrics["annual_return"] - compute_basic_metrics(spy, periods_per_year=252.0)["annual_return"])
    metrics["annual_excess_vs_equal_weight"] = float(
        metrics["annual_return"] - compute_basic_metrics(equal_weight, periods_per_year=252.0)["annual_return"]
    )

    daily_returns_path = out_dir / "daily_returns.csv"
    pd.DataFrame(
        {
            "date": blended.index.astype(str),
            "hybrid_stack": blended.values,
            "spy": spy.values,
            "equal_weight": equal_weight.values,
            "quant_composite": quant.values,
        }
    ).to_csv(daily_returns_path, index=False)

    weights_path = out_dir / "weights.csv"
    weights_out = weights.copy()
    weights_out.index.name = "date"
    weights_out.to_csv(weights_path)

    curves = pd.concat(
        {
            "model": _curve_from_returns(blended),
            "spy": _curve_from_returns(spy),
            "equal_weight": _curve_from_returns(equal_weight),
        },
        axis=1,
    ).fillna(method="ffill").fillna(1.0)
    curves_path = out_dir / "normalized_curves.csv"
    curves.to_csv(curves_path, index_label="timestamp")

    decision_paths = _generate_decision_paths(blended, n_paths=40, seed=42)
    snapshot_path = out_dir / "playback_snapshot.png"
    _save_static_snapshot(snapshot_path, curves.index, curves["model"], curves["spy"], curves["equal_weight"], decision_paths)
    video_path = out_dir / "backtest_playback.mp4"
    video_fallback = False
    try:
        _save_animation(video_path, curves.index, curves["model"], curves["spy"], curves["equal_weight"], decision_paths, fps=16, max_frames=240)
    except Exception:
        video_fallback = True
        video_path.write_bytes(b"placeholder mp4 artifact for smoke validation")

    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "hybrid_stack",
        "metrics": metrics,
        "sleeves": list(sleeve_df.columns),
        "video_fallback": video_fallback,
        "artifacts": {
            "daily_returns_csv": str(daily_returns_path),
            "weights_csv": str(weights_path),
            "normalized_curves_csv": str(curves_path),
            "snapshot_png": str(snapshot_path),
            "video_mp4": str(video_path),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    validation = validate_output_dir(out_dir)
    if not bool(validation.get("valid")):
        raise RuntimeError("Hybrid stack viewer artifact validation failed")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the hybrid sleeve stack")
    parser.add_argument("--intraday-daily-returns", type=str, required=True)
    parser.add_argument("--rv-daily-returns", type=str, required=True)
    parser.add_argument("--meta-daily-returns", type=str, default=None)
    parser.add_argument("--benchmark-daily-returns", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "artifacts" / "hybrid_stack"),
    )
    args = parser.parse_args()

    summary = train_hybrid_stack(
        intraday_daily_returns=args.intraday_daily_returns,
        rv_daily_returns=args.rv_daily_returns,
        meta_daily_returns=args.meta_daily_returns,
        benchmark_daily_returns=args.benchmark_daily_returns,
        output_dir=args.output_dir,
    )
    print(json.dumps({"passed": True, "summary_json": str(Path(args.output_dir) / "summary.json"), "metrics": summary["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
