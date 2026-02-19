"""
Meta-ensemble playback video from walk-forward out-of-sample predictions.

Produces:
- summary JSON
- normalized curve CSV
- static PNG snapshot
- MP4 playback with SPY, model, equal-weight, and gray decision paths
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantum_alpha.backtest_clean import (
    backtest_equal_weight,
    blend_prediction_probabilities,
    compute_signals,
    deduplicate_predictions,
    load_predictions,
)
from quantum_alpha.data.collectors.market_data import DataCollector


def _to_datetime_index(values) -> pd.DatetimeIndex:
    idx = pd.to_datetime(values)
    out = []
    for value in idx:
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        out.append(ts.normalize())
    return pd.DatetimeIndex(out)


def _curve_from_returns(returns: pd.Series) -> pd.Series:
    ret = returns.fillna(0.0).astype(float)
    curve = (1.0 + ret).cumprod()
    if len(curve) > 0:
        curve.iloc[0] = 1.0
    return curve


def _cost_aware_equal_weight_backtest(
    pred: pd.DataFrame,
    commission_bps: float,
    hold_days: int,
) -> dict:
    """
    Equal-weight benchmark with the same commission/hold mechanics as the model
    backtest, but without model selection thresholds.
    """
    bench = pred[["date", "symbol", "forward_return", "y_true"]].copy()
    bench["raw_signal"] = 1.0
    bench["confidence"] = 1.0
    bench["y_proba"] = 1.0

    n_symbols = int(bench["symbol"].nunique())
    return backtest_equal_weight(
        bench,
        max_positions=max(1, n_symbols),
        commission_bps=commission_bps,
        hold_days=hold_days,
        top_k=None,
        initial_capital=100_000.0,
        confidence_weight=False,
        earnings_filter=False,
        pead_boost=False,
        semiconductor_short_gate=False,
        gate_threshold=0.0,
    )


def _generate_decision_paths(
    model_returns: pd.Series,
    n_paths: int = 80,
    block_size: int = 20,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = model_returns.fillna(0.0).to_numpy(dtype=float)
    n = len(base)
    if n == 0:
        return np.empty((0, 0), dtype=float)

    block = max(2, min(block_size, n))
    paths = np.empty((n_paths, n), dtype=float)

    for i in range(n_paths):
        sampled = np.empty(n, dtype=float)
        cursor = 0
        while cursor < n:
            start = int(rng.integers(0, max(1, n - block + 1)))
            segment = base[start : start + block]
            take = min(len(segment), n - cursor)
            sampled[cursor : cursor + take] = segment[:take]
            cursor += take

        sampled *= rng.uniform(0.85, 1.15, size=n)
        hold_mask = rng.random(n) < 0.08
        sampled[hold_mask] *= 0.2
        wrong_mask = rng.random(n) < 0.02
        sampled[wrong_mask] *= -0.6
        sampled = np.clip(sampled, -0.95, 2.0)

        paths[i] = np.cumprod(1.0 + sampled)
        paths[i, 0] = 1.0

    return paths


def _downsample_for_render(
    curves: pd.DataFrame, decision_paths: np.ndarray, max_points: int
) -> tuple[pd.DataFrame, np.ndarray]:
    if max_points <= 0 or len(curves) <= max_points:
        return curves, decision_paths
    idx = np.linspace(0, len(curves) - 1, num=max_points, dtype=int)
    idx = np.unique(idx)
    sampled_curves = curves.iloc[idx]
    if decision_paths.size == 0:
        return sampled_curves, decision_paths
    sampled_paths = decision_paths[:, idx]
    return sampled_curves, sampled_paths


def _save_static_snapshot(
    out_path: Path,
    dates: pd.DatetimeIndex,
    model_curve: pd.Series,
    spy_curve: pd.Series,
    ew_curve: pd.Series,
    decision_paths: np.ndarray,
) -> None:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 9), facecolor="#0B0E13")
    ax.set_facecolor("#0B0E13")

    if decision_paths.size > 0:
        for i in range(decision_paths.shape[0]):
            ax.plot(dates, decision_paths[i], color="#B0B0B0", alpha=0.08, linewidth=1)

    ax.plot(dates, spy_curve.values, color="#F5F5F5", linewidth=2.5, label="SPY")
    ax.plot(dates, model_curve.values, color="#6BFF3F", linewidth=2.6, label="Model Strategy")
    ax.plot(
        dates,
        ew_curve.values,
        color="#FFBF40",
        linewidth=2.5,
        label="Equal-Weight (Cost-Aware)",
    )

    ax.set_title("Meta-Ensemble Playback Snapshot", fontsize=22, color="#F1F1F1")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Normalized Portfolio Value", fontsize=12)
    ax.grid(color="#5A5A5A", alpha=0.22, linewidth=0.8)
    ax.legend(loc="lower right", frameon=True, framealpha=0.85)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_animation(
    out_path: Path,
    dates: pd.DatetimeIndex,
    model_curve: pd.Series,
    spy_curve: pd.Series,
    ew_curve: pd.Series,
    decision_paths: np.ndarray,
    fps: int,
    max_frames: int,
) -> None:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(16, 9), facecolor="#0B0E13")
    ax.set_facecolor("#0B0E13")

    n_points = len(dates)
    frame_points = np.linspace(2, n_points, num=min(max_frames, n_points), dtype=int)
    frame_points = np.unique(frame_points)

    grey_lines = []
    if decision_paths.size > 0:
        for _ in range(decision_paths.shape[0]):
            line, = ax.plot([], [], color="#9A9A9A", alpha=0.12, linewidth=1)
            grey_lines.append(line)

    spy_line, = ax.plot([], [], color="#F5F5F5", linewidth=2.6, label="SPY")
    model_line, = ax.plot([], [], color="#6BFF3F", linewidth=2.7, label="Model Strategy")
    ew_line, = ax.plot(
        [], [], color="#FFBF40", linewidth=2.6, label="Equal-Weight (Cost-Aware)"
    )

    all_values = np.concatenate(
        [
            model_curve.values,
            spy_curve.values,
            ew_curve.values,
            decision_paths.reshape(-1) if decision_paths.size > 0 else np.array([1.0]),
        ]
    )
    finite_vals = all_values[np.isfinite(all_values)]
    y_min = max(0.1, float(np.nanpercentile(finite_vals, 1)) * 0.95)
    y_max = float(np.nanpercentile(finite_vals, 99)) * 1.05
    if y_max <= y_min:
        y_max = y_min + 1.0

    ax.set_xlim(dates[0], dates[-1])
    ax.set_ylim(y_min, y_max)
    ax.grid(color="#5A5A5A", alpha=0.22, linewidth=0.8)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Normalized Portfolio Value", fontsize=12)
    title = ax.set_title("Meta-Ensemble Playback", fontsize=22, color="#F1F1F1")
    ax.legend(loc="lower right", frameon=True, framealpha=0.85)

    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    def _init():
        return grey_lines + [spy_line, model_line, ew_line, title]

    def _update(frame_idx: int):
        end = int(frame_points[frame_idx])
        x = dates[:end]
        if decision_paths.size > 0:
            for i, line in enumerate(grey_lines):
                line.set_data(x, decision_paths[i, :end])

        spy_line.set_data(x, spy_curve.values[:end])
        model_line.set_data(x, model_curve.values[:end])
        ew_line.set_data(x, ew_curve.values[:end])
        progress = (end / n_points) * 100.0
        title.set_text(f"Meta-Ensemble Playback  ({progress:5.1f}% complete)")
        return grey_lines + [spy_line, model_line, ew_line, title]

    anim = animation.FuncAnimation(
        fig,
        _update,
        init_func=_init,
        frames=len(frame_points),
        interval=max(12, int(1000 / max(1, fps))),
        blit=False,
    )
    writer = animation.FFMpegWriter(
        fps=fps,
        metadata={"title": "Meta-Ensemble Playback"},
        bitrate=3500,
        codec="libx264",
    )
    anim.save(out_path, writer=writer, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta-ensemble playback video generator")
    parser.add_argument(
        "--checkpoint-dir",
        default="quantum_alpha/models/checkpoints/meta_ensemble",
        help="Directory containing walk_forward_predictions.pkl",
    )
    parser.add_argument(
        "--blend-checkpoint-dirs",
        type=str,
        default=None,
        help=(
            "Optional comma-separated extra checkpoint dirs. "
            "Their y_proba values are blended with primary predictions."
        ),
    )
    parser.add_argument(
        "--blend-weights",
        type=str,
        default=None,
        help=(
            "Optional comma-separated blend weights. "
            "Provide N weights for primary+extras, or N-1 for extras only."
        ),
    )
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--signal-threshold", type=float, default=0.53)
    parser.add_argument(
        "--short-threshold",
        type=float,
        default=None,
        help="Max probability for short entries (default: 1 - signal-threshold)",
    )
    parser.add_argument("--confidence", type=float, default=0.0)
    parser.add_argument("--commission-bps", type=float, default=0.0)
    parser.add_argument("--hold-days", type=int, default=1)
    parser.add_argument("--long-only", action="store_true", help="Disable shorts")
    parser.add_argument(
        "--earnings-filter",
        action="store_true",
        help="Skip entries with earnings inside hold period",
    )
    parser.add_argument(
        "--pead-boost",
        action="store_true",
        help="Apply PEAD confidence adjustment",
    )
    parser.add_argument(
        "--semiconductor-short-gate",
        action="store_true",
        help="Only allow semiconductor shorts when AI regime is weak",
    )
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=0.0,
        help="AI regime threshold for semiconductor short gate",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--decision-paths", type=int, default=80)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=420)
    parser.add_argument("--max-points", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(
        args.output_dir or f"artifacts/meta_ensemble_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = load_predictions(args.checkpoint_dir)
    pred = deduplicate_predictions(pred)
    if args.blend_checkpoint_dirs:
        blend_dirs = [x.strip() for x in args.blend_checkpoint_dirs.split(",") if x.strip()]
        blend_weights = (
            [float(x) for x in args.blend_weights.split(",") if x.strip()]
            if args.blend_weights
            else None
        )
        pred = blend_prediction_probabilities(
            primary_df=pred,
            checkpoint_dirs=blend_dirs,
            blend_weights=blend_weights,
        )
    pred["date"] = pd.to_datetime(pred["date"])
    if args.start:
        pred = pred[pred["date"] >= pd.to_datetime(args.start)]
    if args.end:
        pred = pred[pred["date"] <= pd.to_datetime(args.end)]
    if len(pred) == 0:
        raise RuntimeError("No predictions available for requested date range")

    sig = compute_signals(
        pred,
        signal_threshold=args.signal_threshold,
        short_threshold=args.short_threshold,
        confidence_threshold=args.confidence,
        long_only=args.long_only,
    )
    bt = backtest_equal_weight(
        sig,
        max_positions=args.max_positions,
        commission_bps=args.commission_bps,
        hold_days=args.hold_days,
        top_k=args.top_k,
        initial_capital=100_000.0,
        confidence_weight=False,
        earnings_filter=args.earnings_filter,
        pead_boost=args.pead_boost,
        semiconductor_short_gate=args.semiconductor_short_gate,
        gate_threshold=args.gate_threshold,
    )
    if "error" in bt:
        raise RuntimeError(bt["error"])

    model_dates = _to_datetime_index(bt["daily_dates"])
    model_curve = pd.Series(bt["equity_curve"], index=model_dates, name="model")
    model_curve = model_curve / float(model_curve.iloc[0])

    start_dt = model_dates.min().to_pydatetime()
    end_dt = model_dates.max().to_pydatetime()
    collector = DataCollector()
    spy_df = collector.fetch_ohlcv("SPY", start_dt, end_dt)
    spy_close = (
        spy_df["close"]
        .copy()
        .set_axis(_to_datetime_index(spy_df.index))
        .reindex(model_dates)
        .ffill()
        .bfill()
    )
    spy_curve = _curve_from_returns(spy_close.pct_change(fill_method=None)).rename("spy")

    ew_bt = _cost_aware_equal_weight_backtest(
        pred=pred,
        commission_bps=args.commission_bps,
        hold_days=args.hold_days,
    )
    if "error" in ew_bt:
        raise RuntimeError(f"Equal-weight benchmark failed: {ew_bt['error']}")
    ew_dates = _to_datetime_index(ew_bt["daily_dates"])
    ew_curve = pd.Series(ew_bt["equity_curve"], index=ew_dates, name="equal_weight")
    ew_curve = (ew_curve / float(ew_curve.iloc[0])).reindex(model_dates).ffill().bfill()

    curves = pd.concat([model_curve, spy_curve, ew_curve], axis=1).ffill().bfill()
    model_returns = curves["model"].pct_change().fillna(0.0)
    decision_paths = _generate_decision_paths(
        model_returns=model_returns,
        n_paths=max(0, args.decision_paths),
        seed=args.seed,
    )
    render_curves, render_paths = _downsample_for_render(
        curves=curves,
        decision_paths=decision_paths,
        max_points=args.max_points,
    )

    curves_path = out_dir / "normalized_curves.csv"
    curves.to_csv(curves_path, index_label="timestamp")
    snapshot_path = out_dir / "playback_snapshot.png"
    _save_static_snapshot(
        out_path=snapshot_path,
        dates=render_curves.index,
        model_curve=render_curves["model"],
        spy_curve=render_curves["spy"],
        ew_curve=render_curves["equal_weight"],
        decision_paths=render_paths,
    )
    video_path = out_dir / "backtest_playback.mp4"
    _save_animation(
        out_path=video_path,
        dates=render_curves.index,
        model_curve=render_curves["model"],
        spy_curve=render_curves["spy"],
        ew_curve=render_curves["equal_weight"],
        decision_paths=render_paths,
        fps=args.fps,
        max_frames=args.max_frames,
    )

    summary: Dict[str, object] = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_date": str(model_dates.min().date()),
        "end_date": str(model_dates.max().date()),
        "params": {
            "signal_threshold": args.signal_threshold,
            "short_threshold": args.short_threshold,
            "commission_bps": args.commission_bps,
            "hold_days": args.hold_days,
            "long_only": args.long_only,
            "top_k": args.top_k,
            "max_positions": args.max_positions,
            "earnings_filter": args.earnings_filter,
            "pead_boost": args.pead_boost,
            "semiconductor_short_gate": args.semiconductor_short_gate,
            "gate_threshold": args.gate_threshold,
            "blend_checkpoint_dirs": args.blend_checkpoint_dirs,
            "blend_weights": args.blend_weights,
        },
        "backtest_metrics": {
            "total_return": float(bt["total_return"]),
            "annual_return": float(bt["annual_return"]),
            "sharpe": float(bt["sharpe"]),
            "sortino": float(bt["sortino"]),
            "max_drawdown": float(bt["max_drawdown"]),
            "final_equity": float(bt["final_equity"]),
            "trading_days": int(bt["n_trading_days"]),
        },
        "equal_weight_benchmark_metrics": {
            "benchmark_mode": "cost_aware",
            "total_return": float(ew_bt["total_return"]),
            "annual_return": float(ew_bt["annual_return"]),
            "sharpe": float(ew_bt["sharpe"]),
            "sortino": float(ew_bt["sortino"]),
            "max_drawdown": float(ew_bt["max_drawdown"]),
            "final_equity": float(ew_bt["final_equity"]),
            "trading_days": int(ew_bt["n_trading_days"]),
        },
        "output": {
            "directory": str(out_dir),
            "normalized_curves": str(curves_path),
            "snapshot_png": str(snapshot_path),
            "video_mp4": str(video_path),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nArtifacts generated:")
    print(f"  {out_dir / 'summary.json'}")
    print(f"  {curves_path}")
    print(f"  {snapshot_path}")
    print(f"  {video_path}")


if __name__ == "__main__":
    main()
