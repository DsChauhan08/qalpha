"""
Animated backtest playback with benchmark overlays.

This script runs a backtest and produces:
- summary JSON
- normalized curve CSV
- static PNG snapshot
- MP4 animation that plays through the full run
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantum_alpha.data.collectors.market_data import DataCollector
from quantum_alpha.main import _resolve_symbols, load_config, run_backtest


def _to_datetime_index(values) -> pd.DatetimeIndex:
    raw = pd.to_datetime(values)
    normalized: List[pd.Timestamp] = []
    for value in raw:
        ts = pd.Timestamp(value)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        normalized.append(ts.normalize())
    return pd.DatetimeIndex(normalized)


def _curve_from_returns(returns: pd.Series) -> pd.Series:
    ret = returns.fillna(0.0).astype(float)
    curve = (1.0 + ret).cumprod()
    if len(curve) > 0:
        curve.iloc[0] = 1.0
    return curve


def _build_benchmarks(
    collector: DataCollector,
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    target_index: pd.DatetimeIndex,
) -> Dict[str, pd.Series]:
    spy_df = collector.fetch_ohlcv("SPY", start_date, end_date)
    spy_close = (
        spy_df["close"]
        .copy()
        .set_axis(_to_datetime_index(spy_df.index))
        .reindex(target_index)
        .ffill()
        .bfill()
    )
    spy_curve = _curve_from_returns(spy_close.pct_change(fill_method=None))

    close_frames = []
    for symbol in symbols:
        try:
            df = collector.fetch_ohlcv(symbol, start_date, end_date)
            series = (
                df["close"]
                .copy()
                .set_axis(_to_datetime_index(df.index))
                .reindex(target_index)
                .ffill()
                .bfill()
                .rename(symbol)
            )
            close_frames.append(series)
        except Exception:
            continue

    if close_frames:
        close_df = pd.concat(close_frames, axis=1)
        ew_returns = close_df.pct_change(fill_method=None).mean(axis=1, skipna=True)
        ew_curve = _curve_from_returns(ew_returns)
    else:
        ew_curve = pd.Series(1.0, index=target_index, name="equal_weight")

    return {"spy": spy_curve, "equal_weight": ew_curve}


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

        # Light perturbations to emulate different decision timing/conviction.
        sampled *= rng.uniform(0.85, 1.15, size=n)
        hold_mask = rng.random(n) < 0.08
        sampled[hold_mask] *= 0.2
        wrong_mask = rng.random(n) < 0.02
        sampled[wrong_mask] *= -0.6
        sampled = np.clip(sampled, -0.95, 2.0)

        paths[i] = np.cumprod(1.0 + sampled)
        paths[i, 0] = 1.0

    return paths


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
    ax.plot(
        dates,
        model_curve.values,
        color="#6BFF3F",
        linewidth=2.6,
        label="Model Strategy",
    )
    ax.plot(
        dates,
        ew_curve.values,
        color="#FFBF40",
        linewidth=2.5,
        label="Equal-Weight Universe",
    )

    ax.set_title("Backtest Playback Snapshot", fontsize=22, color="#F1F1F1")
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
    if n_points < 2:
        raise ValueError("Not enough points for animation")

    frame_points = np.linspace(2, n_points, num=min(max_frames, n_points), dtype=int)
    frame_points = np.unique(frame_points)

    grey_lines = []
    if decision_paths.size > 0:
        for _ in range(decision_paths.shape[0]):
            line, = ax.plot([], [], color="#9A9A9A", alpha=0.12, linewidth=1)
            grey_lines.append(line)

    spy_line, = ax.plot([], [], color="#F5F5F5", linewidth=2.6, label="SPY")
    model_line, = ax.plot([], [], color="#6BFF3F", linewidth=2.7, label="Model Strategy")
    ew_line, = ax.plot([], [], color="#FFBF40", linewidth=2.6, label="Equal-Weight Universe")

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
    title = ax.set_title("Backtest Playback", fontsize=22, color="#F1F1F1")
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
        title.set_text(f"Backtest Playback  ({progress:5.1f}% complete)")
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
        metadata={"title": "Quantum Alpha Backtest Playback"},
        bitrate=3500,
        codec="libx264",
    )
    anim.save(out_path, writer=writer, dpi=140)
    plt.close(fig)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backtest and export animated playback video")
    parser.add_argument("--symbols", nargs="+", default=["AUTO"], help="Universe to backtest")
    parser.add_argument("--strategy", default="enhanced", help="Strategy name for run_backtest")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital")
    parser.add_argument("--config", default=None, help="Optional config path")
    parser.add_argument("--decision-paths", type=int, default=80, help="Number of gray decision paths")
    parser.add_argument("--fps", type=int, default=30, help="Video frames per second")
    parser.add_argument("--max-frames", type=int, default=420, help="Max animation frames")
    parser.add_argument(
        "--max-points",
        type=int,
        default=1500,
        help="Max points used for rendering the PNG/MP4",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for decision paths")
    parser.add_argument("--validate", action="store_true", help="Run MCPT inside backtest")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default artifacts/backtest_video_<timestamp>)",
    )
    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or f"artifacts/backtest_video_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_backtest(
        symbols=args.symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital,
        strategy_type=args.strategy,
        validate=args.validate,
        verbose=True,
        config_path=args.config,
    )
    if not isinstance(results, dict) or "metrics" not in results:
        raise RuntimeError(f"Backtest failed: {results}")

    eq_df = pd.DataFrame(results["equity_curve"]).copy()
    if eq_df.empty:
        raise RuntimeError("Backtest returned empty equity curve")
    eq_df["timestamp"] = _to_datetime_index(eq_df["timestamp"])
    eq_df = eq_df.set_index("timestamp").sort_index()
    model_curve = (eq_df["equity"] / float(eq_df["equity"].iloc[0])).rename("model")

    settings = load_config(args.config)
    collector = DataCollector()
    resolved_symbols = _resolve_symbols(args.symbols, collector, settings)
    benchmarks = _build_benchmarks(
        collector=collector,
        symbols=resolved_symbols,
        start_date=start_date,
        end_date=end_date,
        target_index=model_curve.index,
    )
    spy_curve = benchmarks["spy"].rename("spy")
    ew_curve = benchmarks["equal_weight"].rename("equal_weight")

    model_returns = model_curve.pct_change().fillna(0.0)
    decision_paths = _generate_decision_paths(
        model_returns=model_returns,
        n_paths=max(0, args.decision_paths),
        seed=args.seed,
    )

    curves = pd.concat([model_curve, spy_curve, ew_curve], axis=1).ffill().bfill()
    curves_path = out_dir / "normalized_curves.csv"
    curves.to_csv(curves_path, index_label="timestamp")
    render_curves, render_paths = _downsample_for_render(
        curves=curves,
        decision_paths=decision_paths,
        max_points=args.max_points,
    )

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

    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_date": args.start,
        "end_date": args.end,
        "strategy": args.strategy,
        "input_symbols": args.symbols,
        "resolved_symbols": resolved_symbols,
        "metrics": results["metrics"],
        "output": {
            "directory": str(out_dir),
            "normalized_curves": str(curves_path),
            "snapshot_png": str(snapshot_path),
            "video_mp4": str(video_path),
        },
    }
    if "mcpt" in results:
        summary["mcpt"] = results["mcpt"]

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nArtifacts generated:")
    print(f"  {summary_path}")
    print(f"  {curves_path}")
    print(f"  {snapshot_path}")
    print(f"  {video_path}")


if __name__ == "__main__":
    main()
