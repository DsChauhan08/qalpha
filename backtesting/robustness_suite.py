#!/usr/bin/env python3
"""
Robustness/tuning suite for meta-ensemble execution settings.

Evaluates the model against two baselines:
1) Cost-aware equal-weight benchmark (same commission/hold mechanics)
2) SPY buy-and-hold benchmark

Also reports:
- Full/old/recent/very-recent segment metrics
- Random-window robustness tests
- Simulated regime tests via block-bootstrap paths
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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


@dataclass(frozen=True)
class StrategyConfig:
    signal_threshold: float = 0.53
    short_threshold: float | None = None
    confidence: float = 0.0
    long_only: bool = True
    commission_bps: float = 5.0
    hold_days: int = 10
    top_k: int = 20
    max_positions: int = 20
    confidence_weight: bool = False
    semiconductor_short_gate: bool = False
    gate_threshold: float = 0.0


def _parse_csv_floats(value: str) -> List[float]:
    out = []
    for token in value.split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    return out


def _parse_csv_ints(value: str) -> List[int]:
    out = []
    for token in value.split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    return out


def _to_date_index(values: Iterable) -> pd.DatetimeIndex:
    idx = pd.to_datetime(values)
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def _returns_to_metrics(returns: pd.Series) -> Dict[str, float]:
    r = returns.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n = len(r)
    if n == 0:
        return {
            "n_days": 0,
            "total_return": 0.0,
            "annual_return": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
        }

    equity = (1.0 + r).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    years = max(n / 252.0, 1.0 / 252.0)
    annual_return = float((1.0 + total_return) ** (1.0 / years) - 1.0)

    std = float(r.std(ddof=0))
    mean = float(r.mean())
    sharpe = float((mean / std) * np.sqrt(252.0)) if std > 0 else 0.0

    downside = r[r < 0]
    downside_std = float(downside.std(ddof=0)) if len(downside) > 0 else 0.0
    sortino = (
        float((mean / downside_std) * np.sqrt(252.0)) if downside_std > 0 else 0.0
    )

    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min())

    return {
        "n_days": int(n),
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
    }


def _run_model_returns(pred: pd.DataFrame, cfg: StrategyConfig) -> pd.Series:
    with contextlib.redirect_stdout(io.StringIO()):
        sig = compute_signals(
            pred,
            signal_threshold=cfg.signal_threshold,
            short_threshold=cfg.short_threshold,
            confidence_threshold=cfg.confidence,
            long_only=cfg.long_only,
        )
        bt = backtest_equal_weight(
            sig,
            max_positions=cfg.max_positions,
            commission_bps=cfg.commission_bps,
            hold_days=cfg.hold_days,
            top_k=cfg.top_k,
            initial_capital=100_000.0,
            confidence_weight=cfg.confidence_weight,
            earnings_filter=False,
            pead_boost=False,
            semiconductor_short_gate=cfg.semiconductor_short_gate,
            gate_threshold=cfg.gate_threshold,
        )

    if "error" in bt:
        raise RuntimeError(f"Model backtest failed: {bt['error']}")

    idx = _to_date_index(bt["daily_dates"])
    ret = pd.Series(bt["daily_returns"], index=idx, name="model")
    return ret.sort_index()


def _run_equal_weight_returns(
    pred: pd.DataFrame, commission_bps: float, hold_days: int
) -> pd.Series:
    bench = pred[["date", "symbol", "forward_return", "y_true"]].copy()
    bench["raw_signal"] = 1.0
    bench["confidence"] = 1.0
    bench["y_proba"] = 1.0

    n_symbols = int(bench["symbol"].nunique())
    with contextlib.redirect_stdout(io.StringIO()):
        bt = backtest_equal_weight(
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
    if "error" in bt:
        raise RuntimeError(f"Equal-weight backtest failed: {bt['error']}")

    idx = _to_date_index(bt["daily_dates"])
    ret = pd.Series(bt["daily_returns"], index=idx, name="equal_weight")
    return ret.sort_index()


def _run_spy_returns(
    index: pd.DatetimeIndex, commission_bps: float = 0.0
) -> pd.Series:
    collector = DataCollector()
    start = index.min().to_pydatetime()
    end = index.max().to_pydatetime()
    spy_df = collector.fetch_ohlcv("SPY", start=start, end=end, interval="1d")
    if spy_df is None or len(spy_df) == 0 or "close" not in spy_df.columns:
        raise RuntimeError("Failed to fetch SPY data for benchmark")

    spy_close = pd.Series(
        pd.to_numeric(spy_df["close"], errors="coerce").values,
        index=_to_date_index(spy_df.index),
    ).sort_index()
    spy_ret = spy_close.pct_change(fill_method=None).fillna(0.0)
    spy_ret = spy_ret.reindex(index).ffill().bfill().fillna(0.0)

    # Buy-and-hold: apply one entry + one exit commission drag.
    if len(spy_ret) > 1 and commission_bps > 0:
        commission_rate = float(commission_bps) / 10_000.0
        spy_ret.iloc[0] -= commission_rate
        spy_ret.iloc[-1] -= commission_rate

    spy_ret.name = "spy"
    return spy_ret


def _segment_slice(
    s: pd.Series, start: pd.Timestamp | None, end: pd.Timestamp | None
) -> pd.Series:
    out = s
    if start is not None:
        out = out[out.index >= start]
    if end is not None:
        out = out[out.index <= end]
    return out


def _evaluate_segments(
    returns_map: Dict[str, pd.Series],
    segments: Dict[str, Tuple[pd.Timestamp | None, pd.Timestamp | None]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for seg_name, (start, end) in segments.items():
        out[seg_name] = {}
        for strategy_name, series in returns_map.items():
            seg_series = _segment_slice(series, start, end)
            out[seg_name][strategy_name] = _returns_to_metrics(seg_series)
    return out


def _segment_table(
    segment_results: Dict[str, Dict[str, Dict[str, float]]]
) -> pd.DataFrame:
    rows = []
    for segment, strategy_blob in segment_results.items():
        for strategy, metrics in strategy_blob.items():
            row = {"segment": segment, "strategy": strategy}
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def _score_candidate(
    seg_results: Dict[str, Dict[str, Dict[str, float]]]
) -> float:
    weights = {"full": 0.35, "old": 0.20, "recent": 0.25, "very_recent": 0.20}
    score = 0.0
    for seg, w in weights.items():
        blob = seg_results.get(seg)
        if not blob:
            continue
        m = blob["model"]
        ew = blob["equal_weight"]
        spy = blob["spy"]
        score += w * (
            3.0 * (m["annual_return"] - ew["annual_return"])
            + 1.5 * (m["annual_return"] - spy["annual_return"])
            + 0.4 * (m["sharpe"] - ew["sharpe"])
            + 0.2 * (m["sharpe"] - spy["sharpe"])
        )
        dd_baseline = min(abs(ew["max_drawdown"]), abs(spy["max_drawdown"]))
        dd_penalty = max(0.0, abs(m["max_drawdown"]) - dd_baseline)
        score -= w * dd_penalty
    return float(score)


def _evaluate_random_windows(
    returns_map: Dict[str, pd.Series],
    n_windows: int,
    window_days: int,
    seed: int,
) -> pd.DataFrame:
    common_index = None
    for s in returns_map.values():
        common_index = s.index if common_index is None else common_index.intersection(s.index)
    if common_index is None or len(common_index) == 0:
        return pd.DataFrame()

    common_index = common_index.sort_values()
    n = len(common_index)
    if n <= window_days:
        return pd.DataFrame()

    max_start = n - window_days
    rng = np.random.default_rng(seed)
    starts = rng.choice(max_start + 1, size=min(n_windows, max_start + 1), replace=False)
    starts = np.sort(starts)

    rows = []
    for i, start_idx in enumerate(starts):
        start_dt = common_index[start_idx]
        end_dt = common_index[start_idx + window_days - 1]
        metrics = {}
        for name, series in returns_map.items():
            window_series = series.loc[start_dt:end_dt]
            metrics[name] = _returns_to_metrics(window_series)

        annuals = {k: v["annual_return"] for k, v in metrics.items()}
        winner = max(annuals.items(), key=lambda kv: kv[1])[0]
        rows.append(
            {
                "window_id": i + 1,
                "start": str(start_dt.date()),
                "end": str(end_dt.date()),
                "winner_annual_return": winner,
                "model_annual_return": metrics["model"]["annual_return"],
                "equal_weight_annual_return": metrics["equal_weight"]["annual_return"],
                "spy_annual_return": metrics["spy"]["annual_return"],
                "model_sharpe": metrics["model"]["sharpe"],
                "equal_weight_sharpe": metrics["equal_weight"]["sharpe"],
                "spy_sharpe": metrics["spy"]["sharpe"],
            }
        )

    return pd.DataFrame(rows)


def _sample_block_indices(
    n: int, n_paths: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    block = max(2, min(int(block_size), n))
    out = np.empty((n_paths, n), dtype=np.int32)
    for i in range(n_paths):
        pos = 0
        while pos < n:
            start = int(rng.integers(0, max(1, n - block + 1)))
            take = min(block, n - pos)
            out[i, pos : pos + take] = np.arange(start, start + take, dtype=np.int32)
            pos += take
    return out


def _simulate_block_bootstrap(
    returns_map: Dict[str, pd.Series],
    n_paths: int,
    block_size: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    common_index = None
    for s in returns_map.values():
        common_index = s.index if common_index is None else common_index.intersection(s.index)
    if common_index is None or len(common_index) < 10:
        return {}
    common_index = common_index.sort_values()

    arrays = {
        k: returns_map[k].reindex(common_index).fillna(0.0).to_numpy(dtype=float)
        for k in returns_map
    }
    n = len(common_index)
    rng = np.random.default_rng(seed)
    indices = _sample_block_indices(n=n, n_paths=n_paths, block_size=block_size, rng=rng)

    annual_dists: Dict[str, np.ndarray] = {}
    sharpe_dists: Dict[str, np.ndarray] = {}

    for name, arr in arrays.items():
        ann = np.zeros(n_paths, dtype=float)
        shp = np.zeros(n_paths, dtype=float)
        for i in range(n_paths):
            sample = arr[indices[i]]
            metrics = _returns_to_metrics(pd.Series(sample))
            ann[i] = metrics["annual_return"]
            shp[i] = metrics["sharpe"]
        annual_dists[name] = ann
        sharpe_dists[name] = shp

    summary: Dict[str, Dict[str, float]] = {}
    for name in arrays.keys():
        summary[name] = {
            "annual_return_mean": float(np.mean(annual_dists[name])),
            "annual_return_p10": float(np.percentile(annual_dists[name], 10)),
            "annual_return_p50": float(np.percentile(annual_dists[name], 50)),
            "annual_return_p90": float(np.percentile(annual_dists[name], 90)),
            "sharpe_mean": float(np.mean(sharpe_dists[name])),
            "sharpe_p10": float(np.percentile(sharpe_dists[name], 10)),
            "sharpe_p50": float(np.percentile(sharpe_dists[name], 50)),
            "sharpe_p90": float(np.percentile(sharpe_dists[name], 90)),
        }

    summary["comparisons"] = {
        "p_model_ann_gt_equal_weight": float(
            np.mean(annual_dists["model"] > annual_dists["equal_weight"])
        ),
        "p_model_ann_gt_spy": float(np.mean(annual_dists["model"] > annual_dists["spy"])),
        "p_model_sharpe_gt_equal_weight": float(
            np.mean(sharpe_dists["model"] > sharpe_dists["equal_weight"])
        ),
        "p_model_sharpe_gt_spy": float(np.mean(sharpe_dists["model"] > sharpe_dists["spy"])),
    }
    return summary


def _run_tuning_grid(
    pred: pd.DataFrame,
    base_cfg: StrategyConfig,
    baseline_returns: Dict[str, pd.Series],
    segments: Dict[str, Tuple[pd.Timestamp | None, pd.Timestamp | None]],
    signal_thresholds: List[float],
    top_ks: List[int],
    hold_days: List[int],
    confidence_vals: List[float],
) -> Tuple[pd.DataFrame, StrategyConfig, Dict[str, Dict[str, Dict[str, float]]], pd.Series]:
    rows = []
    best_score = float("-inf")
    best_cfg = base_cfg
    best_seg = {}
    best_model_ret = pd.Series(dtype=float)

    combos = list(
        itertools.product(signal_thresholds, top_ks, hold_days, confidence_vals)
    )
    for i, (sig_thr, top_k, hold_d, conf) in enumerate(combos, start=1):
        cfg = StrategyConfig(
            signal_threshold=float(sig_thr),
            short_threshold=base_cfg.short_threshold,
            confidence=float(conf),
            long_only=base_cfg.long_only,
            commission_bps=base_cfg.commission_bps,
            hold_days=int(hold_d),
            top_k=int(top_k),
            max_positions=base_cfg.max_positions,
            confidence_weight=base_cfg.confidence_weight,
            semiconductor_short_gate=base_cfg.semiconductor_short_gate,
            gate_threshold=base_cfg.gate_threshold,
        )
        try:
            model_ret = _run_model_returns(pred, cfg)
        except Exception:
            continue

        aligned = {
            "model": model_ret,
            "equal_weight": baseline_returns["equal_weight"],
            "spy": baseline_returns["spy"],
        }
        seg = _evaluate_segments(aligned, segments)
        score = _score_candidate(seg)

        rows.append(
            {
                "candidate": i,
                "score": score,
                "signal_threshold": cfg.signal_threshold,
                "top_k": cfg.top_k,
                "hold_days": cfg.hold_days,
                "confidence": cfg.confidence,
                "full_ann_model": seg["full"]["model"]["annual_return"],
                "full_ann_ew": seg["full"]["equal_weight"]["annual_return"],
                "full_ann_spy": seg["full"]["spy"]["annual_return"],
                "recent_ann_model": seg["recent"]["model"]["annual_return"],
                "recent_ann_ew": seg["recent"]["equal_weight"]["annual_return"],
                "recent_ann_spy": seg["recent"]["spy"]["annual_return"],
            }
        )

        if score > best_score:
            best_score = score
            best_cfg = cfg
            best_seg = seg
            best_model_ret = model_ret

    if not rows:
        raise RuntimeError("No valid tuning candidates were produced")

    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return df, best_cfg, best_seg, best_model_ret


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta-ensemble robustness/tuning suite")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/checkpoints/meta_ensemble",
        help="Checkpoint dir containing walk_forward_predictions.pkl",
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
    parser.add_argument("--signal-threshold", type=float, default=0.53)
    parser.add_argument("--confidence", type=float, default=0.0)
    parser.add_argument("--commission-bps", type=float, default=5.0)
    parser.add_argument("--hold-days", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-positions", type=int, default=20)
    parser.add_argument("--long-only", action="store_true", default=True)
    parser.add_argument("--allow-shorts", action="store_true")
    parser.add_argument("--short-threshold", type=float, default=None)
    parser.add_argument("--semiconductor-short-gate", action="store_true")
    parser.add_argument("--gate-threshold", type=float, default=0.0)
    parser.add_argument("--random-windows", type=int, default=12)
    parser.add_argument("--random-window-days", type=int, default=504)
    parser.add_argument("--sim-paths", type=int, default=500)
    parser.add_argument("--sim-block-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-tune", action="store_true", help="Skip parameter grid search")
    parser.add_argument(
        "--tune-signal-thresholds",
        type=str,
        default="0.52,0.53,0.55",
    )
    parser.add_argument("--tune-top-k", type=str, default="10,15,20")
    parser.add_argument("--tune-hold-days", type=str, default="5,10")
    parser.add_argument("--tune-confidence", type=str, default="0.0,0.05")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    long_only = True if not args.allow_shorts else False
    base_cfg = StrategyConfig(
        signal_threshold=float(args.signal_threshold),
        short_threshold=args.short_threshold,
        confidence=float(args.confidence),
        long_only=long_only,
        commission_bps=float(args.commission_bps),
        hold_days=int(args.hold_days),
        top_k=int(args.top_k),
        max_positions=int(args.max_positions),
        confidence_weight=False,
        semiconductor_short_gate=bool(args.semiconductor_short_gate),
        gate_threshold=float(args.gate_threshold),
    )

    out_dir = Path(
        args.output_dir
        or f"artifacts/robustness_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = deduplicate_predictions(load_predictions(args.checkpoint_dir))
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

    model_ret_base = _run_model_returns(pred, base_cfg)
    ew_ret = _run_equal_weight_returns(
        pred=pred,
        commission_bps=base_cfg.commission_bps,
        hold_days=base_cfg.hold_days,
    )
    common_idx = model_ret_base.index.intersection(ew_ret.index).sort_values()
    ew_ret = ew_ret.reindex(common_idx).fillna(0.0)
    model_ret_base = model_ret_base.reindex(common_idx).fillna(0.0)
    spy_ret = _run_spy_returns(common_idx, commission_bps=base_cfg.commission_bps)
    spy_ret = spy_ret.reindex(common_idx).fillna(0.0)

    segments = {
        "full": (None, None),
        "old": (None, pd.Timestamp("2019-12-31")),
        "recent": (pd.Timestamp("2020-01-01"), None),
        "very_recent": (pd.Timestamp("2023-01-01"), None),
    }

    returns_map_base = {"model": model_ret_base, "equal_weight": ew_ret, "spy": spy_ret}
    seg_base = _evaluate_segments(returns_map_base, segments)

    best_cfg = base_cfg
    best_model_ret = model_ret_base
    seg_best = seg_base
    tuning_df = pd.DataFrame()

    if not args.no_tune:
        tuning_df, best_cfg, seg_best, best_model_ret = _run_tuning_grid(
            pred=pred,
            base_cfg=base_cfg,
            baseline_returns={"equal_weight": ew_ret, "spy": spy_ret},
            segments=segments,
            signal_thresholds=_parse_csv_floats(args.tune_signal_thresholds),
            top_ks=_parse_csv_ints(args.tune_top_k),
            hold_days=_parse_csv_ints(args.tune_hold_days),
            confidence_vals=_parse_csv_floats(args.tune_confidence),
        )
        best_model_ret = best_model_ret.reindex(common_idx).fillna(0.0)
        returns_map_best = {
            "model": best_model_ret,
            "equal_weight": ew_ret,
            "spy": spy_ret,
        }
        seg_best = _evaluate_segments(returns_map_best, segments)

    random_base = _evaluate_random_windows(
        returns_map=returns_map_base,
        n_windows=args.random_windows,
        window_days=args.random_window_days,
        seed=args.seed,
    )
    random_best = _evaluate_random_windows(
        returns_map={"model": best_model_ret, "equal_weight": ew_ret, "spy": spy_ret},
        n_windows=args.random_windows,
        window_days=args.random_window_days,
        seed=args.seed + 1,
    )

    sim_base = _simulate_block_bootstrap(
        returns_map=returns_map_base,
        n_paths=args.sim_paths,
        block_size=args.sim_block_size,
        seed=args.seed,
    )
    sim_best = _simulate_block_bootstrap(
        returns_map={"model": best_model_ret, "equal_weight": ew_ret, "spy": spy_ret},
        n_paths=args.sim_paths,
        block_size=args.sim_block_size,
        seed=args.seed + 1,
    )

    seg_base_df = _segment_table(seg_base)
    seg_best_df = _segment_table(seg_best)
    seg_base_df.to_csv(out_dir / "segment_metrics_base.csv", index=False)
    seg_best_df.to_csv(out_dir / "segment_metrics_best.csv", index=False)
    if not tuning_df.empty:
        tuning_df.to_csv(out_dir / "candidate_scores.csv", index=False)
    if not random_base.empty:
        random_base.to_csv(out_dir / "random_windows_base.csv", index=False)
    if not random_best.empty:
        random_best.to_csv(out_dir / "random_windows_best.csv", index=False)

    curves = pd.DataFrame(
        {
            "date": common_idx,
            "model_base": model_ret_base.values,
            "model_best": best_model_ret.values,
            "equal_weight": ew_ret.values,
            "spy": spy_ret.values,
        }
    )
    curves.to_csv(out_dir / "daily_returns.csv", index=False)

    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_dir": args.checkpoint_dir,
        "blend_checkpoint_dirs": args.blend_checkpoint_dirs,
        "blend_weights": args.blend_weights,
        "base_config": asdict(base_cfg),
        "best_config": asdict(best_cfg),
        "segments_base": seg_base,
        "segments_best": seg_best,
        "simulated_base": sim_base,
        "simulated_best": sim_best,
        "output_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nRobustness suite complete")
    print(f"Output: {out_dir}")
    print("Base full annual return:", seg_base["full"]["model"]["annual_return"])
    print("Best full annual return:", seg_best["full"]["model"]["annual_return"])
    print("Equal-weight full annual return:", seg_best["full"]["equal_weight"]["annual_return"])
    print("SPY full annual return:", seg_best["full"]["spy"]["annual_return"])
    if not tuning_df.empty:
        print("Best tuned config:", asdict(best_cfg))


if __name__ == "__main__":
    main()
