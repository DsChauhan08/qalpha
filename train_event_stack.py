"""Combine event sleeves and meta-ensemble into a benchmark-aware event stack."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from quantum_alpha.backtesting.event_sleeve_tools import (
    enrich_summary_metrics,
    validate_viewer_bundle,
    write_json,
    write_viewer_bundle,
)
from quantum_alpha.backtesting.sleeve_metrics import run_mcpt_on_returns


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


def train_event_stack(
    *,
    event_cross_daily_returns: str | Path,
    event_rv_daily_returns: str | Path,
    meta_daily_returns: str | Path | None,
    output_dir: str | Path,
) -> Dict[str, object]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sleeves = {
        "event_cross_sectional": _load_returns_csv(event_cross_daily_returns),
        "event_rv": _load_returns_csv(event_rv_daily_returns),
    }
    benchmark_source = pd.read_csv(event_cross_daily_returns)
    date_col = "date" if "date" in benchmark_source.columns else "timestamp"
    bidx = pd.to_datetime(benchmark_source[date_col])
    spy = pd.Series(pd.to_numeric(benchmark_source.get("benchmark_return", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float), index=bidx)
    equal_weight = pd.Series(pd.to_numeric(benchmark_source.get("equal_weight_return", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float), index=bidx)

    if meta_daily_returns:
        meta_df = pd.read_csv(meta_daily_returns)
        meta_date_col = "date" if "date" in meta_df.columns else "timestamp"
        meta_idx = pd.to_datetime(meta_df[meta_date_col])
        meta_col = "model" if "model" in meta_df.columns else "strategy_return"
        sleeves["meta_ensemble"] = pd.Series(pd.to_numeric(meta_df[meta_col], errors="coerce").fillna(0.0).to_numpy(dtype=float), index=meta_idx)
        if "spy" in meta_df.columns:
            spy = pd.Series(pd.to_numeric(meta_df["spy"], errors="coerce").fillna(0.0).to_numpy(dtype=float), index=meta_idx)
        if "equal_weight" in meta_df.columns:
            equal_weight = pd.Series(pd.to_numeric(meta_df["equal_weight"], errors="coerce").fillna(0.0).to_numpy(dtype=float), index=meta_idx)

    sleeve_df = pd.concat(sleeves, axis=1).fillna(0.0).sort_index()
    blended, weights = _build_weighted_returns(sleeve_df)
    spy = spy.reindex(blended.index).fillna(0.0)
    equal_weight = equal_weight.reindex(blended.index).fillna(0.0)

    metrics = enrich_summary_metrics(
        strategy_returns=blended,
        benchmark_returns=spy,
        equal_weight_returns=equal_weight,
        periods_per_year=252.0,
    )
    metrics["mcpt_p_value"] = float(run_mcpt_on_returns(blended).get("p_value", 1.0))

    beta = float(metrics.get("beta", 0.0))
    hedged_returns = blended - beta * spy
    viewer_dir = out_dir / "viewer"
    viewer_artifacts = write_viewer_bundle(
        output_dir=viewer_dir,
        strategy_returns=blended,
        benchmark_returns=spy,
        equal_weight_returns=equal_weight,
        hedged_returns=hedged_returns,
    )

    daily_returns_path = out_dir / "daily_returns.csv"
    pd.DataFrame(
        {
            "date": blended.index.astype(str),
            "strategy_return": blended.values,
            "benchmark_return": spy.values,
            "equal_weight_return": equal_weight.values,
            "hedged_return": hedged_returns.values,
        }
    ).to_csv(daily_returns_path, index=False)
    weights_path = out_dir / "weights.csv"
    weights.index.name = "date"
    weights.to_csv(weights_path)

    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "event_stack",
        "metrics": metrics,
        "sleeves": list(sleeve_df.columns),
        "artifacts": {
            "daily_returns_csv": str(daily_returns_path),
            "weights_csv": str(weights_path),
            **viewer_artifacts,
        },
    }
    summary_path = out_dir / "summary.json"
    write_json(summary_path, summary)
    validation = validate_viewer_bundle(viewer_dir)
    if not bool(validation.get("valid")):
        raise RuntimeError(f"Event stack viewer artifact validation failed: {validation}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the event-driven stack")
    parser.add_argument("--event-cross-daily-returns", type=str, required=True)
    parser.add_argument("--event-rv-daily-returns", type=str, required=True)
    parser.add_argument("--meta-daily-returns", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "artifacts" / "event_stack"),
    )
    args = parser.parse_args()

    summary = train_event_stack(
        event_cross_daily_returns=args.event_cross_daily_returns,
        event_rv_daily_returns=args.event_rv_daily_returns,
        meta_daily_returns=args.meta_daily_returns,
        output_dir=args.output_dir,
    )
    print(json.dumps({"passed": True, "summary_json": str(Path(args.output_dir) / "summary.json"), "metrics": summary["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
