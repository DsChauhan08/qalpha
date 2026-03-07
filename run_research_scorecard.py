"""Compare baseline and state-graph event models on a shared research spine."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from quantum_alpha.pipeline_runner import RESEARCH_GATE_RULES
from quantum_alpha.research_spine import (
    RESEARCH_LEDGER_FILENAME,
    build_or_load_research_spine,
    load_research_ledger,
)
from quantum_alpha.train_event_cross_sectional import train_event_cross_sectional
from quantum_alpha.train_event_rv import train_event_rv
from quantum_alpha.train_event_stack import train_event_stack

SUPPORTED_STRATEGIES = ["event_cross_sectional", "event_rv", "event_stack"]


def _normalize_strategies(value: str | None) -> List[str]:
    if not value:
        return list(SUPPORTED_STRATEGIES)
    out = [x.strip().lower() for x in value.split(",") if x.strip()]
    invalid = [x for x in out if x not in SUPPORTED_STRATEGIES]
    if invalid:
        raise ValueError(f"Unsupported research scorecard strategies: {', '.join(invalid)}")
    return out


def _normalize_modes(value: str | None) -> List[str]:
    if not value:
        return ["baseline", "state_graph"]
    out = [x.strip().lower() for x in value.split(",") if x.strip()]
    invalid = [x for x in out if x not in {"baseline", "state_graph"}]
    if invalid:
        raise ValueError(f"Unsupported modes: {', '.join(invalid)}")
    return out


def _evaluate_research_gate(strategy: str, metrics: Dict[str, object]) -> List[str]:
    rules = RESEARCH_GATE_RULES.get(strategy, {})
    fail_reasons: List[str] = []
    annual_return = float(metrics.get("annual_return", 0.0))
    sharpe = float(metrics.get("sharpe", 0.0))
    beta = abs(float(metrics.get("beta", 0.0)))
    max_drawdown = abs(float(metrics.get("max_drawdown", 0.0)))
    rolling_positive = float(metrics.get("rolling_positive_ratio_3m", 0.0))
    excess_spy = float(metrics.get("annual_excess_vs_spy", 0.0))
    excess_ew = float(metrics.get("annual_excess_vs_equal_weight", 0.0))

    if annual_return < float(rules.get("annual_return_min", float("-inf"))):
        fail_reasons.append("annual_return_below_research_gate")
    if sharpe < float(rules.get("sharpe_min", float("-inf"))):
        fail_reasons.append("sharpe_below_research_gate")
    if beta > float(rules.get("beta_abs_max", float("inf"))):
        fail_reasons.append("beta_above_research_gate")
    if max_drawdown > float(rules.get("max_drawdown_abs", float("inf"))):
        fail_reasons.append("max_drawdown_above_research_gate")
    if rolling_positive < float(rules.get("rolling_positive_ratio_min", float("-inf"))):
        fail_reasons.append("rolling_positive_ratio_below_research_gate")
    if excess_spy < float(rules.get("excess_vs_spy_min", float("-inf"))):
        fail_reasons.append("excess_vs_spy_below_research_gate")
    if excess_ew < float(rules.get("excess_vs_equal_weight_min", float("-inf"))):
        fail_reasons.append("excess_vs_equal_weight_below_research_gate")
    return fail_reasons


def run_research_scorecard(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    strategies = _normalize_strategies(args.strategies)
    modes = _normalize_modes(args.mode)

    spine = build_or_load_research_spine(
        spine_dir=output_dir / "research_spine",
        symbols=[s.strip().upper() for s in args.event_symbols.split(",") if s.strip()],
        universe_size=int(args.event_universe_size),
        use_fixture=bool(args.event_use_fixture),
        fixture_days=int(args.fixture_days),
        seed=int(args.seed),
    )
    ledger_path = spine.spine_dir / RESEARCH_LEDGER_FILENAME

    records: List[Dict[str, object]] = []
    mode_outputs: Dict[str, Dict[str, object]] = {}
    for mode in modes:
        mode_dir = output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        mode_outputs[mode] = {}

        if "event_cross_sectional" in strategies:
            started = time.time()
            summary = train_event_cross_sectional(
                symbols=[s.strip().upper() for s in args.event_symbols.split(",") if s.strip()],
                output_dir=mode_dir / "event_cross_sectional",
                checkpoint_dir=mode_dir / "models" / "event_cross_sectional",
                universe_size=int(args.event_universe_size),
                use_fixture=bool(args.event_use_fixture),
                fixture_days=int(args.fixture_days),
                seed=int(args.seed),
                quick=bool(args.quick),
                model_family=mode,
                research_spine_dir=spine.spine_dir,
                research_ledger_path=ledger_path,
            )
            fail_reasons = _evaluate_research_gate("event_cross_sectional", summary["metrics"])
            records.append(
                {
                    "strategy": "event_cross_sectional",
                    "mode": mode,
                    "runtime_sec": round(time.time() - started, 3),
                    "feature_families": summary.get("feature_families", []),
                    "feature_count": int(summary.get("feature_count", 0)),
                    "metrics": summary["metrics"],
                    "uncertainty_veto_rate": float(summary["metrics"].get("uncertainty_veto_rate", 0.0)),
                    "passed": len(fail_reasons) == 0,
                    "fail_reasons": fail_reasons,
                    "summary_json": str(Path(mode_dir / "event_cross_sectional" / "summary.json")),
                }
            )
            mode_outputs[mode]["cross"] = mode_dir / "event_cross_sectional" / "daily_returns.csv"

        if "event_rv" in strategies:
            started = time.time()
            summary = train_event_rv(
                symbols=[s.strip().upper() for s in args.event_symbols.split(",") if s.strip()],
                output_dir=mode_dir / "event_rv",
                checkpoint_dir=mode_dir / "models" / "event_rv",
                universe_size=int(args.event_universe_size),
                use_fixture=bool(args.event_use_fixture),
                fixture_days=int(args.fixture_days),
                seed=int(args.seed),
                quick=bool(args.quick),
                model_family=mode,
                research_spine_dir=spine.spine_dir,
                research_ledger_path=ledger_path,
            )
            fail_reasons = _evaluate_research_gate("event_rv", summary["metrics"])
            records.append(
                {
                    "strategy": "event_rv",
                    "mode": mode,
                    "runtime_sec": round(time.time() - started, 3),
                    "feature_families": summary.get("feature_families", []),
                    "feature_count": int(summary.get("feature_count", 0)),
                    "metrics": summary["metrics"],
                    "uncertainty_veto_rate": float(summary["metrics"].get("uncertainty_veto_rate", 0.0)),
                    "passed": len(fail_reasons) == 0,
                    "fail_reasons": fail_reasons,
                    "summary_json": str(Path(mode_dir / "event_rv" / "summary.json")),
                }
            )
            mode_outputs[mode]["rv"] = mode_dir / "event_rv" / "daily_returns.csv"

        if "event_stack" in strategies:
            cross_path = mode_outputs[mode].get("cross")
            rv_path = mode_outputs[mode].get("rv")
            if cross_path and rv_path:
                started = time.time()
                summary = train_event_stack(
                    event_cross_daily_returns=cross_path,
                    event_rv_daily_returns=rv_path,
                    meta_daily_returns=None,
                    output_dir=mode_dir / "event_stack",
                    model_family=mode,
                )
                fail_reasons = _evaluate_research_gate("event_stack", summary["metrics"])
                records.append(
                    {
                        "strategy": "event_stack",
                        "mode": mode,
                        "runtime_sec": round(time.time() - started, 3),
                        "feature_families": [],
                        "feature_count": 0,
                        "metrics": summary["metrics"],
                        "uncertainty_veto_rate": 0.0,
                        "passed": len(fail_reasons) == 0,
                        "fail_reasons": fail_reasons,
                        "summary_json": str(Path(mode_dir / "event_stack" / "summary.json")),
                    }
                )

    deltas: List[Dict[str, object]] = []
    for strategy in strategies:
        baseline = next((r for r in records if r["strategy"] == strategy and r["mode"] == "baseline"), None)
        upgraded = next((r for r in records if r["strategy"] == strategy and r["mode"] == "state_graph"), None)
        if baseline is None or upgraded is None:
            continue
        deltas.append(
            {
                "strategy": strategy,
                "annual_return_delta": float(upgraded["metrics"].get("annual_return", 0.0) - baseline["metrics"].get("annual_return", 0.0)),
                "sharpe_delta": float(upgraded["metrics"].get("sharpe", 0.0) - baseline["metrics"].get("sharpe", 0.0)),
                "beta_delta": float(upgraded["metrics"].get("beta", 0.0) - baseline["metrics"].get("beta", 0.0)),
                "turnover_delta": float(upgraded["metrics"].get("turnover_mean", 0.0) - baseline["metrics"].get("turnover_mean", 0.0)),
                "runtime_delta_sec": float(upgraded["runtime_sec"] - baseline["runtime_sec"]),
            }
        )

    passed = all(bool(record.get("passed")) for record in records)
    manifest = {
        "run_at_utc": pd.Timestamp.utcnow().isoformat(),
        "strategies": strategies,
        "modes": modes,
        "research_spine": {
            "spine_dir": str(spine.spine_dir),
            "panel_path": str(spine.panel_path),
            "metadata_path": str(spine.metadata_path),
            "dataset_hash": str(spine.dataset_hash),
        },
        "records": records,
        "deltas": deltas,
        "ledger_path": str(ledger_path),
        "ledger_records": load_research_ledger(ledger_path),
        "passed": passed,
    }
    manifest_path = output_dir / "comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline vs state-graph research scorecard")
    parser.add_argument("--strategies", type=str, default="event_cross_sectional,event_rv,event_stack")
    parser.add_argument("--mode", type=str, default="baseline,state_graph")
    parser.add_argument("--event-symbols", type=str, default="SPY,AAPL,MSFT,NVDA,AMZN,XOM")
    parser.add_argument("--event-universe-size", type=int, default=200)
    parser.add_argument("--event-use-fixture", action="store_true")
    parser.add_argument("--fixture-days", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "artifacts" / "research_scorecard"),
    )
    args = parser.parse_args()
    manifest = run_research_scorecard(args)
    print(json.dumps({"passed": manifest["passed"], "manifest": str(Path(args.output_dir) / "comparison_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
