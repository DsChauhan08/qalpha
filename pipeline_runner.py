"""
Unified pipeline runner for research, validation, and promotion checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from quantum_alpha.execution.live_graph_dashboard import (
    load_equity_curve,
    load_status,
    load_trades,
)
from quantum_alpha.visualization.meta_ensemble_video import validate_output_dir as validate_video_output_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = Path(__file__).resolve().parent / "config"
DEFAULT_STRATEGIES = ["enhanced", "meta_ensemble", "news_lstm", "intraday"]
DEFAULT_META_HORIZONS = [5, 21]
DEFAULT_PROMOTION_SYMBOLS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA"]
META_PROMOTION_RULES = {
    "max_drawdown_abs": 0.25,
    "sharpe_min": 0.75,
}
NEWS_PROMOTION_RULES = {
    "max_drawdown_abs": 0.25,
    "sharpe_min": 0.75,
}

SMOKE_TESTS = {
    "enhanced": [
        "quantum_alpha/tests/test_run_paper_smoke.py",
        "quantum_alpha/tests/test_validation_objectives.py",
        "quantum_alpha/tests/test_hardening_protocol.py",
    ],
    "meta_ensemble": [
        "quantum_alpha/tests/test_regime_path_shape.py",
        "quantum_alpha/tests/test_meta_ensemble_feature_sets.py",
        "quantum_alpha/tests/test_meta_ensemble_portfolio_eval.py",
        "quantum_alpha/tests/test_backtest_clean_predictions.py",
    ],
    "news_lstm": [
        "quantum_alpha/tests/test_news_lstm_pipeline.py",
        "quantum_alpha/tests/test_realtime_paper_news_mode.py",
    ],
    "intraday": [
        "quantum_alpha/tests/test_live_paper_core.py",
        "quantum_alpha/tests/test_market_data_intraday_period.py",
    ],
    "dashboard": [
        "quantum_alpha/tests/test_live_graph_dashboard.py",
        "quantum_alpha/tests/test_realtime_paper_meta_blend.py",
    ],
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_config_hashes() -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    if not CONFIG_ROOT.exists():
        return hashes
    for path in sorted(CONFIG_ROOT.glob("*.yaml")):
        hashes[str(path.relative_to(PROJECT_ROOT))] = _sha256_file(path)
    return hashes


def _normalize_strategies(value: str | None) -> List[str]:
    if not value:
        return list(DEFAULT_STRATEGIES)
    out = [x.strip().lower() for x in value.split(",") if x.strip()]
    allowed = set(DEFAULT_STRATEGIES)
    invalid = [x for x in out if x not in allowed]
    if invalid:
        raise ValueError(f"Unsupported strategies: {', '.join(invalid)}")
    return out


def _parse_csv_ints(value: str | None, default: Iterable[int]) -> List[int]:
    if not value:
        return [int(x) for x in default]
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _normalize_symbols(value: str | None, default: Iterable[str]) -> List[str]:
    if not value:
        out = [str(x).upper() for x in default]
    else:
        out = [x.strip().upper() for x in value.split(",") if x.strip()]
    if not out:
        raise ValueError("Universe selection is empty")
    return out


def _run_command(
    cmd: List[str],
    *,
    cwd: Path = PROJECT_ROOT,
    timeout_s: int = 3600,
    env: Dict[str, str] | None = None,
) -> Dict[str, object]:
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    ended = time.time()
    return {
        "command": cmd,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout[-12000:],
        "stderr": proc.stderr[-12000:],
        "duration_sec": round(ended - started, 3),
    }


def _record_command(
    strategy: str,
    phase: str,
    result: Dict[str, object],
    *,
    artifacts: Dict[str, str] | None = None,
    metrics: Dict[str, object] | None = None,
    fail_reasons: List[str] | None = None,
) -> Dict[str, object]:
    failed = int(result.get("returncode", 1)) != 0
    reasons = list(fail_reasons or [])
    if failed and not reasons:
        reasons.append("nonzero_returncode")
    return {
        "strategy": strategy,
        "phase": phase,
        "command": list(result.get("command", [])),
        "returncode": int(result.get("returncode", 1)),
        "duration_sec": float(result.get("duration_sec", 0.0)),
        "stdout_tail": str(result.get("stdout", "")),
        "stderr_tail": str(result.get("stderr", "")),
        "artifacts": artifacts or {},
        "metrics": metrics or {},
        "passed": not failed and not reasons,
        "fail_reasons": reasons,
    }


def _discover_news_checkpoint() -> str | None:
    ckpt_dir = PROJECT_ROOT / "quantum_alpha" / "models" / "checkpoints" / "news_lstm"
    if not ckpt_dir.exists():
        return None
    candidates = sorted(ckpt_dir.glob("*.pt"))
    if not candidates:
        return None
    return candidates[-1].stem


def _discover_intraday_checkpoint() -> Tuple[str, Path] | None:
    root = PROJECT_ROOT / "quantum_alpha" / "models" / "intraday_checkpoints"
    if not root.exists():
        return None
    subdirs = [p for p in sorted(root.iterdir()) if p.is_dir() and (p / "latest_intraday.json").exists()]
    if not subdirs:
        return None
    symbol = subdirs[0].name.upper()
    return symbol, subdirs[0]


def _validate_required_files(paths: Dict[str, Path]) -> Tuple[bool, List[str]]:
    missing = [name for name, path in paths.items() if not path.exists()]
    return len(missing) == 0, missing


def _validate_live_session(session_dir: Path) -> Dict[str, object]:
    required = {
        "live_status": session_dir / "live_status.json",
        "equity_curve": session_dir / "equity_curve.csv",
        "live_trades": session_dir / "live_trades.jsonl",
    }
    ok, missing = _validate_required_files(required)
    if not ok:
        return {"valid": False, "fail_reasons": [f"missing_{m}" for m in missing]}

    status = load_status(session_dir)
    _ = load_equity_curve(session_dir)
    _ = load_trades(session_dir)
    pipeline_health = str(status.get("pipeline_health", "ok")).lower()
    fail_reasons: List[str] = []
    if pipeline_health in {"degraded", "failed"}:
        fail_reasons.append(f"pipeline_health_{pipeline_health}")
    if int(status.get("symbols_tracked", 0) or 0) <= 0 and status.get("symbols"):
        fail_reasons.append("empty_universe")
    return {
        "valid": len(fail_reasons) == 0,
        "status": status,
        "fail_reasons": fail_reasons,
    }


def _meta_checkpoint_artifacts(checkpoint_dir: Path, prediction_file: str) -> Dict[str, Path]:
    return {
        "model": checkpoint_dir / "meta_ensemble_hybrid_math_model.pkl",
        "results": checkpoint_dir / "meta_ensemble_hybrid_math_walk_forward_results.json",
        "predictions": checkpoint_dir / prediction_file,
    }


def _meta_candidate_verdict(robust_summary: Dict[str, object]) -> Tuple[bool, List[str], Dict[str, object]]:
    segments = robust_summary.get("segments_best", {}) or {}
    full = segments.get("full", {}) or {}
    model = full.get("model", {}) or {}
    equal_weight = full.get("equal_weight", {}) or {}
    spy = full.get("spy", {}) or {}
    quant = full.get("quant_composite", {}) or {}

    fail_reasons: List[str] = []
    if float(model.get("annual_return", 0.0)) <= float(equal_weight.get("annual_return", 0.0)):
        fail_reasons.append("annual_return_not_above_equal_weight")
    if float(model.get("annual_return", 0.0)) <= float(spy.get("annual_return", 0.0)):
        fail_reasons.append("annual_return_not_above_spy")
    if float(model.get("annual_return", 0.0)) <= float(quant.get("annual_return", 0.0)):
        fail_reasons.append("annual_return_not_above_quant_composite")
    if abs(float(model.get("max_drawdown", 0.0))) > META_PROMOTION_RULES["max_drawdown_abs"]:
        fail_reasons.append("max_drawdown_above_cap")
    if float(model.get("sharpe", 0.0)) < META_PROMOTION_RULES["sharpe_min"]:
        fail_reasons.append("sharpe_below_floor")

    metrics = {
        "annual_return": float(model.get("annual_return", 0.0)),
        "equal_weight_annual_return": float(equal_weight.get("annual_return", 0.0)),
        "spy_annual_return": float(spy.get("annual_return", 0.0)),
        "quant_annual_return": float(quant.get("annual_return", 0.0)),
        "sharpe": float(model.get("sharpe", 0.0)),
        "max_drawdown": float(model.get("max_drawdown", 0.0)),
    }
    return len(fail_reasons) == 0, fail_reasons, metrics


def _news_candidate_verdict(payload: Dict[str, object]) -> Tuple[bool, List[str], Dict[str, object]]:
    metrics = (payload.get("metrics") or {}) if isinstance(payload, dict) else {}
    fail_reasons: List[str] = []
    if float(metrics.get("total_return", 0.0)) <= 0.0:
        fail_reasons.append("nonpositive_total_return")
    if float(metrics.get("sharpe_ratio", 0.0)) < NEWS_PROMOTION_RULES["sharpe_min"]:
        fail_reasons.append("sharpe_below_floor")
    if abs(float(metrics.get("max_drawdown", 0.0))) > NEWS_PROMOTION_RULES["max_drawdown_abs"]:
        fail_reasons.append("max_drawdown_above_cap")
    if "mcpt_p_value" in metrics and float(metrics.get("mcpt_p_value", 1.0)) >= 0.05:
        fail_reasons.append("mcpt_p_value_not_significant")
    return len(fail_reasons) == 0, fail_reasons, metrics


def _intraday_candidate_verdict(payload: Dict[str, object]) -> Tuple[bool, List[str], Dict[str, object]]:
    stats = (payload.get("stats") or {}) if isinstance(payload, dict) else {}
    fail_reasons: List[str] = []
    if float(stats.get("strategy_return", 0.0)) <= float(stats.get("benchmark_return", 0.0)):
        fail_reasons.append("strategy_return_not_above_benchmark")
    return len(fail_reasons) == 0, fail_reasons, stats


def _run_smoke_suite(
    strategies: List[str],
    output_dir: Path,
    timeout_s: int,
) -> Tuple[List[Dict[str, object]], bool]:
    records: List[Dict[str, object]] = []
    for strategy in strategies:
        tests = list(SMOKE_TESTS.get(strategy, []))
        if strategy == "meta_ensemble":
            tests.extend(SMOKE_TESTS["dashboard"])
        if not tests:
            continue
        cmd = [sys.executable, "-m", "pytest", "-q", *tests]
        result = _run_command(cmd, timeout_s=timeout_s)
        records.append(_record_command(strategy, "smoke", result))
    passed = all(bool(r.get("passed")) for r in records)
    _write_json(output_dir / "smoke_manifest.json", {"records": records, "passed": passed})
    return records, passed


def _run_meta_candidate(
    args: argparse.Namespace,
    output_dir: Path,
    timeout_s: int,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    horizons = _parse_csv_ints(args.meta_horizons, DEFAULT_META_HORIZONS)
    for horizon in horizons:
        tag = f"ps_h{int(horizon)}d"
        checkpoint_dir = PROJECT_ROOT / "quantum_alpha" / "models" / "checkpoints" / f"meta_ensemble_{tag}"
        prediction_file = "meta_ensemble_hybrid_math_walk_forward_predictions.pkl"
        horizon_dir = output_dir / f"meta_ensemble_h{int(horizon)}d"
        horizon_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = [
            sys.executable,
            "-m",
            "quantum_alpha.meta_ensemble",
            "--phase",
            "all",
            "--feature-set",
            "hybrid_math",
            "--forward-period",
            str(int(horizon)),
            "--run-tag",
            tag,
            "--n-symbols",
            str(int(args.meta_n_symbols)),
        ]
        if args.quick:
            train_cmd.append("--quick")
        if args.n_folds is not None:
            train_cmd.extend(["--n-folds", str(int(args.n_folds))])
        train_result = _run_command(train_cmd, timeout_s=timeout_s)
        artifacts = {k: str(v) for k, v in _meta_checkpoint_artifacts(checkpoint_dir, prediction_file).items()}
        train_record = _record_command("meta_ensemble", f"train_h{int(horizon)}d", train_result, artifacts=artifacts)
        records.append(train_record)
        if not train_record["passed"]:
            continue

        model_artifacts = _meta_checkpoint_artifacts(checkpoint_dir, prediction_file)
        ok, missing = _validate_required_files(model_artifacts)
        if not ok:
            train_record["passed"] = False
            train_record["fail_reasons"] = [f"missing_{m}" for m in missing]
            continue

        train_summary = _load_json(model_artifacts["results"])
        recommended = ((train_summary.get("summary") or {}).get("recommended_config") or {})
        signal_threshold = float(recommended.get("min_up_probability", 0.55))
        hold_days = int(recommended.get("hold_days", max(5, int(horizon))))
        top_k = int(recommended.get("top_k", 5))
        max_positions = max(1, int(top_k))

        robust_dir = horizon_dir / "robustness"
        robust_cmd = [
            sys.executable,
            "-m",
            "quantum_alpha.backtesting.robustness_suite",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--prediction-file",
            prediction_file,
            "--signal-threshold",
            str(signal_threshold),
            "--commission-bps",
            "10",
            "--hold-days",
            str(hold_days),
            "--top-k",
            str(top_k),
            "--max-positions",
            str(max_positions),
            "--no-tune",
            "--output-dir",
            str(robust_dir),
        ]
        robust_result = _run_command(robust_cmd, timeout_s=timeout_s)
        robust_summary_path = robust_dir / "summary.json"
        robust_record = _record_command(
            "meta_ensemble",
            f"robustness_h{int(horizon)}d",
            robust_result,
            artifacts={"summary_json": str(robust_summary_path)},
        )
        if robust_record["passed"] and robust_summary_path.exists():
            robust_summary = _load_json(robust_summary_path)
            passed, fail_reasons, metrics = _meta_candidate_verdict(robust_summary)
            robust_record["metrics"] = metrics
            robust_record["passed"] = passed
            robust_record["fail_reasons"] = fail_reasons
        records.append(robust_record)

        stress_dir = horizon_dir / "synthetic_stress"
        stress_cmd = [
            sys.executable,
            "-m",
            "quantum_alpha.backtesting.synthetic_stress_suite",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--prediction-file",
            prediction_file,
            "--signal-threshold",
            str(signal_threshold),
            "--commission-bps",
            "10",
            "--hold-days",
            str(hold_days),
            "--top-k",
            str(top_k),
            "--max-positions",
            str(max_positions),
            "--output-dir",
            str(stress_dir),
        ]
        stress_result = _run_command(stress_cmd, timeout_s=timeout_s)
        stress_record = _record_command(
            "meta_ensemble",
            f"synthetic_h{int(horizon)}d",
            stress_result,
            artifacts={"summary_json": str(stress_dir / "summary.json")},
        )
        records.append(stress_record)

        video_dir = horizon_dir / "viewer"
        video_cmd = [
            sys.executable,
            "-m",
            "quantum_alpha.visualization.meta_ensemble_video",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--prediction-file",
            prediction_file,
            "--signal-threshold",
            str(signal_threshold),
            "--commission-bps",
            "10",
            "--hold-days",
            str(hold_days),
            "--top-k",
            str(top_k),
            "--max-positions",
            str(max_positions),
            "--long-only",
            "--output-dir",
            str(video_dir),
        ]
        video_result = _run_command(video_cmd, timeout_s=timeout_s)
        validation = validate_video_output_dir(video_dir)
        video_record = _record_command(
            "meta_ensemble",
            f"viewer_h{int(horizon)}d",
            video_result,
            artifacts={k: str(v) for k, v in validation.get("required_files", {}).items()},
            metrics={
                "curve_rows": int(validation.get("curve_rows", 0)),
            },
            fail_reasons=[] if validation.get("valid") else [f"missing_{m}" for m in validation.get("missing", [])],
        )
        video_record["passed"] = bool(video_record["passed"] and validation.get("valid"))
        records.append(video_record)

    return records


def _run_enhanced_candidate(args: argparse.Namespace, output_dir: Path, timeout_s: int) -> List[Dict[str, object]]:
    symbols = _normalize_symbols(args.promotion_symbols, DEFAULT_PROMOTION_SYMBOLS[:3])
    out_path = output_dir / "enhanced_hardening.json"
    cmd = [
        sys.executable,
        "-m",
        "quantum_alpha.backtesting.hardening_protocol",
        "--symbols",
        ",".join(symbols[:3]),
        "--strategy",
        "enhanced",
        "--output-path",
        str(out_path),
    ]
    result = _run_command(cmd, timeout_s=timeout_s)
    record = _record_command(
        "enhanced",
        "hardening",
        result,
        artifacts={"summary_json": str(out_path)},
    )
    if record["passed"] and out_path.exists():
        summary = _load_json(out_path)
        record["metrics"] = {
            "promotion_passed": bool(summary.get("promotion_passed", False)),
        }
        record["passed"] = bool(summary.get("promotion_passed", False))
        if not record["passed"]:
            record["fail_reasons"] = list(summary.get("promotion_fail_reasons", []))
    return [record]


def _run_news_candidate(args: argparse.Namespace, output_dir: Path, timeout_s: int) -> List[Dict[str, object]]:
    checkpoint = args.news_checkpoint or _discover_news_checkpoint()
    if not checkpoint:
        return [
            {
                "strategy": "news_lstm",
                "phase": "backtest",
                "command": [],
                "returncode": 1,
                "duration_sec": 0.0,
                "stdout_tail": "",
                "stderr_tail": "",
                "artifacts": {},
                "metrics": {},
                "passed": False,
                "fail_reasons": ["missing_checkpoint"],
            }
        ]
    out_path = output_dir / "news_lstm_summary.json"
    cmd = [
        sys.executable,
        "-m",
        "quantum_alpha.backtest_news_lstm",
        "--symbols",
        "SPY",
        "QQQ",
        "AAPL",
        "--years",
        "2",
        "--checkpoint",
        checkpoint,
        "--validate",
        "--output-json",
        str(out_path),
    ]
    result = _run_command(cmd, timeout_s=timeout_s)
    record = _record_command(
        "news_lstm",
        "backtest",
        result,
        artifacts={"summary_json": str(out_path)},
    )
    if record["passed"] and out_path.exists():
        payload = _load_json(out_path)
        passed, fail_reasons, metrics = _news_candidate_verdict(payload)
        record["metrics"] = metrics
        record["passed"] = passed
        record["fail_reasons"] = fail_reasons
    return [record]


def _run_intraday_candidate(args: argparse.Namespace, output_dir: Path, timeout_s: int) -> List[Dict[str, object]]:
    discovered = _discover_intraday_checkpoint()
    if discovered is None:
        return [
            {
                "strategy": "intraday",
                "phase": "backtest",
                "command": [],
                "returncode": 1,
                "duration_sec": 0.0,
                "stdout_tail": "",
                "stderr_tail": "",
                "artifacts": {},
                "metrics": {},
                "passed": False,
                "fail_reasons": ["missing_checkpoint"],
            }
        ]
    symbol, ckpt_dir = discovered
    chart_path = output_dir / f"{symbol.lower()}_intraday.png"
    csv_path = output_dir / f"{symbol.lower()}_intraday.csv"
    summary_path = output_dir / f"{symbol.lower()}_intraday_summary.json"
    cmd = [
        sys.executable,
        "-m",
        "quantum_alpha.report_intraday_backtest",
        "--symbol",
        symbol,
        "--checkpoint-dir",
        str(ckpt_dir),
        "--checkpoint-file",
        str(ckpt_dir / "latest_intraday.json"),
        "--output",
        str(chart_path),
        "--csv",
        str(csv_path),
        "--summary-json",
        str(summary_path),
    ]
    result = _run_command(cmd, timeout_s=timeout_s)
    record = _record_command(
        "intraday",
        "backtest",
        result,
        artifacts={
            "chart_png": str(chart_path),
            "csv_path": str(csv_path),
            "summary_json": str(summary_path),
        },
    )
    if record["passed"] and summary_path.exists():
        payload = _load_json(summary_path)
        passed, fail_reasons, metrics = _intraday_candidate_verdict(payload)
        record["metrics"] = metrics
        record["passed"] = passed
        record["fail_reasons"] = fail_reasons
    return [record]


def _run_realtime_probe(args: argparse.Namespace, output_dir: Path, timeout_s: int) -> List[Dict[str, object]]:
    if not args.live_probe:
        return []
    symbols = _normalize_symbols(args.live_probe_symbols, DEFAULT_PROMOTION_SYMBOLS[:3])
    probe_root = output_dir / "live_probe_artifacts"
    probe_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "quantum_alpha.execution.realtime_paper",
        "--symbols",
        *symbols,
        "--interval",
        "5m",
        "--duration-minutes",
        "1",
        "--poll-seconds",
        "60",
        "--lookback-days",
        "10",
        "--capital",
        "100000",
        "--output-dir",
        str(probe_root),
        "--strategy-type",
        "meta_blend_hybrid",
        "--ab-group",
        "B",
    ]
    result = _run_command(cmd, timeout_s=timeout_s)
    latest_sessions = sorted(probe_root.glob("realtime_paper_*"))
    session_dir = latest_sessions[-1] if latest_sessions else probe_root
    validation = _validate_live_session(session_dir) if session_dir.exists() else {"valid": False, "fail_reasons": ["missing_session_dir"]}
    record = _record_command(
        "meta_ensemble",
        "realtime_probe",
        result,
        artifacts={"session_dir": str(session_dir)},
        metrics={"pipeline_health": (validation.get("status") or {}).get("pipeline_health")},
        fail_reasons=list(validation.get("fail_reasons", [])),
    )
    record["passed"] = bool(record["passed"] and validation.get("valid"))
    return [record]


def _aggregate_verdict(records: List[Dict[str, object]]) -> Tuple[bool, List[str]]:
    fail_reasons: List[str] = []
    for record in records:
        if not bool(record.get("passed")):
            fail_reasons.extend(
                [f"{record.get('strategy')}:{record.get('phase')}:{r}" for r in record.get("fail_reasons", [])]
            )
    return len(fail_reasons) == 0, fail_reasons


def run_pipeline_suite(args: argparse.Namespace) -> Dict[str, object]:
    strategies = _normalize_strategies(args.strategies)
    output_dir = Path(args.output_dir or (PROJECT_ROOT / "artifacts" / f"pipeline_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "run_at_utc": _now_utc(),
        "suite": args.suite,
        "strategies": strategies,
        "offline": bool(args.offline),
        "live_probe": bool(args.live_probe),
        "config_hashes": _collect_config_hashes(),
        "records": [],
    }

    records: List[Dict[str, object]] = []
    timeout_s = int(args.timeout_seconds)

    if args.suite == "smoke":
        smoke_records, _ = _run_smoke_suite(strategies, output_dir, timeout_s)
        records.extend(smoke_records)
    else:
        if "enhanced" in strategies:
            records.extend(_run_enhanced_candidate(args, output_dir / "enhanced", timeout_s))
        if "meta_ensemble" in strategies:
            records.extend(_run_meta_candidate(args, output_dir / "meta_ensemble", timeout_s))
        if "news_lstm" in strategies:
            records.extend(_run_news_candidate(args, output_dir / "news_lstm", timeout_s))
        if "intraday" in strategies:
            records.extend(_run_intraday_candidate(args, output_dir / "intraday", timeout_s))

        if args.suite == "promotion":
            smoke_records, _ = _run_smoke_suite(strategies, output_dir / "smoke", timeout_s)
            records.extend(smoke_records)
            records.extend(_run_realtime_probe(args, output_dir / "promotion_checks", timeout_s))

    passed, fail_reasons = _aggregate_verdict(records)
    verdict = {
        "suite": args.suite,
        "strategies": strategies,
        "passed": passed,
        "fail_reasons": fail_reasons,
        "generated_at_utc": _now_utc(),
    }
    manifest["records"] = records
    manifest["passed"] = passed
    manifest["fail_reasons"] = fail_reasons
    manifest["output_dir"] = str(output_dir)

    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "verdict.json", verdict)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Quantum Alpha pipeline runner")
    parser.add_argument("--suite", choices=["smoke", "candidate", "promotion"], default="smoke")
    parser.add_argument(
        "--strategies",
        type=str,
        default="enhanced,meta_ensemble,news_lstm,intraday",
        help="Comma-separated strategy families to include",
    )
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--live-probe", action="store_true", help="Run a short realtime paper probe in promotion mode")
    parser.add_argument("--live-probe-symbols", type=str, default="SPY,QQQ,AAPL")
    parser.add_argument("--meta-horizons", type=str, default="5,21")
    parser.add_argument("--meta-n-symbols", type=int, default=100)
    parser.add_argument("--promotion-symbols", type=str, default="SPY,QQQ,IWM,AAPL,MSFT,NVDA")
    parser.add_argument("--news-checkpoint", type=str, default=None)
    parser.add_argument("--quick", action="store_true", help="Pass quick flags to supported research jobs")
    parser.add_argument("--n-folds", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    manifest = run_pipeline_suite(args)
    print(json.dumps({"passed": manifest["passed"], "output_dir": manifest["output_dir"]}, indent=2))


if __name__ == "__main__":
    main()
