"""
Unified pipeline runner for research, validation, and promotion checks.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from quantum_alpha.backtesting.event_sleeve_tools import validate_viewer_bundle as validate_event_viewer_bundle
from quantum_alpha.execution.live_graph_dashboard import (
    load_equity_curve,
    load_status,
    load_trades,
)
from quantum_alpha.research_spine import (
    RESEARCH_LEDGER_FILENAME,
    build_or_load_research_spine,
)
from quantum_alpha.visualization.meta_ensemble_video import validate_output_dir as validate_video_output_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = Path(__file__).resolve().parent / "config"
DEFAULT_STRATEGIES = [
    "enhanced",
    "meta_ensemble",
    "news_lstm",
    "intraday",
    "intraday_microstructure",
    "rv_stat_arb",
    "hybrid_stack",
    "event_cross_sectional",
    "event_rv",
    "event_stack",
]
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
INTRADAY_MICRO_RULES = {
    "max_drawdown_abs": 0.12,
    "sharpe_min": 1.25,
    "beta_abs_max": 0.15,
}
RV_STAT_ARB_RULES = {
    "max_drawdown_abs": 0.12,
    "sharpe_min": 1.00,
    "beta_abs_max": 0.10,
}
HYBRID_STACK_RULES = {
    "max_drawdown_abs": 0.20,
    "sharpe_min": 1.10,
    "annual_excess_min": 0.02,
    "mcpt_pvalue_max": 0.05,
    "beat_ratio_min": 0.55,
}
EVENT_CROSS_RULES = {
    "annual_return_min": 0.0,
    "sharpe_min": 1.25,
    "beta_abs_max": 0.10,
    "max_drawdown_abs": 0.12,
    "rolling_positive_ratio_min": 0.55,
}
EVENT_RV_RULES = {
    "annual_return_min": 0.0,
    "sharpe_min": 1.10,
    "beta_abs_max": 0.08,
    "max_drawdown_abs": 0.10,
}
EVENT_STACK_RULES = {
    "annual_return_min": 0.0,
    "sharpe_min": 1.10,
}
RESEARCH_GATE_RULES = {
    "event_cross_sectional": {
        "annual_return_min": 0.0,
        "sharpe_min": 0.90,
        "beta_abs_max": 0.10,
        "rolling_positive_ratio_min": 0.55,
    },
    "event_rv": {
        "annual_return_min": 0.0,
        "sharpe_min": 0.80,
        "beta_abs_max": 0.08,
        "max_drawdown_abs": 0.12,
    },
    "event_stack": {
        "excess_vs_spy_min": 0.0,
        "excess_vs_equal_weight_min": 0.0,
        "sharpe_min": 0.85,
    },
}
SMOKE_SUITE_RUNTIME_TARGET_SEC = 120.0
SMOKE_STRATEGY_RUNTIME_TARGET_SEC = 15.0
QUICK_CANDIDATE_RUNTIME_TARGET_SEC = 20.0 * 60.0

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
    "intraday_microstructure": [
        "quantum_alpha/tests/test_intraday_replay_store.py",
        "quantum_alpha/tests/test_intraday_feature_builder.py",
        "quantum_alpha/tests/test_intraday_microstructure_pipeline.py",
    ],
    "rv_stat_arb": [
        "quantum_alpha/tests/test_rv_stat_arb_pipeline.py",
    ],
    "hybrid_stack": [
        "quantum_alpha/tests/test_hybrid_stack_pipeline.py",
    ],
    "event_cross_sectional": [
        "quantum_alpha/tests/test_event_feature_builder.py",
        "quantum_alpha/tests/test_event_cross_sectional_pipeline.py",
    ],
    "event_rv": [
        "quantum_alpha/tests/test_event_rv_pipeline.py",
    ],
    "event_stack": [
        "quantum_alpha/tests/test_event_stack_pipeline.py",
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


def _prepare_event_research_spine(args: argparse.Namespace, output_dir: Path) -> Dict[str, object]:
    spine = build_or_load_research_spine(
        spine_dir=output_dir / "research_spine",
        symbols=_normalize_symbols(args.event_symbols, ["SPY", "AAPL", "MSFT"]),
        universe_size=int(args.event_universe_size),
        use_fixture=bool(getattr(args, "event_use_fixture", False)),
        fixture_days=int(args.fixture_days),
        seed=42,
    )
    return {
        "spine_dir": str(spine.spine_dir),
        "panel_path": str(spine.panel_path),
        "metadata_path": str(spine.metadata_path),
        "dataset_hash": str(spine.dataset_hash),
        "ledger_path": str(spine.spine_dir / RESEARCH_LEDGER_FILENAME),
    }


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


def _discover_meta_daily_returns(candidate_root: Path, horizon: int) -> Path | None:
    path = candidate_root / "meta_ensemble" / f"meta_ensemble_h{int(horizon)}d" / "robustness" / "daily_returns.csv"
    return path if path.exists() else None


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


def _intraday_micro_candidate_verdict(payload: Dict[str, object]) -> Tuple[bool, List[str], Dict[str, object]]:
    metrics = (payload.get("metrics") or {}) if isinstance(payload, dict) else {}
    quality = (payload.get("data_quality") or {}) if isinstance(payload, dict) else {}
    fail_reasons: List[str] = []
    if float(metrics.get("annual_return", 0.0)) <= 0.0:
        fail_reasons.append("annual_return_nonpositive")
    if float(metrics.get("sharpe", 0.0)) < INTRADAY_MICRO_RULES["sharpe_min"]:
        fail_reasons.append("sharpe_below_floor")
    if abs(float(metrics.get("beta", 0.0))) > INTRADAY_MICRO_RULES["beta_abs_max"]:
        fail_reasons.append("beta_above_cap")
    if abs(float(metrics.get("max_drawdown", 0.0))) > INTRADAY_MICRO_RULES["max_drawdown_abs"]:
        fail_reasons.append("max_drawdown_above_cap")
    if float(quality.get("depth_completeness", 0.0)) < 0.95:
        fail_reasons.append("depth_completeness_below_floor")
    if float(quality.get("median_quote_staleness_ms", 1e9)) > 2000.0:
        fail_reasons.append("quote_staleness_above_cap")
    if float(quality.get("crossed_market_rate", 1.0)) > 0.001:
        fail_reasons.append("crossed_market_rate_above_cap")
    if float(quality.get("negative_spread_rate", 1.0)) > 0.001:
        fail_reasons.append("negative_spread_rate_above_cap")
    return len(fail_reasons) == 0, fail_reasons, {**metrics, **quality}


def _rv_stat_arb_candidate_verdict(payload: Dict[str, object]) -> Tuple[bool, List[str], Dict[str, object]]:
    metrics = (payload.get("metrics") or {}) if isinstance(payload, dict) else {}
    quality = (payload.get("data_quality") or {}) if isinstance(payload, dict) else {}
    fail_reasons: List[str] = []
    if float(metrics.get("annual_return", 0.0)) <= 0.0:
        fail_reasons.append("annual_return_nonpositive")
    if float(metrics.get("sharpe", 0.0)) < RV_STAT_ARB_RULES["sharpe_min"]:
        fail_reasons.append("sharpe_below_floor")
    if abs(float(metrics.get("beta", 0.0))) > RV_STAT_ARB_RULES["beta_abs_max"]:
        fail_reasons.append("beta_above_cap")
    if abs(float(metrics.get("max_drawdown", 0.0))) > RV_STAT_ARB_RULES["max_drawdown_abs"]:
        fail_reasons.append("max_drawdown_above_cap")
    if float(quality.get("depth_completeness", 0.0)) < 0.95:
        fail_reasons.append("depth_completeness_below_floor")
    return len(fail_reasons) == 0, fail_reasons, {**metrics, **quality}


def _hybrid_stack_candidate_verdict(payload: Dict[str, object]) -> Tuple[bool, List[str], Dict[str, object]]:
    metrics = (payload.get("metrics") or {}) if isinstance(payload, dict) else {}
    fail_reasons: List[str] = []
    if float(metrics.get("annual_excess_vs_spy", -1.0)) < HYBRID_STACK_RULES["annual_excess_min"]:
        fail_reasons.append("annual_excess_vs_spy_below_floor")
    if float(metrics.get("annual_excess_vs_equal_weight", -1.0)) < HYBRID_STACK_RULES["annual_excess_min"]:
        fail_reasons.append("annual_excess_vs_equal_weight_below_floor")
    if float(metrics.get("sharpe", 0.0)) < HYBRID_STACK_RULES["sharpe_min"]:
        fail_reasons.append("sharpe_below_floor")
    if abs(float(metrics.get("max_drawdown", 0.0))) > HYBRID_STACK_RULES["max_drawdown_abs"]:
        fail_reasons.append("max_drawdown_above_cap")
    if float(metrics.get("mcpt_p_value", 1.0)) >= HYBRID_STACK_RULES["mcpt_pvalue_max"]:
        fail_reasons.append("mcpt_pvalue_not_significant")
    if float(metrics.get("beat_ratio_spy_3m", 0.0)) <= HYBRID_STACK_RULES["beat_ratio_min"]:
        fail_reasons.append("beat_ratio_spy_below_floor")
    if float(metrics.get("beat_ratio_equal_weight_3m", 0.0)) <= HYBRID_STACK_RULES["beat_ratio_min"]:
        fail_reasons.append("beat_ratio_equal_weight_below_floor")
    return len(fail_reasons) == 0, fail_reasons, metrics


def _event_cross_candidate_verdict(payload: Dict[str, object], *, require_paid: bool) -> Tuple[bool, List[str], Dict[str, object]]:
    metrics = (payload.get("metrics") or {}) if isinstance(payload, dict) else {}
    quality = (payload.get("data_quality") or {}) if isinstance(payload, dict) else {}
    artifacts = (payload.get("artifacts") or {}) if isinstance(payload, dict) else {}
    fail_reasons: List[str] = []
    if float(metrics.get("annual_return", -1.0)) <= EVENT_CROSS_RULES["annual_return_min"]:
        fail_reasons.append("annual_return_nonpositive")
    if float(metrics.get("sharpe", 0.0)) < EVENT_CROSS_RULES["sharpe_min"]:
        fail_reasons.append("sharpe_below_floor")
    if abs(float(metrics.get("beta", 0.0))) > EVENT_CROSS_RULES["beta_abs_max"]:
        fail_reasons.append("beta_above_cap")
    if abs(float(metrics.get("max_drawdown", 0.0))) > EVENT_CROSS_RULES["max_drawdown_abs"]:
        fail_reasons.append("max_drawdown_above_cap")
    if float(metrics.get("rolling_positive_ratio_3m", 0.0)) < EVENT_CROSS_RULES["rolling_positive_ratio_min"]:
        fail_reasons.append("rolling_positive_ratio_below_floor")
    if int(metrics.get("eligible_event_count", 0) or 0) <= 0:
        fail_reasons.append("empty_event_set")
    congress_cov = float((quality.get("coverage_by_domain") or {}).get("congress", 0.0))
    if congress_cov > 0.05 and not bool(quality.get("event_lag_ok", False)):
        fail_reasons.append("event_lag_invalid")
    if require_paid and not bool(quality.get("paid_data_eligible", False)):
        fail_reasons.append("paid_data_ineligible")
    if float((quality.get("coverage_by_domain") or {}).get("earnings", 0.0)) <= 0.0:
        fail_reasons.append("earnings_coverage_missing")
    if float((quality.get("staleness_days") or {}).get("earnings", 9999.0)) > 45.0:
        fail_reasons.append("earnings_staleness_above_cap")
    required_artifacts = {
        "summary_json": payload.get("summary_json"),
        "daily_returns_csv": artifacts.get("daily_returns_csv"),
        "robustness_json": artifacts.get("robustness_json"),
        "synthetic_stress_json": artifacts.get("synthetic_stress_json"),
        "regime_report_json": artifacts.get("regime_report_json"),
    }
    for name, path in required_artifacts.items():
        if not path or not Path(str(path)).exists():
            fail_reasons.append(f"missing_{name}")
    return len(fail_reasons) == 0, fail_reasons, {**metrics, **quality}


def _event_rv_candidate_verdict(payload: Dict[str, object], *, require_paid: bool) -> Tuple[bool, List[str], Dict[str, object]]:
    metrics = (payload.get("metrics") or {}) if isinstance(payload, dict) else {}
    quality = (payload.get("data_quality") or {}) if isinstance(payload, dict) else {}
    artifacts = (payload.get("artifacts") or {}) if isinstance(payload, dict) else {}
    fail_reasons: List[str] = []
    if float(metrics.get("annual_return", -1.0)) <= EVENT_RV_RULES["annual_return_min"]:
        fail_reasons.append("annual_return_nonpositive")
    if float(metrics.get("sharpe", 0.0)) < EVENT_RV_RULES["sharpe_min"]:
        fail_reasons.append("sharpe_below_floor")
    if abs(float(metrics.get("beta", 0.0))) > EVENT_RV_RULES["beta_abs_max"]:
        fail_reasons.append("beta_above_cap")
    if abs(float(metrics.get("max_drawdown", 0.0))) > EVENT_RV_RULES["max_drawdown_abs"]:
        fail_reasons.append("max_drawdown_above_cap")
    if int(metrics.get("eligible_event_count", 0) or 0) <= 0:
        fail_reasons.append("empty_event_set")
    congress_cov = float((quality.get("coverage_by_domain") or {}).get("congress", 0.0))
    if congress_cov > 0.05 and not bool(quality.get("event_lag_ok", False)):
        fail_reasons.append("event_lag_invalid")
    if require_paid and not bool(quality.get("paid_data_eligible", False)):
        fail_reasons.append("paid_data_ineligible")
    required_artifacts = {
        "daily_returns_csv": artifacts.get("daily_returns_csv"),
        "pairs_json": artifacts.get("pairs_json"),
        "robustness_json": artifacts.get("robustness_json"),
        "synthetic_stress_json": artifacts.get("synthetic_stress_json"),
        "regime_report_json": artifacts.get("regime_report_json"),
    }
    for name, path in required_artifacts.items():
        if not path or not Path(str(path)).exists():
            fail_reasons.append(f"missing_{name}")
    return len(fail_reasons) == 0, fail_reasons, {**metrics, **quality}


def _event_stack_candidate_verdict(payload: Dict[str, object]) -> Tuple[bool, List[str], Dict[str, object]]:
    metrics = (payload.get("metrics") or {}) if isinstance(payload, dict) else {}
    fail_reasons: List[str] = []
    if float(metrics.get("annual_return", -1.0)) <= EVENT_STACK_RULES["annual_return_min"]:
        fail_reasons.append("annual_return_nonpositive")
    if float(metrics.get("sharpe", 0.0)) < EVENT_STACK_RULES["sharpe_min"]:
        fail_reasons.append("sharpe_below_floor")
    return len(fail_reasons) == 0, fail_reasons, metrics


def _run_smoke_suite(
    strategies: List[str],
    output_dir: Path,
    timeout_s: int,
) -> Tuple[List[Dict[str, object]], bool]:
    jobs: List[Tuple[str, List[str]]] = []
    for strategy in strategies:
        tests = list(SMOKE_TESTS.get(strategy, []))
        if strategy == "meta_ensemble":
            tests.extend(SMOKE_TESTS["dashboard"])
        if not tests:
            continue
        jobs.append((strategy, [sys.executable, "-m", "pytest", "-q", *tests]))

    if not jobs:
        _write_json(output_dir / "smoke_manifest.json", {"records": [], "passed": True})
        return [], True

    records_by_strategy: Dict[str, Dict[str, object]] = {}
    started = time.time()
    max_workers = max(1, min(len(jobs), 4, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(_run_command, cmd, timeout_s=timeout_s): strategy for strategy, cmd in jobs
        }
        for future in as_completed(future_map):
            strategy = future_map[future]
            result = future.result()
            record = _record_command(strategy, "smoke", result)
            record["runtime_target_sec"] = SMOKE_STRATEGY_RUNTIME_TARGET_SEC
            record["runtime_target_passed"] = float(record.get("duration_sec", 0.0)) <= SMOKE_STRATEGY_RUNTIME_TARGET_SEC
            records_by_strategy[strategy] = record

    records = [records_by_strategy[strategy] for strategy, _ in jobs]
    total_runtime = round(time.time() - started, 3)
    passed = all(bool(r.get("passed")) for r in records) and total_runtime <= SMOKE_SUITE_RUNTIME_TARGET_SEC
    fail_reasons = []
    if total_runtime > SMOKE_SUITE_RUNTIME_TARGET_SEC:
        fail_reasons.append("smoke_runtime_budget_exceeded")
    payload = {
        "records": records,
        "passed": passed,
        "runtime_sec": total_runtime,
        "runtime_target_sec": SMOKE_SUITE_RUNTIME_TARGET_SEC,
        "fail_reasons": fail_reasons,
    }
    _write_json(output_dir / "smoke_manifest.json", payload)
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


def _run_intraday_micro_candidate(args: argparse.Namespace, output_dir: Path, timeout_s: int) -> List[Dict[str, object]]:
    summary_path = output_dir / "summary.json"
    daily_returns_path = output_dir / "daily_returns.csv"
    checkpoint_dir = PROJECT_ROOT / "quantum_alpha" / "models" / "intraday_microstructure"
    cmd = [
        sys.executable,
        "-m",
        "quantum_alpha.train_intraday_microstructure",
        "--symbols",
        args.intraday_symbols,
        "--market-symbol",
        args.intraday_market_symbol,
        "--sector-symbol",
        args.intraday_sector_symbol,
        "--fixture-days",
        str(int(args.fixture_days)),
        "--top-k",
        str(int(args.intraday_top_k)),
        "--output-dir",
        str(output_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if args.intraday_replay_dir:
        cmd.extend(["--replay-dir", args.intraday_replay_dir])
    result = _run_command(cmd, timeout_s=timeout_s)
    record = _record_command(
        "intraday_microstructure",
        "train_backtest",
        result,
        artifacts={
            "summary_json": str(summary_path),
            "daily_returns_csv": str(daily_returns_path),
            "checkpoint_dir": str(checkpoint_dir),
        },
    )
    if record["passed"] and summary_path.exists():
        payload = _load_json(summary_path)
        passed, fail_reasons, metrics = _intraday_micro_candidate_verdict(payload)
        record["metrics"] = metrics
        record["passed"] = passed
        record["fail_reasons"] = fail_reasons
    return [record]


def _run_rv_stat_arb_candidate(args: argparse.Namespace, output_dir: Path, timeout_s: int) -> List[Dict[str, object]]:
    summary_path = output_dir / "summary.json"
    daily_returns_path = output_dir / "daily_returns.csv"
    checkpoint_dir = PROJECT_ROOT / "quantum_alpha" / "models" / "rv_stat_arb"
    cmd = [
        sys.executable,
        "-m",
        "quantum_alpha.train_rv_stat_arb",
        "--symbols",
        args.intraday_symbols,
        "--fixture-days",
        str(int(args.fixture_days)),
        "--output-dir",
        str(output_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if args.intraday_replay_dir:
        cmd.extend(["--replay-dir", args.intraday_replay_dir])
    result = _run_command(cmd, timeout_s=timeout_s)
    record = _record_command(
        "rv_stat_arb",
        "train_backtest",
        result,
        artifacts={
            "summary_json": str(summary_path),
            "daily_returns_csv": str(daily_returns_path),
            "checkpoint_dir": str(checkpoint_dir),
        },
    )
    if record["passed"] and summary_path.exists():
        payload = _load_json(summary_path)
        passed, fail_reasons, metrics = _rv_stat_arb_candidate_verdict(payload)
        record["metrics"] = metrics
        record["passed"] = passed
        record["fail_reasons"] = fail_reasons
    return [record]


def _run_hybrid_stack_candidate(args: argparse.Namespace, output_dir: Path, timeout_s: int, candidate_root: Path) -> List[Dict[str, object]]:
    intraday_path = candidate_root / "intraday_microstructure" / "daily_returns.csv"
    rv_path = candidate_root / "rv_stat_arb" / "daily_returns.csv"
    meta_path = _discover_meta_daily_returns(candidate_root, _parse_csv_ints(args.meta_horizons, DEFAULT_META_HORIZONS)[0])
    if not intraday_path.exists():
        return [_record_command("hybrid_stack", "train_backtest", {"command": [], "returncode": 1, "stdout": "", "stderr": "", "duration_sec": 0.0}, fail_reasons=["missing_intraday_microstructure_daily_returns"])]
    if not rv_path.exists():
        return [_record_command("hybrid_stack", "train_backtest", {"command": [], "returncode": 1, "stdout": "", "stderr": "", "duration_sec": 0.0}, fail_reasons=["missing_rv_stat_arb_daily_returns"])]

    summary_path = output_dir / "summary.json"
    cmd = [
        sys.executable,
        "-m",
        "quantum_alpha.train_hybrid_stack",
        "--intraday-daily-returns",
        str(intraday_path),
        "--rv-daily-returns",
        str(rv_path),
        "--output-dir",
        str(output_dir),
    ]
    if meta_path is not None:
        cmd.extend(["--meta-daily-returns", str(meta_path), "--benchmark-daily-returns", str(meta_path)])
    result = _run_command(cmd, timeout_s=timeout_s)
    record = _record_command(
        "hybrid_stack",
        "train_backtest",
        result,
        artifacts={
            "summary_json": str(summary_path),
            "daily_returns_csv": str(output_dir / "daily_returns.csv"),
            "normalized_curves": str(output_dir / "normalized_curves.csv"),
            "snapshot_png": str(output_dir / "playback_snapshot.png"),
            "video_mp4": str(output_dir / "backtest_playback.mp4"),
        },
    )
    if record["passed"] and summary_path.exists():
        payload = _load_json(summary_path)
        passed, fail_reasons, metrics = _hybrid_stack_candidate_verdict(payload)
        record["metrics"] = metrics
        record["passed"] = passed
        record["fail_reasons"] = fail_reasons
    return [record]


def _run_event_cross_candidate(
    args: argparse.Namespace,
    output_dir: Path,
    timeout_s: int,
    *,
    research_spine: Dict[str, object] | None = None,
    model_family: str = "state_graph",
) -> List[Dict[str, object]]:
    summary_path = output_dir / "summary.json"
    checkpoint_dir = PROJECT_ROOT / "quantum_alpha" / "models" / "event_cross_sectional"
    viewer_dir = output_dir / "viewer"
    cmd = [
        sys.executable,
        "-m",
        "quantum_alpha.train_event_cross_sectional",
        "--symbols",
        args.event_symbols,
        "--universe-size",
        str(int(args.event_universe_size)),
        "--model-family",
        model_family,
        "--fixture-days",
        str(int(args.fixture_days)),
        "--output-dir",
        str(output_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if research_spine:
        cmd.extend(
            [
                "--research-spine-dir",
                str(research_spine["spine_dir"]),
                "--research-ledger-path",
                str(research_spine["ledger_path"]),
            ]
        )
    if bool(getattr(args, "event_use_fixture", False)):
        cmd.append("--use-fixture")
    if args.quick:
        cmd.append("--quick")
    result = _run_command(cmd, timeout_s=timeout_s)
    validation = validate_event_viewer_bundle(viewer_dir) if viewer_dir.exists() else {"valid": False, "missing": ["viewer_dir"]}
    record = _record_command(
        "event_cross_sectional",
        "train_backtest",
        result,
        artifacts={
            "summary_json": str(summary_path),
            "daily_returns_csv": str(output_dir / "daily_returns.csv"),
            "viewer_dir": str(viewer_dir),
        },
        fail_reasons=[] if validation.get("valid") else [f"missing_{m}" for m in validation.get("missing", [])],
    )
    if record["passed"] and summary_path.exists():
        payload = _load_json(summary_path)
        payload["summary_json"] = str(summary_path)
        viewer_artifacts = payload.setdefault("artifacts", {})
        viewer_artifacts.update(
            {
                "normalized_curves_csv": str(viewer_dir / "normalized_curves.csv"),
                "hedged_curves_csv": str(viewer_dir / "hedged_curves.csv"),
                "snapshot_png": str(viewer_dir / "playback_snapshot.png"),
                "video_mp4": str(viewer_dir / "backtest_playback.mp4"),
            }
        )
        passed, fail_reasons, metrics = _event_cross_candidate_verdict(payload, require_paid=args.suite == "promotion")
        if not bool(validation.get("valid")):
            fail_reasons.extend([f"missing_{m}" for m in validation.get("missing", [])])
        record["metrics"] = metrics
        record["passed"] = bool(passed and validation.get("valid"))
        record["fail_reasons"] = list(dict.fromkeys(fail_reasons))
    return [record]


def _run_event_rv_candidate(
    args: argparse.Namespace,
    output_dir: Path,
    timeout_s: int,
    *,
    research_spine: Dict[str, object] | None = None,
    model_family: str = "state_graph",
) -> List[Dict[str, object]]:
    summary_path = output_dir / "summary.json"
    checkpoint_dir = PROJECT_ROOT / "quantum_alpha" / "models" / "event_rv"
    viewer_dir = output_dir / "viewer"
    cmd = [
        sys.executable,
        "-m",
        "quantum_alpha.train_event_rv",
        "--symbols",
        args.event_symbols,
        "--universe-size",
        str(int(args.event_universe_size)),
        "--model-family",
        model_family,
        "--fixture-days",
        str(int(args.fixture_days)),
        "--output-dir",
        str(output_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if research_spine:
        cmd.extend(
            [
                "--research-spine-dir",
                str(research_spine["spine_dir"]),
                "--research-ledger-path",
                str(research_spine["ledger_path"]),
            ]
        )
    if bool(getattr(args, "event_use_fixture", False)):
        cmd.append("--use-fixture")
    if args.quick:
        cmd.append("--quick")
    result = _run_command(cmd, timeout_s=timeout_s)
    validation = validate_event_viewer_bundle(viewer_dir) if viewer_dir.exists() else {"valid": False, "missing": ["viewer_dir"]}
    record = _record_command(
        "event_rv",
        "train_backtest",
        result,
        artifacts={
            "summary_json": str(summary_path),
            "daily_returns_csv": str(output_dir / "daily_returns.csv"),
            "viewer_dir": str(viewer_dir),
        },
        fail_reasons=[] if validation.get("valid") else [f"missing_{m}" for m in validation.get("missing", [])],
    )
    if record["passed"] and summary_path.exists():
        payload = _load_json(summary_path)
        payload["summary_json"] = str(summary_path)
        viewer_artifacts = payload.setdefault("artifacts", {})
        viewer_artifacts.update(
            {
                "normalized_curves_csv": str(viewer_dir / "normalized_curves.csv"),
                "hedged_curves_csv": str(viewer_dir / "hedged_curves.csv"),
                "snapshot_png": str(viewer_dir / "playback_snapshot.png"),
                "video_mp4": str(viewer_dir / "backtest_playback.mp4"),
            }
        )
        passed, fail_reasons, metrics = _event_rv_candidate_verdict(payload, require_paid=args.suite == "promotion")
        if not bool(validation.get("valid")):
            fail_reasons.extend([f"missing_{m}" for m in validation.get("missing", [])])
        record["metrics"] = metrics
        record["passed"] = bool(passed and validation.get("valid"))
        record["fail_reasons"] = list(dict.fromkeys(fail_reasons))
    return [record]


def _run_event_stack_candidate(
    args: argparse.Namespace,
    output_dir: Path,
    timeout_s: int,
    candidate_root: Path,
    *,
    model_family: str = "state_graph",
) -> List[Dict[str, object]]:
    cross_path = candidate_root / "event_cross_sectional" / "daily_returns.csv"
    rv_path = candidate_root / "event_rv" / "daily_returns.csv"
    if not cross_path.exists():
        return [_record_command("event_stack", "train_backtest", {"command": [], "returncode": 1, "stdout": "", "stderr": "", "duration_sec": 0.0}, fail_reasons=["missing_event_cross_sectional_daily_returns"])]
    if not rv_path.exists():
        return [_record_command("event_stack", "train_backtest", {"command": [], "returncode": 1, "stdout": "", "stderr": "", "duration_sec": 0.0}, fail_reasons=["missing_event_rv_daily_returns"])]

    meta_path = _discover_meta_daily_returns(candidate_root, _parse_csv_ints(args.meta_horizons, DEFAULT_META_HORIZONS)[0])
    summary_path = output_dir / "summary.json"
    viewer_dir = output_dir / "viewer"
    cmd = [
        sys.executable,
        "-m",
        "quantum_alpha.train_event_stack",
        "--event-cross-daily-returns",
        str(cross_path),
        "--event-rv-daily-returns",
        str(rv_path),
        "--model-family",
        model_family,
        "--output-dir",
        str(output_dir),
    ]
    if meta_path is not None:
        cmd.extend(["--meta-daily-returns", str(meta_path)])
    result = _run_command(cmd, timeout_s=timeout_s)
    validation = validate_event_viewer_bundle(viewer_dir) if viewer_dir.exists() else {"valid": False, "missing": ["viewer_dir"]}
    record = _record_command(
        "event_stack",
        "train_backtest",
        result,
        artifacts={
            "summary_json": str(summary_path),
            "daily_returns_csv": str(output_dir / "daily_returns.csv"),
            "viewer_dir": str(viewer_dir),
        },
        fail_reasons=[] if validation.get("valid") else [f"missing_{m}" for m in validation.get("missing", [])],
    )
    if record["passed"] and summary_path.exists():
        payload = _load_json(summary_path)
        passed, fail_reasons, metrics = _event_stack_candidate_verdict(payload)
        if not bool(validation.get("valid")):
            fail_reasons.extend([f"missing_{m}" for m in validation.get("missing", [])])
        record["metrics"] = metrics
        record["passed"] = bool(passed and validation.get("valid"))
        record["fail_reasons"] = list(dict.fromkeys(fail_reasons))
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
    suite_started = time.time()

    manifest: Dict[str, object] = {
        "run_at_utc": _now_utc(),
        "suite": args.suite,
        "strategies": strategies,
        "offline": bool(args.offline),
        "live_probe": bool(args.live_probe),
        "config_hashes": _collect_config_hashes(),
        "runtime_targets": {
            "smoke_suite_sec": SMOKE_SUITE_RUNTIME_TARGET_SEC,
            "smoke_strategy_sec": SMOKE_STRATEGY_RUNTIME_TARGET_SEC,
            "quick_candidate_sec": QUICK_CANDIDATE_RUNTIME_TARGET_SEC,
        },
        "records": [],
    }

    records: List[Dict[str, object]] = []
    timeout_s = int(args.timeout_seconds)
    research_spine: Dict[str, object] | None = None

    if args.suite == "smoke":
        smoke_records, _ = _run_smoke_suite(strategies, output_dir, timeout_s)
        records.extend(smoke_records)
    else:
        if any(s in strategies for s in ("event_cross_sectional", "event_rv", "event_stack")):
            research_spine = _prepare_event_research_spine(args, output_dir)
            manifest["research_spine"] = research_spine
        if "enhanced" in strategies:
            records.extend(_run_enhanced_candidate(args, output_dir / "enhanced", timeout_s))
        if "intraday_microstructure" in strategies:
            records.extend(_run_intraday_micro_candidate(args, output_dir / "intraday_microstructure", timeout_s))
        if "rv_stat_arb" in strategies:
            records.extend(_run_rv_stat_arb_candidate(args, output_dir / "rv_stat_arb", timeout_s))
        if "event_cross_sectional" in strategies:
            records.extend(
                _run_event_cross_candidate(
                    args,
                    output_dir / "event_cross_sectional",
                    timeout_s,
                    research_spine=research_spine,
                    model_family="state_graph",
                )
            )
        if "event_rv" in strategies:
            records.extend(
                _run_event_rv_candidate(
                    args,
                    output_dir / "event_rv",
                    timeout_s,
                    research_spine=research_spine,
                    model_family="state_graph",
                )
            )
        if "meta_ensemble" in strategies:
            records.extend(_run_meta_candidate(args, output_dir / "meta_ensemble", timeout_s))
        if "news_lstm" in strategies:
            records.extend(_run_news_candidate(args, output_dir / "news_lstm", timeout_s))
        if "intraday" in strategies:
            records.extend(_run_intraday_candidate(args, output_dir / "intraday", timeout_s))
        if "hybrid_stack" in strategies:
            records.extend(_run_hybrid_stack_candidate(args, output_dir / "hybrid_stack", timeout_s, output_dir))
        if "event_stack" in strategies:
            records.extend(
                _run_event_stack_candidate(
                    args,
                    output_dir / "event_stack",
                    timeout_s,
                    output_dir,
                    model_family="state_graph",
                )
            )

        if args.suite == "promotion":
            smoke_records, _ = _run_smoke_suite(strategies, output_dir / "smoke", timeout_s)
            records.extend(smoke_records)
            records.extend(_run_realtime_probe(args, output_dir / "promotion_checks", timeout_s))

    passed, fail_reasons = _aggregate_verdict(records)
    total_runtime = round(time.time() - suite_started, 3)
    if args.suite == "smoke" and total_runtime > SMOKE_SUITE_RUNTIME_TARGET_SEC:
        fail_reasons.append("smoke_suite:runtime:smoke_runtime_budget_exceeded")
        passed = False
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
    manifest["runtime_sec"] = total_runtime

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
    parser.add_argument("--intraday-replay-dir", type=str, default=None)
    parser.add_argument("--intraday-symbols", type=str, default="SPY,XLK,AAPL,MSFT")
    parser.add_argument("--intraday-market-symbol", type=str, default="SPY")
    parser.add_argument("--intraday-sector-symbol", type=str, default="XLK")
    parser.add_argument("--intraday-top-k", type=int, default=2)
    parser.add_argument("--event-symbols", type=str, default="SPY,AAPL,MSFT,NVDA,AMZN,XOM,JPM")
    parser.add_argument("--event-universe-size", type=int, default=800)
    parser.add_argument("--event-use-fixture", action="store_true")
    parser.add_argument("--fixture-days", type=int, default=5)
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
