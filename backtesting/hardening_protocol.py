"""
R&D protocol runner for enhanced-engine hardening.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


WINDOWS = {
    "full": ("2006-01-01", "2026-01-01"),
    "old": ("2006-01-01", "2013-12-31"),
    "mid": ("2014-01-01", "2019-12-31"),
    "recent": ("2020-01-01", "2026-01-01"),
    "stress_2008": ("2008-01-01", "2008-12-31"),
    "stress_2020": ("2020-01-01", "2020-12-31"),
    "stress_2022": ("2022-01-01", "2022-12-31"),
}


def _parse_date(d: str) -> datetime:
    return datetime.strptime(d, "%Y-%m-%d")


def _passes_promotion(metrics: Dict[str, object]) -> Tuple[bool, List[str]]:
    fail: List[str] = []
    max_dd = abs(float(metrics.get("max_drawdown", 0.0)))
    excess = float(metrics.get("excess_total_return_vs_quant", 0.0))
    info = float(metrics.get("quant_information_ratio", 0.0))
    sharpe = float(metrics.get("sharpe_ratio", 0.0))
    if max_dd > 0.20:
        fail.append("max_drawdown_gt_20pct")
    if excess <= 0.0:
        fail.append("excess_total_return_vs_quant_nonpositive")
    if info < 0.10:
        fail.append("quant_information_ratio_below_0.10")
    if sharpe < 0.50:
        fail.append("sharpe_below_0.50")
    return len(fail) == 0, fail


def run_hardening_rnd_protocol(
    symbols: List[str],
    strategy_type: str = "enhanced",
    initial_capital: float = 100000,
    config_path: Optional[str] = None,
    output_path: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, object]:
    """
    Run standard R&D windows and emit a promotion-ready summary.
    """
    from quantum_alpha.main import run_backtest

    segment_results: Dict[str, Dict[str, object]] = {}
    for name, (start_s, end_s) in WINDOWS.items():
        result = run_backtest(
            symbols=symbols,
            start_date=_parse_date(start_s),
            end_date=_parse_date(end_s),
            initial_capital=initial_capital,
            strategy_type=strategy_type,
            validate=True,
            verbose=verbose,
            config_path=config_path,
        )
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        segment_results[name] = {
            "start": start_s,
            "end": end_s,
            "metrics": metrics,
            "promotion_verdict": result.get("promotion_verdict", {}),
        }

    full_metrics = segment_results.get("full", {}).get("metrics", {})
    passes, fail_reasons = _passes_promotion(full_metrics if isinstance(full_metrics, dict) else {})
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "windows": WINDOWS,
        "segments": segment_results,
        "candidate_selection_objective": (
            "maximize benchmark-relative edge first (excess + IR), then sharpe"
        ),
        "promotion_criteria": {
            "max_drawdown_lte": 0.20,
            "excess_total_return_vs_quant_gt": 0.0,
            "quant_information_ratio_gte": 0.10,
            "sharpe_gte": 0.50,
        },
        "promotion_passed": bool(passes),
        "promotion_fail_reasons": fail_reasons,
        "escalate_meta_migration": bool(not passes),
    }

    if output_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output = Path(__file__).resolve().parents[1] / "artifacts" / f"hardening_protocol_{stamp}.json"
    else:
        output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["output_path"] = str(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced strategy hardening protocol")
    parser.add_argument(
        "--symbols",
        type=str,
        default="SPY,QQQ,IWM",
        help="Comma-separated symbols",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="enhanced",
        help="Strategy type passed to run_backtest",
    )
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--config-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    if not symbols:
        raise SystemExit("No symbols provided")

    summary = run_hardening_rnd_protocol(
        symbols=symbols,
        strategy_type=str(args.strategy).strip().lower(),
        initial_capital=float(args.capital),
        config_path=args.config_path,
        output_path=args.output_path,
        verbose=bool(args.verbose),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
