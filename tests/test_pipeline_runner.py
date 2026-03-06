import argparse
import json
from pathlib import Path

import pandas as pd

import quantum_alpha.pipeline_runner as pr


def _args(tmp_path: Path, suite: str = "smoke", strategies: str = "meta_ensemble"):
    return argparse.Namespace(
        suite=suite,
        strategies=strategies,
        offline=True,
        live_probe=False,
        live_probe_symbols="SPY,QQQ,AAPL",
        meta_horizons="5,21",
        meta_n_symbols=10,
        promotion_symbols="SPY,QQQ,IWM,AAPL,MSFT,NVDA",
        news_checkpoint=None,
        quick=True,
        n_folds=1,
        timeout_seconds=60,
        output_dir=str(tmp_path),
    )


def test_pipeline_runner_smoke_writes_manifest(monkeypatch, tmp_path: Path):
    def _fake_run_command(cmd, **kwargs):
        return {
            "command": cmd,
            "returncode": 0,
            "stdout": "ok",
            "stderr": "",
            "duration_sec": 0.1,
        }

    monkeypatch.setattr(pr, "_run_command", _fake_run_command)
    manifest = pr.run_pipeline_suite(_args(tmp_path, suite="smoke", strategies="meta_ensemble"))

    assert manifest["passed"] is True
    assert (tmp_path / "manifest.json").exists()
    payload = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert payload["records"]


def test_pipeline_runner_candidate_reports_missing_checkpoint(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(pr, "_discover_news_checkpoint", lambda: None)
    manifest = pr.run_pipeline_suite(_args(tmp_path, suite="candidate", strategies="news_lstm"))

    assert manifest["passed"] is False
    assert any("missing_checkpoint" in r for r in manifest["fail_reasons"])


def test_validate_live_session_flags_degraded_pipeline(tmp_path: Path):
    session = tmp_path / "realtime_paper_20260306_000000"
    session.mkdir()
    (session / "live_status.json").write_text(
        json.dumps({"pipeline_health": "degraded", "symbols_tracked": 3}),
        encoding="utf-8",
    )
    (session / "equity_curve.csv").write_text(
        "timestamp,equity,cash\n2026-03-06T15:30:00Z,10000,5000\n",
        encoding="utf-8",
    )
    (session / "live_trades.jsonl").write_text("", encoding="utf-8")

    validation = pr._validate_live_session(session)

    assert validation["valid"] is False
    assert "pipeline_health_degraded" in validation["fail_reasons"]


def test_normalize_symbols_rejects_empty_universe():
    try:
        pr._normalize_symbols("", [])
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for empty universe")
