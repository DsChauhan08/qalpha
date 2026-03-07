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
        intraday_replay_dir=None,
        intraday_symbols="SPY,XLK,AAPL,MSFT",
        intraday_market_symbol="SPY",
        intraday_sector_symbol="XLK",
        intraday_top_k=2,
        event_symbols="SPY,AAPL,MSFT,NVDA,AMZN",
        event_universe_size=50,
        event_use_fixture=True,
        fixture_days=1,
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


def test_pipeline_runner_smoke_accepts_new_strategy_names(monkeypatch, tmp_path: Path):
    def _fake_run_command(cmd, **kwargs):
        return {
            "command": cmd,
            "returncode": 0,
            "stdout": "ok",
            "stderr": "",
            "duration_sec": 0.1,
        }

    monkeypatch.setattr(pr, "_run_command", _fake_run_command)
    manifest = pr.run_pipeline_suite(
        _args(tmp_path, suite="smoke", strategies="intraday_microstructure,rv_stat_arb,hybrid_stack")
    )

    assert manifest["passed"] is True
    assert len(manifest["records"]) == 3


def test_pipeline_runner_candidate_hybrid_requires_dependencies(tmp_path: Path):
    manifest = pr.run_pipeline_suite(_args(tmp_path, suite="candidate", strategies="hybrid_stack"))
    assert manifest["passed"] is False
    assert any("missing_intraday_microstructure_daily_returns" in r for r in manifest["fail_reasons"])


def test_pipeline_runner_smoke_accepts_event_strategy_names(monkeypatch, tmp_path: Path):
    def _fake_run_command(cmd, **kwargs):
        return {
            "command": cmd,
            "returncode": 0,
            "stdout": "ok",
            "stderr": "",
            "duration_sec": 0.1,
        }

    monkeypatch.setattr(pr, "_run_command", _fake_run_command)
    manifest = pr.run_pipeline_suite(
        _args(tmp_path, suite="smoke", strategies="event_cross_sectional,event_rv,event_stack")
    )

    assert manifest["passed"] is True
    assert len(manifest["records"]) == 3


def test_pipeline_runner_event_cross_promotion_blocks_free_data(monkeypatch, tmp_path: Path):
    def _fake_run_command(cmd, **kwargs):
        if "--output-dir" not in cmd:
            return {
                "command": cmd,
                "returncode": 0,
                "stdout": "ok",
                "stderr": "",
                "duration_sec": 0.1,
            }
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        viewer_dir = output_dir / "viewer"
        viewer_dir.mkdir(parents=True, exist_ok=True)
        (viewer_dir / "summary.json").write_text("{}", encoding="utf-8")
        (viewer_dir / "normalized_curves.csv").write_text(
            "timestamp,model,spy,equal_weight\n2025-01-01,1.0,1.0,1.0\n",
            encoding="utf-8",
        )
        (viewer_dir / "hedged_curves.csv").write_text(
            "timestamp,hedged_model,spy\n2025-01-01,1.0,1.0\n",
            encoding="utf-8",
        )
        (viewer_dir / "playback_snapshot.png").write_bytes(b"png")
        (viewer_dir / "backtest_playback.mp4").write_bytes(b"mp4")
        (output_dir / "daily_returns.csv").write_text(
            "date,strategy_return,benchmark_return,equal_weight_return\n2025-01-01,0.01,0.0,0.0\n",
            encoding="utf-8",
        )
        (output_dir / "robustness.json").write_text("{}", encoding="utf-8")
        (output_dir / "synthetic_stress.json").write_text("{}", encoding="utf-8")
        (output_dir / "regime_report.json").write_text("{}", encoding="utf-8")
        payload = {
            "strategy": "event_cross_sectional",
            "metrics": {
                "annual_return": 0.2,
                "sharpe": 1.5,
                "beta": 0.01,
                "max_drawdown": -0.05,
                "rolling_positive_ratio_3m": 0.7,
                "eligible_event_count": 10,
            },
            "data_quality": {
                "event_lag_ok": True,
                "paid_data_eligible": False,
                "coverage_by_domain": {"earnings": 1.0, "congress": 0.0},
                "staleness_days": {"earnings": 0.0},
            },
            "artifacts": {
                "daily_returns_csv": str(output_dir / "daily_returns.csv"),
                "robustness_json": str(output_dir / "robustness.json"),
                "synthetic_stress_json": str(output_dir / "synthetic_stress.json"),
                "regime_report_json": str(output_dir / "regime_report.json"),
            },
        }
        (output_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
        return {
            "command": cmd,
            "returncode": 0,
            "stdout": "ok",
            "stderr": "",
            "duration_sec": 0.1,
        }

    monkeypatch.setattr(pr, "_run_command", _fake_run_command)
    manifest = pr.run_pipeline_suite(
        _args(tmp_path, suite="promotion", strategies="event_cross_sectional")
    )

    assert manifest["passed"] is False
    assert any("paid_data_ineligible" in r for r in manifest["fail_reasons"])
