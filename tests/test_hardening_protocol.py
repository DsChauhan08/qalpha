from datetime import datetime
from pathlib import Path

from quantum_alpha.backtesting import hardening_protocol


def test_hardening_protocol_promotes_when_full_metrics_pass(monkeypatch, tmp_path: Path):
    def _fake_run_backtest(**kwargs):
        return {
            "metrics": {
                "max_drawdown": -0.10,
                "excess_total_return_vs_quant": 0.05,
                "quant_information_ratio": 0.20,
                "sharpe_ratio": 0.90,
            },
            "promotion_verdict": {"eligible": True, "fail_reasons": []},
        }

    monkeypatch.setattr("quantum_alpha.main.run_backtest", _fake_run_backtest)
    out = hardening_protocol.run_hardening_rnd_protocol(
        symbols=["SPY", "QQQ", "IWM"],
        output_path=str(tmp_path / "protocol.json"),
        verbose=False,
    )

    assert out["promotion_passed"] is True
    assert out["escalate_meta_migration"] is False
    assert (tmp_path / "protocol.json").exists()


def test_hardening_protocol_fails_when_dd_or_edge_fails(monkeypatch, tmp_path: Path):
    def _fake_run_backtest(**kwargs):
        return {
            "metrics": {
                "max_drawdown": -0.30,
                "excess_total_return_vs_quant": -0.01,
                "quant_information_ratio": 0.02,
                "sharpe_ratio": 0.10,
            },
            "promotion_verdict": {"eligible": False, "fail_reasons": ["dummy"]},
        }

    monkeypatch.setattr("quantum_alpha.main.run_backtest", _fake_run_backtest)
    out = hardening_protocol.run_hardening_rnd_protocol(
        symbols=["SPY"],
        output_path=str(tmp_path / "protocol_fail.json"),
        verbose=False,
    )

    assert out["promotion_passed"] is False
    assert out["escalate_meta_migration"] is True
    assert "max_drawdown_gt_20pct" in out["promotion_fail_reasons"]
