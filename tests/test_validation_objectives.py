import pandas as pd
import yaml

from quantum_alpha.backtesting.robustness_suite import (
    _benchmark_relative_metrics,
    _score_candidate,
)
from quantum_alpha.main import (
    _init_hard_drawdown_state,
    _rolling_oos_vs_benchmark,
    _select_liquid_subset,
    _normalize_weights,
    load_config,
    _promotion_verdict_from_metrics,
    _strict_promotion_ready,
    _update_hard_drawdown_guard,
    _update_loss_limit_state,
)


def _price_df(close_vals, vol_vals):
    idx = pd.date_range("2025-01-01", periods=len(close_vals), freq="D")
    return pd.DataFrame(
        {
            "open": close_vals,
            "high": close_vals,
            "low": close_vals,
            "close": close_vals,
            "volume": vol_vals,
        },
        index=idx,
    )


def test_select_liquid_subset_by_dollar_volume():
    frames = {
        "LOW": _price_df([10, 10, 10, 10, 10, 10], [100, 100, 100, 100, 100, 100]),
        "MID": _price_df(
            [20, 20, 20, 20, 20, 20],
            [500, 500, 500, 500, 500, 500],
        ),
        "HI": _price_df(
            [30, 30, 30, 30, 30, 30],
            [1000, 1000, 1000, 1000, 1000, 1000],
        ),
    }
    selected = _select_liquid_subset(
        frames,
        subset_size=2,
        adv_window=3,
        min_history=3,
    )
    assert selected == ["HI", "MID"]


def test_rolling_oos_vs_benchmark_passes_with_three_beats():
    idx = pd.date_range("2025-01-01", periods=12, freq="D")
    strat = pd.Series(
        [0.01, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01, 0.00, 0.00, 0.01, 0.00, 0.00],
        index=idx,
    )
    bench = pd.Series(
        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        index=idx,
    )
    res = _rolling_oos_vs_benchmark(
        strategy_returns=strat,
        benchmark_returns=bench,
        window_days=3,
        min_windows=3,
    )
    assert res["available"] is True
    assert res["n_windows"] == 4
    assert res["beats"] == 4
    assert res["passed"] is True


def test_rolling_oos_ratio_requirement_enforced():
    idx = pd.date_range("2025-01-01", periods=15, freq="D")
    strat = pd.Series(
        [
            0.01,
            0.0,
            0.0,  # beat
            0.01,
            0.0,
            0.0,  # beat
            0.01,
            0.0,
            0.0,  # beat
            -0.01,
            0.0,
            0.0,  # lose
            -0.01,
            0.0,
            0.0,  # lose
        ],
        index=idx,
    )
    bench = pd.Series([0.0] * len(idx), index=idx)
    res = _rolling_oos_vs_benchmark(
        strategy_returns=strat,
        benchmark_returns=bench,
        window_days=3,
        min_windows=3,
        min_beat_ratio=0.75,
    )
    assert res["n_windows"] == 5
    assert res["beats"] == 3
    assert res["required_beats"] == 4
    assert res["passed"] is False


def test_benchmark_relative_metrics_information_ratio_positive():
    idx = pd.date_range("2025-01-01", periods=10, freq="D")
    strat = pd.Series([0.012, 0.003, 0.011, 0.004, 0.013, 0.002, 0.010, 0.005, 0.011, 0.004], index=idx)
    bench = pd.Series([0.007, 0.004, 0.008, 0.003, 0.009, 0.003, 0.007, 0.004, 0.008, 0.003], index=idx)
    rel = _benchmark_relative_metrics(strat, bench)
    assert rel["information_ratio"] > 0


def test_score_candidate_rewards_excess_vs_quant():
    seg = {
        "full": {
            "model": {"annual_return": 0.20, "sharpe": 1.0, "max_drawdown": -0.10},
            "equal_weight": {
                "annual_return": 0.08,
                "sharpe": 0.5,
                "max_drawdown": -0.12,
            },
            "spy": {"annual_return": 0.10, "sharpe": 0.6, "max_drawdown": -0.11},
            "quant_composite": {
                "annual_return": 0.15,
                "sharpe": 0.8,
                "max_drawdown": -0.13,
            },
        },
        "old": {
            "model": {"annual_return": 0.18, "sharpe": 0.9, "max_drawdown": -0.12},
            "equal_weight": {
                "annual_return": 0.07,
                "sharpe": 0.4,
                "max_drawdown": -0.11,
            },
            "spy": {"annual_return": 0.09, "sharpe": 0.5, "max_drawdown": -0.10},
            "quant_composite": {
                "annual_return": 0.13,
                "sharpe": 0.7,
                "max_drawdown": -0.12,
            },
        },
        "recent": {
            "model": {"annual_return": 0.22, "sharpe": 1.1, "max_drawdown": -0.09},
            "equal_weight": {
                "annual_return": 0.10,
                "sharpe": 0.6,
                "max_drawdown": -0.11,
            },
            "spy": {"annual_return": 0.12, "sharpe": 0.7, "max_drawdown": -0.10},
            "quant_composite": {
                "annual_return": 0.16,
                "sharpe": 0.9,
                "max_drawdown": -0.12,
            },
        },
        "very_recent": {
            "model": {"annual_return": 0.21, "sharpe": 1.0, "max_drawdown": -0.08},
            "equal_weight": {
                "annual_return": 0.09,
                "sharpe": 0.5,
                "max_drawdown": -0.10,
            },
            "spy": {"annual_return": 0.11, "sharpe": 0.6, "max_drawdown": -0.09},
            "quant_composite": {
                "annual_return": 0.14,
                "sharpe": 0.8,
                "max_drawdown": -0.11,
            },
        },
    }
    score = _score_candidate(seg, rel_full={"information_ratio": 0.2})
    assert score > 0


def test_normalize_weights_respects_budget_and_cap():
    out = _normalize_weights(
        {"A": 3.0, "B": 1.0, "C": 1.0},
        budget=0.10,
        long_only=True,
        max_abs_per_symbol=0.05,
    )
    assert abs(sum(abs(v) for v in out.values()) - 0.10) < 1e-8
    assert max(abs(v) for v in out.values()) <= 0.05 + 1e-12


def test_strict_promotion_ready_enforces_minimum_3_of_4():
    metrics = {
        "mcpt_pass_stage2_0_05": True,
        "benchmark_constraints_passed": True,
        "rolling_oos_n_windows": 4,
        "rolling_oos_beats": 3,
    }
    ready, req_w, req_b = _strict_promotion_ready(
        metrics=metrics,
        required_windows=2,
        required_beats=2,
    )
    assert ready is True
    assert req_w == 4
    assert req_b == 3


def test_strict_promotion_ready_fails_if_under_3_of_4():
    metrics = {
        "mcpt_pass_stage2_0_05": True,
        "benchmark_constraints_passed": True,
        "rolling_oos_n_windows": 4,
        "rolling_oos_beats": 2,
    }
    ready, req_w, req_b = _strict_promotion_ready(
        metrics=metrics,
        required_windows=4,
        required_beats=3,
    )
    assert ready is False
    assert req_w == 4
    assert req_b == 3


def test_update_loss_limit_state_triggers_daily_and_weekly_stops():
    state = {}
    ts0 = pd.Timestamp("2026-01-05")  # Monday
    ts1 = pd.Timestamp("2026-01-05 15:00")
    ts2 = pd.Timestamp("2026-01-06")

    _update_loss_limit_state(
        state=state,
        timestamp=ts0,
        equity=100000.0,
        max_daily_loss=0.03,
        max_weekly_loss=0.07,
    )
    r1 = _update_loss_limit_state(
        state=state,
        timestamp=ts1,
        equity=96000.0,  # -4% day
        max_daily_loss=0.03,
        max_weekly_loss=0.07,
    )
    assert r1["daily_stop_active"] is True
    assert r1["stop_active"] is True

    r2 = _update_loss_limit_state(
        state=state,
        timestamp=ts2,
        equity=92000.0,  # -8% week
        max_daily_loss=0.03,
        max_weekly_loss=0.07,
    )
    assert r2["weekly_stop_active"] is True
    assert r2["stop_active"] is True


def test_hard_drawdown_guard_triggers_and_reenters_after_cooldown():
    state = {}
    _init_hard_drawdown_state(state)
    ts0 = pd.Timestamp("2026-01-02")

    r0 = _update_hard_drawdown_guard(
        state=state,
        timestamp=ts0,
        current_drawdown=-0.21,
        equity_curve=[],
        quant_returns=None,
        limit=0.20,
        action="flatten",
        cooldown_days=5,
        recovery_level=0.10,
        require_positive_quant_ir=False,
        ir_lookback_bars=63,
    )
    assert r0["trigger_now"] is True
    assert r0["halted"] is True
    assert r0["force_flatten"] is True

    r1 = _update_hard_drawdown_guard(
        state=state,
        timestamp=ts0 + pd.Timedelta(days=2),
        current_drawdown=-0.12,
        equity_curve=[],
        quant_returns=None,
        limit=0.20,
        action="flatten",
        cooldown_days=5,
        recovery_level=0.10,
        require_positive_quant_ir=False,
        ir_lookback_bars=63,
    )
    assert r1["halted"] is True
    assert r1["reentered"] is False

    r2 = _update_hard_drawdown_guard(
        state=state,
        timestamp=ts0 + pd.Timedelta(days=6),
        current_drawdown=-0.08,
        equity_curve=[],
        quant_returns=None,
        limit=0.20,
        action="flatten",
        cooldown_days=5,
        recovery_level=0.10,
        require_positive_quant_ir=False,
        ir_lookback_bars=63,
    )
    assert r2["halted"] is False
    assert r2["reentered"] is True
    assert int(state.get("hard_dd_reentries", 0)) == 1


def test_promotion_verdict_collects_fail_reasons():
    verdict = _promotion_verdict_from_metrics(
        {
            "benchmark_constraints_passed": False,
            "benchmark_constraint_fail_reasons": ["information_ratio_below_min"],
            "mcpt_pass_stage2_0_05": False,
            "rolling_oos_n_windows": 2,
            "rolling_oos_beats": 1,
            "promotion_oos_required_windows": 4,
            "promotion_oos_required_beats": 3,
        }
    )
    assert verdict["eligible"] is False
    reasons = verdict["fail_reasons"]
    assert "constraint:information_ratio_below_min" in reasons
    assert "mcpt_stage2_failed" in reasons
    assert "rolling_oos_windows_below_min:2<4" in reasons
    assert "rolling_oos_beats_below_min:1<3" in reasons


def test_load_config_deep_merge_preserves_defensive_defaults(tmp_path):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    override = {
        "strategy": {
            "signal_scale": 1.5,
            "sleeves": {
                "core_budget": 0.9,
            },
        }
    }
    with open(cfg_dir / "settings.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(override, f)

    merged = load_config(str(cfg_dir))
    assert merged["strategy"]["anchor_core_to_quant_composite"] is False
    assert merged["strategy"]["sleeves"]["core_tilt_enabled"] is False
