"""
Configuration validation helpers.
"""

from __future__ import annotations

from typing import Dict, List


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_keys(cfg: Dict, keys: List[str], context: str, issues: List[str]) -> None:
    missing = [k for k in keys if k not in cfg]
    if missing:
        issues.append(f"{context}: missing keys {missing}")


def validate_settings(cfg: Dict) -> List[str]:
    issues: List[str] = []
    _require_keys(cfg, ["data", "backtest", "risk", "strategy", "indicators"], "settings", issues)

    risk = cfg.get("risk", {})
    max_pos = risk.get("max_position_size")
    if max_pos is not None and not (0 < max_pos <= 1):
        issues.append("settings.risk.max_position_size out of range")

    max_dd = risk.get("max_drawdown")
    if max_dd is not None and not (0 < max_dd <= 1):
        issues.append("settings.risk.max_drawdown out of range")

    hard_dd_limit = risk.get("hard_drawdown_limit")
    if hard_dd_limit is not None and not (0 < float(hard_dd_limit) <= 1):
        issues.append("settings.risk.hard_drawdown_limit out of range")

    hard_dd_action = risk.get("hard_drawdown_action")
    if hard_dd_action is not None:
        action = str(hard_dd_action).strip().lower()
        if action not in {"flatten"}:
            issues.append("settings.risk.hard_drawdown_action invalid")

    hard_dd_cooldown = risk.get("hard_drawdown_cooldown_days")
    if hard_dd_cooldown is not None and int(hard_dd_cooldown) < 0:
        issues.append("settings.risk.hard_drawdown_cooldown_days negative")

    hard_dd_recovery = risk.get("hard_drawdown_recovery_level")
    if hard_dd_recovery is not None and not (0 < float(hard_dd_recovery) <= 1):
        issues.append("settings.risk.hard_drawdown_recovery_level out of range")

    core_min = risk.get("core_min_exposure_in_drawdown")
    if core_min is not None and not (0 <= float(core_min) <= 1):
        issues.append("settings.risk.core_min_exposure_in_drawdown out of range")

    backtest = cfg.get("backtest", {})
    commission = backtest.get("commission_rate")
    if commission is not None and commission < 0:
        issues.append("settings.backtest.commission_rate negative")

    validation = cfg.get("validation", {})
    constraint_dd = validation.get("constraint_max_drawdown")
    if constraint_dd is not None and not (0 < float(constraint_dd) <= 1):
        issues.append("settings.validation.constraint_max_drawdown out of range")

    constraint_ir = validation.get("constraint_min_information_ratio")
    if constraint_ir is not None and not _is_number(constraint_ir):
        issues.append("settings.validation.constraint_min_information_ratio invalid")

    constraint_excess = validation.get("constraint_min_excess_total_return_vs_quant")
    if constraint_excess is not None and not _is_number(constraint_excess):
        issues.append(
            "settings.validation.constraint_min_excess_total_return_vs_quant invalid"
        )

    return issues


def validate_strategies(cfg: Dict) -> List[str]:
    issues: List[str] = []
    _require_keys(cfg, ["strategies"], "strategies", issues)
    strategies = cfg.get("strategies", {})
    if "momentum" not in strategies:
        issues.append("strategies.momentum missing")
    if "mean_reversion" not in strategies:
        issues.append("strategies.mean_reversion missing")
    if "trend" not in strategies:
        issues.append("strategies.trend missing")
    return issues


def validate_risk_limits(cfg: Dict) -> List[str]:
    issues: List[str] = []
    _require_keys(cfg, ["limits"], "risk_limits", issues)
    limits = cfg.get("limits", {})
    max_pos = limits.get("max_position_size")
    if max_pos is not None and not (0 < max_pos <= 1):
        issues.append("risk_limits.max_position_size out of range")
    max_dd = limits.get("max_drawdown")
    if max_dd is not None and not (0 < max_dd <= 1):
        issues.append("risk_limits.max_drawdown out of range")
    return issues


def validate_data_sources(cfg: Dict) -> List[str]:
    issues: List[str] = []
    _require_keys(cfg, ["market_data"], "data_sources", issues)
    market = cfg.get("market_data", {})
    if "primary" not in market:
        issues.append("data_sources.market_data.primary missing")
    return issues
