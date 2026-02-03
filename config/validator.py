"""
Configuration validation helpers.
"""

from __future__ import annotations

from typing import Dict, List


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

    backtest = cfg.get("backtest", {})
    commission = backtest.get("commission_rate")
    if commission is not None and commission < 0:
        issues.append("settings.backtest.commission_rate negative")

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
