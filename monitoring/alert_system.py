"""
Alerting utilities for performance monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    level: AlertLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, object] = field(default_factory=dict)


class AlertManager:
    """
    Evaluate metric rules and emit alerts.
    """

    def __init__(self, on_alert: Optional[Callable[[Alert], None]] = None) -> None:
        self.rules: List[Callable[[Dict[str, float]], List[Alert]]] = []
        self.on_alert = on_alert

    def add_rule(self, rule: Callable[[Dict[str, float]], List[Alert]]) -> None:
        self.rules.append(rule)

    def evaluate(self, metrics: Dict[str, float]) -> List[Alert]:
        alerts: List[Alert] = []
        for rule in self.rules:
            try:
                alerts.extend(rule(metrics) or [])
            except Exception as exc:
                logger.warning("Alert rule failed: %s", exc)

        for alert in alerts:
            self._emit(alert)
        return alerts

    def _emit(self, alert: Alert) -> None:
        msg = f"{alert.message}"
        if alert.level == AlertLevel.CRITICAL:
            logger.error(msg)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(msg)
        else:
            logger.info(msg)

        if self.on_alert:
            self.on_alert(alert)


def build_default_rules(thresholds: Optional[Dict[str, float]] = None) -> List[Callable[[Dict[str, float]], List[Alert]]]:
    thresholds = thresholds or {}
    max_drawdown = thresholds.get("max_drawdown", 0.15)
    min_sharpe = thresholds.get("min_sharpe", 0.5)
    min_win_rate = thresholds.get("min_win_rate", 0.4)

    def drawdown_rule(metrics: Dict[str, float]) -> List[Alert]:
        alerts: List[Alert] = []
        dd = metrics.get("max_drawdown")
        if dd is not None and abs(dd) >= max_drawdown:
            alerts.append(
                Alert(
                    level=AlertLevel.WARNING,
                    message=f"Max drawdown {dd:.2%} exceeds threshold {max_drawdown:.2%}",
                    context={"max_drawdown": dd},
                )
            )
        return alerts

    def sharpe_rule(metrics: Dict[str, float]) -> List[Alert]:
        alerts: List[Alert] = []
        sharpe = metrics.get("sharpe_ratio")
        if sharpe is not None and sharpe < min_sharpe:
            alerts.append(
                Alert(
                    level=AlertLevel.WARNING,
                    message=f"Sharpe ratio {sharpe:.2f} below threshold {min_sharpe:.2f}",
                    context={"sharpe_ratio": sharpe},
                )
            )
        return alerts

    def win_rate_rule(metrics: Dict[str, float]) -> List[Alert]:
        alerts: List[Alert] = []
        win_rate = metrics.get("win_rate")
        if win_rate is not None and win_rate < min_win_rate:
            alerts.append(
                Alert(
                    level=AlertLevel.INFO,
                    message=f"Win rate {win_rate:.2%} below threshold {min_win_rate:.2%}",
                    context={"win_rate": win_rate},
                )
            )
        return alerts

    return [drawdown_rule, sharpe_rule, win_rate_rule]
