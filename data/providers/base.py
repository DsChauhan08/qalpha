"""Provider contracts and shared result containers for data routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol


@dataclass
class ProviderResult:
    """Normalized provider fetch result used by the quality router."""

    data: Any
    provider: str
    domain: str
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float = 0.0
    completeness: float = 1.0
    reliability: float = 1.0
    error: Optional[str] = None
    degraded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.data is not None


class MarketDataProvider(Protocol):
    """Contract for market and fundamental data providers."""

    name: str

    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> ProviderResult:
        ...

    def fetch_fundamentals(self, symbol: str) -> ProviderResult:
        ...


class AlternativeDataProvider(Protocol):
    """Contract for optional alternative data domains."""

    name: str

    def fetch_news(self, symbol: str, max_items: int = 50) -> ProviderResult:
        ...

    def fetch_macro(self, series: str, start: Optional[datetime] = None) -> ProviderResult:
        ...

    def fetch_options(self, symbol: str) -> ProviderResult:
        ...

    def fetch_insider(self, symbol: str) -> ProviderResult:
        ...

    def fetch_congress(self, symbol: str) -> ProviderResult:
        ...

    def fetch_earnings(self, symbol: str) -> ProviderResult:
        ...


__all__ = ["ProviderResult", "MarketDataProvider", "AlternativeDataProvider"]
