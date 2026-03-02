"""Provider registry and builder for runtime-mode routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from quantum_alpha.data.providers import (
    FallbackProvider,
    OpenBBAPIProvider,
    OpenBBSDKProvider,
    YFinanceProvider,
)


@dataclass
class ProviderRegistry:
    """Holds domain to provider mappings."""

    domains: Dict[str, List[object]] = field(default_factory=dict)

    def register(self, domain: str, provider: object) -> None:
        self.domains.setdefault(domain, []).append(provider)

    def get(self, domain: str) -> List[object]:
        return list(self.domains.get(domain, []))


def build_provider_registry(
    runtime_mode: str,
    data_cfg: Optional[Dict],
    enable_openbb: Optional[bool] = None,
) -> ProviderRegistry:
    """Build providers by mode and config feature flags."""

    data_cfg = data_cfg or {}
    openbb_cfg = data_cfg.get("openbb", {}) if isinstance(data_cfg, dict) else {}

    enabled = (
        bool(enable_openbb)
        if enable_openbb is not None
        else bool(openbb_cfg.get("enabled", False))
    )
    free_only = bool(openbb_cfg.get("free_only", True))
    provider_hint = openbb_cfg.get("provider")

    sdk_modes = set(str(x).lower() for x in openbb_cfg.get("sdk_modes", ["backtest", "paper", "train"]))
    api_modes = set(str(x).lower() for x in openbb_cfg.get("api_modes", ["realtime", "live"]))

    api_cfg = openbb_cfg.get("api", {}) if isinstance(openbb_cfg, dict) else {}
    api_base = str(api_cfg.get("base_url", "http://127.0.0.1:6900"))
    api_timeout_ms = float(api_cfg.get("timeout_ms", 1500.0))
    api_timeout = max(0.1, api_timeout_ms / 1000.0)

    reg = ProviderRegistry()

    # Always retain baseline providers for failover.
    yf = YFinanceProvider()
    fallback = FallbackProvider()

    if enabled and runtime_mode.lower() in api_modes:
        openbb_provider = OpenBBAPIProvider(
            base_url=api_base, provider=provider_hint, timeout=api_timeout
        )
        reg.register("market_data", openbb_provider)
        reg.register("fundamentals", openbb_provider)
        reg.register("news", openbb_provider)
        reg.register("macro", openbb_provider)
        reg.register("options", openbb_provider)
        reg.register("insider", openbb_provider)
        reg.register("congress", openbb_provider)
        reg.register("earnings", openbb_provider)
    elif enabled and runtime_mode.lower() in sdk_modes:
        openbb_provider = OpenBBSDKProvider(provider=provider_hint, free_only=free_only)
        reg.register("market_data", openbb_provider)
        reg.register("fundamentals", openbb_provider)
        reg.register("news", openbb_provider)
        reg.register("macro", openbb_provider)
        reg.register("options", openbb_provider)
        reg.register("insider", openbb_provider)
        reg.register("congress", openbb_provider)
        reg.register("earnings", openbb_provider)

    reg.register("market_data", yf)
    reg.register("fundamentals", yf)
    reg.register("news", yf)
    reg.register("options", yf)
    reg.register("earnings", yf)
    reg.register("market_data", fallback)
    reg.register("fundamentals", fallback)
    reg.register("news", fallback)
    reg.register("macro", fallback)
    reg.register("options", fallback)
    reg.register("insider", fallback)
    reg.register("congress", fallback)
    reg.register("earnings", fallback)

    for domain in ("news", "macro", "options", "insider", "congress", "earnings"):
        reg.domains.setdefault(domain, [])

    return reg


__all__ = ["ProviderRegistry", "build_provider_registry"]
