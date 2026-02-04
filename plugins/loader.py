"""
Plugin discovery and registry for strategies, models, and data.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Dict, Callable


@dataclass
class PluginRegistry:
    strategies: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)

    def register(self, category: str, name: str, obj: Any) -> None:
        if category not in {"strategies", "models", "data"}:
            raise ValueError(f"Unknown plugin category: {category}")
        getattr(self, category)[name] = obj


REGISTRY = PluginRegistry()


def register_strategy(name: str) -> Callable:
    def decorator(obj: Any) -> Any:
        REGISTRY.register("strategies", name, obj)
        return obj

    return decorator


def register_model(name: str) -> Callable:
    def decorator(obj: Any) -> Any:
        REGISTRY.register("models", name, obj)
        return obj

    return decorator


def register_data(name: str) -> Callable:
    def decorator(obj: Any) -> Any:
        REGISTRY.register("data", name, obj)
        return obj

    return decorator


def _discover(package) -> None:
    for _, module_name, _ in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        importlib.import_module(module_name)


def load_plugins() -> PluginRegistry:
    """
    Discover and load plugins in quantum_alpha.plugins.
    """
    from quantum_alpha.plugins import strategies, models, data

    _discover(strategies)
    _discover(models)
    _discover(data)

    return REGISTRY
