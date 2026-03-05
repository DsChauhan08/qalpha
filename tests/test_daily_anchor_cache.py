from pathlib import Path

import numpy as np
import pandas as pd

from quantum_alpha.execution.daily_anchor_cache import (
    anchor_cache_freshness_minutes,
    refresh_anchor_cache_if_needed,
)


class _FakeStrategy:
    def __init__(self):
        self.calls = 0

    def build_anchor_predictions(self, featured):
        self.calls += 1
        out = {}
        for sym in sorted(featured.keys()):
            out[sym] = {
                "up_probability_blend": 0.6,
                "up_probability": 0.6,
                "confidence": 0.2,
                "missing_feature_ratio_base": 0.0,
                "missing_feature_ratio_mc": 0.0,
                "model_used": "blended",
            }
        return out

    def model_health(self):
        return {
            "base_ok": True,
            "mc_ok": True,
            "base_feature_set": "base",
            "mc_feature_set": "mc_pade",
        }


def _frames():
    idx = pd.date_range("2026-03-01", periods=20, freq="5min")
    close = np.linspace(100, 101, len(idx))
    base = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": np.full(len(idx), 1_000_000.0),
        },
        index=idx,
    )
    return {"AAA": base.copy(), "BBB": base.copy()}


def test_anchor_cache_refresh_and_reuse(tmp_path):
    strategy = _FakeStrategy()
    cache_path = Path(tmp_path) / "anchor_cache.json"
    featured = _frames()

    payload1 = refresh_anchor_cache_if_needed(
        featured=featured,
        strategy=strategy,
        cache_path=cache_path,
        force=False,
        decision_engine="meta_blend_hybrid",
    )
    assert payload1.get("asof_date") == "2026-03-01"
    assert int(payload1.get("symbols_scored", 0)) == 2
    assert payload1.get("model_health", {}).get("mc_feature_set") == "mc_pade"
    assert payload1.get("model_health_mc") is True
    assert strategy.calls == 1

    payload2 = refresh_anchor_cache_if_needed(
        featured=featured,
        strategy=strategy,
        cache_path=cache_path,
        force=False,
        decision_engine="meta_blend_hybrid",
    )
    assert strategy.calls == 1
    assert payload2.get("asof_date") == payload1.get("asof_date")
    assert anchor_cache_freshness_minutes(payload2) is not None
