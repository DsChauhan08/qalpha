"""
Daily anchor cache for meta dual-blend live scoring.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def _to_utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _latest_daily_bar(featured: Dict[str, pd.DataFrame]) -> Optional[pd.Timestamp]:
    latest: Optional[pd.Timestamp] = None
    for df in featured.values():
        if df is None or df.empty:
            continue
        ts = pd.Timestamp(df.index[-1]).normalize()
        if latest is None or ts > latest:
            latest = ts
    return latest


def load_anchor_cache(cache_path: str | Path) -> Dict:
    p = Path(cache_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def anchor_cache_freshness_minutes(payload: Dict) -> Optional[float]:
    generated_at = payload.get("generated_at_utc")
    if not generated_at:
        return None
    try:
        ts = pd.Timestamp(generated_at)
    except Exception:
        return None
    now = pd.Timestamp.now(tz="UTC")
    return float(max((now - ts).total_seconds() / 60.0, 0.0))


def _is_same_asof(payload: Dict, asof_date: str) -> bool:
    cached = str(payload.get("asof_date", "")).strip()
    return bool(cached and cached == asof_date)


def build_anchor_cache_payload(
    featured: Dict[str, pd.DataFrame],
    strategy,
    *,
    decision_engine: str = "meta_blend_hybrid",
) -> Dict:
    latest = _latest_daily_bar(featured)
    if latest is None:
        return {}
    asof_date = str(latest.date())
    predictions = strategy.build_anchor_predictions(featured)
    return {
        "asof_date": asof_date,
        "generated_at_utc": _to_utc_iso(datetime.now(timezone.utc)),
        "decision_engine": str(decision_engine),
        "symbols_scored": int(len(predictions)),
        "predictions": predictions,
        "model_health_base": bool(strategy.model_health().get("base_ok", False)),
        "model_health_mc": bool(strategy.model_health().get("mc_ok", False)),
    }


def refresh_anchor_cache_if_needed(
    featured: Dict[str, pd.DataFrame],
    strategy,
    cache_path: str | Path,
    *,
    force: bool = False,
    decision_engine: str = "meta_blend_hybrid",
) -> Dict:
    latest = _latest_daily_bar(featured)
    if latest is None:
        return {}
    asof_date = str(latest.date())
    cache_file = Path(cache_path)
    current = load_anchor_cache(cache_file)

    if not force and _is_same_asof(current, asof_date):
        return current

    payload = build_anchor_cache_payload(
        featured,
        strategy,
        decision_engine=decision_engine,
    )
    if not payload:
        return current

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload

