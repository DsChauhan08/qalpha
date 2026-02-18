"""
AI/Memory/Datacenter regime feature builder.

The goal is to capture persistent thematic booms (e.g., AI infrastructure
cycles) using only free market data proxies.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_FILE = (
    PROJECT_DIR / "data_store" / "meta_ensemble" / "ai_regime_features.pkl"
)

AI_THEME_BASKET = [
    "NVDA",
    "AMD",
    "AVGO",
    "SMH",
    "SOXX",
    "ANET",
]
MEMORY_BASKET = [
    "MU",
    "WDC",
    "STX",
]
DATACENTER_BASKET = [
    "NVDA",
    "AMD",
    "AVGO",
    "ANET",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
]

AI_REGIME_FEATURES = [
    "ai_theme_mom_63",
    "ai_theme_mom_126",
    "ai_breadth_200dma",
    "ai_breadth_mom63",
    "ai_memory_rel_63",
    "ai_datacenter_rel_63",
    "ai_regime_strength",
    "ai_regime_weak",
]


def _sanitize_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx
    return out.sort_index()


def _fetch_close_series(
    symbols: Iterable[str],
    start: datetime | None = None,
    end: datetime | None = None,
    cache_dir: str | None = None,
) -> Dict[str, pd.Series]:
    from quantum_alpha.data.collectors.market_data import DataCollector

    collector = DataCollector(cache_dir=cache_dir or ".cache")
    if start is None:
        start = datetime(2005, 1, 1)
    if end is None:
        end = datetime.now()

    result: Dict[str, pd.Series] = {}
    for sym in symbols:
        try:
            df = collector.fetch_ohlcv(sym, start=start, end=end, interval="1d")
        except Exception:
            continue
        if df is None or len(df) < 252 or "close" not in df.columns:
            continue
        sdf = _sanitize_index(df)
        result[sym] = pd.to_numeric(sdf["close"], errors="coerce")
    return result


def _safe_avg(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    use = [c for c in cols if c in df.columns]
    if not use:
        return pd.Series(0.0, index=df.index)
    return df[use].mean(axis=1, skipna=True)


def compute_ai_regime_features(
    force_refresh: bool = False,
    cache_file: Path = DEFAULT_CACHE_FILE,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """
    Build daily thematic regime features from ETF/equity proxies.
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists() and not force_refresh:
        try:
            cached = pd.read_pickle(cache_file)
            if isinstance(cached, pd.DataFrame) and len(cached) > 0:
                return cached
        except Exception:
            pass

    symbols = sorted(
        set(AI_THEME_BASKET + MEMORY_BASKET + DATACENTER_BASKET + ["SPY"])
    )
    close_map = _fetch_close_series(symbols=symbols, cache_dir=cache_dir)
    if "SPY" not in close_map:
        raise RuntimeError("Could not load SPY for AI regime features")

    closes = pd.DataFrame(close_map).sort_index()
    closes = closes.ffill()
    spy_close = closes["SPY"]
    returns = closes.pct_change()

    ai_theme_ret = _safe_avg(returns, AI_THEME_BASKET)
    memory_ret = _safe_avg(returns, MEMORY_BASKET)
    datacenter_ret = _safe_avg(returns, DATACENTER_BASKET)
    spy_ret = spy_close.pct_change()

    feat = pd.DataFrame(index=closes.index)
    feat["ai_theme_mom_63"] = ai_theme_ret.rolling(63, min_periods=20).sum()
    feat["ai_theme_mom_126"] = ai_theme_ret.rolling(126, min_periods=40).sum()

    ai_basket_cols = [c for c in AI_THEME_BASKET if c in closes.columns]
    if ai_basket_cols:
        ai_close = closes[ai_basket_cols]
        feat["ai_breadth_200dma"] = (
            ai_close > ai_close.rolling(200, min_periods=120).mean()
        ).mean(axis=1)
        feat["ai_breadth_mom63"] = (ai_close.pct_change(63) > 0).mean(axis=1)
    else:
        feat["ai_breadth_200dma"] = 0.5
        feat["ai_breadth_mom63"] = 0.5

    feat["ai_memory_rel_63"] = (
        memory_ret.rolling(63, min_periods=20).sum()
        - spy_ret.rolling(63, min_periods=20).sum()
    )
    feat["ai_datacenter_rel_63"] = (
        datacenter_ret.rolling(63, min_periods=20).sum()
        - spy_ret.rolling(63, min_periods=20).sum()
    )

    breadth_centered = (feat["ai_breadth_200dma"] - 0.5) * 2.0
    mom_score = np.tanh(feat["ai_theme_mom_126"] / 0.35)
    mem_score = np.tanh(feat["ai_memory_rel_63"] / 0.20)
    dc_score = np.tanh(feat["ai_datacenter_rel_63"] / 0.20)
    feat["ai_regime_strength"] = (
        0.40 * breadth_centered + 0.30 * mom_score + 0.15 * mem_score + 0.15 * dc_score
    )
    feat["ai_regime_weak"] = (feat["ai_regime_strength"] < -0.05).astype(float)

    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.ffill().fillna(0.0)
    feat = feat[AI_REGIME_FEATURES]
    feat.to_pickle(cache_file)

    logger.info(
        "AI regime features built: %d rows (%s -> %s)",
        len(feat),
        feat.index.min().date() if len(feat) else "n/a",
        feat.index.max().date() if len(feat) else "n/a",
    )
    return feat


def load_ai_regime_features(
    cache_file: Path = DEFAULT_CACHE_FILE,
    force_refresh: bool = False,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    return compute_ai_regime_features(
        force_refresh=force_refresh,
        cache_file=cache_file,
        cache_dir=cache_dir,
    )
