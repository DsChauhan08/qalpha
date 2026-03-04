"""
Diagnostics for enhanced strategy factor quality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


DEFAULT_FACTOR_COLUMNS = {
    "momentum": "component_momentum",
    "mean_rev": "component_mean_rev",
    "trend": "component_trend",
    "breakout": "component_breakout",
    "ts_mom": "component_ts_mom",
    "xs_momentum": "component_xs_momentum",
    "stat_arb": "component_stat_arb",
    "regime_mom": "component_regime_mom",
}


@dataclass
class _FactorStats:
    n_obs: int
    n_active: int
    rolling_ic_mean: float
    rolling_ic_std: float
    rolling_ic_ir: float
    hit_rate: float
    avg_forward_spread_top_bottom: float

    def as_dict(self) -> Dict[str, float | int]:
        return {
            "n_obs": int(self.n_obs),
            "n_active": int(self.n_active),
            "rolling_ic_mean": float(self.rolling_ic_mean),
            "rolling_ic_std": float(self.rolling_ic_std),
            "rolling_ic_ir": float(self.rolling_ic_ir),
            "hit_rate": float(self.hit_rate),
            "avg_forward_spread_top_bottom": float(self.avg_forward_spread_top_bottom),
        }


def _segment_label(ts: pd.Timestamp) -> str:
    y = int(ts.year)
    if y == 2008:
        return "stress_2008"
    if y == 2020:
        return "stress_2020"
    if y == 2022:
        return "stress_2022"
    if y <= 2013:
        return "old"
    if y <= 2019:
        return "mid"
    return "recent"


def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def _build_panel(
    frames: Dict[str, pd.DataFrame],
    factor_columns: Dict[str, str],
    forward_periods: int,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for symbol, df in frames.items():
        if not isinstance(df, pd.DataFrame) or df.empty or "close" not in df.columns:
            continue
        local = pd.DataFrame(index=df.index.copy())
        local["symbol"] = str(symbol)
        close = pd.to_numeric(df["close"], errors="coerce")
        local["forward_return"] = close.shift(-forward_periods) / close - 1.0
        for factor_name, col in factor_columns.items():
            if col in df.columns:
                local[factor_name] = pd.to_numeric(df[col], errors="coerce")
            else:
                local[factor_name] = np.nan
        local = local.replace([np.inf, -np.inf], np.nan)
        local = local.dropna(subset=["forward_return"])
        if local.empty:
            continue
        out = local.reset_index().rename(columns={"index": "timestamp"})
        rows.append(out)

    if not rows:
        return pd.DataFrame()

    panel = pd.concat(rows, axis=0, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], errors="coerce")
    panel = panel.dropna(subset=["timestamp", "forward_return"])
    panel["segment"] = panel["timestamp"].map(_segment_label)
    return panel


def _daily_ic(frame: pd.DataFrame, factor: str, min_obs: int = 10) -> pd.Series:
    daily: List[float] = []
    idx: List[pd.Timestamp] = []
    for ts, grp in frame.groupby("timestamp"):
        g = grp[[factor, "forward_return"]].dropna()
        if len(g) < min_obs:
            continue
        if g[factor].nunique(dropna=True) < 2 or g["forward_return"].nunique(dropna=True) < 2:
            continue
        ic = g[factor].corr(g["forward_return"], method="spearman")
        if ic is None or not np.isfinite(ic):
            continue
        idx.append(pd.Timestamp(ts))
        daily.append(float(ic))
    if not daily:
        return pd.Series(dtype=float)
    return pd.Series(daily, index=pd.to_datetime(idx)).sort_index()


def _daily_top_bottom_spread(
    frame: pd.DataFrame,
    factor: str,
    quantile: float = 0.2,
    min_obs: int = 10,
) -> pd.Series:
    spreads: List[float] = []
    idx: List[pd.Timestamp] = []
    q = float(np.clip(quantile, 0.01, 0.49))
    for ts, grp in frame.groupby("timestamp"):
        g = grp[[factor, "forward_return"]].dropna()
        if len(g) < min_obs:
            continue
        hi = g[factor].quantile(1.0 - q)
        lo = g[factor].quantile(q)
        top = g.loc[g[factor] >= hi, "forward_return"]
        bot = g.loc[g[factor] <= lo, "forward_return"]
        if top.empty or bot.empty:
            continue
        spread = float(top.mean() - bot.mean())
        if not np.isfinite(spread):
            continue
        idx.append(pd.Timestamp(ts))
        spreads.append(spread)
    if not spreads:
        return pd.Series(dtype=float)
    return pd.Series(spreads, index=pd.to_datetime(idx)).sort_index()


def _factor_stats(frame: pd.DataFrame, factor: str) -> _FactorStats:
    clean = frame[[factor, "forward_return", "timestamp"]].dropna()
    if clean.empty:
        return _FactorStats(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    active = clean[np.abs(clean[factor]) > 1e-12]
    if active.empty:
        hit_rate = 0.0
    else:
        hit_rate = float(
            (np.sign(active[factor]) == np.sign(active["forward_return"])).mean()
        )

    ic_daily = _daily_ic(clean, factor=factor)
    if ic_daily.empty:
        rolling_mean = 0.0
        rolling_std = 0.0
    else:
        rolling = ic_daily.rolling(window=63, min_periods=20).mean().dropna()
        if rolling.empty:
            rolling_mean = _safe_float(ic_daily.mean())
            rolling_std = _safe_float(ic_daily.std(ddof=0))
        else:
            rolling_mean = _safe_float(rolling.mean())
            rolling_std = _safe_float(rolling.std(ddof=0))
    rolling_ir = rolling_mean / rolling_std if rolling_std > 1e-12 else 0.0

    spread_daily = _daily_top_bottom_spread(clean, factor=factor)
    spread_avg = _safe_float(spread_daily.mean()) if not spread_daily.empty else 0.0

    return _FactorStats(
        n_obs=int(len(clean)),
        n_active=int(len(active)),
        rolling_ic_mean=float(rolling_mean),
        rolling_ic_std=float(rolling_std),
        rolling_ic_ir=float(rolling_ir),
        hit_rate=float(hit_rate),
        avg_forward_spread_top_bottom=float(spread_avg),
    )


def _ablation_summary(
    panel: pd.DataFrame,
    factors: Iterable[str],
    quantile: float = 0.2,
) -> Dict[str, Dict[str, float | int]]:
    out: Dict[str, Dict[str, float | int]] = {}
    for factor in factors:
        clean = panel[[factor, "forward_return", "timestamp"]].dropna()
        if clean.empty:
            out[factor] = {
                "n_obs": 0,
                "n_days": 0,
                "top_quantile_avg_forward_return": 0.0,
                "bottom_quantile_avg_forward_return": 0.0,
                "spread": 0.0,
            }
            continue
        q = float(np.clip(quantile, 0.01, 0.49))
        top_vals: List[float] = []
        bottom_vals: List[float] = []
        for _, grp in clean.groupby("timestamp"):
            hi = grp[factor].quantile(1.0 - q)
            lo = grp[factor].quantile(q)
            top = grp.loc[grp[factor] >= hi, "forward_return"].mean()
            bot = grp.loc[grp[factor] <= lo, "forward_return"].mean()
            if np.isfinite(top):
                top_vals.append(float(top))
            if np.isfinite(bot):
                bottom_vals.append(float(bot))
        top_avg = _safe_float(np.mean(top_vals)) if top_vals else 0.0
        bot_avg = _safe_float(np.mean(bottom_vals)) if bottom_vals else 0.0
        out[factor] = {
            "n_obs": int(len(clean)),
            "n_days": int(len(top_vals)),
            "top_quantile_avg_forward_return": float(top_avg),
            "bottom_quantile_avg_forward_return": float(bot_avg),
            "spread": float(top_avg - bot_avg),
        }
    return out


def run_enhanced_factor_diagnostics(
    frames: Dict[str, pd.DataFrame],
    output_dir: Path,
    factor_columns: Optional[Dict[str, str]] = None,
    forward_periods: int = 5,
    quantile: float = 0.2,
) -> Dict[str, object]:
    """
    Compute and persist enhanced factor diagnostics artifacts.
    """
    cols = dict(DEFAULT_FACTOR_COLUMNS)
    if factor_columns:
        cols.update({str(k): str(v) for k, v in factor_columns.items()})

    panel = _build_panel(frames=frames, factor_columns=cols, forward_periods=forward_periods)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_rows: List[Dict[str, object]] = []
    seg_rows: List[Dict[str, object]] = []
    for factor in cols.keys():
        fs = _factor_stats(panel, factor=factor)
        full_rows.append({"factor": factor, **fs.as_dict()})
        if not panel.empty:
            for segment, seg in panel.groupby("segment"):
                seg_stats = _factor_stats(seg, factor=factor)
                seg_rows.append(
                    {"factor": factor, "segment": str(segment), **seg_stats.as_dict()}
                )

    full_df = pd.DataFrame(full_rows)
    if not full_df.empty and "factor" in full_df.columns:
        full_df = full_df.sort_values("factor")
    seg_df = pd.DataFrame(seg_rows)
    if not seg_df.empty and {"factor", "segment"}.issubset(seg_df.columns):
        seg_df = seg_df.sort_values(["factor", "segment"])
    ablation = _ablation_summary(panel=panel, factors=cols.keys(), quantile=quantile)

    full_path = output_dir / "factor_quality_full.csv"
    seg_path = output_dir / "factor_quality_by_segment.csv"
    ablation_path = output_dir / "factor_ablation_summary.json"

    full_df.to_csv(full_path, index=False)
    seg_df.to_csv(seg_path, index=False)
    with open(ablation_path, "w", encoding="utf-8") as f:
        json.dump(ablation, f, indent=2, sort_keys=True)

    return {
        "output_dir": str(output_dir),
        "factor_quality_full_csv": str(full_path),
        "factor_quality_by_segment_csv": str(seg_path),
        "factor_ablation_summary_json": str(ablation_path),
        "n_rows_panel": int(len(panel)),
        "n_factors": int(len(cols)),
    }
