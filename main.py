"""
Quantum Alpha V1 - Main Entry Point
Single command deployment for backtesting.
"""

import sys
import json
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_alpha.data.collectors.market_data import DataCollector
from quantum_alpha.data.preprocessing.cleaners import DataCleaner
from quantum_alpha.data.preprocessing.imputers import MissingValueImputer
from quantum_alpha.features.technical.indicators import TechnicalFeatureGenerator
from quantum_alpha.strategy.signals import (
    MomentumStrategy,
    CompositeStrategy,
    AdaptiveCompositeStrategy,
    EnhancedCompositeStrategy,
)
from quantum_alpha.backtesting.engine import Backtester, OrderSide, OrderType
from quantum_alpha.backtesting.validation import MCPT, BootstrapAnalysis
from quantum_alpha.backtesting.performance_metrics import (
    compute_metrics,
    compute_metrics_from_returns,
)
from quantum_alpha.backtesting.enhanced_factor_diagnostics import (
    run_enhanced_factor_diagnostics,
)
from quantum_alpha.backtesting.performance_gate import (
    evaluate_gate,
    aggregate_fundamentals,
)
from quantum_alpha.backtesting.benchmark_profiles import (
    evaluate_quant_firm_benchmarks,
    benchmark_rows,
)
from quantum_alpha.risk.position_sizing import PositionSizer, VaRCalculator
from quantum_alpha.risk.drawdown_control import DrawdownController, DrawdownState
from quantum_alpha.execution.paper_trader import PaperTrader
from quantum_alpha.config.validator import (
    validate_settings,
    validate_strategies,
    validate_risk_limits,
    validate_data_sources,
)
from quantum_alpha.monitoring.logging import configure_logging
from quantum_alpha.monitoring.alert_system import AlertManager, build_default_rules
from quantum_alpha.plugins import load_plugins


def _deep_merge_dicts(base: Optional[Dict], override: Optional[Dict]) -> Dict:
    """Recursively merge two mapping objects."""
    out: Dict[str, Any] = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dicts(out.get(key), value)
        else:
            out[key] = value
    return out


def _load_yaml_file(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        blob = yaml.safe_load(f) or {}
    if not isinstance(blob, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return blob


def load_config(config_path: str = None) -> Dict:
    """Load configuration from YAML file."""
    default_settings_path = PROJECT_ROOT / "quantum_alpha" / "config" / "settings.yaml"
    base_settings = _load_yaml_file(default_settings_path)

    if config_path is None:
        resolved_config_path = default_settings_path
        override_settings = {}
    else:
        resolved_config_path = Path(config_path)
        if resolved_config_path.is_dir():
            resolved_config_path = resolved_config_path / "settings.yaml"
        override_settings = _load_yaml_file(resolved_config_path)

    settings = _deep_merge_dicts(base_settings, override_settings)

    issues = validate_settings(settings)
    config_dirs: List[Path] = [default_settings_path.parent]
    resolved_dir = resolved_config_path.parent
    if resolved_dir not in config_dirs:
        config_dirs.append(resolved_dir)

    for config_dir in config_dirs:
        strategies_path = config_dir / "strategies.yaml"
        if strategies_path.exists():
            strategies_cfg = _load_yaml_file(strategies_path)
            issues.extend(validate_strategies(strategies_cfg))

        risk_limits_path = config_dir / "risk_limits.yaml"
        if risk_limits_path.exists():
            risk_cfg = _load_yaml_file(risk_limits_path)
            issues.extend(validate_risk_limits(risk_cfg))

        data_sources_path = config_dir / "data_sources.yaml"
        if data_sources_path.exists():
            data_cfg = _load_yaml_file(data_sources_path)
            issues.extend(validate_data_sources(data_cfg))

    if issues:
        raise ValueError(f"Config validation failed: {', '.join(issues)}")

    return settings


def _resolve_config_dir(config_path: Optional[str]) -> Path:
    if config_path is None:
        return PROJECT_ROOT / "quantum_alpha" / "config"

    config_path = Path(config_path)
    return config_path if config_path.is_dir() else config_path.parent


def _load_optional_yaml(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _apply_signal_lag(df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Lag signals and sizing features to avoid lookahead."""
    for col in (
        "signal",
        "signal_confidence",
        "position_signal",
        "atr_pct",
        "mom_12m",
        "mom_3m",
    ):
        if col in df.columns:
            df[col] = df[col].shift(lag)

    if "signal" in df.columns:
        df["signal"] = df["signal"].fillna(0.0)
    if "signal_confidence" in df.columns:
        df["signal_confidence"] = df["signal_confidence"].fillna(0.0)
    if "position_signal" in df.columns:
        df["position_signal"] = df["position_signal"].fillna(0.0)
    if "atr_pct" in df.columns:
        df["atr_pct"] = df["atr_pct"].fillna(0.02)
    if "mom_12m" in df.columns:
        df["mom_12m"] = df["mom_12m"].fillna(0.0)
    if "mom_3m" in df.columns:
        df["mom_3m"] = df["mom_3m"].fillna(0.0)

    return df


def _realized_vol_from_equity(equity_curve, window: int = 63) -> Optional[float]:
    """Compute annualized realized vol from equity curve entries."""
    if not equity_curve or len(equity_curve) < 2:
        return None
    eq = np.array([row["equity"] for row in equity_curve], dtype=float)
    if len(eq) < 2:
        return None
    window = min(window, len(eq) - 1)
    if window < 2:
        return None
    eq = eq[-(window + 1) :]
    rets = np.diff(eq) / eq[:-1]
    if rets.std() == 0:
        return None
    return float(rets.std() * np.sqrt(252))


def _vol_of_vol_from_equity(
    equity_curve, short_window: int = 21, long_window: int = 63
) -> Optional[float]:
    """Estimate volatility of volatility from equity curve."""
    if not equity_curve or len(equity_curve) < (long_window + 2):
        return None
    eq = np.array([row["equity"] for row in equity_curve], dtype=float)
    rets = np.diff(eq) / eq[:-1]
    if len(rets) < long_window:
        return None
    vol_series = pd.Series(rets).rolling(short_window).std().dropna()
    if len(vol_series) < long_window // 2:
        return None
    return float(vol_series.tail(long_window).std())


def _align_signal_frame(
    sig_df: pd.DataFrame, index: pd.Index, limit: int = 10
) -> pd.DataFrame:
    """Align sparse signal frames to price index with a short forward fill."""
    if sig_df.empty:
        return pd.DataFrame(
            {"signal": np.zeros(len(index)), "signal_confidence": np.zeros(len(index))},
            index=index,
        )
    frame = sig_df.copy()
    if "timestamp" in frame.columns:
        frame = frame.set_index("timestamp")
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    signal = frame["signal"].reindex(index, method="ffill", limit=limit).fillna(0.0)
    confidence = (
        frame.get("signal_confidence", pd.Series(0.5, index=frame.index))
        .reindex(index, method="ffill", limit=limit)
        .fillna(0.0)
    )
    return pd.DataFrame(
        {"signal": signal, "signal_confidence": confidence}, index=index
    )


def _resolve_symbols(
    symbols: Optional[list],
    collector: DataCollector,
    settings: Optional[Dict],
) -> list:
    data_cfg = settings.get("data", {}) if settings else {}
    universe_limit = int(data_cfg.get("universe_limit", 0))

    def _limit(values: list) -> list:
        if universe_limit and universe_limit > 0:
            return list(values)[:universe_limit]
        return list(values)

    # Try to load from centralized universe module
    try:
        from quantum_alpha import universe as _u

        _has_universe = True
    except ImportError:
        _has_universe = False

    if symbols:
        if len(symbols) == 1:
            token = symbols[0].upper()
            if token in {"SP500", "S&P500", "SPX"}:
                if _has_universe:
                    return _limit(_u.get_sp500())
                return _limit(collector.get_sp500_symbols())
            if token in {"SP400", "MIDCAP", "MID"}:
                if _has_universe:
                    return _limit(_u.get_sp400())
                logger.warning(
                    "universe.py not available; SP400 token unsupported, falling back to SP500"
                )
                return _limit(collector.get_sp500_symbols())
            if token in {"FULL", "ALL", "UNIVERSE"}:
                if _has_universe:
                    return _limit(_u.get_stocks_only())
                logger.warning(
                    "universe.py not available; FULL token unsupported, falling back to SP500"
                )
                return _limit(collector.get_sp500_symbols())
            if token in {"LIQUID", "LIQUID50"}:
                if _has_universe:
                    return _limit(_u.get_liquid_largecap())
                return _limit(collector.get_sp500_symbols()[:50])
            if token in {"AUTO", "DEFAULT"}:
                return _limit(data_cfg.get("default_universe", ["SPY"]))
        return symbols

    return _limit(data_cfg.get("default_universe", ["SPY"]))


def _format_symbols(symbols: list) -> str:
    if len(symbols) <= 15:
        return str(symbols)
    head = ", ".join(symbols[:10])
    return f"{len(symbols)} symbols (e.g., {head}, ...)"


def _augment_symbols_with_benchmarks(
    symbols: list,
    core_mix: Dict[str, float],
    market_benchmark: Optional[str],
    quant_benchmark: Optional[list],
) -> list:
    """Ensure benchmark/core symbols are always present in the fetch universe."""
    out = []
    seen = set()

    def _add(sym) -> None:
        s = str(sym).upper().strip()
        if not s or s in seen:
            return
        seen.add(s)
        out.append(s)

    for sym in symbols:
        _add(sym)
    for sym in core_mix.keys():
        _add(sym)
    if isinstance(quant_benchmark, list):
        for sym in quant_benchmark:
            _add(sym)
    elif quant_benchmark:
        _add(quant_benchmark)
    if market_benchmark:
        _add(market_benchmark)

    return out


def _composite_returns_from_frames(
    frames: Dict[str, pd.DataFrame],
    symbols: list,
) -> Optional[pd.Series]:
    """Build equal-weight returns from already-loaded OHLCV frames."""
    parts = []
    for sym in symbols:
        s = str(sym).upper()
        df = frames.get(s) or frames.get(str(sym))
        if df is None or df.empty or "close" not in df.columns:
            continue
        close = pd.to_numeric(df["close"], errors="coerce")
        ret = close.pct_change(fill_method=None)
        series = _normalize_return_index(
            pd.Series(ret.values, index=pd.to_datetime(df.index, errors="coerce"))
        )
        series = series.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
        if not series.empty:
            parts.append(series)

    if not parts:
        return None
    composite = pd.concat(parts, axis=1).mean(axis=1).dropna().sort_index()
    composite.name = "composite"
    return composite if not composite.empty else None


def _fallback_core_targets_from_bars(
    bars: Dict[str, Dict[str, float]],
    core_budget: float,
    max_abs_per_symbol: float,
    top_n: int = 12,
) -> Dict[str, float]:
    """Last-resort core basket from most liquid currently tradable symbols."""
    if core_budget <= 0 or not bars:
        return {}

    scores = {}
    for sym, bar in bars.items():
        px = float(bar.get("open", bar.get("close", 0.0)) or 0.0)
        vol = float(bar.get("volume", 0.0) or 0.0)
        if px <= 0 or vol <= 0 or not np.isfinite(px) or not np.isfinite(vol):
            continue
        adv = px * vol
        if np.isfinite(adv) and adv > 0:
            scores[sym] = float(adv)

    if not scores:
        return {}

    n_keep = max(1, int(top_n))
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_keep]
    raw = {sym: val for sym, val in ranked}
    return _normalize_weights(
        raw,
        budget=core_budget,
        long_only=True,
        max_abs_per_symbol=max_abs_per_symbol,
    )


def _select_liquid_subset(
    frames: Dict[str, pd.DataFrame],
    subset_size: int,
    adv_window: int = 20,
    min_history: int = 120,
) -> list:
    if subset_size <= 0 or not frames:
        return list(frames.keys())

    scores = []
    for symbol, df in frames.items():
        if df is None or len(df) < min_history:
            continue
        close = pd.to_numeric(df.get("close"), errors="coerce")
        volume = pd.to_numeric(df.get("volume"), errors="coerce")
        if close is None or volume is None:
            continue
        dollar_volume = (close * volume).replace([np.inf, -np.inf], np.nan).dropna()
        if len(dollar_volume) < max(5, adv_window):
            continue
        adv = float(dollar_volume.tail(adv_window).mean())
        if np.isfinite(adv) and adv > 0:
            scores.append((symbol, adv))

    if not scores:
        return list(frames.keys())

    scores.sort(key=lambda x: x[1], reverse=True)
    selected = [sym for sym, _ in scores[:subset_size]]
    return selected


def _total_return_from_returns(returns: pd.Series) -> float:
    if returns is None:
        return 0.0
    s = pd.Series(returns).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 1:
        return 0.0
    return float((1.0 + s).prod() - 1.0)


def _rolling_oos_vs_benchmark(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    window_days: int = 126,
    min_windows: int = 3,
    min_beat_ratio: float = 0.75,
) -> Dict[str, object]:
    strat = (
        pd.Series(strategy_returns)
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    bench = (
        pd.Series(benchmark_returns)
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    aligned = strat.align(bench, join="inner")
    if aligned[0].empty or window_days <= 1:
        return {
            "available": False,
            "reason": "insufficient_overlap",
            "n_windows": 0,
            "beats": 0,
            "passed": False,
            "windows": [],
        }

    strat_aligned, bench_aligned = aligned
    n = len(strat_aligned)
    n_windows = n // window_days
    if n_windows < 1:
        return {
            "available": False,
            "reason": "insufficient_length",
            "n_windows": 0,
            "beats": 0,
            "passed": False,
            "windows": [],
        }

    start_idx = n - (n_windows * window_days)
    windows = []
    beats = 0
    for w in range(n_windows):
        lo = start_idx + w * window_days
        hi = lo + window_days
        s_win = strat_aligned.iloc[lo:hi]
        b_win = bench_aligned.iloc[lo:hi]
        if s_win.empty or b_win.empty:
            continue
        s_ret = _total_return_from_returns(s_win)
        b_ret = _total_return_from_returns(b_win)
        beat = s_ret > b_ret
        if beat:
            beats += 1
        windows.append(
            {
                "window_id": w + 1,
                "start": str(s_win.index[0].date()),
                "end": str(s_win.index[-1].date()),
                "strategy_total_return": s_ret,
                "benchmark_total_return": b_ret,
                "beat": bool(beat),
            }
        )

    ratio = float(min_beat_ratio)
    if not np.isfinite(ratio):
        ratio = 0.75
    ratio = float(np.clip(ratio, 0.0, 1.0))
    ratio_required = int(np.ceil(ratio * len(windows))) if windows else 0
    required = max(int(min_windows), int(ratio_required))
    passed = len(windows) >= min_windows and beats >= required
    return {
        "available": True,
        "n_windows": int(len(windows)),
        "beats": int(beats),
        "required_beats": int(required),
        "passed": bool(passed),
        "window_days": int(window_days),
        "min_beat_ratio": ratio,
        "windows": windows,
    }


def _should_rebalance(
    ts: datetime, last_ts: Optional[datetime], frequency: str
) -> bool:
    if last_ts is None:
        return True

    freq = (frequency or "daily").lower()
    if freq == "daily":
        return ts.date() != last_ts.date()
    if freq == "weekly":
        ts_iso = ts.isocalendar()
        last_iso = last_ts.isocalendar()
        return (ts_iso.year, ts_iso.week) != (last_iso.year, last_iso.week)
    if freq == "biweekly":
        # Rebalance every 2 weeks (10 trading days)
        delta = (ts - last_ts).days
        return delta >= 14
    if freq == "monthly":
        return (ts.year, ts.month) != (last_ts.year, last_ts.month)
    return True


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(float(x), float(lo), float(hi)))


def _normalize_weights(
    weights: Dict[str, float],
    budget: float,
    long_only: bool,
    max_abs_per_symbol: float = 0.0,
) -> Dict[str, float]:
    b = float(max(budget, 0.0))
    if b <= 0:
        return {}

    clean = {}
    for sym, w in weights.items():
        wv = float(w)
        if not np.isfinite(wv):
            continue
        if long_only:
            wv = max(wv, 0.0)
        if abs(wv) > 0:
            clean[sym] = wv
    if not clean:
        return {}

    total_abs = sum(abs(v) for v in clean.values())
    if total_abs <= 0:
        return {}
    scaled = {s: float(v / total_abs * b) for s, v in clean.items()}

    cap = float(max_abs_per_symbol)
    if cap <= 0:
        return scaled

    clipped: Dict[str, float] = {}
    residual = b
    overflow = {}
    for sym, w in scaled.items():
        aw = abs(w)
        if aw > cap:
            clipped[sym] = float(np.sign(w) * cap)
            residual -= cap
        else:
            overflow[sym] = w

    residual = max(residual, 0.0)
    if overflow:
        rem_total = sum(abs(v) for v in overflow.values())
        if rem_total > 0 and residual > 0:
            for sym, w in overflow.items():
                clipped[sym] = float(w / rem_total * residual)
    return {s: w for s, w in clipped.items() if abs(w) > 1e-12}


def _core_targets_from_mix(
    bars: Dict[str, Dict[str, float]],
    core_mix: Dict[str, float],
    core_budget: float,
) -> Dict[str, float]:
    if core_budget <= 0:
        return {}
    available = {}
    for sym, w in core_mix.items():
        if sym not in bars:
            continue
        price = float(bars[sym].get("open", bars[sym].get("close", 0.0)) or 0.0)
        if price <= 0:
            continue
        available[sym] = float(max(w, 0.0))
    if not available:
        return {}
    return _normalize_weights(available, budget=core_budget, long_only=True)


def _tilt_core_mix(
    core_mix: Dict[str, float],
    data_frames: Dict[str, pd.DataFrame],
    timestamp: datetime,
    lookback_days: int,
    tilt_strength: float,
) -> Dict[str, float]:
    """
    Apply momentum tilt to core mix while preserving long-only normalization.

    The tilt is cross-sectional: symbols with stronger recent momentum are
    overweighted, weaker symbols underweighted.
    """
    if not core_mix:
        return {}

    strength = float(max(0.0, min(tilt_strength, 0.95)))
    lb = int(max(21, lookback_days))
    if strength <= 0.0 or len(core_mix) < 2:
        return {str(k).upper(): float(max(v, 0.0)) for k, v in core_mix.items()}

    base = {str(k).upper(): float(max(v, 0.0)) for k, v in core_mix.items()}
    moms: Dict[str, float] = {}
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)

    for sym in base.keys():
        df = data_frames.get(sym)
        if df is None or df.empty or "close" not in df.columns:
            continue
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        if close.empty:
            continue
        idx = pd.to_datetime(close.index, errors="coerce")
        if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
            idx = idx.tz_localize(None)
        close.index = idx
        close = close[~close.index.isna()].sort_index()
        hist = close[close.index <= ts]
        if len(hist) < lb + 1:
            continue
        prev = float(hist.iloc[-(lb + 1)])
        last = float(hist.iloc[-1])
        if prev <= 0 or not np.isfinite(prev) or not np.isfinite(last):
            continue
        moms[sym] = float(last / prev - 1.0)

    if len(moms) < 2:
        return base

    vals = np.array(list(moms.values()), dtype=float)
    if not np.isfinite(vals).all():
        return base
    centered = {s: float(v - vals.mean()) for s, v in moms.items()}
    denom = float(max(np.max(np.abs(list(centered.values()))), 1e-9))

    tilted = {}
    for sym, bw in base.items():
        score = float(centered.get(sym, 0.0) / denom)
        tilted[sym] = float(max(bw * (1.0 + strength * score), 0.0))

    total = float(sum(tilted.values()))
    if total <= 0:
        return base
    return {s: float(w / total) for s, w in tilted.items()}


def _combine_targets(*targets: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for t in targets:
        for sym, w in t.items():
            out[sym] = float(out.get(sym, 0.0) + float(w))
    return {s: float(w) for s, w in out.items() if abs(w) > 1e-12}


def _current_weights(bt, bars: Dict[str, Dict[str, float]], equity: float) -> Dict[str, float]:
    if equity <= 0:
        return {}
    out = {}
    for sym, pos in bt.positions.items():
        bar = bars.get(sym)
        if bar is None:
            continue
        price = float(bar.get("open", bar.get("close", 0.0)) or 0.0)
        if price <= 0:
            continue
        out[sym] = float(pos.quantity * price / equity)
    return out


def _update_loss_limit_state(
    state: Dict,
    timestamp: datetime,
    equity: float,
    max_daily_loss: float,
    max_weekly_loss: float,
) -> Dict[str, float | bool]:
    day_key = timestamp.date().isoformat()
    iso = timestamp.isocalendar()
    week_key = f"{iso.year:04d}-W{iso.week:02d}"

    if state.get("loss_day_key") != day_key:
        state["loss_day_key"] = day_key
        state["day_start_equity"] = float(equity)
        state["daily_stop_active"] = False

    if state.get("loss_week_key") != week_key:
        state["loss_week_key"] = week_key
        state["week_start_equity"] = float(equity)
        state["weekly_stop_active"] = False

    day_start = float(max(state.get("day_start_equity", equity), 1e-9))
    week_start = float(max(state.get("week_start_equity", equity), 1e-9))
    day_ret = float(equity / day_start - 1.0)
    week_ret = float(equity / week_start - 1.0)

    if max_daily_loss > 0 and day_ret <= -float(max_daily_loss):
        state["daily_stop_active"] = True
    if max_weekly_loss > 0 and week_ret <= -float(max_weekly_loss):
        state["weekly_stop_active"] = True

    stop_active = bool(state.get("daily_stop_active", False) or state.get("weekly_stop_active", False))
    state["loss_stop_active"] = stop_active

    return {
        "day_return": day_ret,
        "week_return": week_ret,
        "daily_stop_active": bool(state.get("daily_stop_active", False)),
        "weekly_stop_active": bool(state.get("weekly_stop_active", False)),
        "stop_active": stop_active,
    }


def _init_hard_drawdown_state(state: Dict) -> None:
    state.setdefault("hard_dd_halted", False)
    state.setdefault("hard_dd_triggered", False)
    state.setdefault("hard_dd_trigger_ts", None)
    state.setdefault("hard_dd_halt_started_ts", None)
    state.setdefault("hard_dd_halt_bars", 0)
    state.setdefault("hard_dd_reentries", 0)
    state.setdefault("hard_dd_reentry_ts", [])
    state.setdefault("hard_dd_last_quant_ir", None)


def _trailing_quant_information_ratio(
    equity_curve: List[Dict],
    quant_returns: Optional[pd.Series],
    lookback_bars: int = 63,
) -> Optional[float]:
    if (
        quant_returns is None
        or quant_returns.empty
        or not equity_curve
        or len(equity_curve) < max(20, lookback_bars)
    ):
        return None

    eq = pd.DataFrame(equity_curve)
    if "timestamp" not in eq.columns or "equity" not in eq.columns:
        return None
    eq["timestamp"] = pd.to_datetime(eq["timestamp"], errors="coerce")
    eq = eq.dropna(subset=["timestamp", "equity"]).set_index("timestamp").sort_index()
    if eq.empty:
        return None

    strategy_returns = eq["equity"].astype(float).pct_change(fill_method=None).dropna()
    strategy_returns = _normalize_return_index(strategy_returns)
    if strategy_returns.empty:
        return None

    strat_aligned, quant_aligned = strategy_returns.align(quant_returns, join="inner")
    if strat_aligned.empty:
        return None

    if lookback_bars > 0:
        strat_aligned = strat_aligned.tail(lookback_bars)
        quant_aligned = quant_aligned.tail(lookback_bars)
    if len(strat_aligned) < 20:
        return None

    rel_metrics = compute_metrics_from_returns(
        strat_aligned,
        benchmark_returns=quant_aligned,
    )
    ir = rel_metrics.get("information_ratio")
    if ir is None or not np.isfinite(ir):
        return None
    return float(ir)


def _update_hard_drawdown_guard(
    state: Dict,
    timestamp: datetime,
    current_drawdown: float,
    equity_curve: List[Dict],
    quant_returns: Optional[pd.Series],
    limit: float,
    action: str,
    cooldown_days: int,
    recovery_level: float,
    require_positive_quant_ir: bool,
    ir_lookback_bars: int = 63,
) -> Dict[str, object]:
    _init_hard_drawdown_state(state)
    action_key = str(action or "flatten").lower().strip()

    if limit <= 0:
        return {
            "halted": False,
            "trigger_now": False,
            "force_flatten": False,
            "reentered": False,
        }

    trigger_now = False
    reentered = False

    if not bool(state.get("hard_dd_halted", False)) and current_drawdown <= -float(limit):
        state["hard_dd_halted"] = True
        state["hard_dd_triggered"] = True
        if state.get("hard_dd_trigger_ts") is None:
            state["hard_dd_trigger_ts"] = timestamp.isoformat()
        state["hard_dd_halt_started_ts"] = timestamp.isoformat()
        trigger_now = True

    halted = bool(state.get("hard_dd_halted", False))
    if halted:
        state["hard_dd_halt_bars"] = int(state.get("hard_dd_halt_bars", 0)) + 1

        halt_started = pd.to_datetime(
            state.get("hard_dd_halt_started_ts"),
            errors="coerce",
        )
        cooldown_ok = True
        if cooldown_days > 0 and pd.notna(halt_started):
            cooldown_ok = bool((timestamp - halt_started).days >= int(cooldown_days))

        recovered_ok = bool(current_drawdown >= -abs(float(recovery_level)))
        ir_ok = True
        trailing_ir: Optional[float] = None
        if require_positive_quant_ir:
            trailing_ir = _trailing_quant_information_ratio(
                equity_curve=equity_curve,
                quant_returns=quant_returns,
                lookback_bars=ir_lookback_bars,
            )
            state["hard_dd_last_quant_ir"] = trailing_ir
            ir_ok = trailing_ir is not None and trailing_ir >= 0.0

        if cooldown_ok and recovered_ok and ir_ok:
            state["hard_dd_halted"] = False
            state["hard_dd_halt_started_ts"] = None
            state["hard_dd_reentries"] = int(state.get("hard_dd_reentries", 0)) + 1
            reentry_ts = list(state.get("hard_dd_reentry_ts", []))
            reentry_ts.append(timestamp.isoformat())
            state["hard_dd_reentry_ts"] = reentry_ts
            halted = False
            reentered = True

    force_flatten = bool(halted and action_key == "flatten")
    return {
        "halted": halted,
        "trigger_now": trigger_now,
        "force_flatten": force_flatten,
        "reentered": reentered,
    }


def _submit_flatten_orders(bt: Backtester, bars: Dict[str, pd.Series]) -> int:
    """Submit market orders to flatten all currently open positions."""
    submitted = 0
    for symbol, pos in list(bt.positions.items()):
        qty = float(getattr(pos, "quantity", 0.0))
        if abs(qty) <= 1e-12:
            continue
        bar = bars.get(symbol)
        if bar is None:
            continue
        side = OrderSide.SELL if qty > 0 else OrderSide.BUY
        bt.submit_order(symbol, side, abs(qty), OrderType.MARKET)
        submitted += 1
    return submitted


def _strict_promotion_ready(
    metrics: Dict[str, float | bool], required_windows: int, required_beats: int
) -> Tuple[bool, int, int]:
    req_windows = max(int(required_windows), 4)
    req_beats = max(int(required_beats), 3)
    ready = bool(
        metrics.get("mcpt_pass_stage2_0_05", False)
        and metrics.get("benchmark_constraints_passed", False)
        and int(metrics.get("rolling_oos_n_windows", 0)) >= req_windows
        and int(metrics.get("rolling_oos_beats", 0)) >= req_beats
    )
    return ready, req_windows, req_beats


def _promotion_verdict_from_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    """Build explicit promotion verdict with concrete fail reasons."""
    fail_reasons: List[str] = []
    constraint_failures = list(metrics.get("benchmark_constraint_fail_reasons", []))
    if not bool(metrics.get("benchmark_constraints_passed", False)):
        if constraint_failures:
            fail_reasons.extend([f"constraint:{r}" for r in constraint_failures])
        else:
            fail_reasons.append("constraint:unknown")

    if not bool(metrics.get("mcpt_pass_stage2_0_05", False)):
        fail_reasons.append("mcpt_stage2_failed")

    req_windows = int(metrics.get("promotion_oos_required_windows", 4) or 4)
    req_beats = int(metrics.get("promotion_oos_required_beats", 3) or 3)
    n_windows = int(metrics.get("rolling_oos_n_windows", 0) or 0)
    beats = int(metrics.get("rolling_oos_beats", 0) or 0)
    if n_windows < req_windows:
        fail_reasons.append(
            f"rolling_oos_windows_below_min:{n_windows}<{req_windows}"
        )
    if beats < req_beats:
        fail_reasons.append(f"rolling_oos_beats_below_min:{beats}<{req_beats}")

    # Deduplicate while preserving order.
    fail_reasons = list(dict.fromkeys(fail_reasons))
    eligible = len(fail_reasons) == 0

    return {
        "eligible": bool(eligible),
        "failed": bool(not eligible),
        "fail_reasons": fail_reasons,
        "benchmark_constraints_passed": bool(
            metrics.get("benchmark_constraints_passed", False)
        ),
        "mcpt_stage2_passed": bool(metrics.get("mcpt_pass_stage2_0_05", False)),
        "rolling_oos_n_windows": n_windows,
        "rolling_oos_beats": beats,
        "rolling_oos_required_windows": req_windows,
        "rolling_oos_required_beats": req_beats,
    }


def _normalize_return_index(series: pd.Series) -> pd.Series:
    """Normalize return series index to tz-naive daily dates."""
    s = pd.Series(series).astype(float).replace([np.inf, -np.inf], np.nan)
    idx = pd.to_datetime(s.index, errors="coerce")
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx
    s = s[~s.index.isna()]
    s.index = s.index.normalize()
    return s.groupby(level=0).mean().sort_index()


def _fetch_symbol_returns(
    collector: DataCollector,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
) -> Optional[pd.Series]:
    """Fetch daily close-to-close returns for a symbol."""
    try:
        df = collector.fetch_ohlcv(symbol, start_date, end_date)
    except Exception:
        return None
    if df is None or df.empty or "close" not in df.columns:
        return None
    close = pd.to_numeric(df["close"], errors="coerce")
    ret = close.pct_change(fill_method=None)
    series = _normalize_return_index(
        pd.Series(ret.values, index=pd.to_datetime(df.index, errors="coerce"))
    ).replace([np.inf, -np.inf], np.nan)
    series = series.dropna().sort_index()
    return series if not series.empty else None


def _fetch_composite_returns(
    collector: DataCollector,
    symbols: list,
    start_date: datetime,
    end_date: datetime,
) -> Optional[pd.Series]:
    """Fetch equal-weighted return composite for a symbol basket."""
    parts = []
    for sym in symbols:
        s = _fetch_symbol_returns(collector, str(sym), start_date, end_date)
        if s is not None and not s.empty:
            parts.append(s)
    if not parts:
        return None
    composite = pd.concat(parts, axis=1).mean(axis=1).dropna().sort_index()
    composite.name = "composite"
    return composite if not composite.empty else None


def _rolling_relative_edge(
    equity_curve: list,
    benchmark_returns: Optional[pd.Series],
    asof: datetime,
    lookback_days: int = 63,
) -> Dict[str, float | bool]:
    """
    Compute rolling relative edge vs benchmark.

    Returns a dict with:
    - available
    - n_obs
    - excess_total_return
    - information_ratio
    - tracking_error
    """
    if benchmark_returns is None or benchmark_returns.empty or len(equity_curve) < 8:
        return {
            "available": False,
            "n_obs": 0,
            "excess_total_return": 0.0,
            "information_ratio": 0.0,
            "tracking_error": 0.0,
        }

    rows = []
    for row in equity_curve:
        ts = row.get("timestamp")
        eq = row.get("equity")
        if ts is None or eq is None:
            continue
        rows.append((pd.to_datetime(ts, errors="coerce"), float(eq)))
    if not rows:
        return {
            "available": False,
            "n_obs": 0,
            "excess_total_return": 0.0,
            "information_ratio": 0.0,
            "tracking_error": 0.0,
        }

    eq_idx = pd.DatetimeIndex([r[0] for r in rows])
    if eq_idx.tz is not None:
        eq_idx = eq_idx.tz_localize(None)
    eq_series = pd.Series([r[1] for r in rows], index=eq_idx)
    eq_series = eq_series.replace([np.inf, -np.inf], np.nan).dropna()
    if eq_series.empty:
        return {
            "available": False,
            "n_obs": 0,
            "excess_total_return": 0.0,
            "information_ratio": 0.0,
            "tracking_error": 0.0,
        }

    strat_ret = eq_series.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    strat_ret = strat_ret.dropna()
    if strat_ret.empty:
        return {
            "available": False,
            "n_obs": 0,
            "excess_total_return": 0.0,
            "information_ratio": 0.0,
            "tracking_error": 0.0,
        }

    strat_ret = strat_ret.groupby(strat_ret.index.normalize()).mean().sort_index()
    bench = benchmark_returns.groupby(benchmark_returns.index.normalize()).mean().sort_index()

    asof_ts = pd.Timestamp(asof)
    if asof_ts.tzinfo is not None:
        asof_ts = asof_ts.tz_convert(None)
    asof_ts = asof_ts.normalize()

    strat_ret = strat_ret[strat_ret.index <= asof_ts].tail(max(lookback_days, 20))
    bench = bench[bench.index <= asof_ts]
    strat_ret, bench = strat_ret.align(bench, join="inner")
    min_obs = max(20, int(lookback_days * 0.4))
    if len(strat_ret) < min_obs:
        return {
            "available": False,
            "n_obs": int(len(strat_ret)),
            "excess_total_return": 0.0,
            "information_ratio": 0.0,
            "tracking_error": 0.0,
        }

    excess_total = float(
        _total_return_from_returns(strat_ret) - _total_return_from_returns(bench)
    )
    diff = (strat_ret - bench).replace([np.inf, -np.inf], np.nan).dropna()
    tracking_error = (
        float(diff.std(ddof=0) * np.sqrt(252.0)) if len(diff) > 1 else 0.0
    )
    ann_excess = float((strat_ret.mean() - bench.mean()) * 252.0)
    info_ratio = float(ann_excess / tracking_error) if tracking_error > 1e-10 else 0.0

    return {
        "available": True,
        "n_obs": int(len(strat_ret)),
        "excess_total_return": excess_total,
        "information_ratio": info_ratio,
        "tracking_error": tracking_error,
    }


def run_backtest(
    symbols: list,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000,
    strategy_type: str = "enhanced",
    validate: bool = False,
    verbose: bool = True,
    config_path: Optional[str] = None,
    checkpoint_name: Optional[str] = None,
    strategy_kwargs: Optional[Dict] = None,
) -> Dict:
    """
    Run a complete backtest.

    Args:
        symbols: List of symbols to trade
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        strategy_type: Trading strategy key (default 'enhanced')
        validate: Whether to run MCPT validation
        verbose: Print progress
        config_path: Path to config directory or settings.yaml
        checkpoint_name: Model checkpoint name (for news_lstm strategy)
        strategy_kwargs: Extra keyword arguments passed to the strategy constructor

    Returns:
        Dict with backtest results
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print("QUANTUM ALPHA V1 - BACKTEST")
        print(f"{'=' * 60}")
        print(f"Symbols: {_format_symbols(symbols)}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Capital: ${initial_capital:,.0f}")
        print(f"Strategy: {strategy_type}")
        print(f"{'=' * 60}\n")

    settings = load_config(config_path)
    risk_cfg = settings.get("risk", {}) if settings else {}
    strategy_cfg = settings.get("strategy", {}) if settings else {}
    validation_cfg = settings.get("validation", {}) if settings else {}
    config_dir = _resolve_config_dir(config_path)
    strategies_cfg = _load_optional_yaml(config_dir / "strategies.yaml")
    sentiment_cfg = {}
    if strategies_cfg and "strategies" in strategies_cfg:
        sentiment_cfg = strategies_cfg["strategies"].get("sentiment", {})
    max_position = float(risk_cfg.get("max_position_size", 0.25))
    max_leverage = float(risk_cfg.get("max_portfolio_leverage", 1.0))
    max_drawdown = float(risk_cfg.get("max_drawdown", 0.10))
    hard_drawdown_limit = float(risk_cfg.get("hard_drawdown_limit", 0.20))
    hard_drawdown_action = str(risk_cfg.get("hard_drawdown_action", "flatten"))
    hard_drawdown_cooldown_days = int(risk_cfg.get("hard_drawdown_cooldown_days", 20))
    hard_drawdown_recovery_level = float(risk_cfg.get("hard_drawdown_recovery_level", 0.10))
    hard_drawdown_require_positive_quant_ir = bool(
        risk_cfg.get("hard_drawdown_require_positive_quant_ir", True)
    )
    hard_drawdown_ir_lookback_bars = int(
        risk_cfg.get("hard_drawdown_ir_lookback_bars", 63)
    )
    max_daily_loss = float(risk_cfg.get("max_daily_loss", 0.03))
    max_weekly_loss = float(risk_cfg.get("max_weekly_loss", 0.07))
    kelly_fraction = float(risk_cfg.get("kelly_fraction", 0.5))
    core_drawdown_scaling_enabled = bool(
        risk_cfg.get("core_drawdown_scaling_enabled", True)
    )
    core_min_exposure_in_drawdown = _clamp(
        float(risk_cfg.get("core_min_exposure_in_drawdown", 0.35)),
        0.0,
        1.0,
    )
    rebalance_frequency = str(strategy_cfg.get("rebalance_frequency", "daily"))
    paper_rebalance = strategy_cfg.get("paper_rebalance_frequency")
    if paper_rebalance:
        rebalance_frequency = str(paper_rebalance)
    momentum_top_pct = float(strategy_cfg.get("momentum_top_pct", 65))
    momentum_bottom_pct = float(strategy_cfg.get("momentum_bottom_pct", 35))
    signal_threshold = float(strategy_cfg.get("signal_threshold", 0.3))
    signal_scale = float(strategy_cfg.get("signal_scale", 1.0))
    min_long_signal = float(strategy_cfg.get("min_long_signal", 0.0))
    market_off_scale = float(strategy_cfg.get("market_off_scale", 1.0))
    anchor_core_to_quant = bool(
        strategy_cfg.get("anchor_core_to_quant_composite", False)
    )
    alpha_deploy_requires_relative_edge = bool(
        strategy_cfg.get("alpha_deploy_requires_relative_edge", True)
    )
    alpha_edge_lookback_days = int(strategy_cfg.get("alpha_edge_lookback_days", 63))
    alpha_min_scale = _clamp(float(strategy_cfg.get("alpha_min_scale", 0.05)), 0.0, 1.0)
    alpha_info_floor = float(strategy_cfg.get("alpha_info_floor", 0.0))
    alpha_excess_floor = float(strategy_cfg.get("alpha_excess_floor", 0.0))
    alpha_recovery_info = float(strategy_cfg.get("alpha_recovery_info", 0.25))
    alpha_recovery_excess = float(strategy_cfg.get("alpha_recovery_excess", 0.03))
    vol_of_vol_threshold = float(risk_cfg.get("vol_of_vol_threshold", 0.03))
    vol_of_vol_scale = float(risk_cfg.get("vol_of_vol_scale", 0.7))
    rs_top_n = int(strategy_cfg.get("relative_strength_top_n", 0))
    rs_min_mom = float(strategy_cfg.get("relative_strength_min_mom", 0.0))
    use_relative_strength = bool(strategy_cfg.get("use_relative_strength", False))
    liquid_subset_size = int(strategy_cfg.get("liquid_subset_size", 0))
    liquid_adv_window = int(strategy_cfg.get("liquid_adv_window", 20))
    liquid_min_history = int(strategy_cfg.get("liquid_min_history", 120))
    liquid_for_full_only = bool(strategy_cfg.get("liquid_for_full_only", True))
    long_only = bool(strategy_cfg.get("long_only", False))
    risk_off_cash = bool(strategy_cfg.get("risk_off_cash", False))
    min_rebalance_notional_frac = float(
        strategy_cfg.get("min_rebalance_notional_frac", 0.015)
    )
    sleeves_cfg = strategy_cfg.get("sleeves", {}) if isinstance(strategy_cfg, dict) else {}
    sleeves_enabled = bool(sleeves_cfg.get("enabled", True))
    core_budget = _clamp(float(sleeves_cfg.get("core_budget", 0.96)), 0.70, 1.00)
    alpha_budget = _clamp(float(sleeves_cfg.get("alpha_budget", 0.04)), 0.00, 0.30)
    lottery_budget = _clamp(float(sleeves_cfg.get("lottery_budget", 0.00)), 0.00, 0.10)
    core_loss_scale = _clamp(float(sleeves_cfg.get("core_loss_scale", 0.50)), 0.10, 1.00)
    lottery_top_n = int(max(1, sleeves_cfg.get("lottery_top_n", 3)))
    lottery_min_signal = float(sleeves_cfg.get("lottery_min_signal", 0.35))
    lottery_min_confidence = float(sleeves_cfg.get("lottery_min_confidence", 0.25))
    lottery_stop_loss = _clamp(float(sleeves_cfg.get("lottery_stop_loss", 0.08)), 0.01, 0.25)
    lottery_max_per_symbol = _clamp(
        float(sleeves_cfg.get("lottery_max_per_symbol", 0.0075)), 0.001, 0.02
    )
    alpha_min_confidence = float(sleeves_cfg.get("alpha_min_confidence", 0.05))
    core_tilt_enabled = bool(sleeves_cfg.get("core_tilt_enabled", False))
    enhanced_debug_components = bool(
        strategy_cfg.get("enhanced_debug_components", False)
    )
    core_tilt_lookback_days = int(sleeves_cfg.get("core_tilt_lookback_days", 126))
    core_tilt_strength = _clamp(
        float(sleeves_cfg.get("core_tilt_strength", 0.35)), 0.0, 0.95
    )
    core_rebalance_band = _clamp(
        float(sleeves_cfg.get("core_rebalance_band", 0.02)), 0.0, 0.10
    )
    core_mix_cfg = sleeves_cfg.get("core_mix", {"SPY": 0.5, "QQQ": 0.3, "IWM": 0.2})
    core_mix = {
        str(k).upper(): float(v)
        for k, v in core_mix_cfg.items()
        if float(v) > 0
    }
    if not core_mix:
        core_mix = {"SPY": 0.5, "QQQ": 0.3, "IWM": 0.2}

    if max_leverage > 0:
        sleeve_total = core_budget + alpha_budget + lottery_budget
        if sleeve_total > max_leverage and sleeve_total > 0:
            scale = max_leverage / sleeve_total
            core_budget *= scale
            alpha_budget *= scale
            lottery_budget *= scale

    if max_leverage <= 0:
        max_leverage = 1.0
    # Volatility target for scaling (annualized) - nudged up to restore exposure
    target_vol = float(risk_cfg.get("target_volatility", 0.15))
    dd_leverage_cap_warning = float(risk_cfg.get("dd_leverage_cap_warning", 1.0))
    dd_leverage_cap_critical = float(risk_cfg.get("dd_leverage_cap_critical", 1.0))
    market_off_leverage_cap = float(risk_cfg.get("market_off_leverage_cap", 1.0))
    faster_derisk_warning_mult = _clamp(
        float(risk_cfg.get("faster_derisk_warning_mult", 0.85)), 0.10, 1.00
    )
    faster_derisk_critical_mult = _clamp(
        float(risk_cfg.get("faster_derisk_critical_mult", 0.70)), 0.10, 1.00
    )

    # Initialize components
    collector = DataCollector(runtime_mode="backtest", config_path=config_path)
    feature_gen = TechnicalFeatureGenerator()
    cleaner = DataCleaner()
    imputer = MissingValueImputer()

    full_token_requested = bool(
        symbols
        and len(symbols) == 1
        and str(symbols[0]).upper() in {"FULL", "ALL", "UNIVERSE"}
    )
    symbols = _resolve_symbols(symbols, collector, settings)

    if strategy_type == "momentum":
        strategy = MomentumStrategy()
    elif strategy_type == "composite":
        strategy = CompositeStrategy()
    elif strategy_type in ("adaptive", "enhanced"):
        strategy = EnhancedCompositeStrategy(
            debug_components=enhanced_debug_components
        )
    elif strategy_type == "sentiment":
        from quantum_alpha.strategy.sentiment_strategies import SocialSentimentStrategy

        strategy = SocialSentimentStrategy()
    elif strategy_type == "ml":
        from quantum_alpha.strategy.ml_strategies import MLTradingStrategy

        strategy = MLTradingStrategy(**(strategy_kwargs or {}))
    elif strategy_type == "news_lstm":
        from quantum_alpha.strategy.news_lstm_strategy import NewsLSTMStrategy

        kwargs = dict(strategy_kwargs or {})
        if checkpoint_name is not None:
            kwargs.setdefault("checkpoint_name", checkpoint_name)
        strategy = NewsLSTMStrategy(**kwargs)
    elif strategy_type == "meta_ensemble":
        from quantum_alpha.strategy.meta_ensemble_strategy import MetaEnsembleStrategy

        strategy = MetaEnsembleStrategy(**(strategy_kwargs or {}))
    else:
        strategy = MomentumStrategy()

    use_enhanced = isinstance(strategy, EnhancedCompositeStrategy)
    use_sentiment = strategy_type == "sentiment"

    if use_sentiment:
        from quantum_alpha.data.collectors.alternative import (
            load_social_sentiment,
            load_options_sentiment,
            load_insider_trades,
            load_congress_trades,
        )
        from quantum_alpha.strategy.sentiment_strategies import (
            SocialSentimentStrategy,
            OptionsSentimentStrategy,
            InsiderTradingStrategy,
            CongressTradingStrategy,
        )

        social_strategy = SocialSentimentStrategy(**sentiment_cfg.get("social", {}))
        options_strategy = OptionsSentimentStrategy(**sentiment_cfg.get("options", {}))
        insider_strategy = InsiderTradingStrategy(**sentiment_cfg.get("insider", {}))
        congress_strategy = CongressTradingStrategy(**sentiment_cfg.get("congress", {}))
        sentiment_weights = sentiment_cfg.get(
            "combined_weights",
            {"social": 0.4, "options": 0.2, "insider": 0.2, "congress": 0.2},
        )
    use_sentiment = strategy_type == "sentiment"

    if use_sentiment:
        from quantum_alpha.data.collectors.alternative import (
            load_social_sentiment,
            load_options_sentiment,
            load_insider_trades,
            load_congress_trades,
        )
        from quantum_alpha.strategy.sentiment_strategies import (
            SocialSentimentStrategy,
            OptionsSentimentStrategy,
            InsiderTradingStrategy,
            CongressTradingStrategy,
        )

        social_strategy = SocialSentimentStrategy(**sentiment_cfg.get("social", {}))
        options_strategy = OptionsSentimentStrategy(**sentiment_cfg.get("options", {}))
        insider_strategy = InsiderTradingStrategy(**sentiment_cfg.get("insider", {}))
        congress_strategy = CongressTradingStrategy(**sentiment_cfg.get("congress", {}))
        sentiment_weights = sentiment_cfg.get(
            "combined_weights",
            {"social": 0.4, "options": 0.2, "insider": 0.2, "congress": 0.2},
        )

    if use_sentiment:
        signal_threshold = float(
            sentiment_cfg.get("signal_threshold", signal_threshold)
        )

    position_sizer = PositionSizer(
        max_position=max_position,
        kelly_fraction=kelly_fraction,
        max_drawdown=max_drawdown,
    )
    backtester = Backtester(initial_capital=initial_capital)

    bench_cfg = settings.get("benchmarks", {})
    market_benchmark = str(bench_cfg.get("market", "SPY")).upper()
    quant_raw = bench_cfg.get("quant_composite", ["QQQ", "IWM"])
    if isinstance(quant_raw, list):
        quant_benchmark = [str(s).upper() for s in quant_raw if str(s).strip()]
    elif quant_raw:
        quant_benchmark = [str(quant_raw).upper()]
    else:
        quant_benchmark = []

    if anchor_core_to_quant and quant_benchmark:
        eq_w = 1.0 / len(quant_benchmark)
        core_mix = {sym: eq_w for sym in quant_benchmark}
        if verbose:
            print(f"Core sleeve anchored to quant composite: {core_mix}")

    symbols = _augment_symbols_with_benchmarks(
        symbols=symbols,
        core_mix=core_mix,
        market_benchmark=market_benchmark,
        quant_benchmark=quant_benchmark,
    )
    reserved_benchmark_symbols = set(core_mix.keys()) | set(quant_benchmark) | {
        market_benchmark
    }
    market_df = None
    market_trend = None
    quant_returns_control = None
    try:
        market_df = collector.fetch_ohlcv(market_benchmark, start_date, end_date)
        m_close = market_df["close"]
        m_ma = m_close.rolling(200).mean()
        market_trend = m_close.shift(1) > m_ma.shift(1)
    except Exception:
        market_df = None
        market_trend = None
    if quant_benchmark:
        quant_returns_control = _fetch_composite_returns(
            collector=collector,
            symbols=quant_benchmark,
            start_date=start_date,
            end_date=end_date,
        )

    # Collect data
    if verbose:
        print("Collecting price data...")

    data = {}
    raw_featured = {}  # For enhanced strategy: feature-generated but pre-signal
    for symbol in symbols:
        try:
            df = collector.fetch_ohlcv(symbol, start_date, end_date)
            df = cleaner.clean(df)
            df = imputer.impute(df)
            df = feature_gen.generate(df)
            raw_featured[symbol] = df
            if use_sentiment:
                social_df = load_social_sentiment(symbol, use_live=False)
                options_df = load_options_sentiment(symbol, use_live=False)
                insider_df = load_insider_trades(symbol, use_live=True)
                congress_df = load_congress_trades(symbol, use_live=True)

                if "symbol" in social_df.columns:
                    social_df = social_df[social_df["symbol"] == symbol]
                if "symbol" in options_df.columns:
                    options_df = options_df[options_df["symbol"] == symbol]
                if "symbol" in insider_df.columns:
                    insider_df = insider_df[insider_df["symbol"] == symbol]
                if "symbol" in congress_df.columns:
                    congress_df = congress_df[congress_df["symbol"] == symbol]

                frames = {}
                if not social_df.empty:
                    frames["social"] = _align_signal_frame(
                        social_strategy.generate_signals(social_df), df.index
                    )
                if not options_df.empty:
                    if "timestamp" in options_df.columns:
                        options_df = options_df.copy()
                        options_df["timestamp"] = pd.to_datetime(
                            options_df["timestamp"]
                        )
                        options_df = options_df.set_index("timestamp")
                    elif "date" in options_df.columns:
                        options_df = options_df.copy()
                        options_df["date"] = pd.to_datetime(options_df["date"])
                        options_df = options_df.set_index("date")
                    options_sig = options_strategy.generate_signals(options_df)
                    frames["options"] = _align_signal_frame(options_sig, df.index)
                if not insider_df.empty:
                    frames["insider"] = _align_signal_frame(
                        insider_strategy.generate_signals(insider_df),
                        df.index,
                        limit=20,
                    )
                if not congress_df.empty:
                    frames["congress"] = _align_signal_frame(
                        congress_strategy.generate_signals(congress_df),
                        df.index,
                        limit=20,
                    )

                combined_signal = pd.Series(0.0, index=df.index)
                combined_conf = pd.Series(0.0, index=df.index)
                total_w = 0.0
                for key, sig_frame in frames.items():
                    weight = float(sentiment_weights.get(key, 0.0))
                    if weight <= 0:
                        continue
                    total_w += weight
                    combined_signal += weight * sig_frame["signal"]
                    combined_conf += weight * sig_frame["signal_confidence"]

                if total_w > 0:
                    combined_signal = combined_signal / total_w
                    combined_conf = combined_conf / total_w
                else:
                    combined_signal = combined_signal * 0.0
                    combined_conf = combined_conf * 0.0

                df["signal"] = combined_signal
                df["signal_confidence"] = combined_conf
                df["position_signal"] = np.where(
                    np.abs(df["signal"]) >= signal_threshold,
                    np.sign(df["signal"]),
                    0.0,
                )
                df = _apply_signal_lag(df)
                data[symbol] = df
            elif not use_enhanced:
                # Pass symbol kwarg for strategies that need it (e.g. news_lstm)
                import inspect

                sig = inspect.signature(strategy.generate_signals)
                if "symbol" in sig.parameters:
                    df = strategy.generate_signals(df, symbol=symbol)
                else:
                    df = strategy.generate_signals(df)
                df = _apply_signal_lag(df)
                data[symbol] = df
            if verbose:
                print(f"  {symbol}: {len(df)} bars")
        except Exception as e:
            if verbose:
                print(f"  {symbol}: FAILED - {e}")

    apply_liquid_filter = (
        liquid_subset_size > 0
        and len(raw_featured) > liquid_subset_size
        and (full_token_requested or not liquid_for_full_only)
    )
    if apply_liquid_filter:
        selected = _select_liquid_subset(
            raw_featured,
            subset_size=liquid_subset_size,
            adv_window=liquid_adv_window,
            min_history=liquid_min_history,
        )
        selected_set = set(selected) | {
            sym for sym in reserved_benchmark_symbols if sym in raw_featured
        }
        raw_featured = {
            sym: df for sym, df in raw_featured.items() if sym in selected_set
        }
        data = {sym: df for sym, df in data.items() if sym in selected_set}
        if verbose:
            print(
                "  Liquidity filter applied: "
                f"trading top {len(selected)} symbols by {liquid_adv_window}D ADV"
            )

    # Enhanced strategy: fit cross-asset signals, then generate per-symbol
    if use_enhanced and raw_featured:
        if verbose:
            print("  Computing cross-asset signals...")
        strategy.fit_cross_asset(raw_featured)
        for symbol, df in raw_featured.items():
            df = strategy.generate_signals(df, symbol=symbol)
            df = _apply_signal_lag(df)
            data[symbol] = df

    if not data:
        return {"error": "No data collected"}

    if (market_df is None or market_df.empty) and market_benchmark in data:
        market_df = data[market_benchmark]
        try:
            m_close = pd.to_numeric(market_df["close"], errors="coerce")
            m_ma = m_close.rolling(200).mean()
            market_trend = m_close.shift(1) > m_ma.shift(1)
        except Exception:
            market_trend = None

    if quant_returns_control is None or quant_returns_control.empty:
        quant_returns_control = _composite_returns_from_frames(
            frames=data,
            symbols=quant_benchmark,
        )

    if verbose:
        print(f"\nRunning backtest...")

    # Track state for strategy
    state = {
        "positions": {},
        "trade_history": [],
        "current_drawdown": 0,
        "peak_equity": initial_capital,
        "last_rebalance": None,
        "loss_day_key": None,
        "loss_week_key": None,
        "day_start_equity": float(initial_capital),
        "week_start_equity": float(initial_capital),
        "daily_stop_active": False,
        "weekly_stop_active": False,
        "loss_stop_active": False,
        "lottery_entry_price": {},
        "lottery_blocked_day": {},
        "alpha_scale_last": 1.0,
        "alpha_scale_history": [],
    }
    _init_hard_drawdown_state(state)

    # Dynamic drawdown control for continuous exposure scaling.
    # Hard circuit breaker (flatten + cooldown) is handled separately
    # via _update_hard_drawdown_guard().
    dd_warning = float(risk_cfg.get("dd_warning", 0.15))
    dd_critical = float(risk_cfg.get("dd_critical", 0.25))
    dd_breaker = float(risk_cfg.get("dd_circuit_breaker", 0.40))
    dd_limit = float(risk_cfg.get("dd_max_limit", 0.50))
    dd_min_exposure = float(risk_cfg.get("dd_min_exposure", 0.25))
    dd_controller = DrawdownController(
        warning_threshold=dd_warning,
        critical_threshold=dd_critical,
        circuit_breaker_threshold=dd_breaker,
        max_drawdown_limit=dd_limit,
        scaling_method="linear",
        min_exposure=dd_min_exposure,
        cooldown_days=10,
    )
    dd_controller.reset(initial_capital)

    def trading_strategy(timestamp, bars, bt):
        """Strategy execution function."""
        equity = bt._total_equity()
        state["peak_equity"] = max(float(state.get("peak_equity", equity)), equity)
        peak = float(state.get("peak_equity", equity))
        state["current_drawdown"] = (equity - peak) / peak if peak > 0 else 0.0

        loss_status = _update_loss_limit_state(
            state=state,
            timestamp=timestamp,
            equity=equity,
            max_daily_loss=max_daily_loss,
            max_weekly_loss=max_weekly_loss,
        )
        hard_dd = _update_hard_drawdown_guard(
            state=state,
            timestamp=timestamp,
            current_drawdown=float(state.get("current_drawdown", 0.0)),
            equity_curve=bt.equity_curve,
            quant_returns=quant_returns_control,
            limit=hard_drawdown_limit,
            action=hard_drawdown_action,
            cooldown_days=hard_drawdown_cooldown_days,
            recovery_level=hard_drawdown_recovery_level,
            require_positive_quant_ir=hard_drawdown_require_positive_quant_ir,
            ir_lookback_bars=hard_drawdown_ir_lookback_bars,
        )

        # Update drawdown controller every bar
        dd_metrics = dd_controller.update(equity, timestamp)

        # Exposure multiplier is soft scaling; hard breaker behavior is
        # enforced separately through hard_dd state.
        exposure_mult = dd_metrics.exposure_multiplier

        force_rebalance = bool(
            loss_status["stop_active"]
            or hard_dd.get("trigger_now", False)
            or hard_dd.get("force_flatten", False)
        )
        if (
            not _should_rebalance(timestamp, state["last_rebalance"], rebalance_frequency)
            and not force_rebalance
        ):
            return

        state["last_rebalance"] = timestamp
        if bool(hard_dd.get("force_flatten", False)):
            _submit_flatten_orders(bt, bars)
            return

        target_positions = {}
        signal_strengths = {}
        signal_confidences = {}
        volatilities = {}
        rp_drawdown_mult = 1.0
        trade_history = (
            np.array([t["pnl"] for t in bt.trades]) if bt.trades else np.array([0])
        )
        realized_vol = _realized_vol_from_equity(bt.equity_curve)
        vol_of_vol = _vol_of_vol_from_equity(bt.equity_curve)
        vol_scale = 1.0
        if realized_vol and realized_vol > 0:
            vol_scale = float(np.clip(target_vol / realized_vol, 0.5, 1.5))
        if (
            vol_of_vol is not None
            and vol_of_vol > vol_of_vol_threshold
            and dd_metrics.state
            in {
                DrawdownState.WARNING,
                DrawdownState.CRITICAL,
                DrawdownState.RECOVERY,
                DrawdownState.CIRCUIT_BREAKER,
            }
        ):
            vol_scale *= vol_of_vol_scale
        market_risk_on = True
        if market_trend is not None and timestamp in market_trend.index:
            market_risk_on = bool(market_trend.loc[timestamp])
        allow_shorts = not long_only
        if not market_risk_on:
            allow_shorts = False

        mom_scores = {}
        for sym, df in data.items():
            if timestamp in df.index:
                mom_val = df.loc[timestamp].get("mom_12m", 0.0)
                if pd.notna(mom_val):
                    mom_scores[sym] = float(mom_val)
        top_cut = None
        bottom_cut = None
        top_syms = None
        if len(mom_scores) >= 3:
            vals = np.array(list(mom_scores.values()), dtype=float)
            top_cut = np.nanpercentile(vals, momentum_top_pct)
            bottom_cut = np.nanpercentile(vals, momentum_bottom_pct)
            if use_relative_strength and rs_top_n > 0:
                ranked = sorted(mom_scores.items(), key=lambda x: x[1], reverse=True)
                top_syms = {sym for sym, val in ranked[:rs_top_n] if val >= rs_min_mom}

        use_rp_allocation = top_syms is not None

        for symbol, bar in bars.items():
            if symbol not in data:
                continue

            df = data[symbol]
            if timestamp not in df.index:
                continue

            row = df.loc[timestamp]
            signal = (
                row.get("position_signal")
                if "position_signal" in row
                else row.get("signal", 0)
            )
            confidence = float(row.get("signal_confidence", 0.5))
            if top_syms is not None:
                if use_enhanced:
                    # Enhanced strategy: use top_syms as universe filter,
                    # but preserve strategy's signal magnitude & direction
                    if symbol not in top_syms:
                        signal = 0.0
                    else:
                        # Ensure long-only compliance; keep magnitude
                        signal = max(signal, 0.0) if long_only else signal
                        # Floor at a small positive so selected symbols trade
                        if signal == 0.0:
                            signal = 0.5
                else:
                    signal = 1.0 if symbol in top_syms else 0.0
            else:
                if signal < 0 and not allow_shorts:
                    signal = 0.0
                if top_cut is not None and bottom_cut is not None:
                    mom_val = mom_scores.get(symbol)
                    if signal > 0 and (mom_val is None or mom_val < top_cut):
                        signal = 0.0
                    if signal < 0 and (
                        mom_val is None or mom_val > bottom_cut or long_only
                    ):
                        signal = 0.0
            if signal_scale != 1.0 and signal != 0.0:
                signal = float(np.clip(signal * signal_scale, -1.0, 1.0))
            if long_only and signal > 0 and min_long_signal > 0:
                signal = max(signal, min_long_signal)
            signal_strengths[symbol] = float(signal)
            signal_confidences[symbol] = float(confidence)
            volatility = row.get("atr_pct", 0.02)
            if pd.isna(volatility) or volatility <= 0:
                volatility = 0.02
            volatility = volatility * np.sqrt(252)
            volatilities[symbol] = max(volatility, 1e-6)

            confidence = row.get("signal_confidence", 0.5)

            # Calculate position size
            sizing = position_sizer.calculate(
                trade_history=trade_history,
                current_volatility=max(volatility, 0.01),
                current_drawdown=state["current_drawdown"],
                signal_strength=signal,
                signal_confidence=confidence,
            )

            if sizing["halt_trading"]:
                return

            rp_drawdown_mult = min(rp_drawdown_mult, float(sizing.get("dd_multiplier", 1.0)))
            if use_rp_allocation:
                continue

            target_positions[symbol] = sizing["position_size"]

        if use_rp_allocation:
            # Only allocate to symbols selected by relative strength
            selected_vols = {
                s: v
                for s, v in volatilities.items()
                if top_syms is None or s in top_syms
            }
            if selected_vols:
                inv = {s: 1 / v for s, v in selected_vols.items()}
                total_inv = sum(inv.values())
                if total_inv > 0:
                    target_positions = {
                        s: (inv[s] / total_inv) * max_leverage * rp_drawdown_mult
                        for s in inv
                    }
            if not target_positions:
                return
        if risk_off_cash and not market_risk_on:
            target_positions = {s: 0.0 for s in bars.keys()}
        elif not target_positions:
            return
        if not market_risk_on and market_off_scale < 1.0:
            target_positions = {
                s: w * market_off_scale for s, w in target_positions.items()
            }

        if volatilities and not use_rp_allocation:
            inv = {s: 1 / v for s, v in volatilities.items()}
            total_inv = sum(inv.values())
            if total_inv > 0:
                rp_weights = {s: inv[s] / total_inv for s in inv}
                weight_scale = len(rp_weights)
                target_positions = {
                    s: target_positions[s] * rp_weights.get(s, 0) * weight_scale
                    for s in target_positions
                }

        alpha_targets = dict(target_positions)

        # Apply drawdown-based exposure scaling to alpha sleeve.
        if exposure_mult < 1.0:
            alpha_targets = {s: w * exposure_mult for s, w in alpha_targets.items()}

        # Volatility targeting overlay on alpha sleeve.
        if vol_scale != 1.0:
            alpha_targets = {s: w * vol_scale for s, w in alpha_targets.items()}

        cap = max_leverage
        if dd_metrics.state == DrawdownState.WARNING:
            cap = min(cap, dd_leverage_cap_warning * faster_derisk_warning_mult)
        elif dd_metrics.state in {DrawdownState.CRITICAL, DrawdownState.RECOVERY}:
            cap = min(cap, dd_leverage_cap_critical * faster_derisk_critical_mult)
        if not market_risk_on:
            cap = min(cap, market_off_leverage_cap * faster_derisk_warning_mult)
        if anchor_core_to_quant and not loss_status["stop_active"]:
            cap = max(cap, min(max_leverage, core_budget))

        if sleeves_enabled:
            today_key = timestamp.date().isoformat()
            blocked = state.setdefault("lottery_blocked_day", {})
            entries = state.setdefault("lottery_entry_price", {})

            for sym in list(blocked.keys()):
                if blocked.get(sym) != today_key:
                    blocked.pop(sym, None)

            for sym, entry in list(entries.items()):
                bar = bars.get(sym)
                if bar is None:
                    entries.pop(sym, None)
                    continue
                px = float(bar.get("open", bar.get("close", 0.0)) or 0.0)
                if px > 0 and px <= float(entry) * (1.0 - lottery_stop_loss):
                    blocked[sym] = today_key
                    entries.pop(sym, None)

            core_drawdown_mult = 1.0
            if core_drawdown_scaling_enabled:
                core_drawdown_mult = max(
                    core_min_exposure_in_drawdown,
                    min(1.0, float(exposure_mult)),
                )
            core_budget_eff = (
                core_budget
                * (core_loss_scale if loss_status["stop_active"] else 1.0)
                * core_drawdown_mult
            )
            alpha_budget_eff = 0.0 if loss_status["stop_active"] else alpha_budget
            lottery_budget_eff = 0.0 if loss_status["stop_active"] else lottery_budget

            alpha_scale = 1.0
            if (
                alpha_budget_eff > 0
                and alpha_deploy_requires_relative_edge
                and quant_returns_control is not None
                and not quant_returns_control.empty
            ):
                edge = _rolling_relative_edge(
                    equity_curve=bt.equity_curve[-(max(alpha_edge_lookback_days, 20) + 10) :],
                    benchmark_returns=quant_returns_control,
                    asof=timestamp,
                    lookback_days=alpha_edge_lookback_days,
                )
                if bool(edge.get("available", False)):
                    info_ratio = float(edge.get("information_ratio", 0.0))
                    excess_total = float(edge.get("excess_total_return", 0.0))
                    info_den = max(alpha_recovery_info - alpha_info_floor, 1e-6)
                    excess_den = max(alpha_recovery_excess - alpha_excess_floor, 1e-6)
                    info_score = float(
                        np.clip((info_ratio - alpha_info_floor) / info_den, 0.0, 1.0)
                    )
                    excess_score = float(
                        np.clip(
                            (excess_total - alpha_excess_floor) / excess_den, 0.0, 1.0
                        )
                    )
                    alpha_scale = float(
                        alpha_min_scale
                        + (1.0 - alpha_min_scale) * (0.5 * info_score + 0.5 * excess_score)
                    )
                else:
                    alpha_scale = float(alpha_min_scale)

            alpha_budget_eff *= alpha_scale
            lottery_budget_eff *= alpha_scale
            state["alpha_scale_last"] = float(alpha_scale)
            state.setdefault("alpha_scale_history", []).append(float(alpha_scale))

            effective_core_mix = core_mix
            if core_tilt_enabled:
                effective_core_mix = _tilt_core_mix(
                    core_mix=core_mix,
                    data_frames=data,
                    timestamp=timestamp,
                    lookback_days=core_tilt_lookback_days,
                    tilt_strength=core_tilt_strength,
                )
            core_targets = _core_targets_from_mix(
                bars=bars,
                core_mix=effective_core_mix,
                core_budget=core_budget_eff,
            )
            alpha_candidates = {
                s: w
                for s, w in alpha_targets.items()
                if s not in core_mix
                and float(signal_confidences.get(s, 0.0)) >= alpha_min_confidence
            }

            if core_budget_eff > 0 and not core_targets:
                fallback_anchors = {}
                for sym in list(quant_benchmark) + [market_benchmark]:
                    bar = bars.get(sym)
                    if bar is None:
                        continue
                    px = float(bar.get("open", bar.get("close", 0.0)) or 0.0)
                    if px > 0:
                        fallback_anchors[sym] = 1.0
                if fallback_anchors:
                    core_targets = _normalize_weights(
                        fallback_anchors,
                        budget=core_budget_eff,
                        long_only=True,
                        max_abs_per_symbol=max_position,
                    )

            if core_budget_eff > 0 and not core_targets:
                fallback_from_alpha = {
                    s: max(float(w), 0.0)
                    for s, w in alpha_candidates.items()
                    if float(w) > 0
                }
                if fallback_from_alpha:
                    core_targets = _normalize_weights(
                        fallback_from_alpha,
                        budget=core_budget_eff,
                        long_only=True,
                        max_abs_per_symbol=max_position,
                    )

            if core_budget_eff > 0 and not core_targets:
                core_targets = _fallback_core_targets_from_bars(
                    bars=bars,
                    core_budget=core_budget_eff,
                    max_abs_per_symbol=max_position,
                    top_n=12,
                )

            if core_targets:
                alpha_candidates = {
                    s: w for s, w in alpha_candidates.items() if s not in core_targets
                }

            alpha_targets = _normalize_weights(
                alpha_candidates,
                budget=alpha_budget_eff,
                long_only=long_only,
                max_abs_per_symbol=max_position,
            )

            lottery_targets = {}
            if lottery_budget_eff > 0:
                lottery_scores = {}
                for sym, sig in signal_strengths.items():
                    if sym in core_mix or sym in alpha_targets:
                        continue
                    if blocked.get(sym) == today_key:
                        continue
                    conf = float(signal_confidences.get(sym, 0.0))
                    if conf < lottery_min_confidence:
                        continue
                    if long_only and sig < lottery_min_signal:
                        continue
                    if (not long_only) and abs(sig) < lottery_min_signal:
                        continue
                    vol = float(volatilities.get(sym, 0.0))
                    if vol <= 0:
                        continue
                    lottery_scores[sym] = abs(float(sig)) * max(conf, 0.05) * vol

                ranked = sorted(lottery_scores.items(), key=lambda x: x[1], reverse=True)
                raw_lottery = {sym: 1.0 for sym, _ in ranked[:lottery_top_n]}
                lottery_targets = _normalize_weights(
                    raw_lottery,
                    budget=lottery_budget_eff,
                    long_only=True,
                    max_abs_per_symbol=lottery_max_per_symbol,
                )

            target_positions = _combine_targets(core_targets, alpha_targets, lottery_targets)

            for sym, w in lottery_targets.items():
                if w <= 0:
                    continue
                bar = bars.get(sym)
                if bar is None:
                    continue
                px = float(bar.get("open", bar.get("close", 0.0)) or 0.0)
                if px > 0 and sym not in entries:
                    entries[sym] = px
            for sym in list(entries.keys()):
                if sym not in lottery_targets:
                    entries.pop(sym, None)

            if loss_status["stop_active"]:
                cap = min(cap, max(core_budget_eff, 0.05))
        else:
            target_positions = alpha_targets
            if loss_status["stop_active"]:
                target_positions = {}
                cap = min(cap, 0.05)

        total_abs = sum(abs(w) for w in target_positions.values())
        if total_abs > cap and total_abs > 0:
            if sleeves_enabled:
                core_symbols = set(core_mix.keys())
                core_part = {
                    s: float(w) for s, w in target_positions.items() if s in core_symbols
                }
                sat_part = {
                    s: float(w) for s, w in target_positions.items() if s not in core_symbols
                }
                core_abs = sum(abs(w) for w in core_part.values())
                sat_abs = sum(abs(w) for w in sat_part.values())
                if core_abs >= cap and core_abs > 0:
                    c_scale = cap / core_abs
                    target_positions = {s: float(w * c_scale) for s, w in core_part.items()}
                else:
                    sat_budget = max(cap - core_abs, 0.0)
                    if sat_abs > 0:
                        s_scale = sat_budget / sat_abs
                        sat_part = {s: float(w * s_scale) for s, w in sat_part.items()}
                    else:
                        sat_part = {}
                    target_positions = _combine_targets(core_part, sat_part)
            else:
                scale = cap / total_abs
                target_positions = {s: float(w * scale) for s, w in target_positions.items()}

        if sleeves_enabled and core_rebalance_band > 0:
            current_w = _current_weights(bt, bars=bars, equity=bt._total_equity())
            core_symbols = set(core_mix.keys())
            for sym in core_symbols:
                cur = float(current_w.get(sym, 0.0))
                tgt = float(target_positions.get(sym, 0.0))
                if abs(cur) > 1e-6 and abs(tgt - cur) < core_rebalance_band:
                    target_positions[sym] = cur

        equity = bt._total_equity()
        all_symbols = set(target_positions.keys()) | set(bt.positions.keys())
        for symbol in sorted(all_symbols):
            target_position = float(target_positions.get(symbol, 0.0))
            bar = bars.get(symbol)
            if bar is None:
                continue
            current_pos = bt.positions.get(symbol)
            current_qty = current_pos.quantity if current_pos else 0
            price = bar.get("open", bar.get("close", 0))
            target_value = equity * target_position
            target_qty = target_value / price if price > 0 else 0

            qty_diff = target_qty - current_qty

            if price > 0 and abs(qty_diff) > min_rebalance_notional_frac * equity / price:
                if qty_diff > 0:
                    bt.submit_order(
                        symbol, OrderSide.BUY, abs(qty_diff), OrderType.MARKET
                    )
                else:
                    bt.submit_order(
                        symbol, OrderSide.SELL, abs(qty_diff), OrderType.MARKET
                    )

        # Update drawdown tracking
        equity = bt._total_equity()
        state["peak_equity"] = max(state["peak_equity"], equity)
        state["current_drawdown"] = (equity - state["peak_equity"]) / state[
            "peak_equity"
        ]

    # Run backtest
    backtester.run(data, trading_strategy)

    # Get results
    metrics = backtester.get_metrics()

    # Extended metrics and gating
    def _returns(df: pd.DataFrame) -> pd.Series:
        out = df["close"].pct_change(fill_method=None).dropna()
        return _normalize_return_index(out)

    market_returns = None
    quant_returns = quant_returns_control

    try:
        if market_df is None:
            market_df = collector.fetch_ohlcv(market_benchmark, start_date, end_date)
        market_returns = _returns(market_df)
    except Exception:
        market_returns = _composite_returns_from_frames(
            frames=data,
            symbols=[market_benchmark],
        )

    if quant_returns is None and quant_benchmark:
        returns_list = []
        for sym in quant_benchmark:
            try:
                qdf = collector.fetch_ohlcv(sym, start_date, end_date)
                returns_list.append(_returns(qdf))
            except Exception:
                continue
        if returns_list:
            quant_returns = pd.concat(returns_list, axis=1).mean(axis=1).dropna()
        if quant_returns is None or quant_returns.empty:
            quant_returns = _composite_returns_from_frames(
                frames=data,
                symbols=quant_benchmark,
            )

    extended_metrics = compute_metrics(
        backtester.equity_curve,
        trades=backtester.trades,
        benchmark_returns=market_returns,
    )
    metrics.update(extended_metrics)

    fundamentals = []
    for symbol in symbols:
        try:
            fundamentals.append(collector.fetch_fundamentals(symbol))
        except Exception:
            continue
    metrics.update(aggregate_fundamentals(fundamentals))
    metrics["sleeves_enabled"] = bool(sleeves_enabled)
    metrics["sleeve_core_budget"] = float(core_budget)
    metrics["sleeve_alpha_budget"] = float(alpha_budget)
    metrics["sleeve_lottery_budget"] = float(lottery_budget)
    alpha_hist = state.get("alpha_scale_history", [])
    if alpha_hist:
        metrics["alpha_scale_last"] = float(alpha_hist[-1])
        metrics["alpha_scale_avg"] = float(np.mean(alpha_hist))
    else:
        metrics["alpha_scale_last"] = 1.0
        metrics["alpha_scale_avg"] = 1.0
    metrics["max_daily_loss_limit"] = float(max_daily_loss)
    metrics["max_weekly_loss_limit"] = float(max_weekly_loss)
    metrics["hard_dd_limit"] = float(hard_drawdown_limit)
    metrics["hard_dd_triggered"] = bool(state.get("hard_dd_triggered", False))
    metrics["hard_dd_trigger_ts"] = state.get("hard_dd_trigger_ts")
    metrics["hard_dd_halt_bars"] = int(state.get("hard_dd_halt_bars", 0))
    metrics["hard_dd_reentries"] = int(state.get("hard_dd_reentries", 0))
    metrics["hard_dd_reentry_ts"] = list(state.get("hard_dd_reentry_ts", []))
    metrics["hard_dd_last_quant_ir"] = state.get("hard_dd_last_quant_ir")

    market_metrics = (
        compute_metrics_from_returns(market_returns, benchmark_returns=market_returns)
        if market_returns is not None
        else {}
    )
    quant_metrics = (
        compute_metrics_from_returns(quant_returns, benchmark_returns=market_returns)
        if quant_returns is not None
        else {}
    )

    constraint_fail_reasons: List[str] = []
    metrics["benchmark_constraints_passed"] = False
    if quant_returns is not None and not quant_returns.empty:
        strategy_returns = backtester.get_equity_series().pct_change(fill_method=None).dropna()
        strategy_returns = _normalize_return_index(strategy_returns)
        strat_aligned, quant_aligned = strategy_returns.align(quant_returns, join="inner")
        if not strat_aligned.empty:
            rel_metrics = compute_metrics_from_returns(
                strat_aligned, benchmark_returns=quant_aligned
            )
            quant_total = _total_return_from_returns(quant_aligned)
            strat_total = _total_return_from_returns(strat_aligned)
            metrics["excess_total_return_vs_quant"] = float(strat_total - quant_total)
            metrics["quant_beta"] = float(rel_metrics.get("beta", 0.0))
            metrics["quant_information_ratio"] = float(
                rel_metrics.get("information_ratio", 0.0)
            )
            md_limit = float(validation_cfg.get("constraint_max_drawdown", 0.20))
            beta_min = float(validation_cfg.get("constraint_beta_min", 0.8))
            beta_max = float(validation_cfg.get("constraint_beta_max", 1.2))
            min_info = float(validation_cfg.get("constraint_min_information_ratio", 0.10))
            min_excess = float(
                validation_cfg.get("constraint_min_excess_total_return_vs_quant", 0.0)
            )
            metrics["constraint_max_drawdown_limit"] = float(md_limit)
            metrics["constraint_beta_min"] = float(beta_min)
            metrics["constraint_beta_max"] = float(beta_max)
            metrics["constraint_min_information_ratio"] = float(min_info)
            metrics["constraint_min_excess_total_return_vs_quant"] = float(min_excess)

            max_dd_ok = abs(float(metrics.get("max_drawdown", 0.0))) <= md_limit
            beta_ok = beta_min <= float(rel_metrics.get("beta", 0.0)) <= beta_max
            info_ok = float(rel_metrics.get("information_ratio", 0.0)) >= min_info
            excess_ok = float(metrics.get("excess_total_return_vs_quant", 0.0)) >= min_excess

            if not max_dd_ok:
                constraint_fail_reasons.append(
                    "max_drawdown_exceeded"
                )
            if not beta_ok:
                constraint_fail_reasons.append("quant_beta_out_of_bounds")
            if not info_ok:
                constraint_fail_reasons.append("information_ratio_below_min")
            if not excess_ok:
                constraint_fail_reasons.append("excess_total_return_vs_quant_below_min")
            metrics["benchmark_constraints_passed"] = len(constraint_fail_reasons) == 0
        else:
            constraint_fail_reasons.append("insufficient_overlap_with_quant_benchmark")
    else:
        constraint_fail_reasons.append("quant_benchmark_returns_unavailable")
    metrics["benchmark_constraint_fail_reasons"] = constraint_fail_reasons

    try:
        market_fund = collector.fetch_fundamentals(market_benchmark)
        market_metrics.update(aggregate_fundamentals([market_fund]))
    except Exception:
        pass

    if quant_benchmark:
        quant_funds = []
        for sym in quant_benchmark:
            try:
                quant_funds.append(collector.fetch_fundamentals(sym))
            except Exception:
                continue
        if quant_funds:
            quant_metrics.update(aggregate_fundamentals(quant_funds))

    gate_details = None
    if market_returns is not None and quant_returns is not None:
        gate = evaluate_gate(metrics, market_metrics, quant_metrics)
        metrics["gate_passed"] = gate.passed
        metrics["gate_ratio_good"] = gate.ratio_good
        metrics["gate_coverage"] = gate.coverage
        metrics["gate_good_count"] = gate.good_count
        metrics["gate_available"] = gate.available
        metrics["gate_required"] = gate.required
        metrics["gate_relaxed"] = gate.relaxed
        gate_details = gate.details

    quant_firm_benchmarks = {}
    quant_firm_rows = []
    quant_bench_cfg = bench_cfg.get("quant_firm_proxies", {})
    if isinstance(quant_bench_cfg, dict) and quant_bench_cfg.get("enabled", False):
        profile_names = quant_bench_cfg.get("profiles")
        if isinstance(profile_names, str):
            profile_names = [p.strip() for p in profile_names.split(",") if p.strip()]
        strategy_returns = backtester.get_equity_series().pct_change().dropna()
        try:
            quant_firm_benchmarks = evaluate_quant_firm_benchmarks(
                strategy_returns=strategy_returns,
                collector=collector,
                start_date=start_date,
                end_date=end_date,
                profile_names=profile_names,
                interval=str(quant_bench_cfg.get("interval", "1d")),
                min_assets=int(quant_bench_cfg.get("min_assets", 2)),
            )
            quant_firm_rows = benchmark_rows(quant_firm_benchmarks)
            metrics["quant_firm_profiles_evaluated"] = len(quant_firm_rows)
            metrics["quant_firm_profiles_outperformed"] = int(
                sum(1 for r in quant_firm_rows if r["excess_total_return"] > 0)
            )
            if quant_firm_rows:
                excess_vals = np.array(
                    [float(r["excess_total_return"]) for r in quant_firm_rows],
                    dtype=float,
                )
                metrics["quant_firm_best_excess_total_return"] = float(
                    excess_vals.max()
                )
                metrics["quant_firm_median_excess_total_return"] = float(
                    np.median(excess_vals)
                )
        except Exception as exc:
            quant_firm_benchmarks = {"error": str(exc)}

    factor_diagnostics = None
    if use_enhanced and enhanced_debug_components:
        try:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_dir = (
                PROJECT_ROOT / "quantum_alpha" / "artifacts" / f"factor_diag_{stamp}"
            )
            factor_diagnostics = run_enhanced_factor_diagnostics(
                frames=data,
                output_dir=output_dir,
            )
            metrics["factor_diagnostics_generated"] = True
            metrics["factor_diagnostics_dir"] = str(
                factor_diagnostics.get("output_dir", "")
            )
        except Exception as exc:
            factor_diagnostics = {"error": str(exc)}
            metrics["factor_diagnostics_generated"] = False
            metrics["factor_diagnostics_error"] = str(exc)

    if verbose:
        print(f"\n{'=' * 60}")
        print("BACKTEST RESULTS")
        print(f"{'=' * 60}")
        print(f"Total Return:    {metrics['total_return'] * 100:>10.2f}%")
        print(f"Annual Return:   {metrics['annual_return'] * 100:>10.2f}%")
        print(f"Volatility:      {metrics['volatility'] * 100:>10.2f}%")
        print(f"Sharpe Ratio:    {metrics['sharpe_ratio']:>10.2f}")
        print(f"Sortino Ratio:   {metrics['sortino_ratio']:>10.2f}")
        print(f"Max Drawdown:    {metrics['max_drawdown'] * 100:>10.2f}%")
        print(f"Calmar Ratio:    {metrics['calmar_ratio']:>10.2f}")
        print(f"Win Rate:        {metrics['win_rate'] * 100:>10.2f}%")
        print(f"Profit Factor:   {metrics['profit_factor']:>10.2f}")
        print(f"Total Trades:    {metrics['n_trades']:>10d}")
        print(f"Final Equity:    ${metrics['final_equity']:>10,.2f}")
        if "excess_total_return_vs_quant" in metrics:
            print(
                "Excess vs QQQ/IWM:"
                f" {metrics['excess_total_return_vs_quant'] * 100:>9.2f}%"
            )
            print(f"Quant Beta:      {metrics.get('quant_beta', 0.0):>10.2f}")
            print(
                "Quant InfoRatio: "
                f"{metrics.get('quant_information_ratio', 0.0):>10.2f}"
            )
            if "benchmark_constraints_passed" in metrics:
                print(
                    "Bench Constraints: "
                    f"{str(metrics['benchmark_constraints_passed']).upper():>7}"
                )
                if not metrics.get("benchmark_constraints_passed", False):
                    reasons = metrics.get("benchmark_constraint_fail_reasons", [])
                    if reasons:
                        print(f"Constraint Fails: {', '.join(str(r) for r in reasons)}")
        if "gate_passed" in metrics:
            print(f"Gate Passed:     {str(metrics['gate_passed']).upper():>10}")
            print(f"Gate Coverage:   {metrics['gate_coverage']:>10d}")
            if "gate_available" in metrics and "gate_required" in metrics:
                print(f"Gate Available:  {metrics['gate_available']:>10d}")
                print(f"Gate Required:   {metrics['gate_required']:>10d}")
                if metrics.get("gate_relaxed"):
                    print(f"Gate Relaxed:    {'TRUE':>10}")
            print(f"Gate Good:       {metrics['gate_good_count']:>10d}")
            print(f"Gate Ratio:      {metrics['gate_ratio_good'] * 100:>9.2f}%")
        print(f"{'=' * 60}\n")

        # Print gate detail breakdown
        if gate_details:
            print(f"{'GATE DETAIL BREAKDOWN':^60}")
            print(f"{'Metric':<28} {'Value':>8} {'Market':>8} {'Quant':>8} {'Pass':>6}")
            print("-" * 60)
            for m, d in sorted(gate_details.items()):
                if d.get("reason") == "missing":
                    continue
                v = d.get("value")
                mk = d.get("market")
                qt = d.get("quant")
                ok = d.get("good", False)
                fmt = (
                    lambda x: f"{x:>8.4f}"
                    if x is not None and abs(x) < 100
                    else f"{x:>8.1f}"
                    if x is not None
                    else f"{'N/A':>8}"
                )
                print(
                    f"{m:<28} {fmt(v)} {fmt(mk)} {fmt(qt)} {'  YES' if ok else '   NO':>6}"
                )
            print()

        if quant_firm_rows:
            print(f"{'QUANT-FIRM PROXY SCOREBOARD':^60}")
            print(
                f"{'Profile':<30} {'Excess':>9} {'IR':>7} {'DownCap':>9} {'HitRate':>9}"
            )
            print("-" * 60)
            for row in quant_firm_rows:
                print(
                    f"{row['profile_label']:<30} "
                    f"{row['excess_total_return'] * 100:>8.2f}% "
                    f"{row['information_ratio']:>7.2f} "
                    f"{row['downside_capture']:>9.2f} "
                    f"{row['hit_rate_vs_profile'] * 100:>8.2f}%"
                )
            print()

    results = {
        "metrics": metrics,
        "gate_details": gate_details,
        "quant_firm_benchmarks": quant_firm_benchmarks,
        "quant_firm_rows": quant_firm_rows,
        "factor_diagnostics": factor_diagnostics,
        "equity_curve": backtester.equity_curve,
        "trades": backtester.trades,
        "fills": backtester.fills,
    }

    # Run validation if requested
    if validate and len(data) > 0:
        if verbose:
            print("Running MCPT validation...")

        equity_series = backtester.get_equity_series()
        returns = equity_series.pct_change().dropna().values
        mcpt_p10 = float(validation_cfg.get("mcpt_threshold_stage1", 0.10))
        mcpt_p05 = float(validation_cfg.get("mcpt_threshold_stage2", 0.05))
        oos_window_days = int(validation_cfg.get("rolling_oos_window_days", 126))
        oos_min_windows = int(validation_cfg.get("rolling_oos_min_windows", 3))
        oos_min_beat_ratio = float(
            validation_cfg.get("rolling_oos_min_beat_ratio", 0.75)
        )
        promo_req_windows = int(
            validation_cfg.get("promotion_oos_required_windows", 4)
        )
        promo_req_beats = int(validation_cfg.get("promotion_oos_required_beats", 3))

        mcpt = MCPT(n_permutations=500, test_statistic="sharpe")
        mcpt_results = mcpt.run_on_returns(
            returns, show_progress=verbose, block_size=5, method="sign_flip"
        )
        p_value = float(mcpt_results.get("p_value", 1.0))
        mcpt_results["passes_stage1_0_10"] = p_value < mcpt_p10
        mcpt_results["passes_stage2_0_05"] = p_value < mcpt_p05

        results["mcpt"] = mcpt_results
        metrics["mcpt_p_value"] = p_value
        metrics["mcpt_pass_stage1_0_10"] = bool(mcpt_results["passes_stage1_0_10"])
        metrics["mcpt_pass_stage2_0_05"] = bool(mcpt_results["passes_stage2_0_05"])

        if quant_returns is not None and not quant_returns.empty:
            strategy_returns = equity_series.pct_change(fill_method=None).dropna()
            strategy_returns = _normalize_return_index(strategy_returns)
            rolling_oos = _rolling_oos_vs_benchmark(
                strategy_returns=strategy_returns,
                benchmark_returns=quant_returns,
                window_days=oos_window_days,
                min_windows=oos_min_windows,
                min_beat_ratio=oos_min_beat_ratio,
            )
            results["rolling_oos_vs_quant"] = rolling_oos
            metrics["rolling_oos_n_windows"] = int(rolling_oos.get("n_windows", 0))
            metrics["rolling_oos_beats"] = int(rolling_oos.get("beats", 0))
            metrics["rolling_oos_pass"] = bool(rolling_oos.get("passed", False))
            metrics["rolling_oos_required_beats"] = int(
                rolling_oos.get("required_beats", 0)
            )
            metrics["rolling_oos_min_beat_ratio"] = float(
                rolling_oos.get("min_beat_ratio", oos_min_beat_ratio)
            )
            strict_ready, strict_windows, strict_beats = _strict_promotion_ready(
                metrics=metrics,
                required_windows=promo_req_windows,
                required_beats=promo_req_beats,
            )
            metrics["promotion_oos_required_windows"] = int(strict_windows)
            metrics["promotion_oos_required_beats"] = int(strict_beats)
            metrics["promotion_ready"] = bool(strict_ready)
        else:
            metrics["promotion_oos_required_windows"] = max(int(promo_req_windows), 4)
            metrics["promotion_oos_required_beats"] = max(int(promo_req_beats), 3)
            metrics["promotion_ready"] = False

        if verbose:
            print(f"\nMCPT Results:")
            print(f"  P-Value: {mcpt_results['p_value']:.4f}")
            print(f"  Significant: {'YES' if mcpt_results['is_significant'] else 'NO'}")
            print(f"  Percentile: {mcpt_results['percentile']:.1f}%")
            print(
                f"  Stage1 (p<{mcpt_p10:.2f}): "
                f"{'PASS' if mcpt_results['passes_stage1_0_10'] else 'FAIL'}"
            )
            print(
                f"  Stage2 (p<{mcpt_p05:.2f}): "
                f"{'PASS' if mcpt_results['passes_stage2_0_05'] else 'FAIL'}"
            )
            if "rolling_oos_vs_quant" in results:
                oos = results["rolling_oos_vs_quant"]
                print("\nRolling OOS vs QQQ/IWM:")
                print(f"  Windows: {oos.get('n_windows', 0)}")
                print(f"  Beats:   {oos.get('beats', 0)}")
                print(f"  Needed:  {oos.get('required_beats', 0)}")
                print(f"  Passed:  {'YES' if oos.get('passed', False) else 'NO'}")
                print(
                    "  Promotion (>=3/4 + constraints + MCPT stage2): "
                    f"{'YES' if metrics.get('promotion_ready', False) else 'NO'}"
                )
    else:
        metrics.setdefault("mcpt_pass_stage2_0_05", False)
        metrics.setdefault("rolling_oos_n_windows", 0)
        metrics.setdefault("rolling_oos_beats", 0)
        metrics["promotion_oos_required_windows"] = max(
            int(validation_cfg.get("promotion_oos_required_windows", 4)),
            4,
        )
        metrics["promotion_oos_required_beats"] = max(
            int(validation_cfg.get("promotion_oos_required_beats", 3)),
            3,
        )
        metrics["promotion_ready"] = False

    promotion_verdict = _promotion_verdict_from_metrics(metrics)
    if not validate:
        reasons = list(promotion_verdict.get("fail_reasons", []))
        if "validation_not_run" not in reasons:
            reasons.insert(0, "validation_not_run")
        promotion_verdict["fail_reasons"] = reasons
        promotion_verdict["failed"] = True
        promotion_verdict["eligible"] = False
    metrics["promotion_ready"] = bool(promotion_verdict.get("eligible", False))
    metrics["promotion_fail_reasons"] = list(
        promotion_verdict.get("fail_reasons", [])
    )
    results["promotion_verdict"] = promotion_verdict

    if verbose:
        verdict = results["promotion_verdict"]
        print("Promotion Verdict:")
        print(f"  Eligible: {'YES' if verdict.get('eligible') else 'NO'}")
        reasons = verdict.get("fail_reasons", [])
        if reasons:
            print(f"  Fail Reasons: {', '.join(str(r) for r in reasons)}")

    return results


def run_paper(
    symbols: list,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 100000,
    strategy_type: str = "enhanced",
    paper_bars: int = 120,
    strategy_kwargs: Optional[Dict] = None,
    config_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run a paper trading simulation on the most recent bars.
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print("QUANTUM ALPHA V1 - PAPER TRADING")
        print(f"{'=' * 60}")
        print(f"Symbols: {_format_symbols(symbols)}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Capital: ${initial_capital:,.0f}")
        print(f"Strategy: {strategy_type}")
        print(f"Paper Bars: {paper_bars}")
        print(f"{'=' * 60}\n")

    settings = load_config(config_path)
    risk_cfg = settings.get("risk", {}) if settings else {}
    strategy_cfg = settings.get("strategy", {}) if settings else {}
    validation_cfg = settings.get("validation", {}) if settings else {}
    config_dir = _resolve_config_dir(config_path)
    strategies_cfg = _load_optional_yaml(config_dir / "strategies.yaml")
    sentiment_cfg = {}
    if strategies_cfg and "strategies" in strategies_cfg:
        sentiment_cfg = strategies_cfg["strategies"].get("sentiment", {})
    max_position = float(risk_cfg.get("max_position_size", 0.25))
    max_leverage = float(risk_cfg.get("max_portfolio_leverage", 1.0))
    max_drawdown = float(risk_cfg.get("max_drawdown", 0.10))
    hard_drawdown_limit = float(risk_cfg.get("hard_drawdown_limit", 0.20))
    hard_drawdown_action = str(risk_cfg.get("hard_drawdown_action", "flatten"))
    hard_drawdown_cooldown_days = int(risk_cfg.get("hard_drawdown_cooldown_days", 20))
    hard_drawdown_recovery_level = float(risk_cfg.get("hard_drawdown_recovery_level", 0.10))
    hard_drawdown_require_positive_quant_ir = bool(
        risk_cfg.get("hard_drawdown_require_positive_quant_ir", True)
    )
    hard_drawdown_ir_lookback_bars = int(
        risk_cfg.get("hard_drawdown_ir_lookback_bars", 63)
    )
    max_daily_loss = float(risk_cfg.get("max_daily_loss", 0.03))
    max_weekly_loss = float(risk_cfg.get("max_weekly_loss", 0.07))
    kelly_fraction = float(risk_cfg.get("kelly_fraction", 0.5))
    core_drawdown_scaling_enabled = bool(
        risk_cfg.get("core_drawdown_scaling_enabled", True)
    )
    core_min_exposure_in_drawdown = _clamp(
        float(risk_cfg.get("core_min_exposure_in_drawdown", 0.35)),
        0.0,
        1.0,
    )
    rebalance_frequency = str(strategy_cfg.get("rebalance_frequency", "daily"))
    momentum_top_pct = float(strategy_cfg.get("momentum_top_pct", 80))
    momentum_bottom_pct = float(strategy_cfg.get("momentum_bottom_pct", 35))
    signal_threshold = float(strategy_cfg.get("signal_threshold", 0.3))
    signal_scale = float(strategy_cfg.get("signal_scale", 1.0))
    min_long_signal = float(strategy_cfg.get("min_long_signal", 0.0))
    market_off_scale = float(strategy_cfg.get("market_off_scale", 1.0))
    anchor_core_to_quant = bool(
        strategy_cfg.get("anchor_core_to_quant_composite", False)
    )
    alpha_deploy_requires_relative_edge = bool(
        strategy_cfg.get("alpha_deploy_requires_relative_edge", True)
    )
    alpha_edge_lookback_days = int(strategy_cfg.get("alpha_edge_lookback_days", 63))
    alpha_min_scale = _clamp(float(strategy_cfg.get("alpha_min_scale", 0.05)), 0.0, 1.0)
    alpha_info_floor = float(strategy_cfg.get("alpha_info_floor", 0.0))
    alpha_excess_floor = float(strategy_cfg.get("alpha_excess_floor", 0.0))
    alpha_recovery_info = float(strategy_cfg.get("alpha_recovery_info", 0.25))
    alpha_recovery_excess = float(strategy_cfg.get("alpha_recovery_excess", 0.03))
    rs_top_n = int(strategy_cfg.get("relative_strength_top_n", 0))
    rs_min_mom = float(strategy_cfg.get("relative_strength_min_mom", 0.0))
    use_relative_strength = bool(strategy_cfg.get("use_relative_strength", False))
    liquid_subset_size = int(strategy_cfg.get("liquid_subset_size", 0))
    liquid_adv_window = int(strategy_cfg.get("liquid_adv_window", 20))
    liquid_min_history = int(strategy_cfg.get("liquid_min_history", 120))
    liquid_for_full_only = bool(strategy_cfg.get("liquid_for_full_only", True))
    long_only = bool(strategy_cfg.get("long_only", False))
    risk_off_cash = bool(strategy_cfg.get("risk_off_cash", False))
    min_rebalance_notional_frac = float(
        strategy_cfg.get("min_rebalance_notional_frac", 0.015)
    )
    sleeves_cfg = strategy_cfg.get("sleeves", {}) if isinstance(strategy_cfg, dict) else {}
    sleeves_enabled = bool(sleeves_cfg.get("enabled", True))
    core_budget = _clamp(float(sleeves_cfg.get("core_budget", 0.96)), 0.70, 1.00)
    alpha_budget = _clamp(float(sleeves_cfg.get("alpha_budget", 0.04)), 0.00, 0.30)
    lottery_budget = _clamp(float(sleeves_cfg.get("lottery_budget", 0.00)), 0.00, 0.10)
    core_loss_scale = _clamp(float(sleeves_cfg.get("core_loss_scale", 0.50)), 0.10, 1.00)
    lottery_top_n = int(max(1, sleeves_cfg.get("lottery_top_n", 3)))
    lottery_min_signal = float(sleeves_cfg.get("lottery_min_signal", 0.35))
    lottery_min_confidence = float(sleeves_cfg.get("lottery_min_confidence", 0.25))
    lottery_stop_loss = _clamp(float(sleeves_cfg.get("lottery_stop_loss", 0.08)), 0.01, 0.25)
    lottery_max_per_symbol = _clamp(
        float(sleeves_cfg.get("lottery_max_per_symbol", 0.0075)), 0.001, 0.02
    )
    alpha_min_confidence = float(sleeves_cfg.get("alpha_min_confidence", 0.05))
    core_tilt_enabled = bool(sleeves_cfg.get("core_tilt_enabled", False))
    enhanced_debug_components = bool(
        strategy_cfg.get("enhanced_debug_components", False)
    )
    core_tilt_lookback_days = int(sleeves_cfg.get("core_tilt_lookback_days", 126))
    core_tilt_strength = _clamp(
        float(sleeves_cfg.get("core_tilt_strength", 0.35)), 0.0, 0.95
    )
    core_rebalance_band = _clamp(
        float(sleeves_cfg.get("core_rebalance_band", 0.02)), 0.0, 0.10
    )
    core_mix_cfg = sleeves_cfg.get("core_mix", {"SPY": 0.5, "QQQ": 0.3, "IWM": 0.2})
    core_mix = {
        str(k).upper(): float(v)
        for k, v in core_mix_cfg.items()
        if float(v) > 0
    }
    if not core_mix:
        core_mix = {"SPY": 0.5, "QQQ": 0.3, "IWM": 0.2}

    if max_leverage > 0:
        sleeve_total = core_budget + alpha_budget + lottery_budget
        if sleeve_total > max_leverage and sleeve_total > 0:
            scale = max_leverage / sleeve_total
            core_budget *= scale
            alpha_budget *= scale
            lottery_budget *= scale

    target_vol = float(risk_cfg.get("target_volatility", 0.15))
    dd_warning = float(risk_cfg.get("dd_warning", 0.15))
    dd_critical = float(risk_cfg.get("dd_critical", 0.25))
    dd_breaker = float(risk_cfg.get("dd_circuit_breaker", 0.40))
    dd_limit = float(risk_cfg.get("dd_max_limit", 0.50))
    dd_min_exposure = float(risk_cfg.get("dd_min_exposure", 0.25))
    dd_leverage_cap_warning = float(risk_cfg.get("dd_leverage_cap_warning", 0.85))
    dd_leverage_cap_critical = float(risk_cfg.get("dd_leverage_cap_critical", 0.8))
    market_off_leverage_cap = float(risk_cfg.get("market_off_leverage_cap", 0.85))
    faster_derisk_warning_mult = _clamp(
        float(risk_cfg.get("faster_derisk_warning_mult", 0.85)), 0.10, 1.00
    )
    faster_derisk_critical_mult = _clamp(
        float(risk_cfg.get("faster_derisk_critical_mult", 0.70)), 0.10, 1.00
    )
    if max_leverage <= 0:
        max_leverage = 1.0
    min_paper_bars = int(validation_cfg.get("paper_min_bars", 120))
    if paper_bars < min_paper_bars:
        if verbose:
            print(
                f"paper-bars {paper_bars} is below validation minimum "
                f"{min_paper_bars}; using {min_paper_bars}"
            )
        paper_bars = min_paper_bars

    collector = DataCollector(runtime_mode="paper", config_path=config_path)
    feature_gen = TechnicalFeatureGenerator()
    cleaner = DataCleaner()
    imputer = MissingValueImputer()

    full_token_requested = bool(
        symbols
        and len(symbols) == 1
        and str(symbols[0]).upper() in {"FULL", "ALL", "UNIVERSE"}
    )
    symbols = _resolve_symbols(symbols, collector, settings)

    if strategy_type == "momentum":
        strategy = MomentumStrategy()
    elif strategy_type == "composite":
        strategy = CompositeStrategy()
    elif strategy_type in ("adaptive", "enhanced"):
        strategy = EnhancedCompositeStrategy(
            debug_components=enhanced_debug_components
        )
    elif strategy_type == "sentiment":
        from quantum_alpha.strategy.sentiment_strategies import SocialSentimentStrategy

        strategy = SocialSentimentStrategy()
    elif strategy_type == "ml":
        from quantum_alpha.strategy.ml_strategies import MLTradingStrategy

        strategy = MLTradingStrategy(**(strategy_kwargs or {}))
    elif strategy_type == "news_lstm":
        from quantum_alpha.strategy.news_lstm_strategy import NewsLSTMStrategy

        strategy = NewsLSTMStrategy(**(strategy_kwargs or {}))
    elif strategy_type == "meta_ensemble":
        from quantum_alpha.strategy.meta_ensemble_strategy import MetaEnsembleStrategy

        strategy = MetaEnsembleStrategy(**(strategy_kwargs or {}))
    else:
        strategy = MomentumStrategy()

    use_enhanced = isinstance(strategy, EnhancedCompositeStrategy)
    use_sentiment = strategy_type == "sentiment"

    if use_sentiment:
        from quantum_alpha.data.collectors.alternative import (
            load_social_sentiment,
            load_options_sentiment,
            load_insider_trades,
            load_congress_trades,
        )
        from quantum_alpha.strategy.sentiment_strategies import (
            SocialSentimentStrategy,
            OptionsSentimentStrategy,
            InsiderTradingStrategy,
            CongressTradingStrategy,
        )

        social_strategy = SocialSentimentStrategy(**sentiment_cfg.get("social", {}))
        options_strategy = OptionsSentimentStrategy(**sentiment_cfg.get("options", {}))
        insider_strategy = InsiderTradingStrategy(**sentiment_cfg.get("insider", {}))
        congress_strategy = CongressTradingStrategy(**sentiment_cfg.get("congress", {}))
        sentiment_weights = sentiment_cfg.get(
            "combined_weights",
            {"social": 0.4, "options": 0.2, "insider": 0.2, "congress": 0.2},
        )
        signal_threshold = float(
            sentiment_cfg.get("signal_threshold", signal_threshold)
        )

    position_sizer = PositionSizer(
        max_position=max_position,
        kelly_fraction=kelly_fraction,
        max_drawdown=max_drawdown,
    )
    paper_trader = PaperTrader(initial_capital=initial_capital, paper_bars=paper_bars)

    bench_cfg = settings.get("benchmarks", {})
    market_benchmark = str(bench_cfg.get("market", "SPY")).upper()
    quant_raw = bench_cfg.get("quant_composite", ["QQQ", "IWM"])
    if isinstance(quant_raw, list):
        quant_benchmark = [str(s).upper() for s in quant_raw if str(s).strip()]
    elif quant_raw:
        quant_benchmark = [str(quant_raw).upper()]
    else:
        quant_benchmark = []

    if anchor_core_to_quant and quant_benchmark:
        eq_w = 1.0 / len(quant_benchmark)
        core_mix = {sym: eq_w for sym in quant_benchmark}
        if verbose:
            print(f"Core sleeve anchored to quant composite: {core_mix}")

    symbols = _augment_symbols_with_benchmarks(
        symbols=symbols,
        core_mix=core_mix,
        market_benchmark=market_benchmark,
        quant_benchmark=quant_benchmark,
    )
    reserved_benchmark_symbols = set(core_mix.keys()) | set(quant_benchmark) | {
        market_benchmark
    }
    market_df = None
    market_trend = None
    quant_returns_control = None
    try:
        market_df = collector.fetch_ohlcv(market_benchmark, start_date, end_date)
        m_close = market_df["close"]
        m_ma = m_close.rolling(200).mean()
        market_trend = m_close.shift(1) > m_ma.shift(1)
    except Exception:
        market_df = None
        market_trend = None
    if quant_benchmark:
        quant_returns_control = _fetch_composite_returns(
            collector=collector,
            symbols=quant_benchmark,
            start_date=start_date,
            end_date=end_date,
        )

    if verbose:
        print("Collecting price data...")

    data = {}
    for symbol in symbols:
        try:
            df = collector.fetch_ohlcv(symbol, start_date, end_date)
            df = cleaner.clean(df)
            df = imputer.impute(df)
            df = feature_gen.generate(df)
            if use_sentiment:
                social_df = load_social_sentiment(symbol, use_live=True)
                options_df = load_options_sentiment(symbol, use_live=True)
                insider_df = load_insider_trades(symbol, use_live=True)
                congress_df = load_congress_trades(symbol, use_live=True)

                if "symbol" in social_df.columns:
                    social_df = social_df[social_df["symbol"] == symbol]
                if "symbol" in options_df.columns:
                    options_df = options_df[options_df["symbol"] == symbol]
                if "symbol" in insider_df.columns:
                    insider_df = insider_df[insider_df["symbol"] == symbol]
                if "symbol" in congress_df.columns:
                    congress_df = congress_df[congress_df["symbol"] == symbol]

                frames = {}
                if not social_df.empty:
                    frames["social"] = _align_signal_frame(
                        social_strategy.generate_signals(social_df), df.index
                    )
                if not options_df.empty:
                    if "timestamp" in options_df.columns:
                        options_df = options_df.copy()
                        options_df["timestamp"] = pd.to_datetime(
                            options_df["timestamp"]
                        )
                        options_df = options_df.set_index("timestamp")
                    elif "date" in options_df.columns:
                        options_df = options_df.copy()
                        options_df["date"] = pd.to_datetime(options_df["date"])
                        options_df = options_df.set_index("date")
                    options_sig = options_strategy.generate_signals(options_df)
                    frames["options"] = _align_signal_frame(options_sig, df.index)
                if not insider_df.empty:
                    frames["insider"] = _align_signal_frame(
                        insider_strategy.generate_signals(insider_df),
                        df.index,
                        limit=20,
                    )
                if not congress_df.empty:
                    frames["congress"] = _align_signal_frame(
                        congress_strategy.generate_signals(congress_df),
                        df.index,
                        limit=20,
                    )

                combined_signal = pd.Series(0.0, index=df.index)
                combined_conf = pd.Series(0.0, index=df.index)
                total_w = 0.0
                for key, sig_frame in frames.items():
                    weight = float(sentiment_weights.get(key, 0.0))
                    if weight <= 0:
                        continue
                    total_w += weight
                    combined_signal += weight * sig_frame["signal"]
                    combined_conf += weight * sig_frame["signal_confidence"]

                if total_w > 0:
                    combined_signal = combined_signal / total_w
                    combined_conf = combined_conf / total_w
                else:
                    combined_signal = combined_signal * 0.0
                    combined_conf = combined_conf * 0.0

                df["signal"] = combined_signal
                df["signal_confidence"] = combined_conf
                df["position_signal"] = np.where(
                    np.abs(df["signal"]) >= signal_threshold,
                    np.sign(df["signal"]),
                    0.0,
                )
                df = _apply_signal_lag(df)
                data[symbol] = df
            else:
                df = strategy.generate_signals(df)
                df = _apply_signal_lag(df)
                data[symbol] = df
            if verbose:
                print(f"  {symbol}: {len(df)} bars")
        except Exception as e:
            if verbose:
                print(f"  {symbol}: FAILED - {e}")

    if liquid_subset_size > 0 and len(data) > liquid_subset_size:
        if full_token_requested or not liquid_for_full_only:
            selected = _select_liquid_subset(
                data,
                subset_size=liquid_subset_size,
                adv_window=liquid_adv_window,
                min_history=liquid_min_history,
            )
            selected_set = set(selected) | {
                sym for sym in reserved_benchmark_symbols if sym in data
            }
            data = {sym: df for sym, df in data.items() if sym in selected_set}
            if verbose:
                print(
                    "  Liquidity filter applied: "
                    f"trading top {len(selected)} symbols by {liquid_adv_window}D ADV"
                )

    if not data:
        return {"error": "No data collected"}

    if (market_df is None or market_df.empty) and market_benchmark in data:
        market_df = data[market_benchmark]
        try:
            m_close = pd.to_numeric(market_df["close"], errors="coerce")
            m_ma = m_close.rolling(200).mean()
            market_trend = m_close.shift(1) > m_ma.shift(1)
        except Exception:
            market_trend = None

    if quant_returns_control is None or quant_returns_control.empty:
        quant_returns_control = _composite_returns_from_frames(
            frames=data,
            symbols=quant_benchmark,
        )

    if verbose:
        print(f"\nRunning paper trading simulation...")

    state = {
        "positions": {},
        "trade_history": [],
        "current_drawdown": 0,
        "peak_equity": initial_capital,
        "last_rebalance": None,
        "loss_day_key": None,
        "loss_week_key": None,
        "day_start_equity": float(initial_capital),
        "week_start_equity": float(initial_capital),
        "daily_stop_active": False,
        "weekly_stop_active": False,
        "loss_stop_active": False,
        "lottery_entry_price": {},
        "lottery_blocked_day": {},
        "alpha_scale_last": 1.0,
        "alpha_scale_history": [],
    }
    _init_hard_drawdown_state(state)
    dd_controller = DrawdownController(
        warning_threshold=dd_warning,
        critical_threshold=dd_critical,
        circuit_breaker_threshold=dd_breaker,
        max_drawdown_limit=dd_limit,
        scaling_method="linear",
        min_exposure=dd_min_exposure,
        cooldown_days=10,
    )
    dd_controller.reset(initial_capital)

    def trading_strategy(timestamp, bars, bt):
        equity = bt._total_equity()
        state["peak_equity"] = max(float(state.get("peak_equity", equity)), equity)
        peak = float(state.get("peak_equity", equity))
        state["current_drawdown"] = (equity - peak) / peak if peak > 0 else 0.0

        loss_status = _update_loss_limit_state(
            state=state,
            timestamp=timestamp,
            equity=equity,
            max_daily_loss=max_daily_loss,
            max_weekly_loss=max_weekly_loss,
        )
        hard_dd = _update_hard_drawdown_guard(
            state=state,
            timestamp=timestamp,
            current_drawdown=float(state.get("current_drawdown", 0.0)),
            equity_curve=bt.equity_curve,
            quant_returns=quant_returns_control,
            limit=hard_drawdown_limit,
            action=hard_drawdown_action,
            cooldown_days=hard_drawdown_cooldown_days,
            recovery_level=hard_drawdown_recovery_level,
            require_positive_quant_ir=hard_drawdown_require_positive_quant_ir,
            ir_lookback_bars=hard_drawdown_ir_lookback_bars,
        )
        dd_metrics = dd_controller.update(equity, timestamp)
        exposure_mult = dd_metrics.exposure_multiplier

        force_rebalance = bool(
            loss_status["stop_active"]
            or hard_dd.get("trigger_now", False)
            or hard_dd.get("force_flatten", False)
        )
        if (
            not _should_rebalance(timestamp, state["last_rebalance"], rebalance_frequency)
            and not force_rebalance
        ):
            return

        state["last_rebalance"] = timestamp
        if bool(hard_dd.get("force_flatten", False)):
            _submit_flatten_orders(bt, bars)
            return

        target_positions = {}
        signal_strengths = {}
        signal_confidences = {}
        volatilities = {}
        rp_drawdown_mult = 1.0
        trade_history = (
            np.array([t["pnl"] for t in bt.trades]) if bt.trades else np.array([0])
        )
        realized_vol = _realized_vol_from_equity(bt.equity_curve)
        vol_scale = 1.0
        if realized_vol and realized_vol > 0:
            vol_scale = float(np.clip(target_vol / realized_vol, 0.5, 1.5))
        market_risk_on = True
        if market_trend is not None and timestamp in market_trend.index:
            market_risk_on = bool(market_trend.loc[timestamp])
        allow_shorts = not long_only
        if not market_risk_on:
            allow_shorts = False

        mom_scores = {}
        for sym, df in data.items():
            if timestamp in df.index:
                mom_val = df.loc[timestamp].get("mom_12m", 0.0)
                if pd.notna(mom_val):
                    mom_scores[sym] = float(mom_val)
        top_cut = None
        bottom_cut = None
        top_syms = None
        if len(mom_scores) >= 3:
            vals = np.array(list(mom_scores.values()), dtype=float)
            top_cut = np.nanpercentile(vals, momentum_top_pct)
            bottom_cut = np.nanpercentile(vals, momentum_bottom_pct)
            if use_relative_strength and rs_top_n > 0:
                ranked = sorted(mom_scores.items(), key=lambda x: x[1], reverse=True)
                top_syms = {sym for sym, val in ranked[:rs_top_n] if val >= rs_min_mom}
        use_rp_allocation = top_syms is not None
        for symbol, bar in bars.items():
            if symbol not in data:
                continue

            df = data[symbol]
            if timestamp not in df.index:
                continue

            row = df.loc[timestamp]
            signal = (
                row.get("position_signal")
                if "position_signal" in row
                else row.get("signal", 0)
            )
            confidence = float(row.get("signal_confidence", 0.5))
            if top_syms is not None:
                signal = 1.0 if symbol in top_syms else 0.0
            else:
                if signal < 0 and not allow_shorts:
                    signal = 0.0
                if top_cut is not None and bottom_cut is not None:
                    mom_val = mom_scores.get(symbol)
                    if signal > 0 and (mom_val is None or mom_val < top_cut):
                        signal = 0.0
                    if signal < 0 and (
                        mom_val is None or mom_val > bottom_cut or long_only
                    ):
                        signal = 0.0
            if signal_scale != 1.0 and signal != 0.0:
                signal = float(np.clip(signal * signal_scale, -1.0, 1.0))
            if long_only and signal > 0 and min_long_signal > 0:
                signal = max(signal, min_long_signal)
            signal_strengths[symbol] = float(signal)
            signal_confidences[symbol] = float(confidence)
            volatility = row.get("atr_pct", 0.02)
            if pd.isna(volatility) or volatility <= 0:
                volatility = 0.02
            volatility = volatility * np.sqrt(252)
            volatilities[symbol] = max(volatility, 1e-6)

            confidence = row.get("signal_confidence", 0.5)

            sizing = position_sizer.calculate(
                trade_history=trade_history,
                current_volatility=max(volatility, 0.01),
                current_drawdown=state["current_drawdown"],
                signal_strength=signal,
                signal_confidence=confidence,
            )

            if sizing["halt_trading"]:
                return

            rp_drawdown_mult = min(rp_drawdown_mult, float(sizing.get("dd_multiplier", 1.0)))
            if use_rp_allocation:
                continue

            target_positions[symbol] = sizing["position_size"]

        if use_rp_allocation:
            # Only allocate to symbols selected by relative strength
            selected_vols = {
                s: v
                for s, v in volatilities.items()
                if top_syms is None or s in top_syms
            }
            if selected_vols:
                inv = {s: 1 / v for s, v in selected_vols.items()}
                total_inv = sum(inv.values())
                if total_inv > 0:
                    target_positions = {
                        s: (inv[s] / total_inv) * max_leverage * rp_drawdown_mult
                        for s in inv
                    }
            if not target_positions:
                return
        if risk_off_cash and not market_risk_on:
            target_positions = {s: 0.0 for s in bars.keys()}
        elif not target_positions:
            return
        if not market_risk_on and market_off_scale < 1.0:
            target_positions = {
                s: w * market_off_scale for s, w in target_positions.items()
            }

        if volatilities and not use_rp_allocation:
            inv = {s: 1 / v for s, v in volatilities.items()}
            total_inv = sum(inv.values())
            if total_inv > 0:
                rp_weights = {s: inv[s] / total_inv for s in inv}
                weight_scale = len(rp_weights)
                target_positions = {
                    s: target_positions[s] * rp_weights.get(s, 0) * weight_scale
                    for s in target_positions
                }

        alpha_targets = dict(target_positions)
        if vol_scale != 1.0:
            alpha_targets = {s: w * vol_scale for s, w in alpha_targets.items()}

        if exposure_mult < 1.0:
            alpha_targets = {
                s: w * exposure_mult for s, w in alpha_targets.items()
            }

        cap = max_leverage
        if dd_metrics.state == DrawdownState.WARNING:
            cap = min(cap, dd_leverage_cap_warning * faster_derisk_warning_mult)
        elif dd_metrics.state in {DrawdownState.CRITICAL, DrawdownState.RECOVERY}:
            cap = min(cap, dd_leverage_cap_critical * faster_derisk_critical_mult)
        if not market_risk_on:
            cap = min(cap, market_off_leverage_cap * faster_derisk_warning_mult)
        if anchor_core_to_quant and not loss_status["stop_active"]:
            cap = max(cap, min(max_leverage, core_budget))

        if sleeves_enabled:
            today_key = timestamp.date().isoformat()
            blocked = state.setdefault("lottery_blocked_day", {})
            entries = state.setdefault("lottery_entry_price", {})

            for sym in list(blocked.keys()):
                if blocked.get(sym) != today_key:
                    blocked.pop(sym, None)

            for sym, entry in list(entries.items()):
                bar = bars.get(sym)
                if bar is None:
                    entries.pop(sym, None)
                    continue
                px = float(bar.get("open", bar.get("close", 0.0)) or 0.0)
                if px > 0 and px <= float(entry) * (1.0 - lottery_stop_loss):
                    blocked[sym] = today_key
                    entries.pop(sym, None)

            core_drawdown_mult = 1.0
            if core_drawdown_scaling_enabled:
                core_drawdown_mult = max(
                    core_min_exposure_in_drawdown,
                    min(1.0, float(exposure_mult)),
                )
            core_budget_eff = (
                core_budget
                * (core_loss_scale if loss_status["stop_active"] else 1.0)
                * core_drawdown_mult
            )
            alpha_budget_eff = 0.0 if loss_status["stop_active"] else alpha_budget
            lottery_budget_eff = 0.0 if loss_status["stop_active"] else lottery_budget

            alpha_scale = 1.0
            if (
                alpha_budget_eff > 0
                and alpha_deploy_requires_relative_edge
                and quant_returns_control is not None
                and not quant_returns_control.empty
            ):
                edge = _rolling_relative_edge(
                    equity_curve=bt.equity_curve[-(max(alpha_edge_lookback_days, 20) + 10) :],
                    benchmark_returns=quant_returns_control,
                    asof=timestamp,
                    lookback_days=alpha_edge_lookback_days,
                )
                if bool(edge.get("available", False)):
                    info_ratio = float(edge.get("information_ratio", 0.0))
                    excess_total = float(edge.get("excess_total_return", 0.0))
                    info_den = max(alpha_recovery_info - alpha_info_floor, 1e-6)
                    excess_den = max(alpha_recovery_excess - alpha_excess_floor, 1e-6)
                    info_score = float(
                        np.clip((info_ratio - alpha_info_floor) / info_den, 0.0, 1.0)
                    )
                    excess_score = float(
                        np.clip(
                            (excess_total - alpha_excess_floor) / excess_den, 0.0, 1.0
                        )
                    )
                    alpha_scale = float(
                        alpha_min_scale
                        + (1.0 - alpha_min_scale) * (0.5 * info_score + 0.5 * excess_score)
                    )
                else:
                    alpha_scale = float(alpha_min_scale)

            alpha_budget_eff *= alpha_scale
            lottery_budget_eff *= alpha_scale
            state["alpha_scale_last"] = float(alpha_scale)
            state.setdefault("alpha_scale_history", []).append(float(alpha_scale))

            effective_core_mix = core_mix
            if core_tilt_enabled:
                effective_core_mix = _tilt_core_mix(
                    core_mix=core_mix,
                    data_frames=data,
                    timestamp=timestamp,
                    lookback_days=core_tilt_lookback_days,
                    tilt_strength=core_tilt_strength,
                )
            core_targets = _core_targets_from_mix(
                bars=bars,
                core_mix=effective_core_mix,
                core_budget=core_budget_eff,
            )
            alpha_candidates = {
                s: w
                for s, w in alpha_targets.items()
                if s not in core_mix
                and float(signal_confidences.get(s, 0.0)) >= alpha_min_confidence
            }

            if core_budget_eff > 0 and not core_targets:
                fallback_anchors = {}
                for sym in list(quant_benchmark) + [market_benchmark]:
                    bar = bars.get(sym)
                    if bar is None:
                        continue
                    px = float(bar.get("open", bar.get("close", 0.0)) or 0.0)
                    if px > 0:
                        fallback_anchors[sym] = 1.0
                if fallback_anchors:
                    core_targets = _normalize_weights(
                        fallback_anchors,
                        budget=core_budget_eff,
                        long_only=True,
                        max_abs_per_symbol=max_position,
                    )

            if core_budget_eff > 0 and not core_targets:
                fallback_from_alpha = {
                    s: max(float(w), 0.0)
                    for s, w in alpha_candidates.items()
                    if float(w) > 0
                }
                if fallback_from_alpha:
                    core_targets = _normalize_weights(
                        fallback_from_alpha,
                        budget=core_budget_eff,
                        long_only=True,
                        max_abs_per_symbol=max_position,
                    )

            if core_budget_eff > 0 and not core_targets:
                core_targets = _fallback_core_targets_from_bars(
                    bars=bars,
                    core_budget=core_budget_eff,
                    max_abs_per_symbol=max_position,
                    top_n=12,
                )

            if core_targets:
                alpha_candidates = {
                    s: w for s, w in alpha_candidates.items() if s not in core_targets
                }

            alpha_targets = _normalize_weights(
                alpha_candidates,
                budget=alpha_budget_eff,
                long_only=long_only,
                max_abs_per_symbol=max_position,
            )

            lottery_targets = {}
            if lottery_budget_eff > 0:
                lottery_scores = {}
                for sym, sig in signal_strengths.items():
                    if sym in core_mix or sym in alpha_targets:
                        continue
                    if blocked.get(sym) == today_key:
                        continue
                    conf = float(signal_confidences.get(sym, 0.0))
                    if conf < lottery_min_confidence:
                        continue
                    if long_only and sig < lottery_min_signal:
                        continue
                    if (not long_only) and abs(sig) < lottery_min_signal:
                        continue
                    vol = float(volatilities.get(sym, 0.0))
                    if vol <= 0:
                        continue
                    lottery_scores[sym] = abs(float(sig)) * max(conf, 0.05) * vol

                ranked = sorted(lottery_scores.items(), key=lambda x: x[1], reverse=True)
                raw_lottery = {sym: 1.0 for sym, _ in ranked[:lottery_top_n]}
                lottery_targets = _normalize_weights(
                    raw_lottery,
                    budget=lottery_budget_eff,
                    long_only=True,
                    max_abs_per_symbol=lottery_max_per_symbol,
                )

            target_positions = _combine_targets(core_targets, alpha_targets, lottery_targets)

            for sym, w in lottery_targets.items():
                if w <= 0:
                    continue
                bar = bars.get(sym)
                if bar is None:
                    continue
                px = float(bar.get("open", bar.get("close", 0.0)) or 0.0)
                if px > 0 and sym not in entries:
                    entries[sym] = px
            for sym in list(entries.keys()):
                if sym not in lottery_targets:
                    entries.pop(sym, None)

            if loss_status["stop_active"]:
                cap = min(cap, max(core_budget_eff, 0.05))
        else:
            target_positions = alpha_targets
            if loss_status["stop_active"]:
                target_positions = {}
                cap = min(cap, 0.05)

        total_abs = sum(abs(w) for w in target_positions.values())
        if total_abs > cap and total_abs > 0:
            if sleeves_enabled:
                core_symbols = set(core_mix.keys())
                core_part = {
                    s: float(w) for s, w in target_positions.items() if s in core_symbols
                }
                sat_part = {
                    s: float(w) for s, w in target_positions.items() if s not in core_symbols
                }
                core_abs = sum(abs(w) for w in core_part.values())
                sat_abs = sum(abs(w) for w in sat_part.values())
                if core_abs >= cap and core_abs > 0:
                    c_scale = cap / core_abs
                    target_positions = {s: float(w * c_scale) for s, w in core_part.items()}
                else:
                    sat_budget = max(cap - core_abs, 0.0)
                    if sat_abs > 0:
                        s_scale = sat_budget / sat_abs
                        sat_part = {s: float(w * s_scale) for s, w in sat_part.items()}
                    else:
                        sat_part = {}
                    target_positions = _combine_targets(core_part, sat_part)
            else:
                scale = cap / total_abs
                target_positions = {s: float(w * scale) for s, w in target_positions.items()}

        if sleeves_enabled and core_rebalance_band > 0:
            current_w = _current_weights(bt, bars=bars, equity=bt._total_equity())
            core_symbols = set(core_mix.keys())
            for sym in core_symbols:
                cur = float(current_w.get(sym, 0.0))
                tgt = float(target_positions.get(sym, 0.0))
                if abs(cur) > 1e-6 and abs(tgt - cur) < core_rebalance_band:
                    target_positions[sym] = cur

        equity = bt._total_equity()
        all_symbols = set(target_positions.keys()) | set(bt.positions.keys())
        for symbol in sorted(all_symbols):
            target_position = float(target_positions.get(symbol, 0.0))
            bar = bars.get(symbol)
            if bar is None:
                continue
            current_pos = bt.positions.get(symbol)
            current_qty = current_pos.quantity if current_pos else 0
            price = bar.get("open", bar.get("close", 0))
            target_value = equity * target_position
            target_qty = target_value / price if price > 0 else 0

            qty_diff = target_qty - current_qty

            if price > 0 and abs(qty_diff) > min_rebalance_notional_frac * equity / price:
                if qty_diff > 0:
                    bt.submit_order(
                        symbol, OrderSide.BUY, abs(qty_diff), OrderType.MARKET
                    )
                else:
                    bt.submit_order(
                        symbol, OrderSide.SELL, abs(qty_diff), OrderType.MARKET
                    )

        equity = bt._total_equity()
        state["peak_equity"] = max(state["peak_equity"], equity)
        state["current_drawdown"] = (equity - state["peak_equity"]) / state[
            "peak_equity"
        ]

    metrics, paper_start = paper_trader.run(data, trading_strategy)
    if isinstance(metrics, dict) and "error" not in metrics:
        metrics["sleeves_enabled"] = bool(sleeves_enabled)
        metrics["sleeve_core_budget"] = float(core_budget)
        metrics["sleeve_alpha_budget"] = float(alpha_budget)
        metrics["sleeve_lottery_budget"] = float(lottery_budget)
        alpha_hist = state.get("alpha_scale_history", [])
        if alpha_hist:
            metrics["alpha_scale_last"] = float(alpha_hist[-1])
            metrics["alpha_scale_avg"] = float(np.mean(alpha_hist))
        else:
            metrics["alpha_scale_last"] = 1.0
            metrics["alpha_scale_avg"] = 1.0
        metrics["max_daily_loss_limit"] = float(max_daily_loss)
        metrics["max_weekly_loss_limit"] = float(max_weekly_loss)
        metrics["hard_dd_limit"] = float(hard_drawdown_limit)
        metrics["hard_dd_triggered"] = bool(state.get("hard_dd_triggered", False))
        metrics["hard_dd_trigger_ts"] = state.get("hard_dd_trigger_ts")
        metrics["hard_dd_halt_bars"] = int(state.get("hard_dd_halt_bars", 0))
        metrics["hard_dd_reentries"] = int(state.get("hard_dd_reentries", 0))
        metrics["hard_dd_reentry_ts"] = list(state.get("hard_dd_reentry_ts", []))
        metrics["hard_dd_last_quant_ir"] = state.get("hard_dd_last_quant_ir")
        if quant_returns_control is not None and not quant_returns_control.empty:
            strategy_returns = (
                paper_trader.backtester.get_equity_series()
                .pct_change(fill_method=None)
                .dropna()
            )
            strategy_returns = _normalize_return_index(strategy_returns)
            strat_aligned, quant_aligned = strategy_returns.align(
                quant_returns_control, join="inner"
            )
            if not strat_aligned.empty:
                rel_metrics = compute_metrics_from_returns(
                    strat_aligned, benchmark_returns=quant_aligned
                )
                quant_total = _total_return_from_returns(quant_aligned)
                strat_total = _total_return_from_returns(strat_aligned)
                metrics["excess_total_return_vs_quant"] = float(strat_total - quant_total)
                metrics["quant_information_ratio"] = float(
                    rel_metrics.get("information_ratio", 0.0)
                )
                metrics["quant_beta"] = float(rel_metrics.get("beta", 0.0))

    if verbose and "error" not in metrics:
        print(f"\n{'=' * 60}")
        print("PAPER RESULTS")
        print(f"{'=' * 60}")
        print(f"Paper Start:    {paper_start.date()}")
        print(f"Total Return:   {metrics['total_return'] * 100:>10.2f}%")
        print(f"Annual Return:  {metrics['annual_return'] * 100:>10.2f}%")
        print(f"Volatility:     {metrics['volatility'] * 100:>10.2f}%")
        print(f"Sharpe Ratio:   {metrics['sharpe_ratio']:>10.2f}")
        print(f"Sortino Ratio:  {metrics['sortino_ratio']:>10.2f}")
        print(f"Max Drawdown:   {metrics['max_drawdown'] * 100:>10.2f}%")
        print(f"Calmar Ratio:   {metrics['calmar_ratio']:>10.2f}")
        print(f"Win Rate:       {metrics['win_rate'] * 100:>10.2f}%")
        print(f"Profit Factor:  {metrics['profit_factor']:>10.2f}")
        print(f"Total Trades:   {metrics['n_trades']:>10d}")
        print(f"Final Equity:   ${metrics['final_equity']:>10,.2f}")
        if "excess_total_return_vs_quant" in metrics:
            print(
                "Excess vs QQQ/IWM:"
                f" {metrics['excess_total_return_vs_quant'] * 100:>9.2f}%"
            )
            print(
                "Quant InfoRatio: "
                f"{metrics.get('quant_information_ratio', 0.0):>10.2f}"
            )
            print(f"Quant Beta:      {metrics.get('quant_beta', 0.0):>10.2f}")
        print(f"{'=' * 60}\n")

    return {
        "metrics": metrics,
        "paper_start": paper_start,
        "equity_curve": paper_trader.backtester.equity_curve,
        "trades": paper_trader.backtester.trades,
        "fills": paper_trader.backtester.fills,
    }


def run_paper_ab_comparison(
    symbols: list,
    start_date: datetime,
    end_date: datetime,
    config_a: str,
    config_b: str,
    initial_capital: float = 100000,
    strategy_type: str = "enhanced",
    paper_bars: int = 120,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run paper A/B comparison (Group A baseline vs Group B hardened).
    """
    res_a = run_paper(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        strategy_type=strategy_type,
        paper_bars=paper_bars,
        config_path=config_a,
        verbose=verbose,
    )
    res_b = run_paper(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        strategy_type=strategy_type,
        paper_bars=paper_bars,
        config_path=config_b,
        verbose=verbose,
    )

    m_a = res_a.get("metrics", {}) if isinstance(res_a, dict) else {}
    m_b = res_b.get("metrics", {}) if isinstance(res_b, dict) else {}

    compare_keys = [
        "total_return",
        "max_drawdown",
        "sharpe_ratio",
        "win_rate",
        "excess_total_return_vs_quant",
        "quant_information_ratio",
        "hard_dd_triggered",
        "hard_dd_halt_bars",
    ]

    comparisons: Dict[str, Dict[str, object]] = {}
    for key in compare_keys:
        a_val = m_a.get(key)
        b_val = m_b.get(key)
        delta = None
        if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
            delta = float(b_val) - float(a_val)
        comparisons[key] = {
            "A": a_val,
            "B": b_val,
            "delta_b_minus_a": delta,
        }

    dd_cap = 0.20
    b_dd = abs(float(m_b.get("max_drawdown", 0.0)))
    objective_checks = {
        "b_within_dd_cap_20pct": bool(b_dd <= dd_cap),
        "b_excess_vs_quant_gt_a": bool(
            float(m_b.get("excess_total_return_vs_quant", -np.inf))
            > float(m_a.get("excess_total_return_vs_quant", -np.inf))
        ),
        "b_quant_ir_gt_a": bool(
            float(m_b.get("quant_information_ratio", -np.inf))
            > float(m_a.get("quant_information_ratio", -np.inf))
        ),
        "b_drawdown_not_worse_than_a": bool(
            b_dd <= abs(float(m_a.get("max_drawdown", np.inf)))
        ),
    }
    promote_b = all(objective_checks.values())

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "group_a": {"config_path": str(config_a), "metrics": m_a},
        "group_b": {"config_path": str(config_b), "metrics": m_b},
        "comparisons": comparisons,
        "objective_checks": objective_checks,
        "promotion_decision": {
            "promote_group_b": bool(promote_b),
            "reason": (
                "group_b_passed"
                if promote_b
                else "group_b_failed_objective_or_dd_cap"
            ),
        },
    }

    if output_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output = PROJECT_ROOT / "quantum_alpha" / "artifacts" / f"paper_ab_{stamp}.json"
    else:
        output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    summary["output_path"] = str(output)

    if verbose:
        print("\nPAPER A/B SUMMARY")
        print(f"  Output: {output}")
        print(
            "  Promote Group B: "
            f"{'YES' if summary['promotion_decision']['promote_group_b'] else 'NO'}"
        )

    return summary


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Quantum Alpha V1 - Algorithmic Trading System"
    )
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper", "live"],
        default="backtest",
        help="Operating mode",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=["SPY"], help="Symbols to trade"
    )
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")
    parser.add_argument(
        "--start", type=str, default=None, help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--strategy",
        choices=[
            "momentum",
            "mean_reversion",
            "composite",
            "adaptive",
            "enhanced",
            "sentiment",
            "ml",
            "news_lstm",
            "meta_ensemble",
        ],
        default="enhanced",
        help="Strategy type",
    )
    parser.add_argument(
        "--firm-mode",
        action="store_true",
        help="Enable firm-grade execution safeguards",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch local dashboard (if available)",
    )
    parser.add_argument(
        "--paper-bars",
        type=int,
        default=120,
        help="Number of most recent bars to simulate in paper mode",
    )
    parser.add_argument(
        "--paper-ab",
        action="store_true",
        help="Run paper A/B comparison (requires --paper-config-a and --paper-config-b)",
    )
    parser.add_argument(
        "--paper-config-a",
        type=str,
        default=None,
        help="Config path for paper Group A",
    )
    parser.add_argument(
        "--paper-config-b",
        type=str,
        default=None,
        help="Config path for paper Group B",
    )
    parser.add_argument(
        "--paper-ab-output",
        type=str,
        default=None,
        help="Output path for paper A/B summary JSON",
    )
    parser.add_argument("--validate", action="store_true", help="Run MCPT validation")
    parser.add_argument("--config", type=str, default=None, help="Config file path")

    args = parser.parse_args()

    try:
        settings = load_config(args.config)
    except Exception as e:
        print(f"Config error: {e}")
        return None

    load_plugins()

    config_dir = _resolve_config_dir(args.config)
    log_cfg = settings.get("logging", {}) if settings else {}
    configure_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file", "quantum_alpha.log"),
    )
    thresholds = {}
    risk_cfg = _load_optional_yaml(config_dir / "risk_limits.yaml")
    if risk_cfg and "limits" in risk_cfg:
        limits = risk_cfg["limits"]
        if "max_drawdown" in limits:
            thresholds["max_drawdown"] = limits["max_drawdown"]
        if "min_sharpe" in limits:
            thresholds["min_sharpe"] = limits["min_sharpe"]
        if "min_win_rate" in limits:
            thresholds["min_win_rate"] = limits["min_win_rate"]

    alert_manager = AlertManager()
    for rule in build_default_rules(thresholds):
        alert_manager.add_rule(rule)

    if args.firm_mode and args.mode != "live":
        print(
            "Firm mode requested, but live execution is not enabled. Firm mode disabled."
        )
        args.firm_mode = False

    if args.firm_mode and args.mode == "live":
        print(
            "Firm mode requested, but live execution is not implemented in this phase."
        )
        return None

    if args.dashboard:
        print("Dashboard flag enabled. Local dashboard is not implemented in V1.")

    # Parse dates
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    else:
        end_date = datetime.now()

    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=365 * 2)  # 2 years default

    if args.mode == "backtest":
        results = run_backtest(
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.capital,
            strategy_type=args.strategy,
            validate=args.validate,
            verbose=True,
            config_path=args.config,
        )
        if isinstance(results, dict) and "metrics" in results:
            alert_manager.evaluate(results["metrics"])
        return results

    elif args.mode == "paper":
        if args.paper_ab:
            if not args.paper_config_a or not args.paper_config_b:
                print(
                    "paper-ab requires both --paper-config-a and --paper-config-b"
                )
                return None
            return run_paper_ab_comparison(
                symbols=args.symbols,
                start_date=start_date,
                end_date=end_date,
                config_a=args.paper_config_a,
                config_b=args.paper_config_b,
                initial_capital=args.capital,
                strategy_type=args.strategy,
                paper_bars=args.paper_bars,
                output_path=args.paper_ab_output,
                verbose=True,
            )
        results = run_paper(
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=args.capital,
            strategy_type=args.strategy,
            paper_bars=args.paper_bars,
            config_path=args.config,
            verbose=True,
        )
        if isinstance(results, dict) and "metrics" in results:
            alert_manager.evaluate(results["metrics"])
        return results

    elif args.mode == "live":
        print("Live trading requires additional safety checks")
        print("Coming in Phase 2...")
        return None


if __name__ == "__main__":
    main()
