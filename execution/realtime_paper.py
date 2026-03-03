"""
Real-time paper trading runner.

Uses live market data polling with fake-money execution.
Supports:
- Enhanced composite cross-asset strategy
- News-LSTM + Gemini LLM gate for intraday signal adjudication
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from zoneinfo import ZoneInfo

from quantum_alpha.data.collectors.market_data import DataCollector
from quantum_alpha.data.collectors.news_collector import NewsCollector
from quantum_alpha.data.preprocessing.cleaners import DataCleaner
from quantum_alpha.data.preprocessing.imputers import MissingValueImputer
from quantum_alpha.execution.daily_anchor_cache import (
    anchor_cache_freshness_minutes,
    refresh_anchor_cache_if_needed,
)
from quantum_alpha.execution.openbb_api_manager import OpenBBAPIManager
from quantum_alpha.features.technical.indicators import TechnicalFeatureGenerator
from quantum_alpha.portfolio import PortfolioAllocatorEngine
from quantum_alpha.strategy.meta_dual_blend_strategy import MetaDualBlendStrategy
from quantum_alpha.strategy.news_lstm_strategy import NewsLSTMStrategy
from quantum_alpha.strategy.signals import EnhancedCompositeStrategy


@dataclass
class SessionConfig:
    symbols: List[str]
    full_universe_requested: bool
    interval: str
    duration_minutes: int
    poll_seconds: int
    lookback_days: int
    capital: float
    max_position_size: float
    max_portfolio_leverage: float
    signal_threshold: float
    min_long_signal: float
    long_only: bool
    strategy_type: str
    checkpoint_name: Optional[str]
    llm_enabled: bool
    llm_mode: Optional[str]
    llm_models: Optional[List[str]]
    llm_min_alignment: float
    llm_fail_mode: str
    llm_scope: Optional[str]
    llm_max_calls: int
    llm_decision_interval_cycles: int
    llm_env_path: Optional[str]
    news_poll_seconds: int
    news_max_articles: int
    news_symbol_limit: int
    liquid_subset_size: int
    liquid_adv_window: int
    liquid_min_history: int
    liquid_for_full_only: bool
    universe_refresh_cycles: int
    commission_rate: float
    slippage_bps: float
    min_commission: float
    min_trade_notional: float
    output_dir: Path
    config_path: Optional[str] = None
    openbb_enabled: bool = False
    openbb_manage_api_process: bool = True
    openbb_api_base_url: str = "http://127.0.0.1:6900"
    openbb_api_start_cmd: Optional[str] = None
    openbb_api_startup_timeout_s: float = 20.0
    portfolio_allocator_cfg: Optional[Dict] = None
    meta_base_checkpoint_dir: Optional[str] = None
    meta_mc_checkpoint_dir: Optional[str] = None
    meta_blend_weights: Optional[List[float]] = None
    anchor_cache_path: Optional[str] = None
    ab_group: str = "auto"
    latency_budget_ms: int = 20000
    hybrid_gate_top_n: int = 40
    anchor_universe_size: int = 200


class PaperAccount:
    def __init__(
        self,
        initial_capital: float,
        commission_rate: float = 0.001,
        slippage_bps: float = 5.0,
        min_commission: float = 1.0,
    ) -> None:
        self.cash = float(initial_capital)
        self.positions: Dict[str, float] = {}
        self.commission_rate = float(commission_rate)
        self.slippage = float(slippage_bps) / 10000.0
        self.min_commission = float(min_commission)
        self.trades: List[Dict] = []

    def equity(self, prices: Dict[str, float]) -> float:
        total = self.cash
        for symbol, qty in self.positions.items():
            price = prices.get(symbol)
            if price is not None and price > 0:
                total += qty * price
        return float(total)

    def rebalance(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        timestamp: pd.Timestamp,
        min_notional: float = 25.0,
    ) -> None:
        symbols = set(self.positions.keys()) | set(target_weights.keys())
        portfolio_equity = self.equity(prices)

        for symbol in sorted(symbols):
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue
            target_w = float(target_weights.get(symbol, 0.0))
            current_qty = float(self.positions.get(symbol, 0.0))
            current_value = current_qty * price
            target_value = portfolio_equity * target_w
            delta_value = target_value - current_value

            if abs(delta_value) < min_notional:
                continue

            side = "BUY" if delta_value > 0 else "SELL"
            exec_price = price * (1 + self.slippage) if side == "BUY" else price * (
                1 - self.slippage
            )
            qty = abs(delta_value) / exec_price
            notional = qty * exec_price
            commission = max(self.min_commission, notional * self.commission_rate)

            if side == "BUY":
                max_affordable = max(self.cash - commission, 0.0)
                if max_affordable <= 0:
                    continue
                qty = min(qty, max_affordable / exec_price)
                notional = qty * exec_price
                commission = max(self.min_commission, notional * self.commission_rate)
                total_cost = notional + commission
                if qty <= 0 or total_cost > self.cash:
                    continue
                self.cash -= total_cost
                self.positions[symbol] = current_qty + qty
            else:
                qty = min(qty, max(current_qty, 0.0))
                if qty <= 0:
                    continue
                notional = qty * exec_price
                commission = max(self.min_commission, notional * self.commission_rate)
                self.cash += max(notional - commission, 0.0)
                new_qty = current_qty - qty
                if abs(new_qty) < 1e-8:
                    self.positions.pop(symbol, None)
                else:
                    self.positions[symbol] = new_qty

            self.trades.append(
                {
                    "timestamp": str(timestamp),
                    "symbol": symbol,
                    "side": side,
                    "qty": float(qty),
                    "exec_price": float(exec_price),
                    "notional": float(notional),
                    "commission": float(commission),
                    "cash_after": float(self.cash),
                }
            )


def _load_settings(config_path: Optional[str]) -> Dict:
    if config_path:
        settings_path = Path(config_path)
    else:
        settings_path = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
    with open(settings_path, "r") as f:
        return yaml.safe_load(f)


def _resolve_symbols(input_symbols: List[str], settings: Dict) -> tuple[List[str], bool]:
    data_cfg = settings.get("data", {})
    universe_limit = int(data_cfg.get("universe_limit", 0))

    def _limit(symbols: List[str]) -> List[str]:
        if universe_limit > 0:
            return symbols[:universe_limit]
        return symbols

    full_requested = False
    if len(input_symbols) == 1:
        token = str(input_symbols[0]).upper()
        if token == "AUTO":
            default = [str(s).upper() for s in data_cfg.get("default_universe", [])]
            return _limit(default), False
        if token in {"FULL", "ALL", "UNIVERSE"}:
            full_requested = True
            try:
                from quantum_alpha import universe as _u

                return _limit([str(s).upper() for s in _u.get_stocks_only()]), full_requested
            except Exception:
                dc = DataCollector()
                return _limit([str(s).upper() for s in dc.get_sp500_symbols()]), full_requested
        if token in {"SP500", "S&P500", "SPX"}:
            dc = DataCollector()
            return _limit([str(s).upper() for s in dc.get_sp500_symbols()]), False

    return [str(s).upper() for s in input_symbols], full_requested


def _minutes_until_close(now_utc: datetime) -> int:
    et = now_utc.astimezone(ZoneInfo("America/New_York"))
    close_et = et.replace(hour=16, minute=0, second=0, microsecond=0)
    if et.weekday() >= 5 or et >= close_et:
        return 1
    remaining = (close_et - et).total_seconds() / 60.0
    return max(1, int(remaining))


def _interval_max_lookback_days(interval: str) -> int:
    iv = str(interval).strip().lower()
    if iv.endswith("m"):
        try:
            minutes = int(iv[:-1])
        except ValueError:
            minutes = 5
        if minutes <= 1:
            return 8
        if minutes <= 5:
            return 59
        if minutes <= 15:
            return 59
        return 180
    if iv.endswith("h"):
        return 730
    return 3650


def _parse_csv_str(value: str | None) -> list[str] | None:
    if not value:
        return None
    out = [x.strip() for x in value.split(",") if x.strip()]
    return out or None


def _parse_csv_float(value: str | None) -> list[float] | None:
    if not value:
        return None
    out: list[float] = []
    for raw in value.split(","):
        val = raw.strip()
        if not val:
            continue
        try:
            out.append(float(val))
        except Exception:
            continue
    return out or None


def _llm_cycle_active(cycle: int, cfg: SessionConfig) -> bool:
    if not cfg.llm_enabled:
        return False
    interval = max(1, int(cfg.llm_decision_interval_cycles))
    return ((int(cycle) - 1) % interval) == 0


def _resolve_ab_group(cfg: SessionConfig, now_utc: datetime) -> str:
    token = str(cfg.ab_group).strip().upper()
    if token in {"A", "B"}:
        return token
    # Auto alternation by date parity (UTC date)
    day_num = int(now_utc.strftime("%Y%m%d"))
    return "A" if (day_num % 2 == 0) else "B"


def _start_live_dashboard(
    cfg: SessionConfig,
    port: int,
    refresh_seconds: float,
    open_browser: bool,
) -> Dict[str, object]:
    if importlib.util.find_spec("streamlit") is None:
        return {"started": False, "reason": "streamlit_not_installed"}

    project_root = Path(__file__).resolve().parents[2]
    script_path = Path(__file__).resolve().parent / "live_graph_dashboard.py"
    if not script_path.exists():
        return {"started": False, "reason": f"missing_script:{script_path}"}

    env = os.environ.copy()
    prior_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = (
        str(project_root)
        if not prior_pythonpath
        else f"{str(project_root)}:{prior_pythonpath}"
    )
    url = f"http://127.0.0.1:{int(port)}"
    log_path = cfg.output_dir / "dashboard.log"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script_path),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(int(port)),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--",
        "--session-dir",
        str(cfg.output_dir),
        "--refresh-seconds",
        str(float(refresh_seconds)),
    ]

    with log_path.open("a", encoding="utf-8") as fh:
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=fh,
            stderr=fh,
        )

    meta = {
        "started": True,
        "url": url,
        "pid": int(proc.pid),
        "log_path": str(log_path),
        "refresh_seconds": float(refresh_seconds),
        "session_dir": str(cfg.output_dir),
    }
    (cfg.output_dir / "dashboard.json").write_text(json.dumps(meta, indent=2))

    if open_browser:
        try:
            webbrowser.open(url, new=2)
        except Exception:
            pass

    return meta


def _build_config(args: argparse.Namespace) -> SessionConfig:
    settings = _load_settings(args.config)
    data_cfg = settings.get("data", {})
    risk_cfg = settings.get("risk", {})
    strategy_cfg = settings.get("strategy", {})
    backtest_cfg = settings.get("backtest", {})
    openbb_cfg = data_cfg.get("openbb", {}) if isinstance(data_cfg, dict) else {}
    openbb_api_cfg = openbb_cfg.get("api", {}) if isinstance(openbb_cfg, dict) else {}
    portfolio_allocator_cfg = settings.get("portfolio_allocator", {})

    symbols, full_requested = _resolve_symbols(args.symbols, settings)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"realtime_paper_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    lookback_days = int(args.lookback_days)
    lookback_days = min(lookback_days, _interval_max_lookback_days(args.interval))

    duration_minutes = int(args.duration_minutes)
    if bool(args.until_close):
        duration_minutes = _minutes_until_close(datetime.now(timezone.utc))

    llm_models = _parse_csv_str(args.llm_models)
    meta_blend_weights = _parse_csv_float(args.meta_blend_weights)

    return SessionConfig(
        symbols=symbols,
        full_universe_requested=full_requested,
        interval=args.interval,
        duration_minutes=duration_minutes,
        poll_seconds=int(args.poll_seconds),
        lookback_days=lookback_days,
        capital=float(args.capital if args.capital is not None else args.default_capital),
        max_position_size=float(risk_cfg.get("max_position_size", 0.25)),
        max_portfolio_leverage=float(risk_cfg.get("max_portfolio_leverage", 1.0)),
        signal_threshold=float(strategy_cfg.get("signal_threshold", 0.3)),
        min_long_signal=float(strategy_cfg.get("min_long_signal", 0.0)),
        long_only=bool(strategy_cfg.get("long_only", True)),
        strategy_type=str(args.strategy_type).strip().lower(),
        checkpoint_name=args.checkpoint_name,
        llm_enabled=bool(args.llm_enable),
        llm_mode=args.llm_mode,
        llm_models=llm_models,
        llm_min_alignment=float(args.llm_min_alignment),
        llm_fail_mode=str(args.llm_fail_mode).lower(),
        llm_scope=args.llm_scope,
        llm_max_calls=int(args.llm_max_calls),
        llm_decision_interval_cycles=max(1, int(args.llm_decision_interval_cycles)),
        llm_env_path=args.llm_env_path,
        news_poll_seconds=max(30, int(args.news_poll_seconds)),
        news_max_articles=max(1, int(args.news_max_articles)),
        news_symbol_limit=max(1, int(args.news_symbol_limit)),
        liquid_subset_size=int(
            args.liquid_subset_size
            if args.liquid_subset_size is not None
            else strategy_cfg.get("liquid_subset_size", 0)
        ),
        liquid_adv_window=int(
            args.liquid_adv_window
            if args.liquid_adv_window is not None
            else strategy_cfg.get("liquid_adv_window", 20)
        ),
        liquid_min_history=int(
            args.liquid_min_history
            if args.liquid_min_history is not None
            else strategy_cfg.get("liquid_min_history", 120)
        ),
        liquid_for_full_only=bool(
            args.liquid_for_full_only
            if args.liquid_for_full_only is not None
            else strategy_cfg.get("liquid_for_full_only", True)
        ),
        universe_refresh_cycles=max(1, int(args.universe_refresh_cycles)),
        commission_rate=float(
            args.commission_rate
            if args.commission_rate is not None
            else backtest_cfg.get("commission_rate", 0.001)
        ),
        slippage_bps=float(
            args.slippage_bps
            if args.slippage_bps is not None
            else backtest_cfg.get("slippage_bps", 5)
        ),
        min_commission=float(
            args.min_commission
            if args.min_commission is not None
            else backtest_cfg.get("min_commission", 1.0)
        ),
        min_trade_notional=float(args.min_trade_notional),
        output_dir=out_dir,
        config_path=args.config,
        openbb_enabled=bool(openbb_cfg.get("enabled", False)),
        openbb_manage_api_process=bool(openbb_api_cfg.get("manage_process", True)),
        openbb_api_base_url=str(openbb_api_cfg.get("base_url", "http://127.0.0.1:6900")),
        openbb_api_start_cmd=os.getenv("OPENBB_API_START_CMD"),
        openbb_api_startup_timeout_s=float(openbb_api_cfg.get("startup_timeout_s", 20.0)),
        portfolio_allocator_cfg=portfolio_allocator_cfg if isinstance(portfolio_allocator_cfg, dict) else {},
        meta_base_checkpoint_dir=args.meta_base_checkpoint_dir,
        meta_mc_checkpoint_dir=args.meta_mc_checkpoint_dir,
        meta_blend_weights=meta_blend_weights,
        anchor_cache_path=args.anchor_cache_path,
        ab_group=str(args.ab_group).upper(),
        latency_budget_ms=max(1000, int(args.latency_budget_ms)),
        hybrid_gate_top_n=max(1, int(args.hybrid_gate_top_n)),
        anchor_universe_size=max(1, int(args.anchor_universe_size)),
    )


def _collect_featured_data(
    collector: DataCollector,
    cleaner: DataCleaner,
    imputer: MissingValueImputer,
    feature_gen: TechnicalFeatureGenerator,
    symbols: List[str],
    end_time: datetime,
    lookback_days: int,
    interval: str,
    use_cache: bool,
    progress_callback: Optional[Callable[[int, int, int, str], None]] = None,
) -> Dict[str, pd.DataFrame]:
    start_time = end_time - timedelta(days=lookback_days)
    featured: Dict[str, pd.DataFrame] = {}
    total = len(symbols)
    for idx, symbol in enumerate(symbols, start=1):
        try:
            df = collector.fetch_ohlcv(
                symbol=symbol,
                start=start_time,
                end=end_time,
                interval=interval,
                use_cache=use_cache,
            )
            df = cleaner.clean(df)
            df = imputer.impute(df)
            df = feature_gen.generate(df)
            if not df.empty:
                featured[symbol] = df
        except Exception:
            pass
        finally:
            if progress_callback is not None and (idx == 1 or idx % 25 == 0 or idx == total):
                progress_callback(idx, total, len(featured), symbol)
    return featured


def _select_liquid_subset(
    frames: Dict[str, pd.DataFrame],
    subset_size: int,
    adv_window: int = 20,
    min_history: int = 120,
) -> List[str]:
    if subset_size <= 0 or not frames:
        return list(frames.keys())

    scores: list[tuple[str, float]] = []
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
    return [sym for sym, _ in scores[:subset_size]]


def _latest_prices(featured: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    for symbol, df in featured.items():
        if not df.empty and "close" in df.columns:
            prices[symbol] = float(df["close"].iloc[-1])
    return prices


def _top_dollar_volume_symbols(featured: Dict[str, pd.DataFrame], n: int) -> List[str]:
    scores: list[tuple[str, float]] = []
    for symbol, df in featured.items():
        if df is None or df.empty:
            continue
        close = pd.to_numeric(df.get("close"), errors="coerce")
        volume = pd.to_numeric(df.get("volume"), errors="coerce")
        if close is None or volume is None:
            continue
        dv = (close * volume).replace([np.inf, -np.inf], np.nan).dropna()
        if dv.empty:
            continue
        scores.append((symbol, float(dv.tail(20).mean())))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in scores[: max(1, n)]]


def _normalize_target_weights(
    raw_scores: Dict[str, float],
    universe_symbols: List[str],
    cfg: SessionConfig,
) -> Dict[str, float]:
    if not raw_scores:
        return {s: 0.0 for s in universe_symbols}

    total_score = sum(abs(v) for v in raw_scores.values())
    if total_score <= 0:
        return {s: 0.0 for s in universe_symbols}

    target_weights: Dict[str, float] = {}
    for symbol, score in raw_scores.items():
        w = abs(score) / total_score if cfg.long_only else score / total_score
        w = min(w, cfg.max_position_size) if cfg.long_only else np.clip(
            w, -cfg.max_position_size, cfg.max_position_size
        )
        target_weights[symbol] = float(w)

    total_abs = sum(abs(v) for v in target_weights.values())
    if total_abs > cfg.max_portfolio_leverage and total_abs > 0:
        scale = cfg.max_portfolio_leverage / total_abs
        target_weights = {s: float(w * scale) for s, w in target_weights.items()}

    for symbol in universe_symbols:
        target_weights.setdefault(symbol, 0.0)
    return target_weights


def _portfolio_allocate(
    allocator: Optional[PortfolioAllocatorEngine],
    raw_scores: Dict[str, float],
    featured: Dict[str, pd.DataFrame],
    cfg: SessionConfig,
    timestamp: Optional[datetime] = None,
) -> tuple[Dict[str, float], Dict[str, object]]:
    if allocator is None:
        return _normalize_target_weights(raw_scores, list(featured.keys()), cfg), {}

    constraint_overrides: Dict[str, object] = {
        "long_short_enabled": not cfg.long_only,
        "max_position_abs": float(cfg.max_position_size),
        "gross_max": float(cfg.max_portfolio_leverage),
    }
    if cfg.long_only:
        constraint_overrides["net_min"] = 0.0
        constraint_overrides["net_max"] = float(cfg.max_portfolio_leverage)

    try:
        output = allocator.allocate(
            signal_scores=raw_scores,
            featured=featured,
            timestamp=timestamp,
            constraint_overrides=constraint_overrides,
        )
        weights = {str(k): float(v) for k, v in output.weights.items()}
        for sym in featured.keys():
            weights.setdefault(sym, 0.0)
        diagnostics = {
            "chosen_stack": output.chosen_stack,
            "risk_snapshot": {
                "var": float(output.risk_snapshot.var),
                "cvar": float(output.risk_snapshot.cvar),
                "drawdown": float(output.risk_snapshot.drawdown),
                "ulcer_index": float(output.risk_snapshot.ulcer_index),
                "vol_estimates": output.risk_snapshot.vol_estimates,
                "net_exposure": float(output.risk_snapshot.net_exposure),
                "gross_exposure": float(output.risk_snapshot.gross_exposure),
            },
            "diagnostics": output.diagnostics,
        }
        return weights, diagnostics
    except Exception as exc:
        # Keep trading continuity if allocator stack fails.
        fallback = _normalize_target_weights(raw_scores, list(featured.keys()), cfg)
        return fallback, {"allocator_error": str(exc)}


def _compute_target_weights_enhanced(
    strategy: EnhancedCompositeStrategy,
    featured: Dict[str, pd.DataFrame],
    cfg: SessionConfig,
    allocator: Optional[PortfolioAllocatorEngine] = None,
    timestamp: Optional[datetime] = None,
) -> tuple[Dict[str, float], List[Dict], Dict[str, object]]:
    strategy.fit_cross_asset(featured)

    raw_scores: Dict[str, float] = {}
    decisions: List[Dict] = []
    for symbol, df in featured.items():
        sig = strategy.generate_signals(df, symbol=symbol)
        if sig.empty:
            continue
        last = sig.iloc[-1]
        signal = float(last.get("signal", 0.0))
        conf = float(last.get("signal_confidence", 0.0))

        if cfg.long_only and signal < 0:
            signal = 0.0

        if signal > 0 and cfg.min_long_signal > 0:
            signal = max(signal, cfg.min_long_signal)

        if abs(signal) < cfg.signal_threshold:
            continue

        score = max(signal * max(conf, 0.1), 0.0) if cfg.long_only else signal * conf
        if cfg.long_only:
            score = max(score, 0.0)
        raw_scores[symbol] = float(score)
        decisions.append(
            {
                "symbol": symbol,
                "signal": float(signal),
                "confidence": float(conf),
                "mode": "enhanced",
            }
        )

    decisions.sort(key=lambda x: abs(float(x.get("signal", 0.0))), reverse=True)
    weights, alloc_diag = _portfolio_allocate(
        allocator=allocator,
        raw_scores=raw_scores,
        featured=featured,
        cfg=cfg,
        timestamp=timestamp,
    )
    return weights, decisions[:20], alloc_diag


def _compute_target_weights_news_lstm(
    strategy: NewsLSTMStrategy,
    featured: Dict[str, pd.DataFrame],
    cfg: SessionConfig,
    news_cache: Dict[str, List[str]],
    llm_active: bool,
    allocator: Optional[PortfolioAllocatorEngine] = None,
    timestamp: Optional[datetime] = None,
) -> tuple[Dict[str, float], List[Dict], Dict[str, object]]:
    raw_scores: Dict[str, float] = {}
    decisions: List[Dict] = []

    for symbol, df in featured.items():
        headlines = news_cache.get(symbol, [])
        live = strategy.generate_signals_live(df, symbol=symbol, headlines=headlines)

        action = float(live.get("action", 0.0))
        signal = float(live.get("signal", action))
        confidence = float(live.get("confidence", 0.0))
        llm_alignment = float(live.get("llm_alignment_score", 0.0))
        llm_gate_pass = bool(live.get("llm_gate_pass", False))

        if cfg.long_only and action < 0:
            action = 0.0
            signal = 0.0

        if signal > 0 and cfg.min_long_signal > 0:
            signal = max(signal, cfg.min_long_signal)

        decisions.append(
            {
                "symbol": symbol,
                "action": int(np.sign(action)) if action != 0 else 0,
                "signal": float(signal),
                "confidence": float(confidence),
                "llm_active_cycle": bool(llm_active),
                "llm_decision": str(live.get("llm_decision", "HOLD")),
                "llm_alignment_score": float(llm_alignment),
                "llm_gate_pass": bool(llm_gate_pass),
                "news_count": int(len(headlines)),
            }
        )

        trade_allowed = abs(signal) >= cfg.signal_threshold and action != 0.0
        if llm_active and not llm_gate_pass:
            trade_allowed = False

        if not trade_allowed:
            continue

        if cfg.long_only:
            score = max(signal, 0.0) * max(confidence, 0.05)
            if llm_active:
                score *= max(llm_alignment, 0.1)
        else:
            score = signal * max(confidence, 0.05)
        if cfg.long_only:
            score = max(score, 0.0)

        if abs(score) > 0:
            raw_scores[symbol] = float(score)

    decisions.sort(key=lambda x: abs(float(x.get("signal", 0.0))), reverse=True)
    weights, alloc_diag = _portfolio_allocate(
        allocator=allocator,
        raw_scores=raw_scores,
        featured=featured,
        cfg=cfg,
        timestamp=timestamp,
    )
    return weights, decisions[:20], alloc_diag


def _compute_target_weights_meta_blend(
    featured: Dict[str, pd.DataFrame],
    cfg: SessionConfig,
    anchor_predictions: Dict[str, Dict[str, object]],
    llm_active: bool,
    news_cache: Dict[str, List[str]],
    gate_strategy: Optional[NewsLSTMStrategy] = None,
    allocator: Optional[PortfolioAllocatorEngine] = None,
    timestamp: Optional[datetime] = None,
    top_n_for_gate: int = 40,
) -> tuple[Dict[str, float], List[Dict], Dict[str, object], Dict[str, object]]:
    raw_scores: Dict[str, float] = {}
    decisions: List[Dict] = []

    eligible: list[tuple[str, Dict[str, object]]] = []
    for symbol, pred in anchor_predictions.items():
        if symbol not in featured:
            continue
        eligible.append((symbol, pred))

    # Rank by confidence for hybrid gating scope.
    eligible.sort(
        key=lambda kv: abs(float(kv[1].get("confidence", 0.0))),
        reverse=True,
    )
    gated_set = {sym for sym, _ in eligible[: max(1, int(top_n_for_gate))]}

    miss_base_vals: list[float] = []
    miss_mc_vals: list[float] = []
    used_blended = 0
    used_base = 0
    used_mc = 0

    for symbol, pred in eligible:
        up_prob = float(pred.get("up_probability_blend", pred.get("up_probability", 0.5)))
        confidence = float(pred.get("confidence", abs(up_prob - 0.5) * 2.0))
        signal = float((up_prob - 0.5) * 2.0)

        miss_base = float(pred.get("missing_feature_ratio_base", 1.0))
        miss_mc = float(pred.get("missing_feature_ratio_mc", 1.0))
        model_used = str(pred.get("model_used", "unknown"))
        miss_base_vals.append(miss_base)
        miss_mc_vals.append(miss_mc)
        if model_used == "blended":
            used_blended += 1
        elif model_used == "base_only":
            used_base += 1
        elif model_used == "mc_only":
            used_mc += 1

        llm_gate_pass = True
        llm_decision = "SKIP"
        if symbol in gated_set and gate_strategy is not None and llm_active:
            headlines = news_cache.get(symbol, [])
            try:
                live = gate_strategy.generate_signals_live(df=featured[symbol], symbol=symbol, headlines=headlines)
                llm_gate_pass = bool(live.get("llm_gate_pass", False))
                llm_decision = str(live.get("llm_decision", "HOLD"))
            except Exception:
                llm_gate_pass = True
                llm_decision = "ERROR_FALLBACK_PASS"

        if cfg.long_only and signal < 0:
            signal = 0.0
        if signal > 0 and cfg.min_long_signal > 0:
            signal = max(signal, cfg.min_long_signal)

        trade_allowed = abs(signal) >= cfg.signal_threshold
        if symbol in gated_set and gate_strategy is not None and llm_active and not llm_gate_pass:
            trade_allowed = False

        decisions.append(
            {
                "symbol": symbol,
                "signal": float(signal),
                "confidence": float(confidence),
                "up_probability_blend": float(up_prob),
                "up_probability_base": pred.get("up_probability_base"),
                "up_probability_mc": pred.get("up_probability_mc"),
                "missing_feature_ratio_base": float(miss_base),
                "missing_feature_ratio_mc": float(miss_mc),
                "model_used": model_used,
                "llm_active_cycle": bool(llm_active and symbol in gated_set),
                "llm_decision": llm_decision,
                "llm_gate_pass": bool(llm_gate_pass),
                "mode": "meta_blend_hybrid",
            }
        )

        if not trade_allowed:
            continue
        if cfg.long_only:
            score = max(signal, 0.0) * max(confidence, 0.05)
        else:
            score = signal * max(confidence, 0.05)
        if abs(score) > 0:
            raw_scores[symbol] = float(score)

    decisions.sort(key=lambda x: abs(float(x.get("signal", 0.0))), reverse=True)
    weights, alloc_diag = _portfolio_allocate(
        allocator=allocator,
        raw_scores=raw_scores,
        featured=featured,
        cfg=cfg,
        timestamp=timestamp,
    )

    health = {
        "feature_missing_ratio_base": float(np.mean(miss_base_vals)) if miss_base_vals else 1.0,
        "feature_missing_ratio_mc": float(np.mean(miss_mc_vals)) if miss_mc_vals else 1.0,
        "model_usage": {
            "blended": int(used_blended),
            "base_only": int(used_base),
            "mc_only": int(used_mc),
        },
        "n_predictions": int(len(eligible)),
    }
    return weights, decisions[:20], alloc_diag, health


def _save_outputs(
    cfg: SessionConfig,
    equity_curve: List[Dict],
    trades: List[Dict],
    summary: Dict,
) -> None:
    out_dir = cfg.output_dir
    equity_path = out_dir / "equity_curve.csv"
    trades_path = out_dir / "trades.csv"
    summary_path = out_dir / "summary.json"
    plot_path = out_dir / "equity_curve.png"

    eq_df = pd.DataFrame(equity_curve)
    tr_df = pd.DataFrame(trades)
    eq_df.to_csv(equity_path, index=False)
    tr_df.to_csv(trades_path, index=False)

    if not eq_df.empty and "timestamp" in eq_df.columns and "equity" in eq_df.columns:
        eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"], errors="coerce")
        eq_df = eq_df.dropna(subset=["timestamp"])
        if not eq_df.empty:
            plt.figure(figsize=(12, 6))
            plt.plot(eq_df["timestamp"], eq_df["equity"], color="#0f766e", linewidth=1.6)
            plt.title("Real-Time Paper Trading Equity Curve")
            plt.xlabel("Time")
            plt.ylabel("Equity ($)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            summary["equity_plot"] = str(plot_path)

    summary_path.write_text(json.dumps(summary, indent=2))


def _write_live_status(cfg: SessionConfig, status: Dict) -> None:
    status_path = cfg.output_dir / "live_status.json"
    status_path.write_text(json.dumps(status, indent=2))


def _pnl_fields(
    equity: Optional[float],
    starting_capital: float,
    previous_equity: Optional[float] = None,
) -> Dict[str, Optional[float] | str]:
    if equity is None:
        return {
            "starting_capital": float(starting_capital),
            "profit_dollars": None,
            "return_pct": None,
            "pnl_direction": "unknown",
            "equity_change_dollars": None,
            "equity_change_pct": None,
            "equity_change_direction": "unknown",
        }

    profit = float(equity) - float(starting_capital)
    ret = (profit / float(starting_capital) * 100.0) if starting_capital else 0.0
    direction = "up" if profit > 0 else "down" if profit < 0 else "flat"

    change = None
    change_pct = None
    change_direction = "unknown"
    if previous_equity is not None:
        change = float(equity) - float(previous_equity)
        denom = abs(float(previous_equity))
        change_pct = (change / denom * 100.0) if denom > 1e-12 else 0.0
        change_direction = "up" if change > 0 else "down" if change < 0 else "flat"

    return {
        "starting_capital": float(starting_capital),
        "profit_dollars": float(profit),
        "return_pct": float(ret),
        "pnl_direction": direction,
        "equity_change_dollars": None if change is None else float(change),
        "equity_change_pct": None if change_pct is None else float(change_pct),
        "equity_change_direction": change_direction,
    }


def _positions_snapshot(account: PaperAccount, prices: Dict[str, float]) -> List[Dict]:
    out: List[Dict] = []
    for symbol, qty in account.positions.items():
        price = prices.get(symbol)
        if price is None:
            continue
        value = float(qty * price)
        out.append(
            {
                "symbol": symbol,
                "qty": float(qty),
                "price": float(price),
                "value": value,
            }
        )
    out.sort(key=lambda x: float(x.get("value", 0.0)), reverse=True)
    return out[:20]


def _append_new_trades(cfg: SessionConfig, trades: List[Dict], from_idx: int) -> int:
    if from_idx >= len(trades):
        return from_idx

    trade_path = cfg.output_dir / "live_trades.jsonl"
    with trade_path.open("a", encoding="utf-8") as f:
        for trade in trades[from_idx:]:
            f.write(json.dumps(trade) + "\n")
            print(
                f"TRADE {trade['timestamp']} {trade['side']} {trade['symbol']} "
                f"qty={trade['qty']:.4f} px={trade['exec_price']:.4f} "
                f"notional=${trade['notional']:.2f} cash=${trade['cash_after']:.2f}",
                flush=True,
            )
    return len(trades)


def run_realtime_paper(cfg: SessionConfig) -> Dict:
    collector = DataCollector(
        runtime_mode="realtime",
        config_path=cfg.config_path,
        enable_openbb=cfg.openbb_enabled,
    )
    cleaner = DataCleaner()
    imputer = MissingValueImputer()
    feature_gen = TechnicalFeatureGenerator()
    news_collector = NewsCollector()
    allocator = PortfolioAllocatorEngine.from_config(cfg.portfolio_allocator_cfg or {})

    openbb_manager = OpenBBAPIManager(
        enabled=bool(cfg.openbb_enabled and cfg.openbb_manage_api_process),
        base_url=str(cfg.openbb_api_base_url),
        startup_timeout_s=float(cfg.openbb_api_startup_timeout_s),
        start_cmd=cfg.openbb_api_start_cmd,
    )
    if cfg.openbb_enabled and cfg.openbb_manage_api_process:
        openbb_boot = openbb_manager.start()
    else:
        openbb_boot = {
            "enabled": bool(cfg.openbb_enabled),
            "started": False,
            "healthy": bool(openbb_manager.health()) if cfg.openbb_enabled else False,
            "reason": "manage_process_disabled",
        }

    strategy: object
    hybrid_gate_strategy: Optional[NewsLSTMStrategy] = None
    meta_blend_strategy: Optional[MetaDualBlendStrategy] = None
    meta_blend_init_error: Optional[str] = None
    if cfg.strategy_type == "news_lstm":
        strategy = NewsLSTMStrategy(
            checkpoint_name=cfg.checkpoint_name,
            signal_threshold=cfg.signal_threshold,
            llm_enabled=cfg.llm_enabled,
            llm_mode=cfg.llm_mode,
            llm_models=cfg.llm_models,
            llm_min_alignment=cfg.llm_min_alignment,
            llm_fail_mode=cfg.llm_fail_mode,
            llm_scope=cfg.llm_scope,
            llm_max_calls=cfg.llm_max_calls,
            llm_env_path=cfg.llm_env_path,
        )
    elif cfg.strategy_type == "meta_blend_hybrid":
        try:
            blend_weights = (
                tuple(cfg.meta_blend_weights[:2])
                if cfg.meta_blend_weights and len(cfg.meta_blend_weights) >= 2
                else (0.35, 0.65)
            )
            meta_blend_strategy = MetaDualBlendStrategy(
                base_checkpoint_dir=cfg.meta_base_checkpoint_dir,
                mc_checkpoint_dir=cfg.meta_mc_checkpoint_dir,
                blend_weights=blend_weights,
            )
            strategy = meta_blend_strategy
        except Exception as exc:
            meta_blend_init_error = str(exc)
            strategy = EnhancedCompositeStrategy()
        # Optional gate model for top-N candidate adjudication.
        if cfg.llm_enabled:
            try:
                hybrid_gate_strategy = NewsLSTMStrategy(
                    checkpoint_name=cfg.checkpoint_name,
                    signal_threshold=cfg.signal_threshold,
                    llm_enabled=cfg.llm_enabled,
                    llm_mode=cfg.llm_mode,
                    llm_models=cfg.llm_models,
                    llm_min_alignment=cfg.llm_min_alignment,
                    llm_fail_mode=cfg.llm_fail_mode,
                    llm_scope=cfg.llm_scope,
                    llm_max_calls=cfg.llm_max_calls,
                    llm_env_path=cfg.llm_env_path,
                )
            except Exception:
                hybrid_gate_strategy = None
    else:
        strategy = EnhancedCompositeStrategy()

    enhanced_fallback_strategy = (
        strategy if isinstance(strategy, EnhancedCompositeStrategy) else EnhancedCompositeStrategy()
    )

    account = PaperAccount(
        initial_capital=cfg.capital,
        commission_rate=cfg.commission_rate,
        slippage_bps=cfg.slippage_bps,
        min_commission=cfg.min_commission,
    )

    session_start = datetime.now(timezone.utc)
    session_end = session_start + timedelta(minutes=cfg.duration_minutes)
    last_bar_time: Optional[pd.Timestamp] = None
    previous_equity: Optional[float] = float(cfg.capital)
    equity_curve: List[Dict] = []
    cycle = 0
    trade_write_idx = 0
    last_prices: Dict[str, float] = {}
    news_cache: Dict[str, List[str]] = {}
    news_last_refresh: Optional[datetime] = None
    latest_decisions: List[Dict] = []
    liquid_symbols_cache: Optional[List[str]] = None
    llm_active_this_cycle = False
    allocator_diag: Dict[str, object] = {}
    ab_group_active = _resolve_ab_group(cfg, session_start)
    decision_engine = cfg.strategy_type
    anchor_cache_payload: Dict[str, object] = {}
    pipeline_health = "ok"
    feature_missing_ratio_base = 0.0
    feature_missing_ratio_mc = 0.0
    model_health_base = bool(meta_blend_strategy is not None and meta_blend_strategy.model_health().get("base_ok"))
    model_health_mc = bool(meta_blend_strategy is not None and meta_blend_strategy.model_health().get("mc_ok"))
    latency_ms = {
        "data_ms": 0.0,
        "decision_ms": 0.0,
        "total_cycle_ms": 0.0,
        "budget_ms": float(cfg.latency_budget_ms),
        "budget_pass": True,
    }

    _write_live_status(
        cfg,
        {
            "status": "running",
            "timestamp_utc": str(datetime.now(timezone.utc)),
            "equity": account.cash,
            "cash": account.cash,
            "trades": 0,
            "positions": 0,
            "cycle": 0,
            "strategy_type": cfg.strategy_type,
            "symbols_total": len(cfg.symbols),
            "llm_enabled": bool(cfg.llm_enabled),
            "llm_active_this_cycle": bool(llm_active_this_cycle),
            "llm_decision_interval_cycles": int(cfg.llm_decision_interval_cycles),
            "allocator_chosen_stack": None,
            "risk_snapshot": {},
            "provider_status": collector.get_provider_diagnostics().get("last_selection", {}),
            "provider_degraded": False,
            "openbb_api": openbb_boot,
            "decision_engine": decision_engine,
            "ab_group": ab_group_active,
            "pipeline_health": pipeline_health,
            "anchor_cache_freshness_minutes": None,
            "model_health_base": bool(model_health_base),
            "model_health_mc": bool(model_health_mc),
            "feature_missing_ratio_base": float(feature_missing_ratio_base),
            "feature_missing_ratio_mc": float(feature_missing_ratio_mc),
            "latency_ms": latency_ms,
            "output_dir": str(cfg.output_dir),
            "note": (
                "Session started, waiting for first data collection cycle"
                if not meta_blend_init_error
                else f"Session started with fallback: {meta_blend_init_error}"
            ),
            **_pnl_fields(
                equity=account.cash,
                starting_capital=cfg.capital,
                previous_equity=previous_equity,
            ),
        },
    )

    try:
        while datetime.now(timezone.utc) < session_end:
            cycle += 1
            cycle_t0 = time.perf_counter()
            now = datetime.now(timezone.utc)
            openbb_health = bool(openbb_manager.health()) if cfg.openbb_enabled else False

            full_scan_cycle = True
            if (
                cfg.full_universe_requested
                and cfg.liquid_subset_size > 0
                and liquid_symbols_cache
                and (cycle % cfg.universe_refresh_cycles) != 1
            ):
                full_scan_cycle = False
                symbols_this_cycle = sorted(
                    set(liquid_symbols_cache) | set(account.positions.keys())
                )
            else:
                symbols_this_cycle = list(cfg.symbols)

            if cfg.strategy_type in {"news_lstm", "meta_blend_hybrid"}:
                llm_active_this_cycle = _llm_cycle_active(cycle=cycle, cfg=cfg)
                if hasattr(strategy, "_llm_router") and getattr(strategy, "_llm_router", None):
                    strategy._llm_router.config.enabled = bool(llm_active_this_cycle)
                if (
                    hybrid_gate_strategy is not None
                    and hasattr(hybrid_gate_strategy, "_llm_router")
                    and getattr(hybrid_gate_strategy, "_llm_router", None)
                ):
                    hybrid_gate_strategy._llm_router.config.enabled = bool(llm_active_this_cycle)
            else:
                llm_active_this_cycle = False

            def _progress_update(
                completed: int, total: int, featured_count: int, last_symbol: str
            ) -> None:
                _write_live_status(
                    cfg,
                    {
                        "status": "running",
                        "timestamp_utc": str(datetime.now(timezone.utc)),
                        "equity": account.cash,
                        "cash": account.cash,
                        "trades": len(account.trades),
                        "positions": len(account.positions),
                        "cycle": cycle,
                        "strategy_type": cfg.strategy_type,
                        "llm_enabled": bool(cfg.llm_enabled),
                        "llm_active_this_cycle": bool(llm_active_this_cycle),
                        "llm_decision_interval_cycles": int(cfg.llm_decision_interval_cycles),
                        "symbols_total": total,
                        "symbols_completed": completed,
                        "featured_symbols": featured_count,
                        "last_symbol": last_symbol,
                        "allocator_chosen_stack": allocator_diag.get("chosen_stack"),
                        "risk_snapshot": allocator_diag.get("risk_snapshot", {}),
                        "provider_status": collector.get_provider_diagnostics().get("last_selection", {}),
                        "decision_engine": decision_engine,
                        "ab_group": ab_group_active,
                        "pipeline_health": pipeline_health,
                        "anchor_cache_freshness_minutes": anchor_cache_freshness_minutes(anchor_cache_payload),
                        "model_health_base": bool(model_health_base),
                        "model_health_mc": bool(model_health_mc),
                        "feature_missing_ratio_base": float(feature_missing_ratio_base),
                        "feature_missing_ratio_mc": float(feature_missing_ratio_mc),
                        "latency_ms": latency_ms,
                        "openbb_api": {
                            "enabled": bool(cfg.openbb_enabled),
                            "healthy": bool(openbb_health),
                        },
                        "note": "Collecting market data for current cycle",
                        **_pnl_fields(
                            equity=account.cash,
                            starting_capital=cfg.capital,
                            previous_equity=previous_equity,
                        ),
                    },
                )

            data_t0 = time.perf_counter()
            featured_all = _collect_featured_data(
                collector=collector,
                cleaner=cleaner,
                imputer=imputer,
                feature_gen=feature_gen,
                symbols=symbols_this_cycle,
                end_time=now,
                lookback_days=cfg.lookback_days,
                interval=cfg.interval,
                use_cache=(cfg.interval.lower() == "1d"),
                progress_callback=_progress_update,
            )
            data_t1 = time.perf_counter()
            latency_ms["data_ms"] = float((data_t1 - data_t0) * 1000.0)

            provider_status = collector.get_provider_diagnostics().get("last_selection", {})
            provider_degraded = any(
                bool(v.get("degraded"))
                for v in provider_status.values()
                if isinstance(v, dict)
            )
            if provider_degraded and pipeline_health == "ok":
                pipeline_health = "degraded"

            if not featured_all:
                cycle_t1 = time.perf_counter()
                latency_ms["decision_ms"] = 0.0
                latency_ms["total_cycle_ms"] = float((cycle_t1 - cycle_t0) * 1000.0)
                latency_ms["budget_pass"] = bool(
                    float(latency_ms.get("total_cycle_ms", 0.0)) <= float(cfg.latency_budget_ms)
                )
                latency_ms["budget_ms"] = float(cfg.latency_budget_ms)
                _write_live_status(
                    cfg,
                    {
                        "status": "running",
                        "timestamp_utc": str(datetime.now(timezone.utc)),
                        "equity": None,
                        "cash": account.cash,
                        "trades": len(account.trades),
                        "positions": len(account.positions),
                        "strategy_type": cfg.strategy_type,
                        "llm_enabled": bool(cfg.llm_enabled),
                        "llm_active_this_cycle": bool(llm_active_this_cycle),
                        "llm_decision_interval_cycles": int(cfg.llm_decision_interval_cycles),
                        "provider_status": provider_status,
                        "provider_degraded": bool(provider_degraded),
                        "decision_engine": decision_engine,
                        "ab_group": ab_group_active,
                        "pipeline_health": "failed",
                        "anchor_cache_freshness_minutes": anchor_cache_freshness_minutes(anchor_cache_payload),
                        "model_health_base": bool(model_health_base),
                        "model_health_mc": bool(model_health_mc),
                        "feature_missing_ratio_base": float(feature_missing_ratio_base),
                        "feature_missing_ratio_mc": float(feature_missing_ratio_mc),
                        "latency_ms": latency_ms,
                        "openbb_api": {
                            "enabled": bool(cfg.openbb_enabled),
                            "healthy": bool(openbb_health),
                        },
                        "note": "No featured data this cycle",
                        **_pnl_fields(
                            equity=None,
                            starting_capital=cfg.capital,
                            previous_equity=previous_equity,
                        ),
                    },
                )
                time.sleep(cfg.poll_seconds)
                continue

            featured_trade = dict(featured_all)
            liquid_filtered = False
            if (
                cfg.liquid_subset_size > 0
                and len(featured_trade) > cfg.liquid_subset_size
                and (cfg.full_universe_requested or not cfg.liquid_for_full_only)
            ):
                selected = _select_liquid_subset(
                    featured_trade,
                    subset_size=cfg.liquid_subset_size,
                    adv_window=cfg.liquid_adv_window,
                    min_history=cfg.liquid_min_history,
                )
                selected_set = set(selected)
                featured_trade = {
                    sym: df for sym, df in featured_trade.items() if sym in selected_set
                }
                liquid_filtered = True
                liquid_symbols_cache = selected
            elif (
                cfg.full_universe_requested
                and cfg.liquid_subset_size > 0
                and liquid_symbols_cache
                and not full_scan_cycle
            ):
                selected_set = set(liquid_symbols_cache)
                featured_trade = {
                    sym: df for sym, df in featured_trade.items() if sym in selected_set
                }
                liquid_filtered = True

            for held_symbol in list(account.positions.keys()):
                if held_symbol in featured_all and held_symbol not in featured_trade:
                    featured_trade[held_symbol] = featured_all[held_symbol]

            if not featured_trade:
                time.sleep(cfg.poll_seconds)
                continue

            latest_ts = max(df.index[-1] for df in featured_trade.values() if not df.empty)
            prices = _latest_prices(featured_trade)
            last_prices.update(prices)

            if last_bar_time is not None and latest_ts <= last_bar_time and cycle > 1:
                current_equity = account.equity(last_prices)
                latency_ms["decision_ms"] = 0.0
                cycle_t1 = time.perf_counter()
                latency_ms["total_cycle_ms"] = float((cycle_t1 - cycle_t0) * 1000.0)
                latency_ms["budget_pass"] = bool(
                    float(latency_ms.get("total_cycle_ms", 0.0)) <= float(cfg.latency_budget_ms)
                )
                latency_ms["budget_ms"] = float(cfg.latency_budget_ms)
                equity_curve.append(
                    {
                        "timestamp": str(pd.Timestamp.now(tz="UTC")),
                        "equity": current_equity,
                    }
                )
                _write_live_status(
                    cfg,
                    {
                        "status": "running",
                        "timestamp_utc": str(datetime.now(timezone.utc)),
                        "latest_bar": str(latest_ts),
                        "equity": current_equity,
                        "cash": account.cash,
                        "trades": len(account.trades),
                        "positions": len(account.positions),
                        "strategy_type": cfg.strategy_type,
                        "llm_enabled": bool(cfg.llm_enabled),
                        "llm_active_this_cycle": bool(llm_active_this_cycle),
                        "llm_decision_interval_cycles": int(cfg.llm_decision_interval_cycles),
                        "symbols_total": len(cfg.symbols),
                        "full_scan_cycle": full_scan_cycle,
                        "symbols_trade": len(featured_trade),
                        "liquidity_filtered": liquid_filtered,
                        "news_last_refresh_utc": str(news_last_refresh) if news_last_refresh else None,
                        "allocator_chosen_stack": allocator_diag.get("chosen_stack"),
                        "allocator_stack_scores": (
                            allocator_diag.get("diagnostics", {}).get("stack_scores", {})
                            if isinstance(allocator_diag.get("diagnostics"), dict)
                            else {}
                        ),
                        "risk_snapshot": allocator_diag.get("risk_snapshot", {}),
                        "provider_status": provider_status,
                        "provider_degraded": bool(provider_degraded),
                        "decision_engine": decision_engine,
                        "ab_group": ab_group_active,
                        "pipeline_health": pipeline_health,
                        "anchor_cache_freshness_minutes": anchor_cache_freshness_minutes(anchor_cache_payload),
                        "model_health_base": bool(model_health_base),
                        "model_health_mc": bool(model_health_mc),
                        "feature_missing_ratio_base": float(feature_missing_ratio_base),
                        "feature_missing_ratio_mc": float(feature_missing_ratio_mc),
                        "latency_ms": latency_ms,
                        "openbb_api": {
                            "enabled": bool(cfg.openbb_enabled),
                            "healthy": bool(openbb_health),
                        },
                        "note": "No new bar yet",
                        "positions_snapshot": _positions_snapshot(account, last_prices),
                        "latest_decisions": latest_decisions[:10],
                        "recent_trades": account.trades[-10:],
                        **_pnl_fields(
                            equity=current_equity,
                            starting_capital=cfg.capital,
                            previous_equity=previous_equity,
                        ),
                    },
                )
                previous_equity = current_equity
                trade_write_idx = _append_new_trades(cfg, account.trades, trade_write_idx)
                time.sleep(cfg.poll_seconds)
                continue

            last_bar_time = latest_ts

            decision_t0 = time.perf_counter()
            if cfg.strategy_type == "news_lstm":
                decision_engine = "news_lstm"
                pipeline_health = "ok"
                should_refresh_news = (
                    news_last_refresh is None
                    or (now - news_last_refresh).total_seconds() >= cfg.news_poll_seconds
                )
                if should_refresh_news:
                    news_symbols = set(
                        _top_dollar_volume_symbols(featured_trade, cfg.news_symbol_limit)
                    )
                    news_symbols.update(account.positions.keys())
                    refreshed = 0
                    for sym in sorted(news_symbols):
                        try:
                            articles = news_collector.fetch_live_news(
                                sym,
                                max_articles=cfg.news_max_articles,
                            )
                        except Exception:
                            articles = []
                        headlines: List[str] = []
                        for article in articles:
                            title = str(article.get("title", "")).strip()
                            summary = str(article.get("summary", "")).strip()
                            text = title if title else summary
                            if text:
                                headlines.append(text[:180])
                        news_cache[sym] = headlines
                        refreshed += 1
                    news_last_refresh = now
                    print(
                        f"[{now}] refreshed news for {refreshed} symbols",
                        flush=True,
                    )

                target_weights, latest_decisions, allocator_diag = _compute_target_weights_news_lstm(
                    strategy=strategy,
                    featured=featured_trade,
                    cfg=cfg,
                    news_cache=news_cache,
                    llm_active=llm_active_this_cycle,
                    allocator=allocator,
                    timestamp=latest_ts.to_pydatetime() if hasattr(latest_ts, "to_pydatetime") else datetime.now(),
                )
                feature_missing_ratio_base = 0.0
                feature_missing_ratio_mc = 0.0
            else:
                if cfg.strategy_type == "meta_blend_hybrid" and ab_group_active == "B":
                    decision_engine = "meta_blend_hybrid"
                    model_health = (
                        meta_blend_strategy.model_health()
                        if meta_blend_strategy is not None
                        else {"base_ok": False, "mc_ok": False}
                    )
                    model_health_base = bool(model_health.get("base_ok", False))
                    model_health_mc = bool(model_health.get("mc_ok", False))
                    if not model_health_base and not model_health_mc:
                        pipeline_health = "failed"
                    elif not model_health_base or not model_health_mc:
                        pipeline_health = "degraded"
                    else:
                        pipeline_health = "ok"

                    # Build/refresh daily anchor cache on top liquid anchor universe.
                    anchor_featured = dict(featured_all)
                    if len(anchor_featured) > cfg.anchor_universe_size:
                        top_anchor = set(
                            _top_dollar_volume_symbols(anchor_featured, cfg.anchor_universe_size)
                        )
                        anchor_featured = {
                            sym: frame for sym, frame in anchor_featured.items() if sym in top_anchor
                        }

                    anchor_cache_path = (
                        Path(cfg.anchor_cache_path)
                        if cfg.anchor_cache_path
                        else (cfg.output_dir / "anchor_cache.json")
                    )
                    if meta_blend_strategy is not None:
                        anchor_cache_payload = refresh_anchor_cache_if_needed(
                            anchor_featured,
                            meta_blend_strategy,
                            cache_path=anchor_cache_path,
                            force=False,
                            decision_engine=decision_engine,
                        )
                    else:
                        anchor_cache_payload = {}

                    anchor_predictions = (
                        anchor_cache_payload.get("predictions", {})
                        if isinstance(anchor_cache_payload, dict)
                        else {}
                    )

                    if not isinstance(anchor_predictions, dict) or len(anchor_predictions) == 0:
                        pipeline_health = "failed"
                        decision_engine = "enhanced_fallback_no_anchor"
                        target_weights, latest_decisions, allocator_diag = _compute_target_weights_enhanced(
                            strategy=enhanced_fallback_strategy,
                            featured=featured_trade,
                            cfg=cfg,
                            allocator=allocator,
                            timestamp=latest_ts.to_pydatetime() if hasattr(latest_ts, "to_pydatetime") else datetime.now(),
                        )
                        feature_missing_ratio_base = 1.0
                        feature_missing_ratio_mc = 1.0
                    else:
                        # Hybrid gating: refresh news only for top-N confidence symbols.
                        candidate_syms = sorted(
                            anchor_predictions.keys(),
                            key=lambda s: abs(float(anchor_predictions[s].get("confidence", 0.0))),
                            reverse=True,
                        )[: max(1, int(cfg.hybrid_gate_top_n))]

                        should_refresh_news = (
                            news_last_refresh is None
                            or (now - news_last_refresh).total_seconds() >= cfg.news_poll_seconds
                        )
                        if should_refresh_news and cfg.llm_enabled and hybrid_gate_strategy is not None:
                            refreshed = 0
                            for sym in candidate_syms:
                                if sym not in featured_trade:
                                    continue
                                try:
                                    articles = news_collector.fetch_live_news(
                                        sym,
                                        max_articles=cfg.news_max_articles,
                                    )
                                except Exception:
                                    articles = []
                                headlines: List[str] = []
                                for article in articles:
                                    title = str(article.get("title", "")).strip()
                                    summary = str(article.get("summary", "")).strip()
                                    text = title if title else summary
                                    if text:
                                        headlines.append(text[:180])
                                news_cache[sym] = headlines
                                refreshed += 1
                            news_last_refresh = now
                            print(
                                f"[{now}] refreshed hybrid gate news for {refreshed} symbols",
                                flush=True,
                            )

                        llm_gate_active = bool(llm_active_this_cycle)
                        if float(latency_ms.get("data_ms", 0.0)) > float(cfg.latency_budget_ms) * 0.8:
                            llm_gate_active = False
                            if pipeline_health != "failed":
                                pipeline_health = "degraded"

                        target_weights, latest_decisions, allocator_diag, meta_health = _compute_target_weights_meta_blend(
                            featured=featured_trade,
                            cfg=cfg,
                            anchor_predictions=anchor_predictions,
                            llm_active=llm_gate_active,
                            news_cache=news_cache,
                            gate_strategy=hybrid_gate_strategy if cfg.llm_enabled else None,
                            allocator=allocator,
                            timestamp=latest_ts.to_pydatetime() if hasattr(latest_ts, "to_pydatetime") else datetime.now(),
                            top_n_for_gate=cfg.hybrid_gate_top_n,
                        )
                        feature_missing_ratio_base = float(meta_health.get("feature_missing_ratio_base", 1.0))
                        feature_missing_ratio_mc = float(meta_health.get("feature_missing_ratio_mc", 1.0))
                        if feature_missing_ratio_base > 0.40 or feature_missing_ratio_mc > 0.40:
                            pipeline_health = "degraded"
                else:
                    # Enhanced path (default + A/B control group A)
                    decision_engine = (
                        "enhanced_ab_a"
                        if cfg.strategy_type == "meta_blend_hybrid" and ab_group_active == "A"
                        else "enhanced"
                    )
                    pipeline_health = "ok"
                    target_weights, latest_decisions, allocator_diag = _compute_target_weights_enhanced(
                        strategy=enhanced_fallback_strategy,
                        featured=featured_trade,
                        cfg=cfg,
                        allocator=allocator,
                        timestamp=latest_ts.to_pydatetime() if hasattr(latest_ts, "to_pydatetime") else datetime.now(),
                    )
                    feature_missing_ratio_base = 0.0
                    feature_missing_ratio_mc = 0.0

            decision_t1 = time.perf_counter()
            latency_ms["decision_ms"] = float((decision_t1 - decision_t0) * 1000.0)

            account.rebalance(
                target_weights=target_weights,
                prices=last_prices,
                timestamp=latest_ts,
                min_notional=cfg.min_trade_notional,
            )
            trade_write_idx = _append_new_trades(cfg, account.trades, trade_write_idx)

            current_equity = account.equity(last_prices)
            equity_curve.append(
                {
                    "timestamp": str(latest_ts),
                    "equity": current_equity,
                    "cash": account.cash,
                    "n_positions": len(account.positions),
                }
            )
            cycle_t1 = time.perf_counter()
            latency_ms["total_cycle_ms"] = float((cycle_t1 - cycle_t0) * 1000.0)
            latency_ms["budget_pass"] = bool(
                float(latency_ms.get("total_cycle_ms", 0.0)) <= float(cfg.latency_budget_ms)
            )
            latency_ms["budget_ms"] = float(cfg.latency_budget_ms)
            if not bool(latency_ms.get("budget_pass", True)):
                pipeline_health = "degraded" if pipeline_health != "failed" else pipeline_health

            _write_live_status(
                cfg,
                {
                    "status": "running",
                    "timestamp_utc": str(datetime.now(timezone.utc)),
                    "latest_bar": str(latest_ts),
                    "equity": current_equity,
                    "cash": account.cash,
                    "trades": len(account.trades),
                    "positions": len(account.positions),
                    "cycle": cycle,
                    "strategy_type": cfg.strategy_type,
                    "llm_enabled": bool(cfg.llm_enabled),
                    "llm_active_this_cycle": bool(llm_active_this_cycle),
                    "llm_decision_interval_cycles": int(cfg.llm_decision_interval_cycles),
                    "symbols_total": len(cfg.symbols),
                    "full_scan_cycle": full_scan_cycle,
                    "symbols_trade": len(featured_trade),
                    "liquidity_filtered": liquid_filtered,
                    "news_last_refresh_utc": str(news_last_refresh) if news_last_refresh else None,
                    "news_cache_symbols": len(news_cache),
                    "allocator_chosen_stack": allocator_diag.get("chosen_stack"),
                    "allocator_stack_scores": (
                        allocator_diag.get("diagnostics", {}).get("stack_scores", {})
                        if isinstance(allocator_diag.get("diagnostics"), dict)
                        else {}
                    ),
                    "risk_snapshot": allocator_diag.get("risk_snapshot", {}),
                    "provider_status": provider_status,
                    "provider_degraded": bool(provider_degraded),
                    "decision_engine": decision_engine,
                    "ab_group": ab_group_active,
                    "pipeline_health": pipeline_health,
                    "anchor_cache_freshness_minutes": anchor_cache_freshness_minutes(anchor_cache_payload),
                    "model_health_base": bool(model_health_base),
                    "model_health_mc": bool(model_health_mc),
                    "feature_missing_ratio_base": float(feature_missing_ratio_base),
                    "feature_missing_ratio_mc": float(feature_missing_ratio_mc),
                    "latency_ms": latency_ms,
                    "openbb_api": {
                        "enabled": bool(cfg.openbb_enabled),
                        "healthy": bool(openbb_health),
                    },
                    "positions_snapshot": _positions_snapshot(account, last_prices),
                    "latest_decisions": latest_decisions[:10],
                    "recent_trades": account.trades[-10:],
                    **_pnl_fields(
                        equity=current_equity,
                        starting_capital=cfg.capital,
                        previous_equity=previous_equity,
                    ),
                },
            )
            previous_equity = current_equity

            print(
                f"[{latest_ts}] cycle={cycle} equity=${current_equity:,.2f} "
                f"pnl=${(current_equity - cfg.capital):,.2f} cash=${account.cash:,.2f} "
                f"positions={len(account.positions)} trades={len(account.trades)} "
                f"llm_cycle={'on' if llm_active_this_cycle else 'off'} "
                f"engine={decision_engine} "
                f"lat={latency_ms.get('total_cycle_ms', 0.0):.0f}ms",
                flush=True,
            )
            time.sleep(cfg.poll_seconds)
    finally:
        openbb_stop = (
            openbb_manager.stop()
            if cfg.openbb_enabled and cfg.openbb_manage_api_process
            else {"stopped": False, "reason": "not_managed"}
        )

    final_equity = account.equity(last_prices)
    profit = final_equity - cfg.capital
    return_pct = (profit / cfg.capital) * 100 if cfg.capital > 0 else 0.0

    summary = {
        "session_start_utc": str(session_start),
        "session_end_utc": str(datetime.now(timezone.utc)),
        "duration_minutes": cfg.duration_minutes,
        "interval": cfg.interval,
        "symbols": cfg.symbols,
        "strategy_type": cfg.strategy_type,
        "starting_capital": cfg.capital,
        "final_equity": final_equity,
        "profit_dollars": profit,
        "return_pct": return_pct,
        "trades": len(account.trades),
        "open_positions": account.positions,
        "commission_rate": cfg.commission_rate,
        "slippage_bps": cfg.slippage_bps,
        "min_commission": cfg.min_commission,
        "min_trade_notional": cfg.min_trade_notional,
        "decision_engine": decision_engine,
        "ab_group": ab_group_active,
        "pipeline_health": pipeline_health,
        "anchor_cache_freshness_minutes": anchor_cache_freshness_minutes(anchor_cache_payload),
        "model_health_base": bool(model_health_base),
        "model_health_mc": bool(model_health_mc),
        "feature_missing_ratio_base": float(feature_missing_ratio_base),
        "feature_missing_ratio_mc": float(feature_missing_ratio_mc),
        "latency_ms": latency_ms,
        "allocator_chosen_stack": allocator_diag.get("chosen_stack"),
        "risk_snapshot": allocator_diag.get("risk_snapshot", {}),
        "provider_status": collector.get_provider_diagnostics().get("last_selection", {}),
        "openbb_api_start": openbb_boot,
        "openbb_api_stop": openbb_stop,
        "output_dir": str(cfg.output_dir),
    }

    _save_outputs(cfg=cfg, equity_curve=equity_curve, trades=account.trades, summary=summary)
    _write_live_status(
        cfg,
        {
            "status": "completed",
            "timestamp_utc": str(datetime.now(timezone.utc)),
            "final_equity": final_equity,
            "profit_dollars": profit,
            "return_pct": return_pct,
            "trades": len(account.trades),
            "decision_engine": decision_engine,
            "ab_group": ab_group_active,
            "pipeline_health": pipeline_health,
            "anchor_cache_freshness_minutes": anchor_cache_freshness_minutes(anchor_cache_payload),
            "model_health_base": bool(model_health_base),
            "model_health_mc": bool(model_health_mc),
            "feature_missing_ratio_base": float(feature_missing_ratio_base),
            "feature_missing_ratio_mc": float(feature_missing_ratio_mc),
            "latency_ms": latency_ms,
            "allocator_chosen_stack": allocator_diag.get("chosen_stack"),
            "risk_snapshot": allocator_diag.get("risk_snapshot", {}),
            "provider_status": collector.get_provider_diagnostics().get("last_selection", {}),
            "openbb_api_stop": openbb_stop,
            "output_dir": str(cfg.output_dir),
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time paper trader with live market data and fake money."
    )
    parser.add_argument("--symbols", nargs="+", default=["AUTO"], help="Symbols, AUTO, SP500, FULL")
    parser.add_argument("--interval", type=str, default="5m", help="Market data interval")
    parser.add_argument(
        "--duration-minutes", type=int, default=60, help="Session duration in minutes"
    )
    parser.add_argument(
        "--until-close",
        action="store_true",
        help="Override duration and run until US market close (16:00 ET)",
    )
    parser.add_argument("--poll-seconds", type=int, default=60, help="Polling cadence")
    parser.add_argument("--lookback-days", type=int, default=60, help="Feature lookback window")
    parser.add_argument("--capital", type=float, default=None, help="Initial fake capital")
    parser.add_argument(
        "--default-capital",
        type=float,
        default=100000,
        help="Capital used when --capital is omitted",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to settings.yaml")
    parser.add_argument(
        "--output-dir", type=str, default="artifacts", help="Output directory for session files"
    )
    parser.add_argument(
        "--strategy-type",
        type=str,
        default="enhanced",
        choices=["enhanced", "news_lstm", "meta_blend_hybrid"],
        help="Signal engine to use",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="News-LSTM checkpoint name (auto-detect when omitted)",
    )
    parser.add_argument(
        "--meta-base-checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory for base meta model",
    )
    parser.add_argument(
        "--meta-mc-checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory for MC/Padé meta model",
    )
    parser.add_argument(
        "--meta-blend-weights",
        type=str,
        default=None,
        help="Comma-separated blend weights: base,mc (default 0.35,0.65)",
    )
    parser.add_argument(
        "--anchor-cache-path",
        type=str,
        default=None,
        help="Path to daily anchor cache JSON (default: session_dir/anchor_cache.json)",
    )
    parser.add_argument(
        "--ab-group",
        type=str,
        default="auto",
        choices=["auto", "A", "B", "a", "b"],
        help="A/B routing group for meta_blend_hybrid",
    )
    parser.add_argument(
        "--latency-budget-ms",
        type=int,
        default=20000,
        help="Cycle latency budget in milliseconds",
    )
    parser.add_argument(
        "--hybrid-gate-top-n",
        type=int,
        default=40,
        help="Apply hybrid news/LLM gate only to top-N anchor candidates",
    )
    parser.add_argument(
        "--anchor-universe-size",
        type=int,
        default=200,
        help="Anchor scoring universe size (top liquid symbols)",
    )
    parser.add_argument(
        "--llm-enable",
        action="store_true",
        help="Enable Gemini LLM gate (news_lstm mode)",
    )
    parser.add_argument(
        "--llm-mode",
        type=str,
        default=None,
        choices=["off", "simulated", "api"],
        help="Gemini mode",
    )
    parser.add_argument(
        "--llm-models",
        type=str,
        default=None,
        help="Comma-separated Gemini model names",
    )
    parser.add_argument(
        "--llm-min-alignment",
        type=float,
        default=0.80,
        help="Minimum LLM alignment required to approve trade",
    )
    parser.add_argument(
        "--llm-fail-mode",
        type=str,
        default="hold",
        choices=["hold", "pass"],
        help="Fallback behavior if LLM fails",
    )
    parser.add_argument(
        "--llm-scope",
        type=str,
        default="latest",
        choices=["all", "latest"],
        help="Apply LLM to all bars or latest only",
    )
    parser.add_argument(
        "--llm-max-calls",
        type=int,
        default=100000,
        help="Max LLM calls per symbol pass",
    )
    parser.add_argument(
        "--llm-decision-interval-cycles",
        type=int,
        default=3,
        help="Run LLM gate every N cycles (1 = every cycle)",
    )
    parser.add_argument(
        "--llm-env-path",
        type=str,
        default=None,
        help="Optional .env path for Gemini keys",
    )
    parser.add_argument(
        "--news-poll-seconds",
        type=int,
        default=300,
        help="Headline refresh cadence in seconds",
    )
    parser.add_argument(
        "--news-max-articles",
        type=int,
        default=6,
        help="Max headlines fetched per symbol at each refresh",
    )
    parser.add_argument(
        "--news-symbol-limit",
        type=int,
        default=60,
        help="Refresh news for top-N liquid symbols each news cycle",
    )
    parser.add_argument(
        "--liquid-subset-size",
        type=int,
        default=None,
        help="If set, trade only top-N liquid names from selected universe",
    )
    parser.add_argument(
        "--liquid-adv-window",
        type=int,
        default=None,
        help="ADV lookback for liquidity subset",
    )
    parser.add_argument(
        "--liquid-min-history",
        type=int,
        default=None,
        help="Minimum bars required for liquidity ranking",
    )
    parser.add_argument(
        "--liquid-for-full-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Apply liquidity subset only when FULL universe token is used",
    )
    parser.add_argument(
        "--universe-refresh-cycles",
        type=int,
        default=30,
        help="When using FULL+liquid subset, rerun full-universe scan every N cycles",
    )
    parser.add_argument(
        "--commission-rate",
        type=float,
        default=None,
        help="Override commission rate (e.g. 0.0 for commission-free paper)",
    )
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=None,
        help="Override slippage in basis points",
    )
    parser.add_argument(
        "--min-commission",
        type=float,
        default=None,
        help="Override minimum commission per order",
    )
    parser.add_argument(
        "--min-trade-notional",
        type=float,
        default=1000.0,
        help="Skip rebalance orders smaller than this dollar notional",
    )
    parser.add_argument(
        "--dashboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-launch live graph dashboard for this session",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8501,
        help="Dashboard port",
    )
    parser.add_argument(
        "--dashboard-refresh-seconds",
        type=float,
        default=1.0,
        help="Dashboard auto-refresh cadence in seconds",
    )
    parser.add_argument(
        "--dashboard-open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open dashboard URL in browser when started",
    )
    args = parser.parse_args()

    cfg = _build_config(args)
    max_lb = _interval_max_lookback_days(cfg.interval)
    if int(args.lookback_days) > int(cfg.lookback_days):
        print(
            f"Lookback capped for interval {cfg.interval}: "
            f"requested={int(args.lookback_days)}d effective={int(cfg.lookback_days)}d "
            f"(max={max_lb}d)",
            flush=True,
        )
    now_et = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York"))
    if args.until_close and cfg.duration_minutes <= 1:
        print(
            f"Market appears closed now ({now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}); "
            "until-close session duration resolved to 1 minute.",
            flush=True,
        )
    print(
        f"Starting realtime paper session: strategy={cfg.strategy_type} "
        f"symbols={len(cfg.symbols)} interval={cfg.interval} "
        f"duration={cfg.duration_minutes}m poll={cfg.poll_seconds}s "
        f"min_notional=${cfg.min_trade_notional:,.0f} comm={cfg.commission_rate} "
        f"min_comm=${cfg.min_commission:.2f} slip={cfg.slippage_bps:.1f}bps "
        f"news_poll={cfg.news_poll_seconds}s llm={'on' if cfg.llm_enabled else 'off'} "
        f"llm_every={cfg.llm_decision_interval_cycles}cy "
        f"ab={cfg.ab_group} latency_budget={cfg.latency_budget_ms}ms",
        flush=True,
    )
    dashboard_meta: Dict[str, object] = {"started": False}
    if args.dashboard:
        dashboard_meta = _start_live_dashboard(
            cfg=cfg,
            port=int(args.dashboard_port),
            refresh_seconds=float(args.dashboard_refresh_seconds),
            open_browser=bool(args.dashboard_open),
        )
        if bool(dashboard_meta.get("started")):
            print(
                "Live graph dashboard started: "
                f"{dashboard_meta.get('url')} "
                f"(pid={dashboard_meta.get('pid')})",
                flush=True,
            )
        else:
            print(
                "Dashboard not started: "
                f"{dashboard_meta.get('reason', 'unknown_error')}",
                flush=True,
            )
    summary = run_realtime_paper(cfg)
    if bool(dashboard_meta.get("started")):
        summary["dashboard_url"] = str(dashboard_meta.get("url"))
        summary["dashboard_pid"] = int(dashboard_meta.get("pid", 0))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
