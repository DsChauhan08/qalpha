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
from quantum_alpha.features.technical.indicators import TechnicalFeatureGenerator
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


def _llm_cycle_active(cycle: int, cfg: SessionConfig) -> bool:
    if not cfg.llm_enabled:
        return False
    interval = max(1, int(cfg.llm_decision_interval_cycles))
    return ((int(cycle) - 1) % interval) == 0


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
    risk_cfg = settings.get("risk", {})
    strategy_cfg = settings.get("strategy", {})
    backtest_cfg = settings.get("backtest", {})

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


def _compute_target_weights_enhanced(
    strategy: EnhancedCompositeStrategy,
    featured: Dict[str, pd.DataFrame],
    cfg: SessionConfig,
) -> tuple[Dict[str, float], List[Dict]]:
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
    weights = _normalize_target_weights(raw_scores, list(featured.keys()), cfg)
    return weights, decisions[:20]


def _compute_target_weights_news_lstm(
    strategy: NewsLSTMStrategy,
    featured: Dict[str, pd.DataFrame],
    cfg: SessionConfig,
    news_cache: Dict[str, List[str]],
    llm_active: bool,
) -> tuple[Dict[str, float], List[Dict]]:
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
    weights = _normalize_target_weights(raw_scores, list(featured.keys()), cfg)
    return weights, decisions[:20]


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
    collector = DataCollector()
    cleaner = DataCleaner()
    imputer = MissingValueImputer()
    feature_gen = TechnicalFeatureGenerator()
    news_collector = NewsCollector()

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
    else:
        strategy = EnhancedCompositeStrategy()

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
            "output_dir": str(cfg.output_dir),
            "note": "Session started, waiting for first data collection cycle",
            **_pnl_fields(
                equity=account.cash,
                starting_capital=cfg.capital,
                previous_equity=previous_equity,
            ),
        },
    )

    while datetime.now(timezone.utc) < session_end:
        cycle += 1
        now = datetime.now(timezone.utc)
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

        if cfg.strategy_type == "news_lstm":
            llm_active_this_cycle = _llm_cycle_active(cycle=cycle, cfg=cfg)
            if hasattr(strategy, "_llm_router") and getattr(strategy, "_llm_router", None):
                strategy._llm_router.config.enabled = bool(llm_active_this_cycle)
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
                    "note": "Collecting market data for current cycle",
                    **_pnl_fields(
                        equity=account.cash,
                        starting_capital=cfg.capital,
                        previous_equity=previous_equity,
                    ),
                },
            )

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

        if not featured_all:
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

        if cfg.strategy_type == "news_lstm":
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

            target_weights, latest_decisions = _compute_target_weights_news_lstm(
                strategy=strategy,
                featured=featured_trade,
                cfg=cfg,
                news_cache=news_cache,
                llm_active=llm_active_this_cycle,
            )
        else:
            target_weights, latest_decisions = _compute_target_weights_enhanced(
                strategy=strategy,
                featured=featured_trade,
                cfg=cfg,
            )

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
            f"llm_cycle={'on' if llm_active_this_cycle else 'off'}",
            flush=True,
        )
        time.sleep(cfg.poll_seconds)

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
        choices=["enhanced", "news_lstm"],
        help="Signal engine to use",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="News-LSTM checkpoint name (auto-detect when omitted)",
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
        f"llm_every={cfg.llm_decision_interval_cycles}cy",
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
