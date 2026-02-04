"""
Live paper trading loop using real market data (no real orders).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, time as dtime
from pathlib import Path
from typing import Dict, List, Optional
import json

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from quantum_alpha.data.collectors.market_data import DataCollector
from quantum_alpha.data.preprocessing.cleaners import DataCleaner
from quantum_alpha.data.preprocessing.imputers import MissingValueImputer
from quantum_alpha.features.technical.indicators import TechnicalFeatureGenerator
from quantum_alpha.monitoring.logging import configure_logging
from quantum_alpha.strategy.signals import MomentumStrategy
from quantum_alpha.models.lstm_v4.architecture import HAS_TF
from quantum_alpha.models.lstm_v4.trainer import LSTMTrainer
from quantum_alpha.models.online.online_learner import OnlineRewardAdapter


DEFAULT_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "returns",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_position",
    "atr",
    "atr_pct",
    "stoch_k",
    "stoch_d",
    "adx",
    "obv",
    "obv_sma",
    "vwap",
]


@dataclass
class PortfolioState:
    name: str
    cash: float
    qty: float = 0.0
    equity: float = 0.0
    peak_equity: float = 0.0
    halted: bool = False
    drawdown: float = 0.0
    last_features: Optional[np.ndarray] = None
    last_signal: float = 0.0
    last_price: Optional[float] = None


@dataclass
class LSTMContext:
    trainer: LSTMTrainer
    horizon: str
    window: int
    scale: float


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    cleaner = DataCleaner()
    imputer = MissingValueImputer()
    feature_gen = TechnicalFeatureGenerator()
    df = cleaner.clean(df)
    df = imputer.impute(df)
    df = feature_gen.generate(df)
    return df


def _is_market_open(now_et: datetime) -> bool:
    if now_et.weekday() >= 5:
        return False
    open_time = dtime(hour=9, minute=30)
    close_time = dtime(hour=16, minute=0)
    return open_time <= now_et.time() <= close_time


def _next_market_open(now_et: datetime) -> datetime:
    open_time = dtime(hour=9, minute=30)
    candidate = now_et.replace(
        hour=open_time.hour, minute=open_time.minute, second=0, microsecond=0
    )
    if now_et.time() < open_time and now_et.weekday() < 5:
        return candidate
    days_ahead = 1
    while True:
        next_day = now_et + timedelta(days=days_ahead)
        if next_day.weekday() < 5:
            return next_day.replace(
                hour=open_time.hour, minute=open_time.minute, second=0, microsecond=0
            )
        days_ahead += 1


def _fetch_recent(
    collector: DataCollector,
    symbol: str,
    end_dt: datetime,
    lookback_days: int,
    interval: str,
) -> pd.DataFrame:
    start_dt = end_dt - timedelta(days=lookback_days)
    return collector.fetch_ohlcv(
        symbol, start_dt, end_dt, interval=interval, use_cache=False
    )


def _cap_lookback(interval: str, lookback_days: int) -> int:
    interval = interval.lower().strip()
    if interval.endswith("m"):
        try:
            minutes = int(interval[:-1])
        except ValueError:
            minutes = 5
        if minutes <= 1:
            return min(lookback_days, 7)
        if minutes <= 5:
            return min(lookback_days, 30)
        if minutes <= 15:
            return min(lookback_days, 60)
    return lookback_days


def _default_decision_bars(interval: str) -> int:
    interval = interval.lower().strip()
    if interval.endswith("m"):
        try:
            minutes = int(interval[:-1])
        except ValueError:
            minutes = 5
        return max(1, int(60 / max(minutes, 1)))
    if interval.endswith("h"):
        return 1
    return 1


def _micro_signal(df: pd.DataFrame, window: int) -> float:
    if len(df) < window + 2:
        return 0.0
    close = df["close"]
    returns = df["returns"].fillna(0.0)
    volume = df["volume"].replace(0, np.nan).ffill().fillna(0.0)

    roll_mean = close.rolling(window).mean()
    roll_std = close.rolling(window).std().replace(0, np.nan)
    z = ((close - roll_mean) / roll_std).fillna(0.0).iloc[-1]

    vol_mean = volume.rolling(window).mean()
    vol_std = volume.rolling(window).std().replace(0, np.nan)
    vol_z = ((volume - vol_mean) / vol_std).fillna(0.0).iloc[-1]

    ret = returns.iloc[-1]
    imbalance = np.sign(ret) * vol_z

    signal = -np.tanh(0.7 * z + 0.3 * imbalance)
    return float(signal)


def _micro_burst_signal(
    df: pd.DataFrame,
    window: int,
    ret_thresh: float,
    vol_thresh: float,
) -> float:
    if len(df) < window + 2:
        return 0.0
    returns = df["returns"].fillna(0.0)
    volume = df["volume"].replace(0, np.nan).ffill().fillna(0.0)

    ret_std = returns.rolling(window).std().replace(0, np.nan)
    ret_z = (returns / ret_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    vol_mean = volume.rolling(window).mean()
    vol_std = volume.rolling(window).std().replace(0, np.nan)
    vol_z = ((volume - vol_mean) / vol_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    last_ret = returns.iloc[-1]
    last_ret_z = ret_z.iloc[-1]
    last_vol_z = vol_z.iloc[-1]

    burst_score = max(0.0, abs(last_ret_z) - ret_thresh) + max(0.0, last_vol_z - vol_thresh)
    if burst_score <= 0:
        return 0.0

    signal = -np.sign(last_ret) * np.tanh(burst_score)
    return float(signal)


def _apply_hft_defaults(args: argparse.Namespace) -> None:
    args.strategy = "burst"
    args.compare_micro = False
    args.decision_bars = 1
    args.position_fraction = min(args.position_fraction, 0.3)
    args.max_trade_fraction = min(args.max_trade_fraction, 0.02)
    args.min_trade_value = min(args.min_trade_value, 100.0)
    args.cost_bps = min(args.cost_bps, 1.0)
    args.signal_threshold = min(args.signal_threshold, 0.05)
    args.burst_window = min(args.burst_window, 12)
    args.burst_ret_thresh = min(args.burst_ret_thresh, 1.0)
    args.burst_vol_thresh = min(args.burst_vol_thresh, 0.8)


def _online_features(df: pd.DataFrame, window: int) -> np.ndarray:
    returns = df["returns"].fillna(0.0)
    volume = df["volume"].replace(0, np.nan).ffill().fillna(0.0)

    rsi = df["rsi"].fillna(50.0)
    macd_hist = df["macd_hist"].fillna(0.0)
    bb_pos = df["bb_position"].fillna(0.5)
    atr_pct = df.get("atr_pct", pd.Series([0.0] * len(df), index=df.index)).fillna(0.0)

    ret_mean = returns.rolling(window).mean().fillna(0.0)
    ret_std = returns.rolling(window).std().replace(0, np.nan).fillna(0.0)

    vol_mean = volume.rolling(window).mean().fillna(0.0)
    vol_std = volume.rolling(window).std().replace(0, np.nan).fillna(0.0)

    r1 = returns.iloc[-1]
    r_mu = ret_mean.iloc[-1]
    r_sigma = ret_std.iloc[-1]
    vol_z = 0.0
    if vol_std.iloc[-1] > 0:
        vol_z = (volume.iloc[-1] - vol_mean.iloc[-1]) / vol_std.iloc[-1]

    rsi_norm = (rsi.iloc[-1] - 50.0) / 50.0
    bb_center = bb_pos.iloc[-1] - 0.5
    macd_norm = macd_hist.iloc[-1]

    features = np.array(
        [
            r1,
            r_mu,
            r_sigma,
            vol_z,
            rsi_norm,
            macd_norm,
            bb_center,
            atr_pct.iloc[-1],
            1.0,
        ],
        dtype=float,
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def _resolve_checkpoint(
    checkpoint_dir: str,
    checkpoint: Optional[str],
    prefer: str,
    checkpoint_file: Optional[str],
) -> str:
    if checkpoint:
        return checkpoint
    if checkpoint_file:
        latest_path = Path(checkpoint_file)
    else:
        latest_path = Path(checkpoint_dir) / "latest.json"
    if not latest_path.exists():
        raise ValueError("Checkpoint not provided and latest.json not found")
    payload = json.loads(latest_path.read_text())
    if "intraday_checkpoint" in payload:
        value = payload.get("intraday_checkpoint")
    elif prefer == "base":
        value = payload.get("base_checkpoint")
    else:
        value = payload.get("live_checkpoint") or payload.get("base_checkpoint")
    if not isinstance(value, str) or not value:
        raise ValueError("latest.json missing checkpoint entries")
    return value


def _load_lstm_context(args: argparse.Namespace) -> LSTMContext:
    if not HAS_TF:
        raise RuntimeError("TensorFlow not installed. Install tensorflow-cpu to use LSTM.")
    trainer = LSTMTrainer(checkpoint_dir=args.checkpoint_dir)
    checkpoint_file = args.lstm_checkpoint_file
    if checkpoint_file is None and args.interval != "1d":
        intraday_path = Path(args.checkpoint_dir) / "latest_intraday.json"
        if intraday_path.exists():
            checkpoint_file = str(intraday_path)
    checkpoint = _resolve_checkpoint(
        args.checkpoint_dir,
        args.checkpoint,
        args.checkpoint_prefer,
        checkpoint_file,
    )
    trainer.load_checkpoint(checkpoint)
    horizon = args.lstm_horizon
    return LSTMContext(trainer=trainer, horizon=horizon, window=args.lstm_window, scale=args.lstm_scale)


def _lstm_signal(df: pd.DataFrame, ctx: LSTMContext) -> float:
    data = df[DEFAULT_FEATURES].dropna()
    if len(data) < ctx.window:
        return 0.0
    window_data = data.values[-ctx.window :]
    X = window_data.reshape(1, ctx.window, window_data.shape[1])
    X_scaled = ctx.trainer.scaler.transform(X)
    preds = ctx.trainer.model.predict(X_scaled)
    if ctx.horizon not in preds:
        raise ValueError(f"Horizon {ctx.horizon} not in predictions")
    mean = float(preds[ctx.horizon]["mean"][0])
    std = float(preds[ctx.horizon]["std"][0])
    score = mean / (std + 1e-6)
    return float(np.tanh(score * ctx.scale))


def _apply_drawdown_guard(
    state: PortfolioState,
    price: float,
    max_drawdown: float,
    action: str,
    cost_bps: float,
) -> bool:
    if state.peak_equity <= 0:
        state.peak_equity = state.equity
    state.peak_equity = max(state.peak_equity, state.equity)
    if state.peak_equity > 0:
        state.drawdown = (state.equity - state.peak_equity) / state.peak_equity

    if max_drawdown <= 0:
        return False
    if state.drawdown <= -max_drawdown and not state.halted:
        if action in {"flatten", "stop"} and state.qty != 0:
            cost = abs(state.qty) * price * (cost_bps / 10000.0)
            state.cash += state.qty * price - cost
            state.qty = 0.0
            state.equity = state.cash
        state.halted = True
        return True
    return False


def _apply_global_drawdown(
    portfolios: Dict[str, PortfolioState],
    price: float,
    state: Dict[str, float],
    max_drawdown: float,
    action: str,
    cost_bps: float,
) -> bool:
    equities = [p.cash + p.qty * price for p in portfolios.values()]
    if not equities:
        return False
    equity = float(np.mean(equities))
    state["equity"] = equity
    if state["peak"] <= 0:
        state["peak"] = equity
    state["peak"] = max(state["peak"], equity)
    state["drawdown"] = (equity - state["peak"]) / state["peak"] if state["peak"] else 0.0
    if max_drawdown <= 0:
        return False
    if state["drawdown"] <= -max_drawdown and not state.get("halted", False):
        for p in portfolios.values():
            if action in {"flatten", "stop"} and p.qty != 0:
                cost = abs(p.qty) * price * (cost_bps / 10000.0)
                p.cash += p.qty * price - cost
                p.qty = 0.0
            p.equity = p.cash + p.qty * price
            p.halted = True
        state["halted"] = True
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Live paper trading loop")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--lookback-days", type=int, default=14)
    parser.add_argument("--duration-hours", type=float, default=24.0)
    parser.add_argument("--sleep-seconds", type=int, default=None)
    parser.add_argument("--capital", type=float, default=100000)
    parser.add_argument("--position-fraction", type=float, default=0.5)
    parser.add_argument("--max-trade-fraction", type=float, default=0.1)
    parser.add_argument("--min-trade-value", type=float, default=500.0)
    parser.add_argument("--cost-bps", type=float, default=2.0)
    parser.add_argument(
        "--strategy",
        choices=["lstm", "momentum", "micro", "burst"],
        default="lstm",
        help="Primary signal model for live paper trading",
    )
    parser.add_argument(
        "--hft-burst",
        action="store_true",
        help="Enable burst-only high-frequency defaults (trade every bar)",
    )
    parser.add_argument("--compare-micro", action="store_true", help="Run micro + burst in parallel")
    parser.add_argument("--signal-threshold", type=float, default=0.15)
    parser.add_argument("--micro-window", type=int, default=12)
    parser.add_argument("--burst-window", type=int, default=20)
    parser.add_argument("--burst-ret-thresh", type=float, default=1.5)
    parser.add_argument("--burst-vol-thresh", type=float, default=1.0)
    parser.add_argument("--decision-bars", type=int, default=None)
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=0.03,
        help="Stop trading if drawdown exceeds this fraction",
    )
    parser.add_argument("--drawdown-micro", type=float, default=None)
    parser.add_argument("--drawdown-burst", type=float, default=None)
    parser.add_argument("--drawdown-lstm", type=float, default=None)
    parser.add_argument("--drawdown-momentum", type=float, default=None)
    parser.add_argument(
        "--global-max-drawdown",
        type=float,
        default=0.0,
        help="Global kill-switch drawdown threshold (0 disables)",
    )
    parser.add_argument(
        "--drawdown-action",
        choices=["halt", "flatten", "stop"],
        default="flatten",
        help="Action on drawdown breach",
    )
    parser.add_argument(
        "--global-drawdown-action",
        choices=["halt", "flatten", "stop"],
        default="stop",
        help="Action on global drawdown breach",
    )
    parser.add_argument(
        "--stop-at-close",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop when the regular market session closes",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(Path(__file__).parent / "models" / "checkpoints"),
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--lstm-checkpoint-file", type=str, default=None)
    parser.add_argument(
        "--checkpoint-prefer",
        choices=["live", "base"],
        default="live",
    )
    parser.add_argument("--lstm-window", type=int, default=90)
    parser.add_argument("--lstm-horizon", type=str, default="1d")
    parser.add_argument("--lstm-scale", type=float, default=6.0)
    parser.add_argument("--online-adapt", action="store_true", help="Enable online reinforcement adapter")
    parser.add_argument("--online-lr", type=float, default=0.05)
    parser.add_argument("--online-l2", type=float, default=0.001)
    parser.add_argument("--online-strength", type=float, default=0.5)
    parser.add_argument("--online-window", type=int, default=20)
    parser.add_argument("--online-pretrain-bars", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "reports" / "live_paper.csv"),
    )
    parser.add_argument(
        "--chart",
        type=str,
        default=str(Path(__file__).parent / "reports" / "live_paper.png"),
    )
    args = parser.parse_args()

    if args.hft_burst:
        _apply_hft_defaults(args)
        print("HFT burst mode enabled: tighter limits and trade every bar.")

    configure_logging()

    collector = DataCollector()
    momentum = MomentumStrategy()

    sleep_seconds = args.sleep_seconds
    if sleep_seconds is None:
        if args.interval.endswith("m"):
            try:
                minutes = int(args.interval[:-1])
            except ValueError:
                minutes = 5
            sleep_seconds = max(60, minutes * 60)
        elif args.interval.endswith("h"):
            sleep_seconds = 3600
        else:
            sleep_seconds = 300

    strategy_order: List[str] = []
    seen = set()
    for name in [args.strategy] + (["micro", "burst"] if args.compare_micro else []):
        if name not in seen:
            seen.add(name)
            strategy_order.append(name)

    strategy_drawdowns = {
        "micro": args.drawdown_micro if args.drawdown_micro is not None else args.max_drawdown,
        "burst": args.drawdown_burst if args.drawdown_burst is not None else args.max_drawdown,
        "lstm": args.drawdown_lstm if args.drawdown_lstm is not None else args.max_drawdown,
        "momentum": args.drawdown_momentum if args.drawdown_momentum is not None else args.max_drawdown,
    }

    lstm_ctx: Optional[LSTMContext] = None
    if "lstm" in strategy_order:
        if args.interval != "1d":
            print("Warning: Ensure an intraday LSTM checkpoint for this interval; daily checkpoints may be unreliable.")
        try:
            lstm_ctx = _load_lstm_context(args)
        except Exception as exc:
            print(f"Failed to load LSTM checkpoint: {exc}")
            return

    adapters: Dict[str, OnlineRewardAdapter] = {}
    pretrain_done = set()

    portfolios = {
        name: PortfolioState(name=name, cash=float(args.capital)) for name in strategy_order
    }
    global_state = {"equity": 0.0, "peak": 0.0, "drawdown": 0.0, "halted": False}

    end_time = time.time() + args.duration_hours * 3600
    last_ts = None
    bench_start = None
    lookback_days = _cap_lookback(args.interval, args.lookback_days)
    decision_bars = (
        args.decision_bars
        if args.decision_bars and args.decision_bars > 0
        else _default_decision_bars(args.interval)
    )
    bar_counter = 0
    session_started = False
    market_tz = ZoneInfo("America/New_York")

    equity_curve = []
    bench_curve = []

    print(
        f"Starting live paper trading for {args.duration_hours:.2f}h "
        f"({args.symbol}) interval {args.interval}"
    )

    while time.time() < end_time:
        now_utc = datetime.now(timezone.utc)
        now_et = now_utc.astimezone(market_tz)
        if not _is_market_open(now_et):
            if args.stop_at_close:
                if session_started:
                    print("Market closed. Stopping live paper session.")
                    break
                next_open = _next_market_open(now_et)
                wait_seconds = max(60, int((next_open - now_et).total_seconds()))
                print(
                    f"Market closed. Waiting for open at {next_open.strftime('%Y-%m-%d %H:%M')} ET."
                )
                time.sleep(min(wait_seconds, sleep_seconds))
                continue
            time.sleep(sleep_seconds)
            continue

        now = now_utc.replace(tzinfo=None)
        try:
            df = _fetch_recent(
                collector, args.symbol, now, lookback_days, args.interval
            )
        except Exception as exc:
            print(f"Fetch error for {args.symbol}: {exc}")
            time.sleep(sleep_seconds)
            continue

        if df.empty or len(df) < 50:
            print("Insufficient data, waiting for more bars.")
            time.sleep(sleep_seconds)
            continue

        df = _prepare(df)
        df = momentum.generate_signals(df)
        latest = df.iloc[-1]
        ts = df.index[-1]

        if last_ts == ts:
            time.sleep(sleep_seconds)
            continue

        price = float(latest["close"])
        features = _online_features(df, args.online_window)

        for name, state in portfolios.items():
            if args.online_adapt and name not in adapters:
                adapters[name] = OnlineRewardAdapter(
                    n_features=len(features),
                    learning_rate=args.online_lr,
                    l2=args.online_l2,
                )
            if (
                args.online_adapt
                and name in adapters
                and name not in pretrain_done
                and args.online_pretrain_bars > 0
                and name != "lstm"
            ):
                adapter = adapters[name]
                pretrain_bars = min(args.online_pretrain_bars, len(df) - 2)
                for i in range(max(1, len(df) - pretrain_bars), len(df) - 1):
                    sub = df.iloc[: i + 1]
                    base_sig = _micro_signal(sub, args.micro_window)
                    if name == "burst":
                        base_sig = _micro_burst_signal(
                            sub, args.burst_window, args.burst_ret_thresh, args.burst_vol_thresh
                        )
                    elif name == "momentum":
                        base_sig = float(sub.iloc[-1].get("position_signal", 0.0))
                    reward = float(df["returns"].iloc[i + 1]) * base_sig
                    adapter.update(_online_features(sub, args.online_window), reward)
                pretrain_done.add(name)

        signals: Dict[str, float] = {}
        for name, state in portfolios.items():
            if args.online_adapt and state.last_features is not None:
                reward = 0.0
                if state.last_price:
                    reward = (price - state.last_price) / state.last_price
                reward *= state.last_signal
                adapters[name].update(state.last_features, reward)

            if name == "micro":
                base_signal = _micro_signal(df, args.micro_window)
            elif name == "burst":
                base_signal = _micro_burst_signal(
                    df, args.burst_window, args.burst_ret_thresh, args.burst_vol_thresh
                )
            elif name == "momentum":
                base_signal = float(latest.get("position_signal", 0.0))
            elif name == "lstm":
                if lstm_ctx is None:
                    base_signal = 0.0
                else:
                    base_signal = _lstm_signal(df, lstm_ctx)
            else:
                base_signal = 0.0

            if abs(base_signal) < args.signal_threshold:
                base_signal = 0.0

            if args.online_adapt:
                adjust = adapters[name].predict(features)
                base_signal = base_signal * (1.0 + args.online_strength * adjust)
                base_signal = float(np.clip(base_signal, -1.0, 1.0))

            signals[name] = base_signal
            state.last_features = features
            state.last_signal = base_signal
            state.last_price = price

        should_trade = bar_counter % decision_bars == 0
        stop_triggered = False
        for name, state in portfolios.items():
            threshold = strategy_drawdowns.get(name, args.max_drawdown)
            if state.halted:
                state.equity = state.cash + state.qty * price
                _apply_drawdown_guard(state, price, threshold, args.drawdown_action, args.cost_bps)
                continue

            signal = signals[name]
            if should_trade:
                equity = state.cash + state.qty * price
                target_value = equity * args.position_fraction * signal
                target_qty = target_value / price if price > 0 else 0.0
                qty_diff = target_qty - state.qty

                max_trade_value = equity * args.max_trade_fraction
                max_qty = max_trade_value / price if price > 0 else 0.0
                if abs(qty_diff) > max_qty:
                    qty_diff = np.sign(qty_diff) * max_qty

                if abs(qty_diff * price) >= args.min_trade_value:
                    cost = abs(qty_diff) * price * (args.cost_bps / 10000.0)
                    state.cash -= qty_diff * price + cost
                    state.qty += qty_diff

            state.equity = state.cash + state.qty * price
            triggered = _apply_drawdown_guard(
                state, price, threshold, args.drawdown_action, args.cost_bps
            )
            if triggered and args.drawdown_action == "stop":
                stop_triggered = True

        global_triggered = _apply_global_drawdown(
            portfolios,
            price,
            global_state,
            args.global_max_drawdown,
            args.global_drawdown_action,
            args.cost_bps,
        )
        if global_triggered and args.global_drawdown_action == "stop":
            stop_triggered = True

        bar_counter += 1
        session_started = True

        if args.benchmark == args.symbol:
            bench_price = price
        else:
            try:
                bench_df = _fetch_recent(
                    collector,
                    args.benchmark,
                    now,
                    lookback_days,
                    args.interval,
                )
                bench_price = float(bench_df["close"].iloc[-1])
            except Exception:
                bench_price = price

        if bench_start is None:
            bench_start = bench_price
        bench_equity = args.capital * (bench_price / bench_start)
        bench_curve.append({"timestamp": ts, "equity": bench_equity, "price": bench_price})

        row = {
            "timestamp": ts,
            "price": price,
            "bench": bench_equity,
            "global_equity": global_state.get("equity", np.nan),
            "global_drawdown": global_state.get("drawdown", np.nan),
        }
        parts = [f"{ts} | price={price:.2f}"]
        for name, state in portfolios.items():
            row[f"{name}_equity"] = state.equity
            row[f"{name}_signal"] = signals[name]
            parts.append(f"{name}:sig={signals[name]:+.2f} eq={state.equity:,.2f}")
        parts.append(f"bench={bench_equity:,.2f}")
        if args.global_max_drawdown > 0:
            parts.append(f"gDD={global_state.get('drawdown', 0.0) * 100:.2f}%")
        print(" | ".join(parts))
        equity_curve.append(row)

        last_ts = ts
        if stop_triggered:
            print("Drawdown stop triggered. Ending session.")
            break
        time.sleep(sleep_seconds)

    if not equity_curve:
        print("No trades executed. Exiting.")
        return

    eq_df = pd.DataFrame(equity_curve).set_index("timestamp")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    eq_df.to_csv(out_path)

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        for name in portfolios:
            plt.plot(eq_df.index, eq_df[f"{name}_equity"], label=name)
        plt.plot(eq_df.index, eq_df["bench"], label=f"Benchmark ({args.benchmark})")
        plt.title("Live Paper Trading Comparison")
        plt.xlabel("Timestamp")
        plt.ylabel("Equity")
        plt.legend()
        plt.tight_layout()
        chart_path = Path(args.chart)
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(chart_path)
        print(f"Chart saved: {chart_path}")
    except Exception as exc:
        print(f"Chart generation failed: {exc}")

    final_bench = eq_df["bench"].iloc[-1]
    print("\nSummary")
    for name, state in portfolios.items():
        final_equity = state.equity
        total_return = (final_equity / args.capital) - 1.0
        result = "BEAT" if final_equity >= final_bench else "DID NOT BEAT"
        print(
            f"{name}: equity={final_equity:,.2f} return={total_return * 100:.2f}% "
            f"drawdown={state.drawdown * 100:.2f}% {result}"
        )
    if args.global_max_drawdown > 0:
        print(
            f"GLOBAL: equity={global_state.get('equity', 0.0):,.2f} "
            f"drawdown={global_state.get('drawdown', 0.0) * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
