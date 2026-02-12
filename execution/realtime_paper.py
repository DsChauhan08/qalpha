"""
Real-time paper trading runner.

Uses live market data polling with fake-money execution.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from quantum_alpha.data.collectors.market_data import DataCollector
from quantum_alpha.data.preprocessing.cleaners import DataCleaner
from quantum_alpha.data.preprocessing.imputers import MissingValueImputer
from quantum_alpha.features.technical.indicators import TechnicalFeatureGenerator
from quantum_alpha.strategy.signals import EnhancedCompositeStrategy


@dataclass
class SessionConfig:
    symbols: List[str]
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
                commission = max(1.0, notional * self.commission_rate)
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


def _resolve_symbols(input_symbols: List[str], settings: Dict) -> List[str]:
    if len(input_symbols) == 1 and input_symbols[0].upper() == "AUTO":
        data_cfg = settings.get("data", {})
        default = data_cfg.get("default_universe", [])
        limit = int(data_cfg.get("universe_limit", len(default)))
        return [str(s).upper() for s in default[:limit]]
    return [str(s).upper() for s in input_symbols]


def _build_config(args: argparse.Namespace) -> SessionConfig:
    settings = _load_settings(args.config)
    risk_cfg = settings.get("risk", {})
    strategy_cfg = settings.get("strategy", {})
    backtest_cfg = settings.get("backtest", {})

    symbols = _resolve_symbols(args.symbols, settings)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"realtime_paper_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    lookback_days = int(args.lookback_days)
    if args.interval.lower() != "1d" and lookback_days >= 60:
        lookback_days = 59

    return SessionConfig(
        symbols=symbols,
        interval=args.interval,
        duration_minutes=int(args.duration_minutes),
        poll_seconds=int(args.poll_seconds),
        lookback_days=lookback_days,
        capital=float(args.capital if args.capital is not None else args.default_capital),
        max_position_size=float(risk_cfg.get("max_position_size", 0.25)),
        max_portfolio_leverage=float(risk_cfg.get("max_portfolio_leverage", 1.0)),
        signal_threshold=float(strategy_cfg.get("signal_threshold", 0.3)),
        min_long_signal=float(strategy_cfg.get("min_long_signal", 0.0)),
        long_only=bool(strategy_cfg.get("long_only", True)),
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
) -> Dict[str, pd.DataFrame]:
    start_time = end_time - timedelta(days=lookback_days)
    featured: Dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            df = collector.fetch_ohlcv(
                symbol=symbol,
                start=start_time,
                end=end_time,
                interval=interval,
                use_cache=False,
            )
            df = cleaner.clean(df)
            df = imputer.impute(df)
            df = feature_gen.generate(df)
            if not df.empty:
                featured[symbol] = df
        except Exception:
            continue
    return featured


def _compute_target_weights(
    strategy: EnhancedCompositeStrategy,
    featured: Dict[str, pd.DataFrame],
    cfg: SessionConfig,
) -> Dict[str, float]:
    strategy.fit_cross_asset(featured)

    raw_scores: Dict[str, float] = {}
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
        raw_scores[symbol] = score

    if not raw_scores:
        return {s: 0.0 for s in featured.keys()}

    total_score = sum(abs(v) for v in raw_scores.values())
    if total_score <= 0:
        return {s: 0.0 for s in featured.keys()}

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

    for symbol in featured.keys():
        target_weights.setdefault(symbol, 0.0)
    return target_weights


def _latest_prices(featured: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    for symbol, df in featured.items():
        if not df.empty and "close" in df.columns:
            prices[symbol] = float(df["close"].iloc[-1])
    return prices


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


def run_realtime_paper(cfg: SessionConfig) -> Dict:
    collector = DataCollector()
    cleaner = DataCleaner()
    imputer = MissingValueImputer()
    feature_gen = TechnicalFeatureGenerator()
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
    equity_curve: List[Dict] = []
    cycle = 0

    while datetime.now(timezone.utc) < session_end:
        cycle += 1
        now = datetime.now(timezone.utc)
        featured = _collect_featured_data(
            collector=collector,
            cleaner=cleaner,
            imputer=imputer,
            feature_gen=feature_gen,
            symbols=cfg.symbols,
            end_time=now,
            lookback_days=cfg.lookback_days,
            interval=cfg.interval,
        )

        if not featured:
            _write_live_status(
                cfg,
                {
                    "status": "running",
                    "timestamp_utc": str(datetime.now(timezone.utc)),
                    "equity": None,
                    "cash": account.cash,
                    "trades": len(account.trades),
                    "positions": len(account.positions),
                    "note": "No featured data this cycle",
                },
            )
            time.sleep(cfg.poll_seconds)
            continue

        latest_ts = max(df.index[-1] for df in featured.values() if not df.empty)
        if last_bar_time is not None and latest_ts <= last_bar_time and cycle > 1:
            prices = _latest_prices(featured)
            current_equity = account.equity(prices)
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
                    "note": "No new bar yet",
                },
            )
            time.sleep(cfg.poll_seconds)
            continue

        last_bar_time = latest_ts
        target_weights = _compute_target_weights(strategy=strategy, featured=featured, cfg=cfg)
        prices = _latest_prices(featured)
        account.rebalance(
            target_weights=target_weights,
            prices=prices,
            timestamp=latest_ts,
            min_notional=cfg.min_trade_notional,
        )

        equity_curve.append(
            {
                "timestamp": str(latest_ts),
                "equity": account.equity(prices),
                "cash": account.cash,
                "n_positions": len(account.positions),
            }
        )

        current_equity = account.equity(prices)
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
            },
        )

        print(
            f"[{latest_ts}] cycle={cycle} equity=${account.equity(prices):,.2f} "
            f"cash=${account.cash:,.2f} positions={len(account.positions)}"
        , flush=True)
        time.sleep(cfg.poll_seconds)

    prices = _latest_prices(
        _collect_featured_data(
            collector=collector,
            cleaner=cleaner,
            imputer=imputer,
            feature_gen=feature_gen,
            symbols=cfg.symbols,
            end_time=datetime.now(timezone.utc),
            lookback_days=cfg.lookback_days,
            interval=cfg.interval,
        )
    )
    final_equity = account.equity(prices)
    profit = final_equity - cfg.capital
    return_pct = (profit / cfg.capital) * 100 if cfg.capital > 0 else 0.0

    summary = {
        "session_start_utc": str(session_start),
        "session_end_utc": str(datetime.now(timezone.utc)),
        "duration_minutes": cfg.duration_minutes,
        "interval": cfg.interval,
        "symbols": cfg.symbols,
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
    parser.add_argument("--symbols", nargs="+", default=["AUTO"], help="Symbols or AUTO")
    parser.add_argument("--interval", type=str, default="5m", help="Market data interval")
    parser.add_argument(
        "--duration-minutes", type=int, default=60, help="Session duration in minutes"
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
    args = parser.parse_args()

    cfg = _build_config(args)
    print(
        f"Starting realtime paper session: symbols={cfg.symbols} interval={cfg.interval} "
        f"duration={cfg.duration_minutes}m poll={cfg.poll_seconds}s "
        f"min_notional=${cfg.min_trade_notional:,.0f} comm={cfg.commission_rate} "
        f"min_comm=${cfg.min_commission:.2f} slip={cfg.slippage_bps:.1f}bps"
    , flush=True)
    summary = run_realtime_paper(cfg)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
