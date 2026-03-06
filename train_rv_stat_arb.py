"""Train and backtest the replay-first relative-value statistical arbitrage sleeve."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from quantum_alpha.backtesting.sleeve_metrics import compute_basic_metrics
from quantum_alpha.data.collectors.intraday_replay import (
    IntradayReplayStore,
    build_synthetic_intraday_replay,
)
from quantum_alpha.features.intraday_feature_builder import UnifiedIntradayFeatureBuilder
from quantum_alpha.strategy.statistical_arbitrage import MultiPairsPortfolio

DEFAULT_SYMBOLS = ("SPY", "XLK", "AAPL", "MSFT")


def _ensure_fixture_replay(root: Path, symbols: Sequence[str], days: int, seed: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp("2025-01-06", tz="UTC")
    for offset in range(max(1, int(days))):
        build_synthetic_intraday_replay(
            root,
            date=(start + pd.Timedelta(days=offset)).strftime("%Y-%m-%d"),
            symbols=symbols,
            minutes=240,
            seed=seed + offset,
        )
    return root


def _load_prices_and_factors(
    replay_dir: str | Path,
    symbols: Sequence[str],
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, float]]:
    store = IntradayReplayStore(replay_dir)
    builder = UnifiedIntradayFeatureBuilder()
    price_frames: Dict[str, List[pd.DataFrame]] = {s.upper(): [] for s in symbols}
    factor_frames: Dict[str, List[pd.DataFrame]] = {s.upper(): [] for s in symbols}
    for day in store.available_dates():
        for symbol in symbols:
            try:
                bundle = store.load_symbol_bundle(day, symbol, domains=("trades", "quotes", "depth", "bars_1m"))
            except FileNotFoundError:
                continue
            bars = bundle["bars_1m"].copy()
            bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
            price_frames[symbol.upper()].append(
                bars.loc[:, ["timestamp", "close"]]
                .rename(columns={"close": symbol.upper()})
                .set_index("timestamp")
            )
            built = builder.build(
                symbol=symbol,
                trades=bundle["trades"],
                quotes=bundle["quotes"],
                depth=bundle["depth"],
                bars_1m=bundle["bars_1m"],
            )
            factors = built.features.reset_index().rename(columns={"index": "timestamp"})
            factor_frames[symbol.upper()].append(factors.loc[:, ["timestamp", "tp_reversion_pressure", "tp_liquidity_fracture"]])

    if not any(price_frames.values()):
        raise ValueError("No replay bars available for RV stat-arb training")

    symbol_series = []
    for symbol, frames in price_frames.items():
        if not frames:
            continue
        series = pd.concat(frames, axis=0).sort_index()
        series = series.loc[~series.index.duplicated(keep="last")]
        symbol_series.append(series)
    prices = pd.concat(symbol_series, axis=1).sort_index().ffill().dropna(how="all")
    factors = {
        symbol: pd.concat(frames, ignore_index=False).drop_duplicates("timestamp").set_index("timestamp").sort_index()
        for symbol, frames in factor_frames.items()
        if frames
    }
    quality = store.summarize_quality(symbols=symbols).to_dict()
    return prices, factors, quality


def _pair_confirmation(factors: Dict[str, pd.DataFrame], sym1: str, sym2: str) -> pd.Series:
    left = factors[sym1][["tp_reversion_pressure", "tp_liquidity_fracture"]].copy()
    right = factors[sym2][["tp_reversion_pressure", "tp_liquidity_fracture"]].copy()
    idx = left.index.union(right.index)
    left = left.reindex(idx).ffill().fillna(0.0)
    right = right.reindex(idx).ffill().fillna(0.0)
    return (
        0.6 * (left["tp_reversion_pressure"] + right["tp_reversion_pressure"]) / 2.0
        - 0.4 * (left["tp_liquidity_fracture"] + right["tp_liquidity_fracture"]) / 2.0
    ).rename("confirmation")


def _backtest_pairs(prices: pd.DataFrame, factors: Dict[str, pd.DataFrame]) -> tuple[pd.Series, List[Dict[str, object]]]:
    selector = MultiPairsPortfolio(max_pairs=3, capital_per_pair=25_000.0, correlation_threshold=0.6)
    pair_df = selector.select_pairs(prices.resample("5min").last().dropna(how="all"))
    if pair_df.empty:
        raise RuntimeError("No cointegrated pairs selected for RV stat-arb")
    pair_df = pair_df.head(3)

    pair_logs: List[Dict[str, object]] = []
    daily_returns: Dict[pd.Timestamp, float] = {}

    for _, pair in pair_df.iterrows():
        sym1 = str(pair["symbol_1"])
        sym2 = str(pair["symbol_2"])
        hedge_ratio = float(pair["hedge_ratio"])
        spread = prices[sym1] - hedge_ratio * prices[sym2]
        spread_mean = spread.rolling(30, min_periods=10).mean()
        spread_std = spread.rolling(30, min_periods=10).std().replace(0, np.nan)
        z = ((spread - spread_mean) / spread_std).fillna(0.0)
        confirm = _pair_confirmation(factors, sym1, sym2).reindex(z.index).ffill().fillna(0.0)

        position = 0
        entry_z = 0.0
        pair_ret = pd.Series(0.0, index=z.index)
        n_trades = 0
        for i in range(1, len(z)):
            signal = float(z.iloc[i])
            confirmation = float(confirm.iloc[i])
            if position == 0:
                if signal > 2.0 and confirmation > 0:
                    position = -1
                    entry_z = signal
                    n_trades += 1
                elif signal < -2.0 and confirmation > 0:
                    position = 1
                    entry_z = signal
                    n_trades += 1
            elif position == 1:
                delta = spread.iloc[i] - spread.iloc[i - 1]
                pair_ret.iloc[i] = delta / max(abs(spread.iloc[i - 1]), 1.0)
                if signal > -0.5 or confirmation < -0.5:
                    position = 0
            elif position == -1:
                delta = spread.iloc[i - 1] - spread.iloc[i]
                pair_ret.iloc[i] = delta / max(abs(spread.iloc[i - 1]), 1.0)
                if signal < 0.5 or confirmation < -0.5:
                    position = 0

        pair_daily = pair_ret.groupby(pair_ret.index.normalize()).sum()
        for day, value in pair_daily.items():
            daily_returns[day] = daily_returns.get(day, 0.0) + float(value) / max(1, len(pair_df))
        pair_logs.append(
            {
                "symbol_1": sym1,
                "symbol_2": sym2,
                "hedge_ratio": hedge_ratio,
                "adf_pvalue": float(pair["adf_pvalue"]),
                "half_life": float(pair.get("half_life", np.nan) or np.nan),
                "n_trades": int(n_trades),
            }
        )

    daily_series = pd.Series(daily_returns).sort_index()
    return daily_series, pair_logs


def train_rv_stat_arb(
    *,
    replay_dir: str | Path | None,
    symbols: Sequence[str],
    output_dir: str | Path,
    checkpoint_dir: str | Path,
    build_fixture: bool = True,
    fixture_days: int = 5,
    seed: int = 42,
) -> Dict[str, object]:
    out_dir = Path(output_dir)
    ckpt_dir = Path(checkpoint_dir)
    if replay_dir is None:
        replay_root = out_dir / "replay_fixture"
        if build_fixture:
            _ensure_fixture_replay(replay_root, symbols, fixture_days, seed)
        replay_dir = replay_root

    prices, factors, quality = _load_prices_and_factors(replay_dir, symbols)
    daily_returns, pairs = _backtest_pairs(prices, factors)
    benchmark = prices["SPY"].pct_change(fill_method=None).groupby(prices.index.normalize()).sum().reindex(daily_returns.index).fillna(0.0)
    metrics = compute_basic_metrics(daily_returns, benchmark, periods_per_year=252.0)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / "rv_stat_arb_model.pkl"
    with open(checkpoint_path, "wb") as f:
        pickle.dump(
            {
                "pairs": pairs,
                "symbols": [s.upper() for s in symbols],
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "quality": quality,
            },
            f,
        )

    daily_returns_path = out_dir / "daily_returns.csv"
    pd.DataFrame(
        {
            "date": daily_returns.index.astype(str),
            "strategy_return": daily_returns.values,
            "benchmark_return": benchmark.values,
            "equity_curve": (1.0 + daily_returns.fillna(0.0)).cumprod().values,
        }
    ).to_csv(daily_returns_path, index=False)

    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "rv_stat_arb",
        "replay_dir": str(replay_dir),
        "checkpoint": str(checkpoint_path),
        "symbols": [s.upper() for s in symbols],
        "pairs": pairs,
        "metrics": metrics,
        "data_quality": quality,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "daily_returns_csv": str(daily_returns_path),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    latest_path = ckpt_dir / "latest_rv_stat_arb.json"
    latest_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "summary_json": str(summary_path),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the relative-value statistical arbitrage sleeve")
    parser.add_argument("--replay-dir", type=str, default=None)
    parser.add_argument("--symbols", type=str, default="SPY,XLK,AAPL,MSFT")
    parser.add_argument("--fixture-days", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-build-fixture", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "artifacts" / "rv_stat_arb"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(Path(__file__).parent / "models" / "rv_stat_arb"),
    )
    args = parser.parse_args()

    summary = train_rv_stat_arb(
        replay_dir=args.replay_dir,
        symbols=[s.strip().upper() for s in args.symbols.split(",") if s.strip()],
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        build_fixture=not args.no_build_fixture,
        fixture_days=args.fixture_days,
        seed=args.seed,
    )
    print(json.dumps({"passed": True, "summary_json": str(Path(args.output_dir) / "summary.json"), "metrics": summary["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
