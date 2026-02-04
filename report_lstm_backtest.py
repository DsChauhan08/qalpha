"""
Generate a backtest chart comparing LSTM strategy vs benchmark.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quantum_alpha.data.collectors.market_data import DataCollector
from quantum_alpha.data.preprocessing.cleaners import DataCleaner
from quantum_alpha.data.preprocessing.imputers import MissingValueImputer
from quantum_alpha.features.technical.indicators import TechnicalFeatureGenerator
from quantum_alpha.models.lstm_v4.trainer import LSTMTrainer
from quantum_alpha.monitoring.logging import configure_logging

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


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_gen = TechnicalFeatureGenerator()
    cleaner = DataCleaner()
    imputer = MissingValueImputer()
    df = cleaner.clean(df)
    df = imputer.impute(df)
    df = feature_gen.generate(df)
    return df


def _load_checkpoint_name(checkpoint_dir: str, checkpoint: str | None) -> str:
    if checkpoint:
        return checkpoint
    latest_path = Path(checkpoint_dir) / "latest.json"
    if latest_path.exists():
        payload = json.loads(latest_path.read_text())
        return payload.get("base_checkpoint")
    raise ValueError("Checkpoint not provided and latest.json not found")


def _build_windows(data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    idx = []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size : i])
        idx.append(i)
    return np.array(X), np.array(idx)


def _strategy_returns(preds: np.ndarray, returns: np.ndarray, cost: float) -> np.ndarray:
    signals = np.tanh(preds)
    if len(signals) < 2:
        return np.array([])
    costs = cost * np.abs(np.diff(signals, prepend=0))
    strat = signals * returns - costs
    return strat


def _calibration_stats(
    collector: DataCollector,
    symbol: str,
    end_date: datetime,
    calib_days: int,
) -> Dict[str, float]:
    start_date = end_date - timedelta(days=calib_days * 2)
    df = collector.fetch_ohlcv(symbol, start_date, end_date)
    if df.empty:
        return {
            "mu": 0.0002,
            "sigma": 0.012,
            "range_mu": 0.015,
            "range_sigma": 0.006,
            "vol_mu": 15.0,
            "vol_sigma": 0.5,
            "last_close": 100.0,
        }

    df = df.tail(calib_days)
    returns = df["close"].pct_change().dropna()
    if returns.empty:
        returns = pd.Series([0.0002] * max(1, len(df)))

    range_pct = (df["high"] - df["low"]) / df["close"]
    range_pct = range_pct.replace([np.inf, -np.inf], np.nan).dropna()
    if range_pct.empty:
        range_pct = pd.Series([0.015] * max(1, len(df)))

    volume = df["volume"].replace(0, np.nan).dropna()
    if volume.empty:
        vol_mu, vol_sigma = 15.0, 0.5
    else:
        logv = np.log(volume)
        vol_mu = float(logv.mean())
        vol_sigma = float(logv.std() if logv.std() > 1e-6 else 0.5)

    return {
        "mu": float(returns.mean()),
        "sigma": float(returns.std() if returns.std() > 1e-6 else 0.012),
        "range_mu": float(range_pct.mean()),
        "range_sigma": float(range_pct.std() if range_pct.std() > 1e-6 else 0.006),
        "vol_mu": vol_mu,
        "vol_sigma": vol_sigma,
        "last_close": float(df["close"].iloc[-1]),
    }


def _simulate_synthetic_market(
    end_date: datetime,
    trading_days: int,
    seed: int,
    calib: Dict[str, float],
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=end_date, periods=trading_days)
    price = max(calib["last_close"], 1.0)

    regimes = [
        {"mu": calib["mu"] + 0.0004, "sigma": calib["sigma"] * 0.8},
        {"mu": calib["mu"] - 0.0006, "sigma": calib["sigma"] * 1.4},
        {"mu": calib["mu"] + 0.0001, "sigma": calib["sigma"] * 0.6},
    ]
    trans = np.array(
        [
            [0.92, 0.04, 0.04],
            [0.08, 0.86, 0.06],
            [0.06, 0.06, 0.88],
        ]
    )
    regime = int(rng.integers(0, 3))
    rows = []
    for _ in range(trading_days):
        if rng.random() < 0.08:
            regime = int(rng.choice([0, 1, 2], p=trans[regime]))
        mu = regimes[regime]["mu"]
        sigma = regimes[regime]["sigma"]
        shock = rng.standard_t(df=4) * sigma
        jump = rng.normal(0, sigma * 3) if rng.random() < 0.02 else 0.0
        ret = mu + shock + jump

        open_price = price * (1 + rng.normal(0, sigma * 0.4))
        close_price = max(price * (1 + ret), 0.5)

        range_mu = calib["range_mu"]
        range_sigma = calib["range_sigma"]
        intraday = abs(rng.normal(range_mu, range_sigma))
        high = max(open_price, close_price) * (1 + intraday)
        low = min(open_price, close_price) * (1 - intraday)

        volume = np.exp(rng.normal(calib["vol_mu"], calib["vol_sigma"]))

        rows.append((open_price, high, low, close_price, volume))
        price = close_price

    df = pd.DataFrame(
        rows, index=idx, columns=["open", "high", "low", "close", "volume"]
    )
    df["returns"] = df["close"].pct_change()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="LSTM backtest chart")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--window", type=int, default=90)
    parser.add_argument("--cost", type=float, default=0.0005)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a synthetic realistic market path instead of historical prices",
    )
    parser.add_argument(
        "--synthetic-days",
        type=int,
        default=252,
        help="Trading days for synthetic path",
    )
    parser.add_argument(
        "--calib-days",
        type=int,
        default=60,
        help="Recent days to calibrate synthetic market statistics",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(Path(__file__).parent / "models" / "checkpoints"),
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "reports" / "lstm_vs_benchmark.png"),
    )
    args = parser.parse_args()

    configure_logging()

    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
    start_date = (
        datetime.strptime(args.start, "%Y-%m-%d")
        if args.start
        else end_date - timedelta(days=365)
    )
    fetch_start = start_date - timedelta(days=args.window * 2)

    collector = DataCollector()
    if args.synthetic:
        calib = _calibration_stats(
            collector, args.symbol, end_date=end_date, calib_days=args.calib_days
        )
        df = _simulate_synthetic_market(
            end_date=end_date,
            trading_days=args.synthetic_days,
            seed=args.seed,
            calib=calib,
        )
        df = df.loc[df.index >= start_date]
    else:
        df = collector.fetch_ohlcv(args.symbol, fetch_start, end_date)
    df = _prepare_features(df)

    data = df[DEFAULT_FEATURES].dropna()
    close_col = DEFAULT_FEATURES.index("close")

    X, idx = _build_windows(data.values, args.window)
    dates = data.index[idx]
    returns = data["close"].pct_change().values

    trainer = LSTMTrainer(checkpoint_dir=args.checkpoint_dir)
    ckpt = _load_checkpoint_name(args.checkpoint_dir, args.checkpoint)
    trainer.load_checkpoint(ckpt)

    X_scaled = trainer.scaler.transform(X)
    preds = trainer.model.predict(X_scaled)["1d"]["mean"]

    # Align returns to prediction day+1
    pred_dates = dates[:-1]
    pred_returns = returns[idx][1:]
    preds = preds[:-1]

    strat_returns = _strategy_returns(preds, pred_returns, args.cost)
    if args.synthetic:
        bench_returns = (
            data["close"].pct_change().reindex(pred_dates).fillna(0.0).values
        )
    else:
        bench_df = collector.fetch_ohlcv(args.benchmark, pred_dates[0], pred_dates[-1])
        bench_returns = (
            bench_df["close"].pct_change().reindex(pred_dates).fillna(0.0).values
        )

    strat_equity = (1 + pd.Series(strat_returns, index=pred_dates)).cumprod()
    bench_equity = (1 + pd.Series(bench_returns, index=pred_dates)).cumprod()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(strat_equity.index, strat_equity.values, label="LSTM Strategy")
    plt.plot(bench_equity.index, bench_equity.values, label=f"Benchmark ({args.benchmark})")
    title = "LSTM Strategy vs Benchmark"
    if args.synthetic:
        title = "LSTM Strategy vs Synthetic Market"
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)

    print(f"Chart saved: {out_path}")
    print(f"Strategy return: {(strat_equity.iloc[-1] - 1) * 100:.2f}%")
    print(f"Benchmark return: {(bench_equity.iloc[-1] - 1) * 100:.2f}%")


if __name__ == "__main__":
    main()
