"""
Rolling intraday backtest for a single LSTM checkpoint.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

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
    cleaner = DataCleaner()
    imputer = MissingValueImputer()
    feature_gen = TechnicalFeatureGenerator()
    df = cleaner.clean(df)
    df = imputer.impute(df)
    df = feature_gen.generate(df)
    return df


def _load_checkpoint_name(checkpoint_dir: str, checkpoint: str | None, checkpoint_file: str | None) -> str:
    if checkpoint:
        return checkpoint
    if checkpoint_file:
        latest_path = Path(checkpoint_file)
    else:
        latest_path = Path(checkpoint_dir) / "latest_intraday.json"
    if latest_path.exists():
        payload = json.loads(latest_path.read_text())
        value = payload.get("intraday_checkpoint")
        if isinstance(value, str) and value:
            return value
    raise ValueError("Checkpoint not provided and latest_intraday.json not found")


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


def run_backtest(
    symbol: str,
    interval: str,
    lookback_days: int,
    checkpoint_dir: str,
    checkpoint: str | None,
    checkpoint_file: str | None,
    horizon: str,
    window: int | None,
    cost_bps: float,
    scale: float,
    output: str,
    csv_path: str,
) -> Dict[str, float]:
    configure_logging()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    collector = DataCollector()
    df = collector.fetch_ohlcv(symbol, start_date, end_date, interval=interval, use_cache=False)
    df = _prepare_features(df)

    data = df[DEFAULT_FEATURES].dropna()
    if data.empty:
        raise ValueError("No data after feature preparation")

    trainer = LSTMTrainer(checkpoint_dir=checkpoint_dir)
    ckpt = _load_checkpoint_name(checkpoint_dir, checkpoint, checkpoint_file)
    trainer.load_checkpoint(ckpt)

    window_size = window or trainer.config.sequence_length
    X, idx = _build_windows(data.values, window_size)
    if len(X) < 5:
        raise ValueError("Not enough windows for backtest")

    dates = data.index[idx]
    returns = data["close"].pct_change().values

    X_scaled = trainer.scaler.transform(X)
    preds = trainer.model.predict(X_scaled)
    if horizon not in preds:
        raise ValueError(f"Horizon {horizon} not in checkpoint predictions")

    mean = preds[horizon]["mean"]
    std = preds[horizon]["std"]
    score = mean / (std + 1e-6)
    score = np.tanh(score * scale)

    pred_dates = dates[:-1]
    pred_returns = returns[idx][1:]
    score = score[:-1]

    strat_returns = _strategy_returns(score, pred_returns, cost_bps / 10000.0)
    bench_returns = data["close"].pct_change().reindex(pred_dates).fillna(0.0).values

    strat_equity = (1 + pd.Series(strat_returns, index=pred_dates)).cumprod()
    bench_equity = (1 + pd.Series(bench_returns, index=pred_dates)).cumprod()

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(strat_equity.index, strat_equity.values, label="LSTM Intraday")
    plt.plot(bench_equity.index, bench_equity.values, label=f"Benchmark ({symbol})")
    plt.title(f"Intraday LSTM Backtest ({symbol})")
    plt.xlabel("Timestamp")
    plt.ylabel("Equity (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)

    csv_out = Path(csv_path)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(
        {
            "timestamp": pred_dates,
            "signal": score,
            "strategy_return": strat_returns,
            "bench_return": bench_returns[: len(strat_returns)],
        }
    ).set_index("timestamp")
    result_df.to_csv(csv_out)

    stats = {
        "strategy_return": float(strat_equity.iloc[-1] - 1),
        "benchmark_return": float(bench_equity.iloc[-1] - 1),
        "n_bars": int(len(strat_returns)),
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Intraday LSTM backtest")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--lookback-days", type=int, default=20)
    parser.add_argument("--checkpoint-dir", type=str, default=str(Path(__file__).parent / "models" / "intraday_checkpoints" / "spy"))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-file", type=str, default=None)
    parser.add_argument("--horizon", type=str, default="1h")
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--cost-bps", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=4.0)
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "reports" / "intraday_backtests" / "intraday_backtest.png"),
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(Path(__file__).parent / "reports" / "intraday_backtests" / "intraday_backtest.csv"),
    )
    args = parser.parse_args()

    stats = run_backtest(
        symbol=args.symbol,
        interval=args.interval,
        lookback_days=args.lookback_days,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint=args.checkpoint,
        checkpoint_file=args.checkpoint_file,
        horizon=args.horizon,
        window=args.window,
        cost_bps=args.cost_bps,
        scale=args.scale,
        output=args.output,
        csv_path=args.csv,
    )

    print(f"Chart saved: {args.output}")
    print(f"Strategy return: {stats['strategy_return'] * 100:.2f}%")
    print(f"Benchmark return: {stats['benchmark_return'] * 100:.2f}%")


if __name__ == "__main__":
    main()
