"""
Train an intraday LSTM on 5m data for live signals.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd

from quantum_alpha.data.collectors.market_data import DataCollector
from quantum_alpha.features.technical.indicators import TechnicalFeatureGenerator
from quantum_alpha.data.preprocessing.cleaners import DataCleaner
from quantum_alpha.data.preprocessing.imputers import MissingValueImputer
from quantum_alpha.models.lstm_v4.trainer import LSTMTrainer, TrainingConfig
from quantum_alpha.models.lstm_v4.architecture import MultiHorizonLSTM, HorizonConfig
from quantum_alpha.models.lstm_v4.architecture import HAS_TF
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


def _write_latest(checkpoint_dir: str, checkpoint: str, interval: str) -> None:
    path = Path(checkpoint_dir) / "latest_intraday.json"
    payload = {
        "intraday_checkpoint": checkpoint,
        "interval": interval,
        "updated_at": datetime.utcnow().isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_gen = TechnicalFeatureGenerator()
    cleaner = DataCleaner()
    imputer = MissingValueImputer()
    df = cleaner.clean(df)
    df = imputer.impute(df)
    df = feature_gen.generate(df)
    return df


def _build_dataset(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    data = df[features].dropna().values
    if data.shape[0] < 500:
        raise ValueError("Not enough intraday data for training. Increase lookback.")
    return data


def _fetch_intraday_history(
    collector: DataCollector,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    interval: str,
    chunk_days: int,
) -> pd.DataFrame:
    def fetch_range(start: datetime, end: datetime, min_days: int = 1) -> pd.DataFrame:
        try:
            return collector.fetch_ohlcv(symbol, start, end, interval=interval, use_cache=False)
        except Exception:
            if (end - start).days <= min_days:
                return pd.DataFrame()
            mid = start + (end - start) / 2
            left = fetch_range(start, mid)
            right = fetch_range(mid + timedelta(minutes=1), end)
            frames = [frame for frame in (left, right) if not frame.empty]
            if not frames:
                return pd.DataFrame()
            return pd.concat(frames).sort_index()

    frames = []
    cursor = start_date
    while cursor < end_date:
        chunk_end = min(cursor + timedelta(days=chunk_days), end_date)
        df = fetch_range(cursor, chunk_end)
        frames.append(df)
        cursor = chunk_end + timedelta(minutes=1)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


def _intraday_horizons(interval: str) -> List[HorizonConfig]:
    interval = interval.lower().strip()
    if interval.endswith("m"):
        try:
            minutes = int(interval[:-1])
        except ValueError:
            minutes = 5
        bars_per_day = max(1, int(390 / max(minutes, 1)))
        one_hour = max(1, int(60 / max(minutes, 1)))
        four_hour = max(1, int(240 / max(minutes, 1)))
        return [
            HorizonConfig("1h", one_hour, 1.0),
            HorizonConfig("4h", four_hour, 1.0),
            HorizonConfig("1d", bars_per_day, 1.0),
            HorizonConfig("5d", bars_per_day * 5, 1.0),
        ]
    return [
        HorizonConfig("1d", 1, 1.0),
        HorizonConfig("1w", 5, 1.0),
        HorizonConfig("1m", 21, 1.0),
        HorizonConfig("6m", 126, 1.0),
    ]


def train_intraday(
    symbol: str,
    end_date: datetime,
    lookback_days: int,
    interval: str,
    epochs: int,
    batch_size: int,
    sequence_length: int,
    window_step_size: int,
    checkpoint_dir: str,
    chunk_days: int,
) -> None:
    if not HAS_TF:
        print("TensorFlow is not installed. LSTM training requires TensorFlow.")
        print("Install with: pip install tensorflow-cpu")
        return

    collector = DataCollector()
    start_date = end_date - timedelta(days=lookback_days)
    print(f"Intraday training window: {start_date} to {end_date} ({interval})")

    df = _fetch_intraday_history(
        collector, symbol, start_date=start_date, end_date=end_date, interval=interval, chunk_days=chunk_days
    )
    if df.empty:
        raise ValueError("No intraday data returned. Check interval and lookback.")

    df = _prepare_features(df)
    data = _build_dataset(df, DEFAULT_FEATURES)
    close_col = DEFAULT_FEATURES.index("close")

    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        sequence_length=sequence_length,
        window_step_size=window_step_size,
    )

    horizons = _intraday_horizons(interval)
    model = MultiHorizonLSTM(
        input_dim=data.shape[1],
        sequence_length=sequence_length,
        horizons=horizons,
    )
    trainer = LSTMTrainer(model=model, config=config, checkpoint_dir=checkpoint_dir)
    X_train, y_train, X_val, y_val = trainer.prepare_data(
        data,
        feature_cols=list(range(len(DEFAULT_FEATURES))),
        close_col=close_col,
        horizons=horizons,
        fit_scaler=True,
    )

    trainer.train(X_train, y_train, X_val, y_val, verbose=1)
    metrics = trainer.evaluate(X_val, y_val, use_uncertainty=True, n_mc_iterations=50)
    ckpt = trainer.save_checkpoint()
    print(f"Saved intraday checkpoint: {ckpt}")
    print(metrics)
    _write_latest(checkpoint_dir, ckpt, interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train intraday LSTM on 5m data")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument("--interval", type=str, default="5m")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=90)
    parser.add_argument("--window-step", type=int, default=5)
    parser.add_argument("--chunk-days", type=int, default=7)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(Path(__file__).parent / "models" / "checkpoints"),
    )
    args = parser.parse_args()

    configure_logging()

    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()

    train_intraday(
        symbol=args.symbol,
        end_date=end_date,
        lookback_days=args.lookback_days,
        interval=args.interval,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        window_step_size=args.window_step,
        checkpoint_dir=args.checkpoint_dir,
        chunk_days=args.chunk_days,
    )


if __name__ == "__main__":
    main()
