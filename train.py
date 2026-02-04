"""
Training entrypoint for Quantum Alpha models.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from quantum_alpha.data.collectors.market_data import DataCollector
from quantum_alpha.data.preprocessing.cleaners import DataCleaner
from quantum_alpha.data.preprocessing.imputers import MissingValueImputer
from quantum_alpha.features.technical.indicators import TechnicalFeatureGenerator
from quantum_alpha.models.lstm_v4.trainer import LSTMTrainer, TrainingConfig
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


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_gen = TechnicalFeatureGenerator()
    cleaner = DataCleaner()
    imputer = MissingValueImputer()

    df = cleaner.clean(df)
    df = imputer.impute(df)
    df = feature_gen.generate(df)

    return df


def _build_dataset(df: pd.DataFrame, features: List[str]) -> np.ndarray:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    data = df[features].dropna().values
    if data.shape[0] < 200:
        raise ValueError("Not enough data for training. Increase lookback period.")

    return data


def train_lstm(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    epochs: int,
    batch_size: int,
    sequence_length: int,
    checkpoint_dir: str,
) -> None:
    if not HAS_TF:
        print("TensorFlow is not installed. LSTM training requires TensorFlow.")
        print("Install with: pip install tensorflow or tensorflow-cpu")
        return

    collector = DataCollector()

    symbol = symbols[0]
    if len(symbols) > 1:
        print(f"LSTM training currently supports one symbol. Using {symbol}.")

    df = collector.fetch_ohlcv(symbol, start_date, end_date)
    df = _prepare_features(df)

    data = _build_dataset(df, DEFAULT_FEATURES)
    close_col = DEFAULT_FEATURES.index("close")

    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )

    trainer = LSTMTrainer(config=config, checkpoint_dir=checkpoint_dir)
    X_train, y_train, X_val, y_val = trainer.prepare_data(
        data,
        feature_cols=list(range(len(DEFAULT_FEATURES))),
        close_col=close_col,
    )

    trainer.train(X_train, y_train, X_val, y_val, verbose=1)
    metrics = trainer.evaluate(X_val, y_val, use_uncertainty=True, n_mc_iterations=50)

    checkpoint = trainer.save_checkpoint()
    print(f"Saved checkpoint: {checkpoint}")
    print(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantum Alpha Training")
    parser.add_argument("--model", choices=["lstm"], default="lstm")
    parser.add_argument("--symbols", nargs="+", default=["SPY"])
    parser.add_argument("--start", type=str, required=False)
    parser.add_argument("--end", type=str, required=False)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=90)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(Path(__file__).parent / "models" / "checkpoints"),
    )
    args = parser.parse_args()

    configure_logging()

    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
    start_date = (
        datetime.strptime(args.start, "%Y-%m-%d")
        if args.start
        else end_date.replace(year=end_date.year - 5)
    )

    if args.model == "lstm":
        train_lstm(
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            checkpoint_dir=args.checkpoint_dir,
        )


if __name__ == "__main__":
    main()
