"""
Train LSTM on past 25 years and fine-tune on recent data (live-style).
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

def _write_latest(checkpoint_dir: str, base: str, live: str) -> None:
    path = Path(checkpoint_dir) / "latest.json"
    payload = {
        "base_checkpoint": base,
        "live_checkpoint": live,
        "updated_at": datetime.utcnow().isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _prepare_features(df):
    feature_gen = TechnicalFeatureGenerator()
    cleaner = DataCleaner()
    imputer = MissingValueImputer()

    df = cleaner.clean(df)
    df = imputer.impute(df)
    df = feature_gen.generate(df)
    return df


def _build_dataset(df, features: List[str]):
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    data = df[features].dropna().values
    if data.shape[0] < 200:
        raise ValueError("Not enough data for training. Increase lookback period.")

    return data


def _compute_dates(end_date: datetime, years: int) -> datetime:
    return end_date - timedelta(days=365 * years)


def _fetch_long_history(
    collector: DataCollector,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    interval: str = "1d",
    chunk_years: int = 5,
) -> pd.DataFrame:
    def fetch_range(start: datetime, end: datetime, min_days: int = 60) -> pd.DataFrame:
        try:
            return collector.fetch_ohlcv(symbol, start, end, interval=interval)
        except Exception:
            if (end - start).days <= min_days:
                raise
            mid = start + (end - start) / 2
            left = fetch_range(start, mid)
            right = fetch_range(mid + timedelta(days=1), end)
            return pd.concat([left, right]).sort_index()

    frames = []
    cursor = start_date
    while cursor < end_date:
        chunk_end = min(cursor + timedelta(days=365 * chunk_years), end_date)
        df = fetch_range(cursor, chunk_end)
        frames.append(df)
        cursor = chunk_end + timedelta(days=1)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


def train_25y_live(
    symbols: List[str],
    end_date: datetime,
    years: int,
    live_years: int,
    epochs: int,
    live_epochs: int,
    batch_size: int,
    sequence_length: int,
    window_step_size: int,
    checkpoint_dir: str,
) -> None:
    if not HAS_TF:
        print("TensorFlow is not installed. LSTM training requires TensorFlow.")
        print("Install with: pip install tensorflow-cpu")
        return

    collector = DataCollector()

    symbol = symbols[0]
    if len(symbols) > 1:
        print(f"LSTM training currently supports one symbol. Using {symbol}.")

    train_start = _compute_dates(end_date, years)
    live_start = _compute_dates(end_date, live_years)

    print(f"Training window: {train_start.date()} to {end_date.date()}")
    print(f"Live fine-tune window: {live_start.date()} to {end_date.date()}")

    df = _fetch_long_history(collector, symbol, train_start, end_date)
    df = _prepare_features(df)
    data = _build_dataset(df, DEFAULT_FEATURES)
    close_col = DEFAULT_FEATURES.index("close")

    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        sequence_length=sequence_length,
        window_step_size=window_step_size,
    )

    trainer = LSTMTrainer(config=config, checkpoint_dir=checkpoint_dir)
    X_train, y_train, X_val, y_val = trainer.prepare_data(
        data,
        feature_cols=list(range(len(DEFAULT_FEATURES))),
        close_col=close_col,
        fit_scaler=True,
    )

    trainer.train(X_train, y_train, X_val, y_val, verbose=1)
    base_metrics = trainer.evaluate(X_val, y_val, use_uncertainty=True, n_mc_iterations=50)
    base_ckpt = trainer.save_checkpoint()
    print(f"Saved base checkpoint: {base_ckpt}")
    print(base_metrics)

    live_df = _fetch_long_history(collector, symbol, live_start, end_date)
    live_df = _prepare_features(live_df)
    live_data = _build_dataset(live_df, DEFAULT_FEATURES)

    live_config = TrainingConfig(
        epochs=live_epochs,
        batch_size=batch_size,
        sequence_length=sequence_length,
        window_step_size=window_step_size,
    )
    trainer.config = live_config

    X_live, y_live, X_live_val, y_live_val = trainer.prepare_data(
        live_data,
        feature_cols=list(range(len(DEFAULT_FEATURES))),
        close_col=close_col,
        fit_scaler=False,
    )

    trainer.train(X_live, y_live, X_live_val, y_live_val, verbose=1)
    live_metrics = trainer.evaluate(
        X_live_val, y_live_val, use_uncertainty=True, n_mc_iterations=50
    )
    live_ckpt = trainer.save_checkpoint()
    print(f"Saved live checkpoint: {live_ckpt}")
    print(live_metrics)
    _write_latest(checkpoint_dir, base_ckpt, live_ckpt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 25y + live fine-tune")
    parser.add_argument("--symbols", nargs="+", default=["SPY"])
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--years", type=int, default=25)
    parser.add_argument("--live-years", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--live-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=90)
    parser.add_argument("--window-step", type=int, default=5)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(Path(__file__).parent / "models" / "checkpoints"),
    )
    args = parser.parse_args()

    configure_logging()

    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()

    train_25y_live(
        symbols=args.symbols,
        end_date=end_date,
        years=args.years,
        live_years=args.live_years,
        epochs=args.epochs,
        live_epochs=args.live_epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        window_step_size=args.window_step,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
