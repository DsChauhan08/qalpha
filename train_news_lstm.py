#!/usr/bin/env python3
"""
News-Driven LSTM Training Script.

End-to-end script that:
1. Fetches historical price data via yfinance
2. Builds sentiment features (real FinBERT or price-proxy)
3. Trains the NewsDrivenLSTM model
4. Evaluates on a held-out validation set
5. Saves checkpoint for use by NewsLSTMStrategy

Usage:
    python train_news_lstm.py
    python train_news_lstm.py --symbols SPY QQQ AAPL --years 10
    python train_news_lstm.py --epochs 150 --seq-len 20

    # With REAL FinBERT sentiment (requires news in DB):
    python train_news_lstm.py --real-sentiment
    python train_news_lstm.py --real-sentiment --fetch-news  # fetch first, then train
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantum_alpha.data.collectors.market_data import DataCollector
from quantum_alpha.data.preprocessing.cleaners import DataCleaner
from quantum_alpha.data.preprocessing.imputers import MissingValueImputer
from quantum_alpha.models.lstm_v4.news_lstm import NewsLSTMConfig
from quantum_alpha.models.lstm_v4.news_trainer import NewsLSTMTrainer


def fetch_price_data(
    symbols: list,
    years: int = 10,
    end_date: datetime = None,
) -> dict:
    """Fetch and clean price data for given symbols."""
    collector = DataCollector()
    cleaner = DataCleaner()
    imputer = MissingValueImputer()

    end_date = end_date or datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    data = {}
    for symbol in symbols:
        try:
            print(f"  Fetching {symbol}...")
            df = collector.fetch_ohlcv(symbol, start_date, end_date)
            df = cleaner.clean(df)
            df = imputer.impute(df)
            if len(df) > 100:
                data[symbol] = df
                print(
                    f"    {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})"
                )
            else:
                print(f"    Skipped: only {len(df)} bars")
        except Exception as e:
            print(f"    FAILED: {e}")

    return data


def train_single_symbol(
    symbol: str,
    price_df: pd.DataFrame,
    config: NewsLSTMConfig,
    buy_threshold: float = 0.005,
    sell_threshold: float = -0.005,
    verbose: int = 1,
) -> dict:
    """Train on a single symbol and return metrics."""
    print(f"\n{'=' * 60}")
    print(f"Training on {symbol} ({len(price_df)} bars)")
    print(f"{'=' * 60}")

    trainer = NewsLSTMTrainer(
        config=config,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )

    # Prepare data
    X_train, y_sig_train, y_conf_train, X_val, y_sig_val, y_conf_val = (
        trainer.prepare_data(price_df, symbol=symbol, val_split=0.2)
    )

    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_val shape:   {X_val.shape}")

    # Train
    history = trainer.train(
        X_train,
        y_sig_train,
        y_conf_train,
        X_val,
        y_sig_val,
        y_conf_val,
        verbose=verbose,
    )

    # Evaluate
    metrics = trainer.evaluate(X_val, y_sig_val, y_conf_val)

    print(f"\n--- Evaluation ({symbol}) ---")
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Trade Accuracy: {metrics['trade_accuracy']:.4f}")
    print(f"Hold Accuracy:  {metrics['hold_accuracy']:.4f}")
    print(f"Selectivity:    {metrics['selectivity']:.4f}")
    print(f"N Trades (val): {metrics['n_trades']}")
    for cls, acc in metrics["class_accuracy"].items():
        print(f"  {cls:>5} acc: {acc:.4f}")

    return {
        "trainer": trainer,
        "metrics": metrics,
        "history": history,
    }


def train_multi_symbol(
    price_data: dict,
    config: NewsLSTMConfig,
    buy_threshold: float = 0.005,
    sell_threshold: float = -0.005,
    forward_period: int = 1,
    verbose: int = 1,
) -> dict:
    """
    Train on concatenated data from multiple symbols.

    Processes each symbol independently (features, labels, windowing)
    then concatenates the windowed sequences for training. This avoids
    cross-symbol contamination in returns and windowing.
    """
    from quantum_alpha.data.collectors.news_collector import (
        NewsCollector,
        SENTIMENT_FEATURE_COLS,
    )

    print(f"\n{'=' * 60}")
    print(f"Multi-symbol training ({len(price_data)} symbols)")
    print(f"{'=' * 60}")

    collector = NewsCollector()
    seq_len = config.sequence_length

    # First pass: collect all raw feature matrices to compute global scaler
    all_raw_features = []
    symbol_data = {}  # symbol -> (features_df, available_cols)

    for symbol, df in price_data.items():
        try:
            features = collector.build_training_features(df, symbol)
            feature_cols = SENTIMENT_FEATURE_COLS
            available = [c for c in feature_cols if c in features.columns]
            if len(available) < 10:
                print(f"  {symbol}: SKIPPED - only {len(available)} features")
                continue

            # Compute forward return PER SYMBOL (critical - not after concat)
            features["forward_return"] = (
                features["close"].pct_change(forward_period).shift(-forward_period)
            )
            features = features.dropna()

            all_raw_features.append(features[available].values)
            symbol_data[symbol] = (features, available)
            print(f"  {symbol}: {len(features)} samples")
        except Exception as e:
            print(f"  {symbol}: FAILED - {e}")

    if not symbol_data:
        raise ValueError("No features could be built from any symbol")

    # Compute global scaler from all symbols
    all_raw = np.concatenate(all_raw_features, axis=0)
    scaler_params = {
        "mean": np.nanmean(all_raw, axis=0),
        "std": np.nanstd(all_raw, axis=0),
    }
    scaler_params["std"] = np.where(
        scaler_params["std"] < 1e-8, 1.0, scaler_params["std"]
    )

    # Second pass: create windowed sequences PER SYMBOL (no cross-symbol windows)
    all_X = []
    all_y_signal = []
    all_y_conf = []

    for symbol, (features, available) in symbol_data.items():
        X_raw = features[available].values
        X_scaled = (X_raw - scaler_params["mean"]) / scaler_params["std"]
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # Labels per symbol
        fwd = features["forward_return"].values
        n_classes = config.n_classes
        if n_classes == 2:
            labels = (fwd > 0).astype(np.int64)
        else:
            labels = np.ones(len(features), dtype=np.int64)
            labels[fwd > buy_threshold] = 2
            labels[fwd < sell_threshold] = 0
        confidence = np.clip(np.abs(fwd) * 20, 0, 1).astype(np.float32)

        # Window per symbol
        n_windows = len(X_scaled) - seq_len
        if n_windows <= 0:
            continue

        X_sym = np.zeros((n_windows, seq_len, X_scaled.shape[1]), dtype=np.float32)
        y_sig_sym = np.zeros(n_windows, dtype=np.int64)
        y_conf_sym = np.zeros(n_windows, dtype=np.float32)

        for i in range(n_windows):
            X_sym[i] = X_scaled[i : i + seq_len]
            y_sig_sym[i] = labels[i + seq_len]
            y_conf_sym[i] = confidence[i + seq_len]

        all_X.append(X_sym)
        all_y_signal.append(y_sig_sym)
        all_y_conf.append(y_conf_sym)

    # Concatenate all symbols' windowed data
    X = np.concatenate(all_X, axis=0)
    y_signal = np.concatenate(all_y_signal, axis=0)
    y_conf = np.concatenate(all_y_conf, axis=0)

    # DO NOT shuffle — use chronological split per symbol.
    # Since data is grouped by symbol and each symbol is sorted by date,
    # we take the last 20% of EACH symbol's windows for validation.
    # This prevents lookahead from overlapping windows.
    train_X_parts, val_X_parts = [], []
    train_ys_parts, val_ys_parts = [], []
    train_yc_parts, val_yc_parts = [], []

    for X_sym, ys_sym, yc_sym in zip(all_X, all_y_signal, all_y_conf):
        n_val = max(1, int(len(X_sym) * 0.2))
        train_X_parts.append(X_sym[:-n_val])
        val_X_parts.append(X_sym[-n_val:])
        train_ys_parts.append(ys_sym[:-n_val])
        val_ys_parts.append(ys_sym[-n_val:])
        train_yc_parts.append(yc_sym[:-n_val])
        val_yc_parts.append(yc_sym[-n_val:])

    X_train = np.concatenate(train_X_parts, axis=0)
    X_val = np.concatenate(val_X_parts, axis=0)
    y_sig_train = np.concatenate(train_ys_parts, axis=0)
    y_sig_val = np.concatenate(val_ys_parts, axis=0)
    y_conf_train = np.concatenate(train_yc_parts, axis=0)
    y_conf_val = np.concatenate(val_yc_parts, axis=0)

    # Shuffle only the training set (not validation — it stays chronological)
    rng = np.random.default_rng(42)
    train_idx = rng.permutation(len(X_train))
    X_train = X_train[train_idx]
    y_sig_train = y_sig_train[train_idx]
    y_conf_train = y_conf_train[train_idx]

    # Update config with actual feature counts
    # Sentiment features come first in SENTIMENT_FEATURE_COLS, then price context
    _sentiment_names = {
        "overnight_gap",
        "overnight_gap_zscore",
        "range_surprise",
        "volume_surprise",
        "returns_accel",
        "sentiment_proxy",
        "sentiment_ma3",
        "sentiment_ma7",
        "sentiment_momentum",
        "news_intensity",
        "fear_greed",
        "return_zscore",
        "vol_regime",
        "trend_strength",
        "vol_price_div",
        "gap_fill_rate",
    }
    # Use the first symbol's available list (all symbols use the same cols)
    first_available = list(symbol_data.values())[0][1]
    n_sent = sum(1 for c in first_available if c in _sentiment_names)
    config.n_sentiment_features = n_sent
    config.n_price_features = X_train.shape[2] - n_sent

    print(f"\nCombined: {len(X_train) + len(X_val)} windowed samples")

    # Split already done above (chronological per symbol)

    for name, y in [("Train", y_sig_train), ("Val", y_sig_val)]:
        total = len(y)
        if config.n_classes == 2:
            downs = (y == 0).sum()
            ups = (y == 1).sum()
            print(
                f"{name}: {total} samples | "
                f"down={downs} ({100 * downs / total:.1f}%) | "
                f"up={ups} ({100 * ups / total:.1f}%)"
            )
        else:
            sells = (y == 0).sum()
            holds = (y == 1).sum()
            buys = (y == 2).sum()
            print(
                f"{name}: {total} samples | "
                f"sell={sells} ({100 * sells / total:.1f}%) | "
                f"hold={holds} ({100 * holds / total:.1f}%) | "
                f"buy={buys} ({100 * buys / total:.1f}%)"
            )

    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_val shape:   {X_val.shape}")

    # Create trainer and set scaler
    trainer = NewsLSTMTrainer(
        config=config,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )
    trainer.scaler_params = scaler_params

    # Train
    history = trainer.train(
        X_train,
        y_sig_train,
        y_conf_train,
        X_val,
        y_sig_val,
        y_conf_val,
        verbose=verbose,
    )

    # Evaluate
    metrics = trainer.evaluate(X_val, y_sig_val, y_conf_val)

    print(f"\n--- Multi-Symbol Evaluation ---")
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Trade Accuracy: {metrics['trade_accuracy']:.4f}")
    print(f"Hold Accuracy:  {metrics['hold_accuracy']:.4f}")
    print(f"Selectivity:    {metrics['selectivity']:.4f}")
    print(f"N Trades (val): {metrics['n_trades']}")

    return {
        "trainer": trainer,
        "metrics": metrics,
        "history": history,
    }


def train_with_real_sentiment(
    price_data: dict,
    config: NewsLSTMConfig,
    buy_threshold: float = 0.005,
    sell_threshold: float = -0.005,
    forward_period: int = 5,
    fetch_news: bool = False,
    verbose: int = 1,
) -> dict:
    """
    Train using REAL FinBERT sentiment features from the sentiment pipeline.

    This replaces price-proxy features with actual news headline sentiment
    scored by FinBERT and stored in SQLite.

    Args:
        price_data: dict of symbol -> price DataFrame
        config: NewsLSTMConfig
        buy_threshold: Forward return threshold for BUY label
        sell_threshold: Forward return threshold for SELL label
        forward_period: Days ahead for labelling
        fetch_news: If True, fetch and score news before training
        verbose: Verbosity level
    """
    from quantum_alpha.data.collectors.sentiment_pipeline import (
        SentimentPipeline,
        REAL_SENTIMENT_FEATURE_COLS,
        ALL_FEATURE_COLS,
    )

    print(f"\n{'=' * 60}")
    print(f"REAL SENTIMENT TRAINING ({len(price_data)} symbols)")
    print(f"Forward period: {forward_period}d | Classes: {config.n_classes}")
    print(f"{'=' * 60}")

    pipeline = SentimentPipeline()

    # Optionally fetch news first
    if fetch_news:
        for symbol in price_data:
            pipeline.collect_and_score(
                symbol, include_gdelt=False, google_lookback="1y"
            )

    print(pipeline.summary())

    seq_len = config.sequence_length
    n_classes = config.n_classes

    # Build features per symbol
    all_X = []
    all_y_signal = []
    all_y_conf = []
    scaler_data_parts = []

    for symbol, price_df in price_data.items():
        print(f"\n--- Building features for {symbol} ---")

        features = pipeline.build_training_features(
            price_df,
            symbol=symbol,
            forward_period=forward_period,
            use_real_sentiment=True,
        )

        if len(features) < seq_len + 10:
            print(f"  SKIP: only {len(features)} samples (need {seq_len + 10})")
            continue

        # Compute forward return + labels
        features["forward_return"] = (
            features["close"].pct_change(forward_period).shift(-forward_period)
        )
        features = features.dropna()

        # Feature columns
        feature_cols = ALL_FEATURE_COLS
        available = [c for c in feature_cols if c in features.columns]
        print(f"  Features: {len(available)} | Samples: {len(features)}")

        if len(available) < 10:
            print(f"  SKIP: only {len(available)} features")
            continue

        X_raw = features[available].values
        scaler_data_parts.append(X_raw)

        # Labels
        fwd = features["forward_return"].values
        if n_classes == 2:
            labels = (fwd > 0).astype(np.int64)
        else:
            labels = np.ones(len(features), dtype=np.int64)
            labels[fwd > buy_threshold] = 2
            labels[fwd < sell_threshold] = 0
        confidence = np.clip(np.abs(fwd) * 20, 0, 1).astype(np.float32)

        # Windowing
        n_windows = len(X_raw) - seq_len
        if n_windows <= 0:
            print(f"  SKIP: not enough data for windowing")
            continue

        X_sym = np.zeros((n_windows, seq_len, len(available)), dtype=np.float32)
        y_sig_sym = np.zeros(n_windows, dtype=np.int64)
        y_conf_sym = np.zeros(n_windows, dtype=np.float32)

        for i in range(n_windows):
            X_sym[i] = X_raw[i : i + seq_len]
            y_sig_sym[i] = labels[i + seq_len]
            y_conf_sym[i] = confidence[i + seq_len]

        all_X.append(X_sym)
        all_y_signal.append(y_sig_sym)
        all_y_conf.append(y_conf_sym)

    if not all_X:
        raise ValueError("No usable data from any symbol")

    # Global scaler
    all_raw = np.concatenate(scaler_data_parts, axis=0)
    scaler_params = {
        "mean": np.nanmean(all_raw, axis=0),
        "std": np.nanstd(all_raw, axis=0),
    }
    scaler_params["std"] = np.where(
        scaler_params["std"] < 1e-8, 1.0, scaler_params["std"]
    )

    # Apply scaling to all windows
    for k in range(len(all_X)):
        for t in range(all_X[k].shape[1]):
            all_X[k][:, t, :] = (
                all_X[k][:, t, :] - scaler_params["mean"]
            ) / scaler_params["std"]
        all_X[k] = np.nan_to_num(all_X[k], nan=0.0)

    # Chronological split per symbol (last 20% of each symbol for val)
    train_X_parts, val_X_parts = [], []
    train_ys_parts, val_ys_parts = [], []
    train_yc_parts, val_yc_parts = [], []

    for X_sym, ys_sym, yc_sym in zip(all_X, all_y_signal, all_y_conf):
        n_val = max(1, int(len(X_sym) * 0.2))
        train_X_parts.append(X_sym[:-n_val])
        val_X_parts.append(X_sym[-n_val:])
        train_ys_parts.append(ys_sym[:-n_val])
        val_ys_parts.append(ys_sym[-n_val:])
        train_yc_parts.append(yc_sym[:-n_val])
        val_yc_parts.append(yc_sym[-n_val:])

    X_train = np.concatenate(train_X_parts, axis=0)
    X_val = np.concatenate(val_X_parts, axis=0)
    y_sig_train = np.concatenate(train_ys_parts, axis=0)
    y_sig_val = np.concatenate(val_ys_parts, axis=0)
    y_conf_train = np.concatenate(train_yc_parts, axis=0)
    y_conf_val = np.concatenate(val_yc_parts, axis=0)

    # Shuffle training set only
    rng = np.random.default_rng(42)
    train_idx = rng.permutation(len(X_train))
    X_train = X_train[train_idx]
    y_sig_train = y_sig_train[train_idx]
    y_conf_train = y_conf_train[train_idx]

    # Update config dimensions
    n_sent = len(REAL_SENTIMENT_FEATURE_COLS)
    config.n_sentiment_features = n_sent
    config.n_price_features = X_train.shape[2] - n_sent

    print(
        f"\nFeatures: {n_sent} sentiment + {config.n_price_features} price = {X_train.shape[2]} total"
    )
    print(f"X_train: {X_train.shape} | X_val: {X_val.shape}")

    for name, y in [("Train", y_sig_train), ("Val", y_sig_val)]:
        total = len(y)
        if n_classes == 2:
            downs = (y == 0).sum()
            ups = (y == 1).sum()
            print(
                f"{name}: {total} | down={downs} ({100 * downs / total:.1f}%) up={ups} ({100 * ups / total:.1f}%)"
            )
        else:
            sells = (y == 0).sum()
            holds = (y == 1).sum()
            buys = (y == 2).sum()
            print(
                f"{name}: {total} | sell={sells} ({100 * sells / total:.1f}%) hold={holds} ({100 * holds / total:.1f}%) buy={buys} ({100 * buys / total:.1f}%)"
            )

    # Train
    trainer = NewsLSTMTrainer(
        config=config,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )
    trainer.scaler_params = scaler_params

    history = trainer.train(
        X_train,
        y_sig_train,
        y_conf_train,
        X_val,
        y_sig_val,
        y_conf_val,
        verbose=verbose,
    )

    # Evaluate
    metrics = trainer.evaluate(X_val, y_sig_val, y_conf_val)

    print(f"\n--- Real Sentiment Evaluation ---")
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Trade Accuracy: {metrics['trade_accuracy']:.4f}")
    print(f"Hold Accuracy:  {metrics['hold_accuracy']:.4f}")
    print(f"Selectivity:    {metrics['selectivity']:.4f}")
    print(f"N Trades (val): {metrics['n_trades']}")
    for cls, acc in metrics["class_accuracy"].items():
        print(f"  {cls:>5} acc: {acc:.4f}")

    return {
        "trainer": trainer,
        "metrics": metrics,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Train the News-Driven LSTM model")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY"],
        help="Symbols to train on (default: SPY)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=10,
        help="Years of historical data (default: 10)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (default: 100)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=30,
        help="Sequence length in days (default: 30)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--buy-threshold",
        type=float,
        default=0.005,
        help="Forward return threshold for BUY label (default: 0.005 = 0.5%%)",
    )
    parser.add_argument(
        "--sell-threshold",
        type=float,
        default=-0.005,
        help="Forward return threshold for SELL label (default: -0.005 = -0.5%%)",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Train on all symbols combined (multi-symbol training)",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=3,
        choices=[2, 3],
        help="Number of classes: 2=binary (down/up), 3=ternary (sell/hold/buy) (default: 3)",
    )
    parser.add_argument(
        "--forward-period",
        type=int,
        default=1,
        help="Forward return period in days for labelling (default: 1)",
    )
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="Checkpoint name (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0=silent, 1=progress, 2=debug)",
    )
    parser.add_argument(
        "--real-sentiment",
        action="store_true",
        help="Use real FinBERT sentiment from news DB (instead of price proxies)",
    )
    parser.add_argument(
        "--fetch-news",
        action="store_true",
        help="Fetch and score news before training (requires --real-sentiment)",
    )

    args = parser.parse_args()

    print(f"\n{'#' * 60}")
    print("# NEWS-DRIVEN LSTM TRAINING")
    print(f"{'#' * 60}")
    print(f"Symbols:        {args.symbols}")
    print(f"Years:          {args.years}")
    print(f"Epochs:         {args.epochs}")
    print(f"Sequence len:   {args.seq_len}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Buy threshold:  {args.buy_threshold}")
    print(f"Sell threshold: {args.sell_threshold}")
    print(f"Multi-symbol:   {args.multi}")
    print(
        f"N classes:      {args.n_classes} ({'binary' if args.n_classes == 2 else 'ternary'})"
    )
    print(f"Forward period: {args.forward_period} day(s)")
    print(f"Real sentiment: {args.real_sentiment}")
    print(f"Fetch news:     {args.fetch_news}")

    # Build config
    config = NewsLSTMConfig(
        sequence_length=args.seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_classes=args.n_classes,
    )

    # Fetch price data
    print(f"\n--- Fetching Price Data ---")
    # Limit to 1 year for real sentiment (that's our news coverage)
    years = 1 if args.real_sentiment else args.years
    if args.real_sentiment and args.years != 10:
        years = args.years  # User explicitly set years, respect it
    price_data = fetch_price_data(args.symbols, years=years)

    if not price_data:
        print("ERROR: No price data fetched. Exiting.")
        sys.exit(1)

    # Train
    if args.real_sentiment:
        result = train_with_real_sentiment(
            price_data,
            config,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
            forward_period=args.forward_period,
            fetch_news=args.fetch_news,
            verbose=args.verbose,
        )
    elif args.multi and len(price_data) > 1:
        result = train_multi_symbol(
            price_data,
            config,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
            forward_period=args.forward_period,
            verbose=args.verbose,
        )
    else:
        # Train on the first (or only) symbol
        symbol = list(price_data.keys())[0]
        result = train_single_symbol(
            symbol,
            price_data[symbol],
            config,
            buy_threshold=args.buy_threshold,
            sell_threshold=args.sell_threshold,
            verbose=args.verbose,
        )

    # Save checkpoint
    trainer = result["trainer"]
    ckpt_name = (
        args.checkpoint_name or f"news_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    saved_name = trainer.save_checkpoint(name=ckpt_name)

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Checkpoint: {saved_name}")
    print(f"Location:   {trainer.checkpoint_dir}")
    print(f"\nTo use in backtesting:")
    print(f"  python backtest_news_lstm.py --checkpoint {saved_name}")
    print(f"\nTo use in code:")
    print(f"  from quantum_alpha.strategy.news_lstm_strategy import NewsLSTMStrategy")
    print(f'  strategy = NewsLSTMStrategy(checkpoint_name="{saved_name}")')


if __name__ == "__main__":
    main()
