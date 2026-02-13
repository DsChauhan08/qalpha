#!/usr/bin/env python3
"""
Baseline Model Comparison for News Sentiment Trading.

Trains simpler models (logistic regression, random forest, gradient boosting)
on the SAME real FinBERT sentiment features as the LSTM, using the SAME
chronological split, to test whether the LSTM is adding value over
simpler approaches.

With <1000 training samples, simpler models often outperform deep learning.

Usage:
    python train_baselines.py
    python train_baselines.py --symbols SPY QQQ AAPL MSFT NVDA TSLA
    python train_baselines.py --forward-period 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def prepare_data(
    symbols: list[str],
    years: int = 1,
    forward_period: int = 5,
    seq_len: int = 1,
    n_classes: int = 2,
    buy_threshold: float = 0.005,
    sell_threshold: float = -0.005,
) -> dict:
    """
    Prepare data for baseline models using same pipeline as LSTM training.

    For non-sequential models (LR, RF, GB), we use seq_len=1 (single-day features).
    For sequential baselines, we flatten the sequence into a single feature vector.

    Returns dict with X_train, X_val, y_train, y_val, feature_names, scaler_params
    """
    from quantum_alpha.data.collectors.sentiment_pipeline import (
        SentimentPipeline,
        REAL_SENTIMENT_FEATURE_COLS,
        ALL_FEATURE_COLS,
    )
    from quantum_alpha.data.collectors.market_data import DataCollector

    pipeline = SentimentPipeline()
    collector = DataCollector()

    end = datetime.now()
    start = end - timedelta(days=years * 365)

    scaler_data_parts = []
    all_X = []
    all_y = []

    for symbol in symbols:
        print(f"\n--- {symbol} ---")
        price_df = collector.fetch_ohlcv(symbol, start=start, end=end)
        if price_df is None or len(price_df) < 60:
            print(
                f"  SKIP: insufficient price data ({len(price_df) if price_df is not None else 0} bars)"
            )
            continue

        features = pipeline.build_training_features(
            price_df,
            symbol=symbol,
            forward_period=forward_period,
            use_real_sentiment=True,
        )

        if len(features) < 30:
            print(f"  SKIP: only {len(features)} samples")
            continue

        # Forward return + labels
        features["forward_return"] = (
            features["close"].pct_change(forward_period).shift(-forward_period)
        )
        features = features.dropna()

        available = [c for c in ALL_FEATURE_COLS if c in features.columns]
        print(f"  Features: {len(available)} | Samples: {len(features)}")

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

        if seq_len > 1:
            # Flatten sequence windows into a single feature vector
            n_windows = len(X_raw) - seq_len
            if n_windows <= 0:
                print(f"  SKIP: not enough data for seq_len={seq_len}")
                continue
            X_seq = np.zeros((n_windows, seq_len * len(available)), dtype=np.float32)
            y_seq = np.zeros(n_windows, dtype=np.int64)
            for i in range(n_windows):
                X_seq[i] = X_raw[i : i + seq_len].flatten()
                y_seq[i] = labels[i + seq_len]
            all_X.append(X_seq)
            all_y.append(y_seq)
        else:
            all_X.append(X_raw.astype(np.float32))
            all_y.append(labels)

    if not all_X:
        raise ValueError("No usable data from any symbol")

    # Global scaler (on raw features, not flattened)
    all_raw = np.concatenate(scaler_data_parts, axis=0)
    scaler_mean = np.nanmean(all_raw, axis=0)
    scaler_std = np.nanstd(all_raw, axis=0)
    scaler_std = np.where(scaler_std < 1e-8, 1.0, scaler_std)

    # Apply scaling
    for k in range(len(all_X)):
        if seq_len > 1:
            n_feat = len(scaler_mean)
            for t in range(seq_len):
                s = t * n_feat
                e = s + n_feat
                all_X[k][:, s:e] = (all_X[k][:, s:e] - scaler_mean) / scaler_std
        else:
            all_X[k] = (all_X[k] - scaler_mean) / scaler_std
        all_X[k] = np.nan_to_num(all_X[k], nan=0.0)

    # Chronological split per symbol (last 20% for val)
    train_X_parts, val_X_parts = [], []
    train_y_parts, val_y_parts = [], []

    for X_sym, y_sym in zip(all_X, all_y):
        n_val = max(1, int(len(X_sym) * 0.2))
        train_X_parts.append(X_sym[:-n_val])
        val_X_parts.append(X_sym[-n_val:])
        train_y_parts.append(y_sym[:-n_val])
        val_y_parts.append(y_sym[-n_val:])

    X_train = np.concatenate(train_X_parts, axis=0)
    X_val = np.concatenate(val_X_parts, axis=0)
    y_train = np.concatenate(train_y_parts, axis=0)
    y_val = np.concatenate(val_y_parts, axis=0)

    # Shuffle training only
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_train))
    X_train = X_train[idx]
    y_train = y_train[idx]

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "scaler_params": {"mean": scaler_mean.tolist(), "std": scaler_std.tolist()},
        "feature_names": available
        if seq_len == 1
        else [f"t{t}_{f}" for t in range(seq_len) for f in available],
    }


def train_logistic_regression(X_train, y_train, X_val, y_val, n_classes):
    """Logistic Regression baseline."""
    from sklearn.linear_model import LogisticRegression

    results = {}
    for C in [0.01, 0.1, 1.0, 10.0]:
        model = LogisticRegression(
            C=C,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )
        model.fit(X_train, y_train)
        val_acc = model.score(X_val, y_val)
        results[C] = {"model": model, "val_acc": val_acc}

    # Pick best C
    best_C = max(results, key=lambda c: results[c]["val_acc"])
    best = results[best_C]
    return best["model"], best["val_acc"], {"C": best_C}


def train_random_forest(X_train, y_train, X_val, y_val, n_classes):
    """Random Forest baseline."""
    from sklearn.ensemble import RandomForestClassifier

    results = {}
    configs = [
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 8},
        {"n_estimators": 300, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 10},
    ]
    for i, cfg in enumerate(configs):
        model = RandomForestClassifier(**cfg, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        val_acc = model.score(X_val, y_val)
        results[i] = {"model": model, "val_acc": val_acc, "config": cfg}

    best_i = max(results, key=lambda i: results[i]["val_acc"])
    best = results[best_i]
    return best["model"], best["val_acc"], best["config"]


def train_gradient_boosting(X_train, y_train, X_val, y_val, n_classes):
    """Gradient Boosting (HistGradientBoosting) baseline."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    results = {}
    configs = [
        {"max_depth": 3, "learning_rate": 0.1, "max_iter": 100},
        {"max_depth": 4, "learning_rate": 0.05, "max_iter": 200},
        {
            "max_depth": 3,
            "learning_rate": 0.05,
            "max_iter": 300,
            "min_samples_leaf": 10,
        },
        {"max_depth": 5, "learning_rate": 0.01, "max_iter": 500},
    ]
    for i, cfg in enumerate(configs):
        model = HistGradientBoostingClassifier(**cfg, random_state=42)
        model.fit(X_train, y_train)
        val_acc = model.score(X_val, y_val)
        results[i] = {"model": model, "val_acc": val_acc, "config": cfg}

    best_i = max(results, key=lambda i: results[i]["val_acc"])
    best = results[best_i]
    return best["model"], best["val_acc"], best["config"]


def train_simple_threshold(X_train, y_train, X_val, y_val, feature_names):
    """
    Simple threshold rule: if 3-day sentiment momentum > threshold, go long.
    Scans thresholds on training set, evaluates on val.
    """
    # Find the sentiment_momentum_3d feature index
    target_features = ["sentiment_momentum_3d", "mean_sentiment", "weighted_sentiment"]
    results = {}

    for feat_name in target_features:
        if feat_name not in feature_names:
            continue
        feat_idx = feature_names.index(feat_name)
        X_feat_train = X_train[:, feat_idx]
        X_feat_val = X_val[:, feat_idx]

        best_thresh_acc = 0.0
        best_thresh = 0.0
        for thresh in np.linspace(-1.0, 1.0, 41):
            preds = (X_feat_train > thresh).astype(int)
            acc = (preds == y_train).mean()
            if acc > best_thresh_acc:
                best_thresh_acc = acc
                best_thresh = thresh

        # Evaluate on val
        val_preds = (X_feat_val > best_thresh).astype(int)
        val_acc = (val_preds == y_val).mean()
        results[feat_name] = {
            "threshold": best_thresh,
            "train_acc": best_thresh_acc,
            "val_acc": val_acc,
        }

    return results


def evaluate_detailed(model, X_val, y_val, n_classes, name="Model"):
    """Detailed evaluation with per-class accuracy."""
    y_pred = model.predict(X_val)
    acc = (y_pred == y_val).mean()

    if n_classes == 2:
        class_names = ["down", "up"]
    else:
        class_names = ["sell", "hold", "buy"]

    class_acc = {}
    for c, cname in enumerate(class_names):
        mask = y_val == c
        if mask.sum() > 0:
            class_acc[cname] = (y_pred[mask] == c).mean()
        else:
            class_acc[cname] = float("nan")

    # Majority class baseline
    unique, counts = np.unique(y_val, return_counts=True)
    majority_class = unique[counts.argmax()]
    majority_name = class_names[majority_class]
    baseline_acc = counts.max() / len(y_val)
    edge = acc - baseline_acc

    return {
        "accuracy": acc,
        "class_accuracy": class_acc,
        "baseline_acc": baseline_acc,
        "majority_class": majority_name,
        "edge": edge,
        "n_val": len(y_val),
        "predictions": y_pred,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline models on real sentiment features"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA"],
        help="Symbols (default: SPY QQQ AAPL MSFT NVDA TSLA)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=1,
        help="Years of data (default: 1, matching news coverage)",
    )
    parser.add_argument(
        "--forward-period",
        type=int,
        default=5,
        help="Forward return period in days (default: 5, matching v14)",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=2,
        choices=[2, 3],
        help="2=binary (down/up), 3=ternary (sell/hold/buy) (default: 2)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1,
        help="Sequence length: 1=single day features (default), >1=flattened window",
    )

    args = parser.parse_args()

    print(f"\n{'#' * 60}")
    print("# BASELINE MODEL COMPARISON")
    print(f"# Real FinBERT Sentiment Features")
    print(f"{'#' * 60}")
    print(f"Symbols:        {args.symbols}")
    print(f"Forward period: {args.forward_period}d")
    print(f"Classes:        {args.n_classes}")
    print(f"Seq length:     {args.seq_len}")

    # Prepare data
    print(f"\n{'=' * 60}")
    print("DATA PREPARATION")
    print(f"{'=' * 60}")

    data = prepare_data(
        symbols=args.symbols,
        years=args.years,
        forward_period=args.forward_period,
        seq_len=args.seq_len,
        n_classes=args.n_classes,
    )

    X_train = data["X_train"]
    X_val = data["X_val"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    feature_names = data["feature_names"]

    print(f"\nX_train: {X_train.shape} | X_val: {X_val.shape}")

    if args.n_classes == 2:
        for name, y in [("Train", y_train), ("Val", y_val)]:
            total = len(y)
            downs = (y == 0).sum()
            ups = (y == 1).sum()
            print(
                f"{name}: {total} | down={downs} ({100 * downs / total:.1f}%) up={ups} ({100 * ups / total:.1f}%)"
            )
    else:
        for name, y in [("Train", y_train), ("Val", y_val)]:
            total = len(y)
            sells = (y == 0).sum()
            holds = (y == 1).sum()
            buys = (y == 2).sum()
            print(
                f"{name}: {total} | sell={sells} ({100 * sells / total:.1f}%) hold={holds} ({100 * holds / total:.1f}%) buy={buys} ({100 * buys / total:.1f}%)"
            )

    # Majority class baseline
    unique, counts = np.unique(y_val, return_counts=True)
    majority = unique[counts.argmax()]
    baseline_acc = counts.max() / len(y_val)
    class_names = ["down", "up"] if args.n_classes == 2 else ["sell", "hold", "buy"]
    print(f"\nMajority class baseline: {class_names[majority]} = {baseline_acc:.1%}")

    # ================================================================
    # TRAIN ALL BASELINES
    # ================================================================
    results_table = []

    # 1. Logistic Regression
    print(f"\n{'=' * 60}")
    print("1. LOGISTIC REGRESSION")
    print(f"{'=' * 60}")
    t0 = time.time()
    lr_model, lr_acc, lr_cfg = train_logistic_regression(
        X_train, y_train, X_val, y_val, args.n_classes
    )
    lr_time = time.time() - t0
    lr_eval = evaluate_detailed(lr_model, X_val, y_val, args.n_classes, "LogReg")
    print(
        f"  Best C={lr_cfg['C']} | Val acc: {lr_eval['accuracy']:.1%} | Edge: {lr_eval['edge']:+.1%} | Time: {lr_time:.1f}s"
    )
    for cls, acc in lr_eval["class_accuracy"].items():
        print(f"    {cls:>5} acc: {acc:.1%}")
    results_table.append(
        ("Logistic Regression", lr_eval["accuracy"], lr_eval["edge"], lr_cfg)
    )

    # Feature importance (coefficients)
    if args.seq_len == 1 and hasattr(lr_model, "coef_"):
        coefs = (
            np.abs(lr_model.coef_).mean(axis=0)
            if lr_model.coef_.ndim > 1
            else np.abs(lr_model.coef_[0])
        )
        top_idx = np.argsort(coefs)[::-1][:10]
        print("  Top features by |coefficient|:")
        for i in top_idx:
            print(f"    {feature_names[i]:30s}  {coefs[i]:.4f}")

    # 2. Random Forest
    print(f"\n{'=' * 60}")
    print("2. RANDOM FOREST")
    print(f"{'=' * 60}")
    t0 = time.time()
    rf_model, rf_acc, rf_cfg = train_random_forest(
        X_train, y_train, X_val, y_val, args.n_classes
    )
    rf_time = time.time() - t0
    rf_eval = evaluate_detailed(rf_model, X_val, y_val, args.n_classes, "RF")
    print(
        f"  Best config: {rf_cfg} | Val acc: {rf_eval['accuracy']:.1%} | Edge: {rf_eval['edge']:+.1%} | Time: {rf_time:.1f}s"
    )
    for cls, acc in rf_eval["class_accuracy"].items():
        print(f"    {cls:>5} acc: {acc:.1%}")
    results_table.append(
        ("Random Forest", rf_eval["accuracy"], rf_eval["edge"], rf_cfg)
    )

    # Feature importance
    if args.seq_len == 1 and hasattr(rf_model, "feature_importances_"):
        imp = rf_model.feature_importances_
        top_idx = np.argsort(imp)[::-1][:10]
        print("  Top features by importance:")
        for i in top_idx:
            print(f"    {feature_names[i]:30s}  {imp[i]:.4f}")

    # 3. Gradient Boosting
    print(f"\n{'=' * 60}")
    print("3. GRADIENT BOOSTING")
    print(f"{'=' * 60}")
    t0 = time.time()
    gb_model, gb_acc, gb_cfg = train_gradient_boosting(
        X_train, y_train, X_val, y_val, args.n_classes
    )
    gb_time = time.time() - t0
    gb_eval = evaluate_detailed(gb_model, X_val, y_val, args.n_classes, "GB")
    print(
        f"  Best config: {gb_cfg} | Val acc: {gb_eval['accuracy']:.1%} | Edge: {gb_eval['edge']:+.1%} | Time: {gb_time:.1f}s"
    )
    for cls, acc in gb_eval["class_accuracy"].items():
        print(f"    {cls:>5} acc: {acc:.1%}")
    results_table.append(
        ("Gradient Boosting", gb_eval["accuracy"], gb_eval["edge"], gb_cfg)
    )

    # 4. Simple threshold rules (binary only)
    if args.n_classes == 2 and args.seq_len == 1:
        print(f"\n{'=' * 60}")
        print("4. SIMPLE THRESHOLD RULES")
        print(f"{'=' * 60}")
        thresh_results = train_simple_threshold(
            X_train, y_train, X_val, y_val, feature_names
        )
        for feat, res in thresh_results.items():
            edge = res["val_acc"] - baseline_acc
            print(
                f"  {feat}: threshold={res['threshold']:.3f} | Train: {res['train_acc']:.1%} | Val: {res['val_acc']:.1%} | Edge: {edge:+.1%}"
            )
            results_table.append(
                (
                    f"Threshold ({feat})",
                    res["val_acc"],
                    edge,
                    {"threshold": res["threshold"]},
                )
            )

    # ================================================================
    # COMPARISON TABLE
    # ================================================================
    print(f"\n{'=' * 60}")
    print("COMPARISON TABLE")
    print(f"{'=' * 60}")
    print(f"{'Model':<35} {'Val Acc':>8} {'Edge':>8} {'vs LSTM v14':>12}")
    print(f"{'-' * 35} {'-' * 8} {'-' * 8} {'-' * 12}")

    lstm_v14_acc = 0.600
    lstm_v14_edge = 0.081

    # Add LSTM v14 reference
    print(
        f"{'LSTM v14 (reference)':<35} {lstm_v14_acc:>7.1%} {lstm_v14_edge:>+7.1%} {'---':>12}"
    )

    # Majority baseline
    print(
        f"{'Majority class (always ' + class_names[majority] + ')':<35} {baseline_acc:>7.1%} {'+0.0%':>8} {baseline_acc - lstm_v14_acc:>+11.1%}"
    )

    for name, acc, edge, cfg in results_table:
        vs_lstm = acc - lstm_v14_acc
        print(f"{name:<35} {acc:>7.1%} {edge:>+7.1%} {vs_lstm:>+11.1%}")

    # Best model
    if results_table:
        best = max(results_table, key=lambda x: x[1])
        print(f"\nBest baseline: {best[0]} ({best[1]:.1%}, edge {best[2]:+.1%})")
        if best[1] > lstm_v14_acc:
            print(f"  -> BEATS LSTM v14 by {best[1] - lstm_v14_acc:+.1%}")
        elif best[1] == lstm_v14_acc:
            print(f"  -> TIES with LSTM v14")
        else:
            print(f"  -> LSTM v14 still wins by {lstm_v14_acc - best[1]:+.1%}")

    # Save results
    results_path = (
        Path(__file__).parent / "models" / "checkpoints" / "baseline_results.json"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "symbols": args.symbols,
            "forward_period": args.forward_period,
            "n_classes": args.n_classes,
            "seq_len": args.seq_len,
            "n_train": len(X_train),
            "n_val": len(X_val),
        },
        "baseline_acc": float(baseline_acc),
        "lstm_v14_acc": lstm_v14_acc,
        "results": [
            {"model": name, "val_acc": float(acc), "edge": float(edge)}
            for name, acc, edge, _ in results_table
        ],
    }
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
