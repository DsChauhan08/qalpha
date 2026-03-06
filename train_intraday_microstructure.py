"""Train and backtest the replay-first intraday microstructure sleeve."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from quantum_alpha.backtesting.sleeve_metrics import (
    annualize_periods,
    compute_basic_metrics,
)
from quantum_alpha.data.collectors.intraday_replay import (
    IntradayReplayStore,
    build_synthetic_intraday_replay,
)
from quantum_alpha.features.intraday_feature_builder import UnifiedIntradayFeatureBuilder

DEFAULT_SYMBOLS = ("SPY", "XLK", "AAPL", "MSFT")


def _ensure_fixture_replay(root: Path, symbols: Sequence[str], days: int, seed: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp("2025-01-06", tz="UTC")
    for offset in range(max(1, int(days))):
        day = (start + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
        build_synthetic_intraday_replay(
            root,
            date=day,
            symbols=symbols,
            minutes=240,
            seed=seed + offset,
        )
    return root


def load_intraday_features(
    replay_dir: str | Path,
    symbols: Sequence[str],
    *,
    dates: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    store = IntradayReplayStore(replay_dir)
    builder = UnifiedIntradayFeatureBuilder()
    frames: List[pd.DataFrame] = []
    quality_rows: List[Dict[str, float]] = []
    selected_dates = list(dates or store.available_dates())
    for day in selected_dates:
        for symbol in symbols:
            try:
                bundle = store.load_symbol_bundle(day, symbol, domains=("trades", "quotes", "depth", "bars_1m"))
            except FileNotFoundError:
                continue
            built = builder.build(
                symbol=symbol,
                trades=bundle["trades"],
                quotes=bundle["quotes"],
                depth=bundle["depth"],
                bars_1m=bundle["bars_1m"],
            )
            frame = built.features.copy()
            frame["date"] = day
            frames.append(frame.reset_index().rename(columns={"index": "timestamp"}))
            quality_rows.append({k: float(v) for k, v in built.quality.items() if k != "symbol"})

    if not frames:
        raise ValueError("No replay bundles available for requested intraday symbols")

    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    quality_df = pd.DataFrame(quality_rows) if quality_rows else pd.DataFrame()
    aggregate_quality = store.summarize_quality(dates=selected_dates, symbols=symbols).to_dict()
    if not quality_df.empty:
        aggregate_quality["avg_symbol_rows"] = float(quality_df["rows"].mean())
    return out, aggregate_quality


def _target_columns(df: pd.DataFrame, market_symbol: str, sector_symbol: str | None) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    out["forward_return_5m"] = out.groupby("symbol")["close"].shift(-5) / out["close"] - 1.0
    out["forward_return_30m"] = out.groupby("symbol")["close"].shift(-30) / out["close"] - 1.0

    market = (
        out.loc[out["symbol"] == market_symbol, ["timestamp", "forward_return_5m", "forward_return_30m"]]
        .drop_duplicates("timestamp")
        .rename(columns={"forward_return_5m": "market_forward_5m", "forward_return_30m": "market_forward_30m"})
    )
    out = out.merge(market, on="timestamp", how="left")

    sector_name = sector_symbol or market_symbol
    sector = (
        out.loc[out["symbol"] == sector_name, ["timestamp", "forward_return_5m", "forward_return_30m"]]
        .drop_duplicates("timestamp")
        .rename(columns={"forward_return_5m": "sector_forward_5m", "forward_return_30m": "sector_forward_30m"})
    )
    out = out.merge(sector, on="timestamp", how="left")

    out["target_residual_5m"] = (
        out["forward_return_5m"]
        - 0.5 * out["market_forward_5m"].fillna(0.0)
        - 0.5 * out["sector_forward_5m"].fillna(0.0)
    )
    out["target_residual_30m"] = (
        out["forward_return_30m"]
        - 0.5 * out["market_forward_30m"].fillna(0.0)
        - 0.5 * out["sector_forward_30m"].fillna(0.0)
    )
    out["target_cost"] = (
        0.5 * out["ms_relative_spread"].clip(lower=0.0)
        + 0.3 * out["tp_liquidity_fracture"].clip(lower=0.0) * 0.001
        + 0.2 * (1.0 / (out["ms_depth_total"].clip(lower=1.0)))
    )
    return out


def _split_train_test(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = pd.Index(sorted(df["timestamp"].unique()))
    split_at = max(10, int(len(times) * train_frac))
    cutoff = times[min(split_at, len(times) - 1)]
    train = df.loc[df["timestamp"] < cutoff].copy()
    test = df.loc[df["timestamp"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Insufficient timestamps for intraday train/test split")
    return train, test


def _fit_models(train: pd.DataFrame, feature_cols: Sequence[str]) -> Dict[str, object]:
    params = dict(
        learning_rate=0.05,
        max_depth=4,
        max_iter=250,
        min_samples_leaf=20,
        random_state=42,
    )
    model_5m = HistGradientBoostingRegressor(**params)
    model_30m = HistGradientBoostingRegressor(**params)
    cost_model = HistGradientBoostingRegressor(**params)
    X = train.loc[:, feature_cols].to_numpy(dtype=float)
    model_5m.fit(X, train["target_residual_5m"].to_numpy(dtype=float))
    model_30m.fit(X, train["target_residual_30m"].to_numpy(dtype=float))
    cost_model.fit(X, train["target_cost"].to_numpy(dtype=float))
    return {
        "stage1_5m": model_5m,
        "stage1_30m": model_30m,
        "stage2_cost": cost_model,
        "feature_cols": list(feature_cols),
    }


def _score_predictions(bundle: Dict[str, object], df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    X = out.loc[:, bundle["feature_cols"]].to_numpy(dtype=float)
    pred_5 = bundle["stage1_5m"].predict(X)
    pred_30 = bundle["stage1_30m"].predict(X)
    pred_cost = bundle["stage2_cost"].predict(X)
    out["predicted_residual_5m"] = pred_5
    out["predicted_residual_30m"] = pred_30
    out["predicted_cost"] = np.clip(pred_cost, 0.0, 0.01)
    out["expected_residual_edge"] = 0.4 * pred_5 + 0.6 * pred_30
    out["trade_score"] = out["expected_residual_edge"] - out["predicted_cost"]
    out["vetoed"] = (
        (out["ms_relative_spread"] > out["ms_relative_spread"].quantile(0.90))
        | (out["ms_depth_total"] < out["ms_depth_total"].quantile(0.20))
        | (out["tp_liquidity_fracture"] > out["tp_liquidity_fracture"].quantile(0.85))
    ).astype(int)
    out.loc[out["vetoed"] == 1, "trade_score"] = -1e9
    return out


def _intraday_backtest(
    pred: pd.DataFrame,
    *,
    market_symbol: str,
    excluded_symbols: Iterable[str],
    top_k: int,
    rebalance_step: int = 5,
) -> tuple[pd.Series, pd.DataFrame]:
    excluded = {s.upper() for s in excluded_symbols}
    df = pred.copy()
    df["minute_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    candidate = df.loc[~df["symbol"].isin(excluded)].copy()
    candidate = candidate.loc[candidate["minute_of_day"] < (16 * 60 - 5)]
    candidate = candidate.loc[(candidate["minute_of_day"] - candidate["minute_of_day"].min()) % rebalance_step == 0]

    returns: List[Dict[str, float | pd.Timestamp]] = []
    positions: List[Dict[str, float | str | pd.Timestamp]] = []
    grouped = candidate.groupby("timestamp", sort=True)

    for ts, chunk in grouped:
        active = chunk.loc[chunk["vetoed"] == 0].copy()
        if active.empty:
            continue
        longs = active.sort_values("trade_score", ascending=False).head(max(1, int(top_k)))
        shorts = active.sort_values("trade_score", ascending=True).head(max(1, int(top_k)))
        longs = longs.loc[longs["trade_score"] > 0]
        shorts = shorts.loc[shorts["trade_score"] < 0]

        long_weight = 0.5 / max(1, len(longs)) if len(longs) else 0.0
        short_weight = -0.5 / max(1, len(shorts)) if len(shorts) else 0.0

        period_ret = 0.0
        for _, row in longs.iterrows():
            realized = float(row["forward_return_5m"])
            cost = float(row["target_cost"])
            period_ret += long_weight * (realized - cost)
            positions.append({"timestamp": ts, "symbol": row["symbol"], "side": "long", "weight": long_weight})
        for _, row in shorts.iterrows():
            realized = float(row["forward_return_5m"])
            cost = float(row["target_cost"])
            period_ret += short_weight * (realized + cost)
            positions.append({"timestamp": ts, "symbol": row["symbol"], "side": "short", "weight": short_weight})

        benchmark_row = df.loc[(df["timestamp"] == ts) & (df["symbol"] == market_symbol)]
        benchmark_ret = float(benchmark_row["forward_return_5m"].iloc[0]) if not benchmark_row.empty else 0.0
        returns.append({"timestamp": ts, "strategy_return": period_ret, "benchmark_return": benchmark_ret})

    ret_df = pd.DataFrame(returns)
    pos_df = pd.DataFrame(positions)
    if ret_df.empty:
        return pd.Series(dtype=float), pos_df
    ret_series = pd.Series(ret_df["strategy_return"].to_numpy(dtype=float), index=pd.to_datetime(ret_df["timestamp"], utc=True))
    return ret_series.sort_index(), pos_df


def train_intraday_microstructure(
    *,
    replay_dir: str | Path | None,
    symbols: Sequence[str],
    market_symbol: str,
    sector_symbol: str | None,
    output_dir: str | Path,
    checkpoint_dir: str | Path,
    build_fixture: bool = True,
    fixture_days: int = 5,
    top_k: int = 2,
    seed: int = 42,
) -> Dict[str, object]:
    out_dir = Path(output_dir)
    ckpt_dir = Path(checkpoint_dir)
    if replay_dir is None:
        replay_root = out_dir / "replay_fixture"
        if build_fixture:
            _ensure_fixture_replay(replay_root, symbols, fixture_days, seed)
        replay_dir = replay_root

    feature_df, replay_quality = load_intraday_features(replay_dir, symbols)
    feature_df = _target_columns(feature_df, market_symbol=market_symbol, sector_symbol=sector_symbol)
    feature_df = feature_df.dropna(subset=["target_residual_5m", "target_residual_30m"]).copy()

    excluded = {market_symbol.upper()}
    if sector_symbol:
        excluded.add(sector_symbol.upper())
    feature_cols = [c for c in feature_df.columns if c.startswith(("ms_", "tp_", "rp_", "ps_"))]
    train_df, test_df = _split_train_test(feature_df)
    bundle = _fit_models(train_df, feature_cols)
    scored = _score_predictions(bundle, test_df)
    returns_5m, positions = _intraday_backtest(
        scored,
        market_symbol=market_symbol.upper(),
        excluded_symbols=excluded,
        top_k=top_k,
    )
    if returns_5m.empty:
        raise RuntimeError("Intraday microstructure backtest produced no returns")

    daily_returns = returns_5m.groupby(returns_5m.index.normalize()).sum()
    benchmark = (
        scored.loc[scored["symbol"] == market_symbol.upper(), ["timestamp", "forward_return_5m"]]
        .drop_duplicates("timestamp")
        .set_index("timestamp")["forward_return_5m"]
    )
    benchmark_daily = benchmark.groupby(pd.to_datetime(benchmark.index).normalize()).sum()
    metrics = compute_basic_metrics(daily_returns, benchmark_daily, periods_per_year=252.0)

    data_quality = dict(replay_quality)
    data_quality["average_participation"] = float(positions["weight"].abs().mean()) if not positions.empty else 0.0
    data_quality["gross_exposure_mean"] = float(positions.groupby("timestamp")["weight"].apply(lambda x: x.abs().sum()).mean()) if not positions.empty else 0.0
    data_quality["net_exposure_mean"] = float(positions.groupby("timestamp")["weight"].sum().mean()) if not positions.empty else 0.0

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / "intraday_microstructure_model.pkl"
    with open(checkpoint_path, "wb") as f:
        pickle.dump(
            {
                "models": bundle,
                "market_symbol": market_symbol.upper(),
                "sector_symbol": sector_symbol.upper() if sector_symbol else None,
                "symbols": [s.upper() for s in symbols],
                "feature_cols": feature_cols,
                "data_quality": data_quality,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            f,
        )

    daily_curve = (1.0 + daily_returns.fillna(0.0)).cumprod()
    daily_returns_path = out_dir / "daily_returns.csv"
    pd.DataFrame(
        {
            "date": daily_returns.index.astype(str),
            "strategy_return": daily_returns.values,
            "benchmark_return": benchmark_daily.reindex(daily_returns.index).fillna(0.0).values,
            "equity_curve": daily_curve.values,
        }
    ).to_csv(daily_returns_path, index=False)
    positions_path = out_dir / "positions.csv"
    positions.to_csv(positions_path, index=False)

    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "intraday_microstructure",
        "replay_dir": str(replay_dir),
        "checkpoint": str(checkpoint_path),
        "symbols": [s.upper() for s in symbols],
        "market_symbol": market_symbol.upper(),
        "sector_symbol": sector_symbol.upper() if sector_symbol else None,
        "feature_count": len(feature_cols),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "metrics": metrics,
        "data_quality": data_quality,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "daily_returns_csv": str(daily_returns_path),
            "positions_csv": str(positions_path),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    latest_path = ckpt_dir / "latest_intraday_microstructure.json"
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
    parser = argparse.ArgumentParser(description="Train the intraday microstructure sleeve")
    parser.add_argument("--replay-dir", type=str, default=None)
    parser.add_argument("--symbols", type=str, default="SPY,XLK,AAPL,MSFT")
    parser.add_argument("--market-symbol", type=str, default="SPY")
    parser.add_argument("--sector-symbol", type=str, default="XLK")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--fixture-days", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-build-fixture", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "artifacts" / "intraday_microstructure"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(Path(__file__).parent / "models" / "intraday_microstructure"),
    )
    args = parser.parse_args()

    summary = train_intraday_microstructure(
        replay_dir=args.replay_dir,
        symbols=[s.strip().upper() for s in args.symbols.split(",") if s.strip()],
        market_symbol=args.market_symbol,
        sector_symbol=args.sector_symbol,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        build_fixture=not args.no_build_fixture,
        fixture_days=args.fixture_days,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(json.dumps({"passed": True, "summary_json": str(Path(args.output_dir) / "summary.json"), "metrics": summary["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
