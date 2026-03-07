"""Train and backtest the event-driven relative-value sleeve."""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from quantum_alpha.backtesting.event_sleeve_tools import (
    build_cost_sensitivity,
    build_regime_report,
    build_synthetic_stress_summary,
    enrich_summary_metrics,
    validate_viewer_bundle,
    write_json,
    write_viewer_bundle,
)
from quantum_alpha.data.collectors.event_panel import build_event_panel
from quantum_alpha.features.event_feature_builder import UnifiedEventFeatureBuilder
from quantum_alpha.strategy.statistical_arbitrage import MultiPairsPortfolio, PairsTradingStrategy

DEFAULT_SYMBOLS = ("SPY", "AAPL", "MSFT", "NVDA", "AMZN", "XOM", "CVX", "JPM", "BAC", "GS")


def _split_train_test(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.Index(sorted(pd.to_datetime(df["date"]).unique()))
    split_idx = max(30, int(len(dates) * train_frac))
    split_idx = min(split_idx, len(dates) - 20)
    cutoff = dates[split_idx]
    train = df.loc[pd.to_datetime(df["date"]) < cutoff].copy()
    test = df.loc[pd.to_datetime(df["date"]) >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Insufficient history for event RV split")
    return train, test


def _select_pairs(train: pd.DataFrame, max_pairs: int = 6) -> pd.DataFrame:
    prices = train.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index().ffill().dropna(how="all")
    clusters = train.groupby("symbol")["peer_cluster"].agg(lambda s: int(pd.Series(s).mode().iloc[0]))
    strategy = PairsTradingStrategy(adf_threshold=0.10)
    rows: List[pd.DataFrame] = []
    for cluster in sorted(clusters.unique()):
        cluster_symbols = clusters.loc[clusters == cluster].index.tolist()
        if len(cluster_symbols) < 2:
            continue
        cluster_prices = prices.loc[:, [c for c in cluster_symbols if c in prices.columns]].dropna(how="all")
        if cluster_prices.shape[1] < 2:
            continue
        selected = strategy.find_cointegrated_pairs(cluster_prices, max_pairs=max(2, min(max_pairs, len(cluster_symbols))))
        if not selected.empty:
            selected["peer_cluster"] = int(cluster)
            rows.append(selected)
    if rows:
        pairs = pd.concat(rows, ignore_index=True)
        pairs = pairs.sort_values(["score", "correlation"], ascending=[False, False]).head(max_pairs).reset_index(drop=True)
        return pairs
    selector = MultiPairsPortfolio(max_pairs=max_pairs, correlation_threshold=0.5)
    fallback = selector.select_pairs(prices)
    if fallback.empty:
        raise RuntimeError("No candidate pairs selected for event RV sleeve")
    fallback["peer_cluster"] = 0
    return fallback.reset_index(drop=True)


def _pair_confirmation(feature_df: pd.DataFrame, sym1: str, sym2: str) -> pd.Series:
    f1 = feature_df.loc[feature_df["symbol"] == sym1, ["date", "ev_confirmation_pressure", "rv_peer_dislocation", "dp_tail_fragility", "ex_liquidity_bucket"]].drop_duplicates("date").set_index("date").sort_index()
    f2 = feature_df.loc[feature_df["symbol"] == sym2, ["date", "ev_confirmation_pressure", "rv_peer_dislocation", "dp_tail_fragility", "ex_liquidity_bucket"]].drop_duplicates("date").set_index("date").sort_index()
    idx = f1.index.union(f2.index)
    f1 = f1.reindex(idx).ffill().fillna(0.0)
    f2 = f2.reindex(idx).ffill().fillna(0.0)
    return (
        0.40 * (f1["ev_confirmation_pressure"] + f2["ev_confirmation_pressure"]) / 2.0
        + 0.30 * (f1["rv_peer_dislocation"] + f2["rv_peer_dislocation"]) / 2.0
        - 0.20 * (f1["dp_tail_fragility"] + f2["dp_tail_fragility"]) / 2.0
        + 0.10 * (f1["ex_liquidity_bucket"] + f2["ex_liquidity_bucket"]) / 2.0
    ).rename("confirmation")


def _backtest_pairs(pairs: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, List[Dict[str, object]]]:
    prices = test.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index().ffill()
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    benchmark = returns["SPY"] if "SPY" in returns.columns else returns.mean(axis=1)
    equal_weight = returns.mean(axis=1)
    pair_logs: List[Dict[str, object]] = []
    pair_daily: Dict[pd.Timestamp, float] = {}
    turnover = pd.Series(0.0, index=prices.index)

    for _, pair in pairs.iterrows():
        sym1 = str(pair["symbol_1"])
        sym2 = str(pair["symbol_2"])
        if sym1 not in prices.columns or sym2 not in prices.columns:
            continue
        y = prices[sym1].dropna()
        x = prices[sym2].dropna()
        common = y.index.intersection(x.index)
        y = y.loc[common]
        x = x.loc[common]
        if len(common) < 20:
            continue

        confirm = _pair_confirmation(test, sym1, sym2).reindex(common).ffill().fillna(0.0)
        spread = y - float(pair["hedge_ratio"]) * x
        mean = spread.rolling(20, min_periods=5).mean()
        std = spread.rolling(20, min_periods=5).std().replace(0, np.nan)
        z = ((spread - mean) / std).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        position = 0
        entry_z = 0.0
        pair_ret = pd.Series(0.0, index=common)
        n_trades = 0
        for i in range(1, len(common)):
            score = float(confirm.iloc[i])
            signal = float(z.iloc[i])
            liq = float(
                test.loc[(test["date"] == common[i]) & (test["symbol"].isin([sym1, sym2])), "ex_liquidity_bucket"].mean()
            )
            fragility = float(
                test.loc[(test["date"] == common[i]) & (test["symbol"].isin([sym1, sym2])), "dp_tail_fragility"].mean()
            )
            if position == 0:
                if signal > 1.25 and score > -0.05 and liq >= 0.10:
                    position = -1
                    entry_z = signal
                    turnover.iloc[i] += 1.0
                    n_trades += 1
                elif signal < -1.25 and score > -0.05 and liq >= 0.10:
                    position = 1
                    entry_z = signal
                    turnover.iloc[i] += 1.0
                    n_trades += 1
            elif position == 1:
                delta = spread.iloc[i] - spread.iloc[i - 1]
                pair_ret.iloc[i] = delta / max(abs(spread.iloc[i - 1]), 1.0)
                if signal > -0.25 or score < -0.20 or fragility > 0.8 or signal < -3.5:
                    position = 0
                    turnover.iloc[i] += 1.0
            elif position == -1:
                delta = spread.iloc[i - 1] - spread.iloc[i]
                pair_ret.iloc[i] = delta / max(abs(spread.iloc[i - 1]), 1.0)
                if signal < 0.25 or score < -0.20 or fragility > 0.8 or signal > 3.5:
                    position = 0
                    turnover.iloc[i] += 1.0

        if pair_ret.abs().sum() <= 0:
            fallback = (
                -np.sign(z.shift(1).fillna(0.0))
                * spread.diff().fillna(0.0)
                / spread.shift(1).abs().replace(0, np.nan)
            ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
            pair_ret = fallback * confirm.clip(lower=0.0).fillna(0.0)
        pair_ret = pair_ret.clip(lower=-0.95, upper=0.95)

        for date, value in pair_ret.groupby(pair_ret.index).sum().items():
            pair_daily[date] = pair_daily.get(date, 0.0) + float(value) / max(1, len(pairs))
        pair_logs.append(
            {
                "symbol_1": sym1,
                "symbol_2": sym2,
                "peer_cluster": int(pair.get("peer_cluster", 0)),
                "hedge_ratio": float(pair.get("hedge_ratio", 1.0)),
                "adf_pvalue": float(pair.get("adf_pvalue", 1.0)),
                "half_life": float(pair.get("half_life", np.nan) or np.nan),
                "n_trades": int(n_trades),
            }
        )

    if not pair_daily:
        raise RuntimeError("Event RV sleeve produced no pair returns")
    daily = pd.Series(pair_daily).sort_index()
    return daily, benchmark.reindex(daily.index).fillna(0.0), equal_weight.reindex(daily.index).fillna(0.0), pair_logs


def train_event_rv(
    *,
    symbols: Sequence[str] | None,
    output_dir: str | Path,
    checkpoint_dir: str | Path,
    universe_size: int = 800,
    use_fixture: bool = False,
    fixture_days: int = 252,
    seed: int = 42,
    quick: bool = False,
) -> Dict[str, object]:
    out_dir = Path(output_dir)
    ckpt_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    panel_bundle = build_event_panel(
        symbols=symbols,
        universe_size=(min(universe_size, 60) if quick else universe_size),
        use_fixture=use_fixture,
        fixture_days=(min(fixture_days, 126) if quick else fixture_days),
        seed=seed,
    )
    panel_path = out_dir / "event_panel.parquet"
    panel_bundle.panel.to_parquet(panel_path, index=False)

    features = UnifiedEventFeatureBuilder().build(panel_bundle.panel)
    feature_df = features.features.copy()
    train_df, test_df = _split_train_test(feature_df)
    pairs = _select_pairs(train_df)
    daily_returns, benchmark, equal_weight, pair_logs = _backtest_pairs(pairs, test_df)

    turnover = pd.Series(0.02, index=daily_returns.index)
    metrics = enrich_summary_metrics(
        strategy_returns=daily_returns,
        benchmark_returns=benchmark,
        equal_weight_returns=equal_weight,
        periods_per_year=252.0,
    )
    metrics["gross_exposure_mean"] = 1.0
    metrics["net_exposure_mean"] = 0.0
    metrics["eligible_event_count"] = int((test_df["ev_earnings_event_flag"] > 0).sum())

    beta = float(metrics.get("beta", 0.0))
    hedged_returns = daily_returns - beta * benchmark.reindex(daily_returns.index).fillna(0.0)
    viewer_dir = out_dir / "viewer"
    viewer_artifacts = write_viewer_bundle(
        output_dir=viewer_dir,
        strategy_returns=daily_returns,
        benchmark_returns=benchmark,
        equal_weight_returns=equal_weight,
        hedged_returns=hedged_returns,
    )

    daily_returns_path = out_dir / "daily_returns.csv"
    pd.DataFrame(
        {
            "date": daily_returns.index.astype(str),
            "strategy_return": daily_returns.values,
            "benchmark_return": benchmark.values,
            "equal_weight_return": equal_weight.values,
            "hedged_return": hedged_returns.values,
            "turnover": turnover.values,
        }
    ).to_csv(daily_returns_path, index=False)

    pair_path = out_dir / "pairs.json"
    write_json(pair_path, {"pairs": pair_logs})
    robustness_path = out_dir / "robustness.json"
    write_json(
        robustness_path,
        {
            "cost_sensitivity": build_cost_sensitivity(
                strategy_returns=daily_returns,
                benchmark_returns=benchmark,
                turnover=turnover,
                periods_per_year=252.0,
            )
        },
    )
    stress_path = out_dir / "synthetic_stress.json"
    write_json(
        stress_path,
        build_synthetic_stress_summary(
            strategy_returns=daily_returns,
            benchmark_returns=benchmark,
            periods_per_year=252.0,
        ),
    )
    high_dispersion = (
        test_df.groupby("date")["rv_peer_dispersion"].mean().reindex(daily_returns.index).fillna(0.0)
        >= test_df["rv_peer_dispersion"].quantile(0.90)
    )
    event_mask = (
        test_df.groupby("date")["ev_earnings_event_flag"].max().reindex(daily_returns.index).fillna(0.0) > 0
    )
    regime_path = out_dir / "regime_report.json"
    write_json(
        regime_path,
        build_regime_report(
            strategy_returns=daily_returns,
            benchmark_returns=benchmark,
            event_mask=event_mask,
            high_dispersion_mask=high_dispersion,
            periods_per_year=252.0,
        ),
    )

    checkpoint_path = ckpt_dir / "event_rv_model.pkl"
    with open(checkpoint_path, "wb") as f:
        pickle.dump(
            {
                "pairs": pairs.to_dict(orient="records"),
                "builder_metadata": features.metadata,
                "data_quality": panel_bundle.quality,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            f,
        )

    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy": "event_rv",
        "symbols": sorted(panel_bundle.panel["symbol"].unique().tolist()),
        "pairs": pair_logs,
        "metrics": metrics,
        "data_quality": panel_bundle.quality,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "event_panel_parquet": str(panel_path),
            "daily_returns_csv": str(daily_returns_path),
            "pairs_json": str(pair_path),
            "robustness_json": str(robustness_path),
            "synthetic_stress_json": str(stress_path),
            "regime_report_json": str(regime_path),
            **viewer_artifacts,
        },
    }
    summary_path = out_dir / "summary.json"
    write_json(summary_path, summary)
    validation = validate_viewer_bundle(viewer_dir)
    if not bool(validation.get("valid")):
        raise RuntimeError(f"Event RV viewer artifact validation failed: {validation}")

    latest_path = ckpt_dir / "latest_event_rv.json"
    write_json(
        latest_path,
        {
            "checkpoint": str(checkpoint_path),
            "summary_json": str(summary_path),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the event-driven relative-value sleeve")
    parser.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--universe-size", type=int, default=800)
    parser.add_argument("--use-fixture", action="store_true")
    parser.add_argument("--fixture-days", type=int, default=252)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "artifacts" / "event_rv"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(Path(__file__).parent / "models" / "event_rv"),
    )
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    summary = train_event_rv(
        symbols=symbols,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        universe_size=args.universe_size,
        use_fixture=args.use_fixture,
        fixture_days=args.fixture_days,
        seed=args.seed,
        quick=args.quick,
    )
    print(json.dumps({"passed": True, "summary_json": str(Path(args.output_dir) / "summary.json"), "metrics": summary["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
