from datetime import datetime, timedelta

import pandas as pd

import quantum_alpha.meta_ensemble as me


def test_ranked_portfolio_uses_non_overlapping_rebalance_periods():
    dates = pd.date_range(datetime(2024, 1, 1), periods=10, freq="B")
    symbols = [f"S{i}" for i in range(10)]

    rows = []
    for date in dates:
        for i, symbol in enumerate(symbols):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "y_true": 1 if i >= 9 else 0,
                    "y_pred": 1 if i >= 9 else 0,
                    "y_proba": 0.05 + 0.09 * i,
                    "forward_return": 0.02 if i >= 9 else (-0.02 if i == 0 else 0.0),
                }
            )

    pred_df = pd.DataFrame(rows)
    metrics = me._evaluate_ranked_portfolio(pred_df, hold_days=5, top_frac=0.10)

    assert int(metrics["n_rebalance_periods"]) == 2
    assert float(metrics["avg_names_per_side"]) == 1.0
    assert float(metrics["total_return"]) > 0.04
    assert float(metrics["avg_spread"]) > 0.0


def test_ranked_portfolio_returns_zero_metrics_on_empty_input():
    metrics = me._evaluate_ranked_portfolio(pd.DataFrame(), hold_days=5, top_frac=0.10)

    assert metrics["n_rebalance_periods"] == 0
    assert metrics["total_return"] == 0.0
    assert metrics["sharpe"] == 0.0
