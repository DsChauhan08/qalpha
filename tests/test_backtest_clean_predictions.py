import pickle

import pandas as pd

from quantum_alpha.backtest_clean import backtest_equal_weight, load_predictions


def test_load_predictions_supports_custom_prediction_file(tmp_path):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["AAA"],
            "y_true": [1],
            "y_pred": [1],
            "y_proba": [0.6],
            "forward_return": [0.01],
        }
    )
    pred_path = tmp_path / "meta_ensemble_mc_pade_walk_forward_predictions.pkl"
    with open(pred_path, "wb") as f:
        pickle.dump(df, f)

    loaded = load_predictions(
        tmp_path,
        prediction_file="meta_ensemble_mc_pade_walk_forward_predictions.pkl",
    )

    assert list(loaded["symbol"]) == ["AAA"]


def test_hold_period_backtest_realizes_forward_return_once_per_rebalance():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    df = pd.DataFrame(
        {
            "date": dates,
            "symbol": ["AAA"] * len(dates),
            "raw_signal": [1.0] * len(dates),
            "confidence": [1.0] * len(dates),
            "forward_return": [0.10] * len(dates),
            "y_true": [1] * len(dates),
            "y_proba": [0.6] * len(dates),
        }
    )

    results = backtest_equal_weight(
        df,
        max_positions=1,
        commission_bps=0.0,
        hold_days=5,
        top_k=1,
        initial_capital=100_000.0,
        confidence_weight=False,
    )

    assert abs(float(results["total_return"]) - 0.10) < 1e-9
