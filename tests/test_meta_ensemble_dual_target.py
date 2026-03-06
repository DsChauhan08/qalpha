import numpy as np
import pandas as pd

import quantum_alpha.meta_ensemble as me


class _IdentityScaler:
    def transform(self, X):
        return np.asarray(X, dtype=float)


class _DummyClassifier:
    def predict_proba(self, X):
        rows = len(X)
        up = np.linspace(0.40, 0.80, rows)
        return np.column_stack([1.0 - up, up])


class _DummyRegressor:
    def predict(self, X):
        rows = len(X)
        return np.linspace(0.001, 0.020, rows)


class _OffsetCalibrator:
    def transform(self, values):
        return np.clip(np.asarray(values, dtype=float) + 0.05, 0.0, 1.0)


def test_predict_dual_target_applies_probability_veto():
    frame = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0],
            "target": [0, 1, 1],
            "forward_return": [0.01, 0.02, 0.03],
            "symbol": ["A", "B", "C"],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    bundle = {
        "classifier": _DummyClassifier(),
        "regressor": _DummyRegressor(),
        "calibrator": _OffsetCalibrator(),
        "scaler": _IdentityScaler(),
        "feature_cols": ["f1"],
        "estimated_round_trip_cost": 0.002,
        "min_up_probability": 0.55,
    }

    out = me._predict_dual_target(bundle, frame)

    assert out.loc[out.index[0], "portfolio_score"] <= -1e8
    assert out.loc[out.index[-1], "portfolio_score"] > 0.0
    assert float(out.loc[out.index[-1], "y_proba"]) > 0.55


def test_select_nested_portfolio_config_returns_allowed_values():
    dates = pd.date_range("2024-01-01", periods=12, freq="B")
    rows = []
    for date in dates:
        for i in range(12):
            rows.append(
                {
                    "date": date,
                    "symbol": f"S{i}",
                    "y_true": 1 if i >= 10 else 0,
                    "y_pred": 1 if i >= 10 else 0,
                    "y_proba": 0.45 + (0.04 * i),
                    "forward_return": 0.015 if i >= 10 else (-0.005 if i == 0 else 0.0),
                    "portfolio_score": -0.01 + (0.004 * i),
                }
            )
    pred_df = pd.DataFrame(rows)

    selection = me._select_nested_portfolio_config(
        pred_df,
        top_ks=[3, 5, 10],
        hold_days_list=[5, 10],
        rebalance_intervals=[1, 5],
        min_up_probs=[0.50, 0.55, 0.60],
        commission_bps=10.0,
    )
    cfg = selection["config"]

    assert cfg["top_k"] in {3, 5, 10}
    assert cfg["hold_days"] in {5, 10}
    assert cfg["rebalance_interval"] in {1, 5}
    assert cfg["min_up_probability"] in {0.50, 0.55, 0.60}
