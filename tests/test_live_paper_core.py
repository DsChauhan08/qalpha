import numpy as np
import pandas as pd

from quantum_alpha.live_paper import (
    _micro_signal,
    _micro_burst_signal,
    _apply_drawdown_guard,
    _apply_global_drawdown,
    _online_features,
    PortfolioState,
)
from quantum_alpha.models.online.online_learner import OnlineRewardAdapter


def _sample_df(n=60):
    idx = pd.date_range("2026-01-01", periods=n, freq="5min")
    close = np.linspace(100, 101, n) + np.sin(np.linspace(0, 3, n))
    open_ = close + 0.05
    high = close + 0.1
    low = close - 0.1
    volume = np.linspace(1000, 1200, n)
    returns = pd.Series(close).pct_change().fillna(0.0).values
    rsi = np.linspace(40, 60, n)
    macd_hist = np.linspace(-0.2, 0.2, n)
    bb_position = np.linspace(0.3, 0.7, n)
    atr_pct = np.full(n, 0.01)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "returns": returns,
            "rsi": rsi,
            "macd_hist": macd_hist,
            "bb_position": bb_position,
            "atr_pct": atr_pct,
        },
        index=idx,
    )
    return df


def test_micro_signal_range():
    df = _sample_df()
    sig = _micro_signal(df, window=12)
    assert -1.0 <= sig <= 1.0


def test_burst_signal_direction():
    df = _sample_df()
    df.loc[df.index[-1], "returns"] = 0.05
    df.loc[df.index[-1], "volume"] = df["volume"].mean() * 3
    sig = _micro_burst_signal(df, window=12, ret_thresh=1.0, vol_thresh=0.5)
    assert sig <= 0.0


def test_drawdown_guard_halts():
    state = PortfolioState(name="micro", cash=1000.0, qty=1.0, equity=900.0, peak_equity=1000.0)
    triggered = _apply_drawdown_guard(state, price=100.0, max_drawdown=0.05, action="flatten", cost_bps=1.0)
    assert triggered is True
    assert state.halted is True
    assert state.qty == 0.0


def test_global_drawdown_halts_all():
    portfolios = {
        "a": PortfolioState(name="a", cash=900.0, qty=0.0, equity=900.0, peak_equity=1000.0),
        "b": PortfolioState(name="b", cash=900.0, qty=0.0, equity=900.0, peak_equity=1000.0),
    }
    state = {"equity": 0.0, "peak": 1000.0, "drawdown": 0.0, "halted": False}
    triggered = _apply_global_drawdown(portfolios, price=100.0, state=state, max_drawdown=0.05, action="halt", cost_bps=1.0)
    assert triggered is True
    assert state["halted"] is True
    assert portfolios["a"].halted is True


def test_online_adapter_updates():
    adapter = OnlineRewardAdapter(n_features=3, learning_rate=0.1)
    features = np.array([0.5, -0.2, 1.0])
    before = adapter.weights.copy()
    adapter.update(features, reward=0.5)
    assert not np.allclose(before, adapter.weights)
    score = adapter.predict(features)
    assert -1.0 <= score <= 1.0


def test_online_features_shape():
    df = _sample_df()
    feats = _online_features(df, window=10)
    assert feats.shape[0] == 9
