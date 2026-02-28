import numpy as np
import pandas as pd

from quantum_alpha.strategy.news_lstm_strategy import NewsLSTMStrategy


def test_llm_context_includes_trimmed_headlines():
    strategy = NewsLSTMStrategy(checkpoint_dir="/nonexistent")
    context = strategy._build_llm_context(
        symbol="TEST",
        timestamp=pd.Timestamp("2026-02-27 10:40:00"),
        action=1,
        confidence=0.75,
        signal_value=0.5,
        signal_probs=np.array([0.2, 0.8]),
        feature_row=pd.Series({"returns": 0.01}),
        window=np.ones((12, 4), dtype=float),
        n_classes=2,
        headlines=["A" * 400, "short headline"],
    )

    headlines = context.get("headlines")
    assert isinstance(headlines, list)
    assert len(headlines) == 2
    assert len(headlines[0]) == 180
    assert headlines[1] == "short headline"
