"""
Paper trading engine using the backtesting execution model.
"""

from __future__ import annotations

from typing import Dict, Callable, Optional, Tuple
from datetime import datetime

import pandas as pd

from quantum_alpha.backtesting.engine import Backtester


class PaperTrader:
    """
    Executes a paper trading run over the most recent bars.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        paper_bars: int = 30,
        backtester: Optional[Backtester] = None,
    ) -> None:
        self.initial_capital = initial_capital
        self.paper_bars = max(1, int(paper_bars))
        self.backtester = backtester or Backtester(initial_capital=initial_capital)

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        strategy: Callable[[datetime, Dict[str, pd.Series], Backtester], None],
    ) -> Tuple[Dict, datetime]:
        timestamps = set()
        for df in data.values():
            timestamps.update(df.index.tolist())
        timestamps = sorted(timestamps)

        if not timestamps:
            return {"error": "No data for paper trading"}, datetime.utcnow()

        start_idx = max(len(timestamps) - self.paper_bars, 0)
        paper_start = timestamps[start_idx]

        def gated_strategy(ts, bars, bt):
            if ts < paper_start:
                return
            strategy(ts, bars, bt)

        self.backtester.run(data, gated_strategy)
        metrics = self.backtester.get_metrics()

        return metrics, paper_start
