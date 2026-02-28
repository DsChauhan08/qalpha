import json
from pathlib import Path

import pandas as pd

from quantum_alpha.execution.live_graph_dashboard import (
    build_trade_flow_by_second,
    load_equity_curve,
    load_status,
    load_trades,
)


def test_dashboard_loaders(tmp_path: Path):
    session = tmp_path / "realtime_paper_20260228_000000"
    session.mkdir()

    (session / "live_status.json").write_text(
        json.dumps({"equity": 10001.0, "trades": 2}),
        encoding="utf-8",
    )
    (session / "equity_curve.csv").write_text(
        "timestamp,equity,cash\n2026-02-28T15:30:00Z,10000,4000\n2026-02-28T15:31:00Z,10005,3900\n",
        encoding="utf-8",
    )
    (session / "live_trades.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-02-28T15:30:01Z",
                        "side": "BUY",
                        "symbol": "SPY",
                        "qty": 1.2,
                        "exec_price": 500.1,
                        "notional": 600.12,
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-02-28T15:30:01Z",
                        "side": "SELL",
                        "symbol": "QQQ",
                        "qty": 0.5,
                        "exec_price": 430.5,
                        "notional": 215.25,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    status = load_status(session)
    eq = load_equity_curve(session)
    trades = load_trades(session)
    flow = build_trade_flow_by_second(trades)

    assert status["equity"] == 10001.0
    assert len(eq) == 2
    assert len(trades) == 2
    assert isinstance(flow, pd.DataFrame)
    assert not flow.empty
