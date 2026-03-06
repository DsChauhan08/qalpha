from datetime import datetime, timezone

from quantum_alpha.data.collectors.intraday_replay import (
    IntradayReplayStore,
    build_synthetic_intraday_replay,
)
from quantum_alpha.data.collectors.order_book import OrderBookCollector
from quantum_alpha.data.providers.replay_deep_provider import ReplayDeepMarketProvider


def test_intraday_replay_store_and_quality(tmp_path):
    replay_root = tmp_path / "replay"
    build_synthetic_intraday_replay(replay_root, date="2025-01-06", symbols=["SPY", "XLK", "AAPL", "MSFT"])

    store = IntradayReplayStore(replay_root)
    quotes = store.load_domain("2025-01-06", "AAPL", "quotes")
    depth = store.load_domain("2025-01-06", "AAPL", "depth")

    assert not quotes.empty
    assert not depth.empty

    quality = store.summarize_quality(dates=["2025-01-06"], symbols=["AAPL"]).to_dict()
    assert quality["completeness"] > 0.0
    assert quality["depth_completeness"] > 0.0
    assert quality["crossed_market_rate"] == 0.0


def test_order_book_collector_and_replay_provider_load_fixture_depth(tmp_path):
    replay_root = tmp_path / "replay"
    build_synthetic_intraday_replay(replay_root, date="2025-01-06", symbols=["AAPL"])

    ts = datetime(2025, 1, 6, 15, 0, tzinfo=timezone.utc)
    collector = OrderBookCollector(provider="fixture", replay_root=str(replay_root), levels=5)
    snapshot = collector.fetch_order_book("AAPL", at=ts)
    assert len(snapshot) == 5
    assert {"bid_price", "ask_price", "bid_size", "ask_size"}.issubset(snapshot.columns)

    provider = ReplayDeepMarketProvider(replay_root)
    quotes = provider.fetch_quotes("AAPL", ts.replace(hour=14, minute=30), ts)
    assert quotes.ok is True
    assert not quotes.data.empty
