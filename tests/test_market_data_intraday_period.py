from datetime import datetime, timedelta

from quantum_alpha.data.collectors.market_data import DataCollector


def test_intraday_period_respects_1m_limit():
    dc = DataCollector(use_sqlite_cache=False, use_parquet_cache=False)
    end = datetime(2026, 2, 28)
    start = end - timedelta(days=40)
    assert dc._intraday_period(start, end, "1m") == "8d"


def test_intraday_period_respects_5m_limit():
    dc = DataCollector(use_sqlite_cache=False, use_parquet_cache=False)
    end = datetime(2026, 2, 28)
    start = end - timedelta(days=120)
    assert dc._intraday_period(start, end, "5m") == "60d"
