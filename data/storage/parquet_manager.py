"""
Parquet storage manager for large datasets.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd


class ParquetManager:
    """
    Store and retrieve OHLCV data in parquet format.
    """

    def __init__(
        self,
        base_dir: str,
        compression: str = "snappy",
        ttl_hours: Optional[int] = None,
        engine: Optional[str] = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.ttl = timedelta(hours=ttl_hours) if ttl_hours is not None else None
        self.engine = engine

    def _path(self, symbol: str, start: datetime, end: datetime, interval: str) -> Path:
        symbol_dir = self.base_dir / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{symbol}_{start.date()}_{end.date()}_{interval}.parquet"
        return symbol_dir / filename

    def load(
        self, symbol: str, start: datetime, end: datetime, interval: str
    ) -> Optional[pd.DataFrame]:
        path = self._path(symbol, start, end, interval)
        if not path.exists():
            return None

        if self.ttl is not None:
            mod_time = datetime.fromtimestamp(path.stat().st_mtime)
            if datetime.now() - mod_time > self.ttl:
                return None

        try:
            return pd.read_parquet(path, engine=self.engine)
        except ImportError as exc:
            raise ImportError("Parquet engine not available. Install pyarrow or fastparquet.") from exc

    def save(
        self, df: pd.DataFrame, symbol: str, start: datetime, end: datetime, interval: str
    ) -> Path:
        path = self._path(symbol, start, end, interval)
        try:
            df.to_parquet(path, compression=self.compression, engine=self.engine)
        except ImportError as exc:
            raise ImportError("Parquet engine not available. Install pyarrow or fastparquet.") from exc
        return path
