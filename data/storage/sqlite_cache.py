"""
SQLite-based cache for pandas DataFrames.
"""

from __future__ import annotations

import sqlite3
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd


class SQLiteCache:
    """
    Simple SQLite cache for serialized DataFrames.
    """

    def __init__(self, db_path: str, ttl_hours: int = 24) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ohlcv_cache (
                    cache_key TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    data BLOB NOT NULL
                )
                """
            )
            conn.commit()

    def get(self, cache_key: str) -> Optional[pd.DataFrame]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT created_at, data FROM ohlcv_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()

            if not row:
                return None

            created_at = datetime.fromisoformat(row[0])
            if datetime.utcnow() - created_at > self.ttl:
                conn.execute(
                    "DELETE FROM ohlcv_cache WHERE cache_key = ?", (cache_key,)
                )
                conn.commit()
                return None

            return pickle.loads(row[1])

    def set(self, cache_key: str, df: pd.DataFrame) -> None:
        payload = pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL)
        created_at = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ohlcv_cache (cache_key, created_at, data)
                VALUES (?, ?, ?)
                """,
                (cache_key, created_at, payload),
            )
            conn.commit()

    def purge(self) -> int:
        cutoff = datetime.utcnow() - self.ttl
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "DELETE FROM ohlcv_cache WHERE created_at < ?", (cutoff.isoformat(),)
            )
            conn.commit()
            return cur.rowcount
