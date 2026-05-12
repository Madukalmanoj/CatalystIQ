"""SQLite-backed query cache with TTL support for external retrieval calls."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from loguru import logger


class QueryCache:
    """Persist and retrieve query responses with time-based expiration."""

    def __init__(self, db_path: str, ttl_seconds: int = 24 * 60 * 60) -> None:
        """Initialize cache store and ensure schema exists.

        Args:
            db_path: SQLite database path for cache persistence.
            ttl_seconds: Time-to-live in seconds for cache entries.

        Returns:
            None.

        Raises:
            sqlite3.Error: If SQLite initialization fails.
        """
        self.db_path = str(Path(db_path).resolve())
        self.ttl_seconds = ttl_seconds
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a SQLite connection.

        Args:
            None.

        Returns:
            A live sqlite3 connection.

        Raises:
            sqlite3.Error: If connection cannot be established.
        """
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """Create cache table if it does not exist.

        Args:
            None.

        Returns:
            None.

        Raises:
            sqlite3.Error: If table creation fails.
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_cache (
                    cache_key TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def _compute_key(source: str, query: dict[str, Any]) -> str:
        """Compute deterministic SHA-256 key from source and query.

        Args:
            source: Source name used by retriever.
            query: Query parameters dictionary.

        Returns:
            SHA-256 hexdigest string.

        Raises:
            TypeError: If query cannot be JSON serialized.
        """
        serialized = json.dumps({"source": source, "query": query}, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, source: str, query: dict[str, Any]) -> dict[str, Any] | None:
        """Read a cache entry when present and not expired.

        Args:
            source: Retriever source key.
            query: Query parameters dictionary.

        Returns:
            Cached payload dictionary or None.

        Raises:
            sqlite3.Error: If database operations fail.
        """
        key = self._compute_key(source, query)
        now = int(time.time())

        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT payload, created_at FROM query_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()

            if row is None:
                return None

            payload, created_at = row
            if now - int(created_at) > self.ttl_seconds:
                conn.execute("DELETE FROM query_cache WHERE cache_key = ?", (key,))
                conn.commit()
                logger.debug("Cache expired for source={} key={}", source, key)
                return None

            return json.loads(payload)

    def set(self, source: str, query: dict[str, Any], value: dict[str, Any]) -> None:
        """Store or replace a cache entry.

        Args:
            source: Retriever source key.
            query: Query parameters dictionary.
            value: Response payload to cache.

        Returns:
            None.

        Raises:
            sqlite3.Error: If insert/update fails.
        """
        key = self._compute_key(source, query)
        now = int(time.time())
        payload = json.dumps(value, sort_keys=True, default=str)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO query_cache (cache_key, source, payload, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    source=excluded.source,
                    payload=excluded.payload,
                    created_at=excluded.created_at
                """,
                (key, source, payload, now),
            )
            conn.commit()
