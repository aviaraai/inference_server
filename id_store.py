"""
id_store.py — SQLite-backed FAISS ID ↔ cattle_id mapping.

Provides atomic, corruption-resistant persistence for the mapping between
FAISS integer IDs and cattle_id strings.

Schema:
    id_map (
        faiss_id        INTEGER PRIMARY KEY,
        cattle_id       TEXT NOT NULL,
        model_version   TEXT NOT NULL,
        registered_at   DATETIME DEFAULT CURRENT_TIMESTAMP
    )
"""

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from godhaar.config import MODEL_VERSION

log = logging.getLogger("godhaar.id_store")


class IDStore:
    """Thread-safe SQLite store for FAISS ID → cattle_id mapping.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file. Created if it doesn't exist.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._lock = threading.Lock()

        # Ensure parent directory exists
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialise schema
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS id_map (
                    faiss_id        INTEGER PRIMARY KEY,
                    cattle_id       TEXT NOT NULL,
                    model_version   TEXT NOT NULL,
                    registered_at   TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cattle_id ON id_map (cattle_id)
            """)
            conn.commit()

        count = self.count()
        log.info(f"IDStore loaded: {self._db_path} ({count} entries)")

    def _connect(self) -> sqlite3.Connection:
        """Create a new connection with WAL mode for concurrent reads."""
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def add(self, faiss_id: int, cattle_id: str, model_version: str = MODEL_VERSION) -> None:
        """Register a new FAISS ID → cattle_id mapping.

        Parameters
        ----------
        faiss_id : int
        cattle_id : str
        model_version : str
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO id_map (faiss_id, cattle_id, model_version, registered_at) "
                    "VALUES (?, ?, ?, ?)",
                    (faiss_id, cattle_id, model_version, now),
                )
                conn.commit()

    def add_batch(self, entries: list[tuple[int, str]], model_version: str = MODEL_VERSION) -> None:
        """Register multiple FAISS ID → cattle_id mappings atomically.

        Parameters
        ----------
        entries : list of (faiss_id, cattle_id) tuples
        model_version : str
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connect() as conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO id_map (faiss_id, cattle_id, model_version, registered_at) "
                    "VALUES (?, ?, ?, ?)",
                    [(fid, cid, model_version, now) for fid, cid in entries],
                )
                conn.commit()

    def lookup(self, faiss_id: int) -> Optional[str]:
        """Look up a cattle_id by its FAISS integer ID.

        Returns None if the ID is not found.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cattle_id FROM id_map WHERE faiss_id = ?", (faiss_id,)
            ).fetchone()
        return row[0] if row else None

    def lookup_batch(self, faiss_ids: list[int]) -> dict[int, str]:
        """Look up multiple FAISS IDs at once.

        Returns
        -------
        dict mapping faiss_id → cattle_id (missing IDs are omitted).
        """
        if not faiss_ids:
            return {}
        placeholders = ",".join("?" for _ in faiss_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT faiss_id, cattle_id FROM id_map WHERE faiss_id IN ({placeholders})",
                faiss_ids,
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def get_ids_for_cattle(self, cattle_id: str) -> list[int]:
        """Get all FAISS IDs registered under a specific cattle_id."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT faiss_id FROM id_map WHERE cattle_id = ?", (cattle_id,)
            ).fetchall()
        return [row[0] for row in rows]

    def count(self) -> int:
        """Total number of registered embeddings."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM id_map").fetchone()
        return row[0] if row else 0

    def get_model_version(self) -> Optional[str]:
        """Return the model version of the most recent entry, or None if empty."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT model_version FROM id_map ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None
