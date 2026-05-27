"""SQLite event store for sessions and agent steps."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    label TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
"""


class EventStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def create_session(self, session_id: str, label: str | None = None) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO sessions(id, created_at, label) VALUES (?, ?, ?)",
            (session_id, datetime.utcnow().isoformat(), label),
        )
        self._conn.commit()

    def log_event(self, session_id: str, kind: str, payload: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT INTO events(session_id, ts, kind, payload) VALUES (?, ?, ?, ?)",
            (
                session_id,
                datetime.utcnow().isoformat(),
                kind,
                json.dumps(payload, default=str),
            ),
        )
        self._conn.commit()

    def get_session_events(self, session_id: str) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT ts, kind, payload FROM events WHERE session_id=? ORDER BY id",
            (session_id,),
        )
        return [
            {"ts": ts, "kind": kind, "payload": json.loads(payload)}
            for ts, kind, payload in cur.fetchall()
        ]

    def close(self) -> None:
        self._conn.close()
