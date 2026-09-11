from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class Event:
    event_id: str
    stream_id: str
    kind: str
    actor_id: str
    payload: dict[str, Any]
    created_at: str
    causation_id: str | None = None
    correlation_id: str | None = None

    @classmethod
    def create(
        cls,
        *,
        stream_id: str,
        kind: str,
        actor_id: str,
        payload: dict[str, Any] | Any,
        causation_id: str | None = None,
        correlation_id: str | None = None,
    ) -> "Event":
        if is_dataclass(payload):
            payload = asdict(payload)
        if not isinstance(payload, dict):
            raise TypeError("event payload must be a dict or dataclass")
        return cls(
            event_id=str(uuid.uuid4()),
            stream_id=stream_id,
            kind=kind,
            actor_id=actor_id,
            payload=payload,
            created_at=datetime.now(timezone.utc).isoformat(),
            causation_id=causation_id,
            correlation_id=correlation_id,
        )


class SQLiteLedger:
    """Append-only event ledger. Updates and deletes are intentionally absent."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                stream_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                causation_id TEXT,
                correlation_id TEXT
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_events_stream ON events(stream_id, sequence)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS ix_events_kind ON events(kind, sequence)"
        )
        self._conn.commit()

    def append(self, event: Event) -> int:
        cursor = self._conn.execute(
            """
            INSERT INTO events (
                event_id, stream_id, kind, actor_id, payload_json,
                created_at, causation_id, correlation_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.stream_id,
                event.kind,
                event.actor_id,
                json.dumps(event.payload, sort_keys=True, separators=(",", ":"), default=str),
                event.created_at,
                event.causation_id,
                event.correlation_id,
            ),
        )
        self._conn.commit()
        return int(cursor.lastrowid)

    def read_stream(self, stream_id: str) -> tuple[Event, ...]:
        rows = self._conn.execute(
            "SELECT * FROM events WHERE stream_id = ? ORDER BY sequence", (stream_id,)
        ).fetchall()
        return tuple(self._row_to_event(row) for row in rows)

    def read_all(self) -> tuple[Event, ...]:
        rows = self._conn.execute("SELECT * FROM events ORDER BY sequence").fetchall()
        return tuple(self._row_to_event(row) for row in rows)

    def events_by_kind(self, kinds: Iterable[str]) -> tuple[Event, ...]:
        kinds = tuple(kinds)
        if not kinds:
            return ()
        marks = ",".join("?" for _ in kinds)
        rows = self._conn.execute(
            f"SELECT * FROM events WHERE kind IN ({marks}) ORDER BY sequence", kinds
        ).fetchall()
        return tuple(self._row_to_event(row) for row in rows)

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Event:
        return Event(
            event_id=row["event_id"],
            stream_id=row["stream_id"],
            kind=row["kind"],
            actor_id=row["actor_id"],
            payload=json.loads(row["payload_json"]),
            created_at=row["created_at"],
            causation_id=row["causation_id"],
            correlation_id=row["correlation_id"],
        )
