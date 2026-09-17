from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .domain import ArtifactRef


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
    ) -> Event:
        if is_dataclass(payload):
            payload = asdict(payload)
        payload = _jsonable(payload)
        if not isinstance(payload, dict):
            raise TypeError("event payload must be a dict or dataclass")
        return cls(
            event_id=str(uuid.uuid4()),
            stream_id=stream_id,
            kind=kind,
            actor_id=actor_id,
            payload=payload,
            created_at=datetime.now(UTC).isoformat(),
            causation_id=causation_id,
            correlation_id=correlation_id,
        )


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    artifact_id: str
    sha256: str
    media_type: str
    artifact_type: str
    byte_length: int
    uri: str | None
    created_at: str


class SQLiteLedger:
    """Append-only event ledger. Updates and deletes are intentionally absent."""

    def close(self) -> None:
        """Release the connection. The events stay; only this handle goes."""
        self._conn.close()

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
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL,
                media_type TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                byte_length INTEGER NOT NULL,
                uri TEXT,
                created_at TEXT NOT NULL
            )
            """
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
        return tuple(_row_to_event(row) for row in rows)

    def read_all(self) -> tuple[Event, ...]:
        rows = self._conn.execute("SELECT * FROM events ORDER BY sequence").fetchall()
        return tuple(_row_to_event(row) for row in rows)

    def events_by_kind(self, kinds: Iterable[str]) -> tuple[Event, ...]:
        kinds = tuple(kinds)
        if not kinds:
            return ()
        marks = ",".join("?" for _ in kinds)
        rows = self._conn.execute(
            f"SELECT * FROM events WHERE kind IN ({marks}) ORDER BY sequence", kinds
        ).fetchall()
        return tuple(_row_to_event(row) for row in rows)

    def register_artifact(
        self,
        ref: ArtifactRef,
        *,
        artifact_type: str,
        byte_length: int,
        created_at: str,
    ) -> ArtifactRecord:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO artifacts (
                artifact_id, sha256, media_type, artifact_type, byte_length, uri, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ref.artifact_id,
                ref.sha256,
                ref.media_type,
                artifact_type,
                byte_length,
                ref.uri,
                created_at,
            ),
        )
        self._conn.commit()
        record = self.read_artifact(ref.artifact_id)
        if record is None:
            raise RuntimeError(f"artifact metadata missing after insert: {ref.artifact_id}")
        return record

    def read_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        row = self._conn.execute(
            "SELECT * FROM artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        if row is None:
            return None
        return ArtifactRecord(
            artifact_id=row["artifact_id"],
            sha256=row["sha256"],
            media_type=row["media_type"],
            artifact_type=row["artifact_type"],
            byte_length=int(row["byte_length"]),
            uri=row["uri"],
            created_at=row["created_at"],
        )


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    return value


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
