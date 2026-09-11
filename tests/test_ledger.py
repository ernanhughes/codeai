import sqlite3

import pytest

from codeai.ledger import Event, SQLiteLedger


def test_ledger_is_append_only_and_ordered():
    ledger = SQLiteLedger()
    first = Event.create(stream_id="d1", kind="directive.opened", actor_id="human", payload={"x": 1})
    second = Event.create(stream_id="d1", kind="task.created", actor_id="runtime", payload={"x": 2})

    ledger.append(first)
    ledger.append(second)

    assert ledger.read_stream("d1") == (first, second)


def test_event_ids_are_unique():
    ledger = SQLiteLedger()
    event = Event.create(stream_id="d1", kind="x", actor_id="a", payload={})
    ledger.append(event)
    with pytest.raises(sqlite3.IntegrityError):
        ledger.append(event)
