"""Chapter 22: retry != replay != duplicate for actions.

Invariant: an idempotency key may replay only the result of the same recorded
action identity, and replay cannot bypass current authority.
"""

import pytest

from codeai.adapters import ActionRequest, ActionResult, ActionStatus
from codeai.domain import Authority, Capability
from codeai.ledger import SQLiteLedger
from codeai.runtime import (
    ACTION_FINGERPRINT_V1,
    IdempotencyConflictError,
    Runtime,
    _action_request_fingerprint,
)


class CountingWriter:
    def __init__(self):
        self.calls = 0

    def execute(self, request: ActionRequest) -> ActionResult:
        self.calls += 1
        return ActionResult(action_id=request.action_id, status=ActionStatus.SUCCEEDED)


class CrashAfterEffectWriter(CountingWriter):
    def execute(self, request: ActionRequest) -> ActionResult:
        self.calls += 1
        raise RuntimeError("effect happened, completion never recorded")


def write_request(action_id="a1", key="k", **overrides):
    fields = {
        "task_id": "t", "capability": "write", "instruction": "append line",
        "precondition_hash": None, "idempotency_key": key, "adapter": "fixture",
    }
    fields.update(overrides)
    return ActionRequest(action_id=action_id, **fields)


def write_auth():
    return Authority(frozenset({Capability.WRITE}))


class TransportDropAfterEffectWriter(CountingWriter):
    def execute(self, request: ActionRequest) -> ActionResult:
        self.calls += 1
        raise ConnectionError("connection reset after the write")


def test_non_runtime_error_after_effect_is_recorded_and_replayed_not_repeated(tmp_path):
    # Regression: only RuntimeError was recorded. A ConnectionError (an OSError)
    # after the effect left action.requested with no completion, so a same-key
    # retry found nothing to replay and executed the effect a second time.
    ledger = SQLiteLedger(tmp_path / "ledger.sqlite")
    runtime = Runtime(ledger)
    writer = TransportDropAfterEffectWriter()
    first = runtime.execute_action(write_request("a1"), authority=write_auth(), adapter=writer)
    second = runtime.execute_action(write_request("a2"), authority=write_auth(), adapter=writer)
    assert writer.calls == 1
    assert first.status == second.status == ActionStatus.FAILED
    assert "connection reset" in (first.error or "")
    assert second.reused_from_action_id == "a1"
    assert [e.kind for e in ledger.read_all()] == [
        # execution_started marks the first attempt; the replay never reaches it.
        "action.requested", "action.execution_started", "action.completed",
        "action.requested", "action.completed",
    ]
    ledger._conn.close()


def test_exact_duplicate_replays_with_one_physical_effect(tmp_path):
    ledger = SQLiteLedger(tmp_path / "ledger.sqlite")
    runtime = Runtime(ledger)
    writer = CountingWriter()
    first = runtime.execute_action(write_request("a1"), authority=write_auth(), adapter=writer)
    second = runtime.execute_action(write_request("a2"), authority=write_auth(), adapter=writer)
    assert writer.calls == 1
    assert first.status == second.status == ActionStatus.SUCCEEDED
    assert second.reused_from_action_id == "a1"
    assert [e.kind for e in ledger.read_all()] == [
        # execution_started marks the first attempt; the replay never reaches it.
        "action.requested", "action.execution_started", "action.completed",
        "action.requested", "action.completed",
    ]
    ledger._conn.close()
    reopened = SQLiteLedger(tmp_path / "ledger.sqlite")
    third = Runtime(reopened).execute_action(
        write_request("a3"), authority=write_auth(), adapter=writer)
    assert writer.calls == 1
    assert third.reused_from_action_id == "a1"
    reopened._conn.close()


@pytest.mark.parametrize("field,value,dimension", [
    ("instruction", "totally different", "instruction"),
    ("capability", "execute", "capability"),
    ("payload", {"line": "other"}, "payload"),
    ("precondition_hash", "state-B", "precondition_hash"),
    ("adapter", "other-worker", "adapter"),
])
def test_key_collision_conflicts_with_zero_new_effects(field, value, dimension):
    ledger = SQLiteLedger()
    runtime = Runtime(ledger)
    writer = CountingWriter()
    runtime.execute_action(write_request("a1"), authority=write_auth(), adapter=writer)
    before = len(ledger.read_all())
    if field == "capability":
        authority = Authority(frozenset({Capability.WRITE, Capability.EXECUTE}))
    else:
        authority = write_auth()
    with pytest.raises(IdempotencyConflictError, match=dimension):
        runtime.execute_action(
            write_request("a2", **{field: value}), authority=authority, adapter=writer)
    assert writer.calls == 1
    refused = ledger.events_by_kind(("action.replay_refused",))
    assert len(refused) == 1
    assert refused[0].payload["original_action_id"] == "a1"
    assert refused[0].payload["idempotency_key"] == "k"
    assert refused[0].payload["fingerprint_version"] == ACTION_FINGERPRINT_V1
    assert dimension in refused[0].payload["mismatched_dimensions"]
    assert len(ledger.read_all()) == before + 2  # requested + replay_refused
    ledger._conn.close()


def test_conflicting_reuse_survives_reopen(tmp_path):
    path = tmp_path / "ledger.sqlite"
    ledger = SQLiteLedger(path)
    runtime = Runtime(ledger)
    writer = CountingWriter()
    runtime.execute_action(write_request("a1"), authority=write_auth(), adapter=writer)
    ledger._conn.close()
    reopened = SQLiteLedger(path)
    with pytest.raises(IdempotencyConflictError, match="instruction"):
        Runtime(reopened).execute_action(
            write_request("a2", instruction="other"), authority=write_auth(), adapter=writer)
    assert writer.calls == 1
    assert len(reopened.events_by_kind(("action.replay_refused",))) == 1
    reopened._conn.close()


def test_replay_does_not_bypass_current_authority():
    ledger = SQLiteLedger()
    runtime = Runtime(ledger)
    writer = CountingWriter()
    runtime.execute_action(write_request("a1"), authority=write_auth(), adapter=writer)
    hing = runtime.execute_action(
        write_request("a2"), authority=Authority(frozenset()), adapter=writer)
    assert hing.status == ActionStatus.DENIED
    assert hing.reused_from_action_id is None
    assert writer.calls == 1
    denied = ledger.events_by_kind(("action.completed",))[-1]
    assert denied.payload["status"] == "denied"
    ledger._conn.close()


def test_unauthorized_colliding_key_is_denied_not_conflicting():
    # Authority is checked before identity: a denied caller learns nothing
    # about the recorded operation beyond the denial.
    ledger = SQLiteLedger()
    runtime = Runtime(ledger)
    writer = CountingWriter()
    runtime.execute_action(write_request("a1"), authority=write_auth(), adapter=writer)
    result = runtime.execute_action(
        write_request("a2", instruction="other"),
        authority=Authority(frozenset()), adapter=writer)
    assert result.status == ActionStatus.DENIED
    assert ledger.events_by_kind(("action.replay_refused",)) == ()
    ledger._conn.close()


def test_failed_result_replays_without_new_effect():
    ledger = SQLiteLedger()
    runtime = Runtime(ledger)
    writer = CrashAfterEffectWriter()
    first = runtime.execute_action(write_request("d1", key="k-crash"),
                                   authority=write_auth(), adapter=writer)
    assert first.status == ActionStatus.FAILED
    second = runtime.execute_action(write_request("d2", key="k-crash"),
                                    authority=write_auth(), adapter=writer)
    assert second.status == ActionStatus.FAILED
    assert second.reused_from_action_id == "d1"
    assert writer.calls == 1  # the effect is not performed again
    ledger._conn.close()


def test_denied_result_replays_until_a_new_key_is_used():
    # A changed authorization context that should produce a new decision
    # needs a new key: replay returns whatever completion was recorded.
    ledger = SQLiteLedger()
    runtime = Runtime(ledger)
    writer = CountingWriter()
    runtime.execute_action(write_request("a1"), authority=Authority(frozenset()), adapter=writer)
    second = runtime.execute_action(write_request("a2"), authority=write_auth(), adapter=writer)
    assert second.status == ActionStatus.DENIED
    assert second.reused_from_action_id == "a1"
    assert writer.calls == 0
    third = runtime.execute_action(write_request("a3", key="k-fresh"),
                                   authority=write_auth(), adapter=writer)
    assert third.status == ActionStatus.SUCCEEDED
    assert writer.calls == 1
    ledger._conn.close()


def test_fingerprint_is_stable_and_sensitive():
    base = _action_request_fingerprint(write_request("a1"))
    assert _action_request_fingerprint(write_request("a2")) == base  # instance id excluded
    assert _action_request_fingerprint(write_request("a3", task_id="other")) == base
    assert _action_request_fingerprint(write_request("a4", instruction="other")) != base
    assert _action_request_fingerprint(write_request("a5", payload={"b": 1, "a": 2})) == \
        _action_request_fingerprint(write_request("a6", payload={"a": 2, "b": 1}))
