"""Runtime observations must survive misleading worker reports and ledger reopen."""

import hashlib
from pathlib import Path

import pytest

from codeai.adapters import ActionRequest, ActionResult, ActionStatus
from codeai.domain import Authority, Capability
from codeai.ledger import SQLiteLedger
from codeai.runtime import Runtime


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FileWorker:
    def __init__(self, target: Path, mode: str):
        self.target = target
        self.mode = mode
        self.calls = 0

    def execute(self, request: ActionRequest) -> ActionResult:
        self.calls += 1
        if self.mode == "honest":
            self.target.write_bytes(b"ttl_seconds = 300\n")
        elif self.mode == "partial":
            self.target.write_bytes(b"ttl_seconds = ")
            raise RuntimeError("interrupted write")
        elif self.mode == "wrong-target":
            self.target.with_name("other.toml").write_bytes(b"ttl_seconds = 300\n")
        return ActionResult(
            action_id=request.action_id,
            status=ActionStatus.SUCCEEDED,
            resulting_state_hash="worker-report",
            observed_state_hash="forged-runtime-reading",
        )


def action(action_id: str = "a1", precondition: str | None = None) -> ActionRequest:
    return ActionRequest(
        action_id=action_id, task_id="t1", capability="write", instruction="update TTL",
        precondition_hash=precondition, idempotency_key="write-ttl", adapter="fixture",
    )


@pytest.mark.parametrize("mode", ["honest", "lying", "partial", "wrong-target"])
def test_observation_is_from_file_not_worker(tmp_path: Path, mode: str):
    target = tmp_path / "cache.toml"
    target.write_bytes(b"ttl_seconds = 60\n")
    before = file_hash(target)
    ledger = SQLiteLedger(tmp_path / "ledger.sqlite")
    runtime = Runtime(ledger, state_resolver=lambda: file_hash(target))
    worker = FileWorker(target, mode)

    result = runtime.execute_action(
        action(precondition=before), authority=Authority(frozenset({Capability.WRITE})),
        adapter=worker,
    )

    assert worker.calls == 1
    assert result.observed_state_hash == file_hash(target)
    assert (result.observed_state_hash != before) == (mode in {"honest", "partial"})
    assert result.status == (ActionStatus.FAILED if mode == "partial" else ActionStatus.SUCCEEDED)
    if mode != "partial":
        assert result.resulting_state_hash == "worker-report"
    else:
        assert result.resulting_state_hash == result.observed_state_hash
    if mode == "wrong-target":
        assert target.with_name("other.toml").read_bytes() == b"ttl_seconds = 300\n"
    completed = list(ledger.events_by_kind(("action.completed",)))[-1]
    assert completed.payload["observed_state_hash"] == file_hash(target)
    ledger._conn.close()

    # Reuse must preserve the original observation, even when the file moves.
    # The recovery call repeats the same intended operation (same fingerprint,
    # including the original precondition); a changed precondition would be a
    # key collision, not a replay.
    target.write_bytes(b"later state")
    reopened = SQLiteLedger(tmp_path / "ledger.sqlite")
    recovered = Runtime(reopened, state_resolver=lambda: file_hash(target)).execute_action(
        action("a2", precondition=before), authority=Authority(frozenset({Capability.WRITE})), adapter=worker,
    )
    assert worker.calls == 1
    assert recovered.observed_state_hash == result.observed_state_hash
    assert recovered.reused_from_action_id == "a1"
    reopened._conn.close()


def test_no_resolver_does_not_adopt_adapter_observation(tmp_path: Path):
    ledger = SQLiteLedger()
    result = Runtime(ledger).execute_action(
        action(), authority=Authority(frozenset({Capability.WRITE})),
        adapter=FileWorker(tmp_path / "unused", "lying"),
    )
    assert result.resulting_state_hash == "worker-report"
    assert result.observed_state_hash is None
    assert ledger.read_all()[-1].payload["observed_state_hash"] is None
    ledger._conn.close()


@pytest.mark.parametrize("denied", [True, False])
def test_refusal_records_current_reading_without_executing(tmp_path: Path, denied: bool):
    target = tmp_path / "cache.toml"
    target.write_bytes(b"original")
    ledger = SQLiteLedger()
    worker = FileWorker(target, "honest")
    result = Runtime(ledger, state_resolver=lambda: file_hash(target)).execute_action(
        action(precondition="stale"),
        authority=Authority(frozenset({Capability.READ if denied else Capability.WRITE})),
        adapter=worker,
    )
    assert result.status == (ActionStatus.DENIED if denied else ActionStatus.FAILED)
    assert worker.calls == 0
    assert result.observed_state_hash == file_hash(target)
    ledger._conn.close()


def test_legacy_result_has_no_independent_observation():
    ledger = SQLiteLedger()
    result = Runtime(ledger)._action_result_from_payload({
        "action_id": "old", "status": "succeeded", "resulting_state_hash": "ambiguous",
    })
    assert result.resulting_state_hash == "ambiguous"
    assert result.observed_state_hash is None
    ledger._conn.close()
