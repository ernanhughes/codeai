"""Chapter 19/22/29 seam: result status is not knowledge of the effect.

The governing question, asked at every interruption point: can the runtime say
whether another execution is safe, unsafe, or unresolved, without guessing?
"""

import pytest
from dataclasses import asdict

from codeai.actions import (
    ACTION_RECOVERY_V1,
    ActionNextOperation,
    EffectState,
    ReconciliationRefused,
    ReconciliationVerdict,
)
from codeai.adapters import ActionRequest, ActionResult, ActionStatus
from codeai.domain import Authority, Capability
from codeai.ledger import Event, SQLiteLedger
from codeai.runtime import Runtime


class CountingWriter:
    def __init__(self):
        self.calls = 0

    def execute(self, request: ActionRequest) -> ActionResult:
        self.calls += 1
        return ActionResult(action_id=request.action_id, status=ActionStatus.SUCCEEDED)


class CrashAfterEffectWriter(CountingWriter):
    """The dangerous shape: the effect happens, then the adapter dies."""

    def execute(self, request: ActionRequest) -> ActionResult:
        self.calls += 1
        raise ConnectionError("connection reset after the write")


def write_request(action_id="a1", key="k", **overrides):
    fields = {
        "task_id": "t", "capability": "write", "instruction": "append line",
        "precondition_hash": None, "idempotency_key": key, "adapter": "fixture",
    }
    fields.update(overrides)
    return ActionRequest(action_id=action_id, **fields)


def write_auth():
    return Authority(frozenset({Capability.WRITE}))


def runtime_with(tmp_path, **kwargs):
    return Runtime(SQLiteLedger(tmp_path / "ledger.sqlite"), **kwargs)


# ---------------- the ordering argument ----------------


def test_execution_started_is_committed_before_the_adapter_acts(tmp_path):
    seen = []
    runtime = runtime_with(tmp_path)

    class ObservingWriter(CountingWriter):
        def execute(self, request):
            seen.extend(e.kind for e in runtime.ledger.read_all())
            return super().execute(request)

    runtime.execute_action(write_request(), authority=write_auth(), adapter=ObservingWriter())
    assert "action.execution_started" in seen, "the effect ran before its intent was durable"
    assert "action.completed" not in seen


def test_requested_without_execution_started_is_safe_to_start(tmp_path):
    runtime = runtime_with(tmp_path)
    request = write_request()
    runtime.ledger.append(
        Event.create(
            stream_id=request.action_id,
            kind="action.requested",
            actor_id="reviewer",
            payload={**asdict(request), "recovery_version": ACTION_RECOVERY_V1},
        )
    )
    state = runtime.action_state("a1")
    assert state.effect_state == EffectState.NONE
    assert state.next_operation == ActionNextOperation.START_ACTION
    assert state.duplicate_effect_risk is False


def test_execution_started_without_completion_is_unknown_and_never_auto_retried(tmp_path):
    runtime = runtime_with(tmp_path)
    request = write_request()
    runtime.ledger.append(
        Event.create(
            stream_id=request.action_id, kind="action.requested", actor_id="reviewer",
            payload={**asdict(request), "recovery_version": ACTION_RECOVERY_V1},
        )
    )
    runtime.ledger.append(
        Event.create(
            stream_id=request.action_id, kind="action.execution_started", actor_id="worker",
            payload={"action_id": "a1", "version": ACTION_RECOVERY_V1},
        )
    )
    state = runtime.action_state("a1")
    assert state.stage == "execution_started"
    assert state.effect_state == EffectState.UNKNOWN
    assert state.next_operation == ActionNextOperation.RECONCILE_EFFECT
    assert state.duplicate_effect_risk is True


def test_success_is_observed_and_terminal(tmp_path):
    runtime = runtime_with(tmp_path, state_resolver=lambda: "after")
    runtime.execute_action(write_request(), authority=write_auth(), adapter=CountingWriter())
    state = runtime.action_state("a1")
    assert (state.stage, state.result_status) == ("completed", "succeeded")
    assert state.effect_state == EffectState.OBSERVED
    assert state.next_operation == ActionNextOperation.NONE
    assert state.observed_state_hash == "after"


def test_denial_means_no_effect_was_possible(tmp_path):
    runtime = runtime_with(tmp_path)
    writer = CountingWriter()
    runtime.execute_action(
        write_request(), authority=Authority(frozenset({Capability.READ})), adapter=writer
    )
    kinds = [e.kind for e in runtime.ledger.read_all()]
    assert "action.execution_started" not in kinds
    assert writer.calls == 0
    state = runtime.action_state("a1")
    assert (state.result_status, state.effect_state) == ("denied", EffectState.NONE)
    assert state.next_operation == ActionNextOperation.NONE


def test_failure_after_execution_started_is_not_evidence_that_nothing_happened(tmp_path):
    runtime = runtime_with(tmp_path)
    writer = CrashAfterEffectWriter()
    result = runtime.execute_action(write_request(), authority=write_auth(), adapter=writer)
    assert result.status == ActionStatus.FAILED
    state = runtime.action_state("a1")
    assert state.stage == "failed_after_execution_started"
    assert state.result_status == "failed"
    assert state.effect_state == EffectState.UNKNOWN, "FAILED was read as 'nothing happened'"
    assert state.next_operation == ActionNextOperation.RECONCILE_EFFECT
    assert state.duplicate_effect_risk is True


def test_precondition_failure_before_execution_leaves_no_effect(tmp_path):
    runtime = runtime_with(tmp_path, state_resolver=lambda: "current")
    writer = CountingWriter()
    runtime.execute_action(
        write_request(precondition_hash="planned-against-something-else"),
        authority=write_auth(),
        adapter=writer,
    )
    assert writer.calls == 0
    state = runtime.action_state("a1")
    assert state.stage == "failed_before_execution_started"
    assert state.effect_state == EffectState.NONE
    assert state.next_operation == ActionNextOperation.FIX_INPUTS


def test_replay_reports_no_new_effect_and_names_the_original(tmp_path):
    runtime = runtime_with(tmp_path)
    writer = CountingWriter()
    runtime.execute_action(write_request("a1"), authority=write_auth(), adapter=writer)
    runtime.execute_action(write_request("a2"), authority=write_auth(), adapter=writer)
    assert writer.calls == 1
    replayed = runtime.action_state("a2")
    assert replayed.stage == "replayed"
    assert replayed.effect_state == EffectState.NONE
    assert replayed.reused_from_action_id == "a1"
    assert runtime.action_state("a1").effect_state == EffectState.OBSERVED


# ---------------- history it cannot see ----------------


def test_records_written_before_execution_marking_stay_unknown(tmp_path):
    runtime = runtime_with(tmp_path)
    request = write_request()
    runtime.ledger.append(
        Event.create(
            stream_id=request.action_id, kind="action.requested", actor_id="reviewer",
            payload=asdict(request),  # no recovery_version
        )
    )
    state = runtime.action_state("a1")
    assert state.effect_state == EffectState.UNKNOWN
    assert state.next_operation == ActionNextOperation.RECONCILE_EFFECT
    assert state.basis == "record:pre-execution-marking"


# ---------------- reconciliation is append-only evidence ----------------


def test_reconciliation_confirming_an_effect_does_not_erase_the_record(tmp_path):
    runtime = runtime_with(tmp_path)
    runtime.execute_action(write_request(), authority=write_auth(), adapter=CrashAfterEffectWriter())
    before = [(e.kind, e.event_id) for e in runtime.ledger.read_all()]

    state = runtime.reconcile_action(
        "a1",
        verdict=ReconciliationVerdict.EFFECT_CONFIRMED,
        actor_id="operator",
        evidence_refs=("file:crash.txt#marker",),
        basis="read the target after the crash",
    )
    after = [(e.kind, e.event_id) for e in runtime.ledger.read_all()]

    assert after[: len(before)] == before, "reconciliation rewrote history"
    assert after[-1][0] == "action.reconciled"
    assert state.effect_state == EffectState.OBSERVED
    assert state.next_operation == ActionNextOperation.NONE
    assert state.duplicate_effect_risk is False
    assert state.basis == "reconciliation:effect_confirmed"
    assert state.reconciliations[-1].evidence_refs == ("file:crash.txt#marker",)
    assert state.result_status == "failed", "the reported status is still what it was"


def test_reconciliation_can_clear_the_way_for_a_new_attempt(tmp_path):
    runtime = runtime_with(tmp_path)
    runtime.execute_action(write_request(), authority=write_auth(), adapter=CrashAfterEffectWriter())
    state = runtime.reconcile_action(
        "a1", verdict=ReconciliationVerdict.NO_EFFECT_CONFIRMED, actor_id="operator",
        evidence_refs=("file:target.txt#unchanged",),
    )
    assert state.effect_state == EffectState.NONE
    assert state.next_operation == ActionNextOperation.START_ACTION
    assert state.duplicate_effect_risk is False


def test_still_unknown_is_a_valid_final_answer(tmp_path):
    runtime = runtime_with(tmp_path)
    runtime.execute_action(write_request(), authority=write_auth(), adapter=CrashAfterEffectWriter())
    state = runtime.reconcile_action(
        "a1", verdict=ReconciliationVerdict.STILL_UNKNOWN, actor_id="operator",
        basis="the target is unreachable",
    )
    assert state.effect_state == EffectState.UNKNOWN
    assert state.next_operation == ActionNextOperation.RECONCILE_EFFECT
    assert state.duplicate_effect_risk is True
    assert state.basis == "reconciliation:still_unknown"


def test_reconciling_a_settled_action_is_refused_durably(tmp_path):
    runtime = runtime_with(tmp_path)
    runtime.execute_action(write_request(), authority=write_auth(), adapter=CountingWriter())
    with pytest.raises(ReconciliationRefused):
        runtime.reconcile_action(
            "a1", verdict=ReconciliationVerdict.NO_EFFECT_CONFIRMED, actor_id="operator"
        )
    kinds = [e.kind for e in runtime.ledger.read_all()]
    assert kinds[-1] == "action.reconcile_refused", "a refusal only a caller can see"
    assert runtime.action_state("a1").effect_state == EffectState.OBSERVED


def test_unknown_verdicts_are_rejected(tmp_path):
    runtime = runtime_with(tmp_path)
    runtime.execute_action(write_request(), authority=write_auth(), adapter=CrashAfterEffectWriter())
    with pytest.raises(ValueError):
        runtime.reconcile_action("a1", verdict="probably fine", actor_id="operator")


# ---------------- orphan discovery ----------------


def test_open_effects_lists_exactly_the_unknown_ones(tmp_path):
    runtime = runtime_with(tmp_path)
    runtime.execute_action(write_request("ok", key="k1"), authority=write_auth(), adapter=CountingWriter())
    runtime.execute_action(
        write_request("crashed", key="k2"), authority=write_auth(), adapter=CrashAfterEffectWriter()
    )
    runtime.execute_action(
        write_request("other-task", key="k3", task_id="t2"),
        authority=write_auth(),
        adapter=CrashAfterEffectWriter(),
    )

    assert [s.action_id for s in runtime.open_effects()] == ["crashed", "other-task"]
    assert [s.action_id for s in runtime.open_effects(task_id="t")] == ["crashed"]

    runtime.reconcile_action(
        "crashed", verdict=ReconciliationVerdict.NO_EFFECT_CONFIRMED, actor_id="operator"
    )
    assert [s.action_id for s in runtime.open_effects()] == ["other-task"]


# ---------------- the governing question ----------------


@pytest.mark.parametrize(
    "interruption,expected_effect,expected_next",
    [
        ("before_request", None, None),
        ("after_request", EffectState.NONE, ActionNextOperation.START_ACTION),
        ("after_execution_started", EffectState.UNKNOWN, ActionNextOperation.RECONCILE_EFFECT),
        ("after_completion", EffectState.OBSERVED, ActionNextOperation.NONE),
    ],
)
def test_every_interruption_point_answers_safe_unsafe_or_unresolved(
    tmp_path, interruption, expected_effect, expected_next
):
    runtime = runtime_with(tmp_path, state_resolver=lambda: "after")
    request = write_request()
    if interruption != "before_request":
        runtime.ledger.append(
            Event.create(
                stream_id=request.action_id, kind="action.requested", actor_id="reviewer",
                payload={**asdict(request), "recovery_version": ACTION_RECOVERY_V1},
            )
        )
    if interruption in ("after_execution_started", "after_completion"):
        runtime.ledger.append(
            Event.create(
                stream_id=request.action_id, kind="action.execution_started", actor_id="worker",
                payload={"action_id": "a1", "version": ACTION_RECOVERY_V1},
            )
        )
    if interruption == "after_completion":
        runtime.ledger.append(
            Event.create(
                stream_id=request.action_id, kind="action.completed", actor_id="worker",
                payload={"action_id": "a1", "status": "succeeded", "observed_state_hash": "after"},
            )
        )

    state = runtime.action_state("a1")
    if expected_effect is None:
        assert state is None
        return
    assert state.effect_state == expected_effect
    assert state.next_operation == expected_next
    # No interruption point leaves the reader guessing.
    assert state.reason
    assert state.duplicate_effect_risk is (expected_effect == EffectState.UNKNOWN)
