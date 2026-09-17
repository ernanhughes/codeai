"""Chapter 28 seam: the scheduler chooses, but it never invents the facts.

    ProcessState = what is true (projected from the ledger)
    scheduler    = what to do about it (a pure function of that state)

The decision is recorded with the facts it rested on, so "what the scheduler
decided then" survives "what today's scheduler would decide".
"""

import json
from dataclasses import replace

import pytest

from codeai.actions import ReconciliationVerdict
from codeai.adapters import ActionRequest, ActionResult, ActionStatus, CheckRequest, CheckResult, CheckVerdict
from codeai.domain import Authority, Budget, Capability, Directive, Task
from codeai.ledger import Event, SQLiteLedger
from codeai.process_state import PROCESS_STATE_V1, project_process_state, state_snapshot
from codeai.runtime import Runtime
from codeai.scheduler import POLICY_VERSION, Operation, SchedulerInput, decide_next_step


class CountingWriter:
    def __init__(self):
        self.calls = 0

    def execute(self, request: ActionRequest) -> ActionResult:
        self.calls += 1
        return ActionResult(action_id=request.action_id, status=ActionStatus.SUCCEEDED)


class CrashAfterEffectWriter(CountingWriter):
    def execute(self, request: ActionRequest) -> ActionResult:
        self.calls += 1
        raise ConnectionError("connection reset after the write")


class PassingVerifier:
    def run(self, request: CheckRequest) -> CheckResult:
        return CheckResult(check_id=request.check_id, verdict=CheckVerdict.PASS, exit_code=0)


def build(tmp_path, *, capabilities=(Capability.WRITE,), criteria=("the marker is gone",),
          max_tokens=None, max_turns=None):
    runtime = Runtime(SQLiteLedger(tmp_path / "ledger.sqlite"))
    runtime.open_directive(
        Directive(
            directive_id="d",
            objective="fixture",
            success_criteria=(),
            budget=Budget(max_tokens=max_tokens, max_turns=max_turns),
            authority=Authority(frozenset(capabilities)),
        )
    )
    runtime.create_task(
        Task(
            task_id="t",
            directive_id="d",
            objective="repair the paragraph",
            success_criteria=criteria,
            budget=Budget(max_tokens=max_tokens),
            authority=Authority(frozenset(capabilities)),
        )
    )
    return runtime


def record_call(runtime, call_id="c1", status="succeeded", tokens=(10, 5)):
    runtime.ledger.append(
        Event.create(
            stream_id=call_id, kind="call.completed", actor_id="model",
            payload={
                "call_id": call_id, "task_id": "t", "call_status": status,
                "total_input_tokens": tokens[0], "total_output_tokens": tokens[1],
            },
            correlation_id="t",
        )
    )


# ---------------- the matrix ----------------


def test_no_proposals_yet_means_call(tmp_path):
    runtime = build(tmp_path, criteria=())
    assert runtime.decide_next_for_task("t").operation == Operation.CALL


def test_an_outstanding_required_check_means_check(tmp_path):
    runtime = build(tmp_path)
    record_call(runtime)
    assert runtime.decide_next_for_task("t").operation == Operation.CHECK


def test_a_satisfied_check_and_missing_accept_grant_means_ask_human(tmp_path):
    runtime = build(tmp_path)  # the directive grants WRITE, never ACCEPT
    record_call(runtime)
    runtime.run_check(CheckRequest(check_id="k1", task_id="t"), verifier=PassingVerifier())

    state = runtime.process_state("t")
    assert state.check_satisfied
    assert state.acceptance_required and not state.acceptance_authority_available
    assert runtime.decide_next_for_task("t").operation == Operation.ASK_HUMAN


def test_an_accept_grant_in_the_record_does_not_need_a_person(tmp_path):
    runtime = build(tmp_path, capabilities=(Capability.WRITE, Capability.ACCEPT))
    record_call(runtime)
    runtime.run_check(CheckRequest(check_id="k1", task_id="t"), verifier=PassingVerifier())

    assert runtime.process_state("t").acceptance_authority_available
    assert runtime.decide_next_for_task("t").operation == Operation.STOP


def test_an_exhausted_model_budget_blocks_calls_but_not_checks(tmp_path):
    runtime = build(tmp_path, max_tokens=10)
    record_call(runtime, tokens=(8, 4))  # 12 >= 10

    state = runtime.process_state("t")
    assert state.model_budget.consumed == 12
    assert state.model_budget_exhausted
    # A check is still owed, and a spent model budget must not silence it.
    assert runtime.decide_next_for_task("t").operation == Operation.CHECK


def test_an_exhausted_process_budget_stops_everything(tmp_path):
    runtime = build(tmp_path, max_turns=1)
    record_call(runtime)

    state = runtime.process_state("t")
    assert state.process_budget.consumed == 1 and state.process_budget_exhausted
    assert runtime.decide_next_for_task("t").operation == Operation.STOP


def test_an_unresolved_effect_outranks_the_ordinary_next_step(tmp_path):
    runtime = build(tmp_path, criteria=())
    runtime.execute_action(
        ActionRequest(
            action_id="a1", task_id="t", directive_id="d", capability="write",
            instruction="write", precondition_hash=None, idempotency_key="k",
            requested_by="human", actor_id="worker", adapter="fixture",
        ),
        adapter=CrashAfterEffectWriter(),
    )

    state = runtime.process_state("t")
    assert state.unresolved_effects == ("a1",)
    decision = runtime.decide_next_for_task("t")
    assert decision.operation == Operation.ASK_HUMAN
    assert decision.reason == "unresolved_effect_requires_reconciliation"

    # Reconciling the effect clears the block without erasing the record.
    runtime.reconcile_action(
        "a1", verdict=ReconciliationVerdict.EFFECT_CONFIRMED, actor_id="operator",
        evidence_refs=("file:target#written",),
    )
    assert runtime.process_state("t").unresolved_effects == ()
    assert runtime.decide_next_for_task("t").operation == Operation.CALL


def test_a_completed_process_stops(tmp_path):
    runtime = build(tmp_path, criteria=())
    state = runtime.process_state("t")
    assert state.process_complete is False
    assert decide_next_step(state.to_scheduler_input()).operation == Operation.CALL

    completed = replace(state, process_complete=True)
    decision = decide_next_step(completed.to_scheduler_input())
    assert decision.operation == Operation.STOP
    assert decision.reason == "the process is already complete"


def test_a_hand_appended_completion_does_not_stop_the_process(tmp_path):
    # Chapter 14: completion counts only when an acceptance caused it. The
    # projection inherits that rule rather than believing the event.
    runtime = build(tmp_path, criteria=())
    for kind in ("task.accepted", "task.completed"):
        runtime.ledger.append(
            Event.create(stream_id="t", kind=kind, actor_id="whoever",
                         payload={"task_id": "t"}, correlation_id="t")
        )
    assert runtime.process_state("t").process_complete is False
    assert runtime.decide_next_for_task("t").operation == Operation.CALL


def test_declared_criteria_are_not_an_owed_check(tmp_path):
    # A criterion is a standing requirement. A check is owed only once there is
    # something to examine, so the first decision is CALL, not CHECK.
    runtime = build(tmp_path)  # criteria declared, nothing produced
    state = runtime.process_state("t")
    assert state.criteria_declared is True
    assert state.check_required is False
    assert any("nothing has been produced to check" in note for note in state.notes)
    assert runtime.decide_next_for_task("t").operation == Operation.CALL

    record_call(runtime)
    after = runtime.process_state("t")
    assert after.criteria_declared and after.check_required
    assert runtime.decide_next_for_task("t").operation == Operation.CHECK


# ---------------- the facts are not the caller's to invent ----------------


def test_the_projection_contradicts_a_caller_who_says_no_check_is_owed(tmp_path):
    runtime = build(tmp_path)  # criteria declared, no check run
    record_call(runtime)

    fabricated = SchedulerInput(has_required_verification=False)
    assert decide_next_step(fabricated).operation != Operation.CHECK

    # The task-level API does not accept that claim: it reads the ledger.
    state = runtime.process_state("t")
    assert state.check_required and not state.check_satisfied
    assert runtime.decide_next_for_task("t").operation == Operation.CHECK


def test_an_unknown_budget_is_never_reported_as_exhausted(tmp_path):
    runtime = build(tmp_path, criteria=())
    record_call(runtime, tokens=(10_000, 10_000))

    state = runtime.process_state("t")
    assert state.model_budget.known is False
    assert state.model_budget_exhausted is False
    assert any("token limit" in note for note in state.notes)


def test_the_state_carries_the_events_it_was_read_from(tmp_path):
    runtime = build(tmp_path)
    record_call(runtime)
    state = runtime.process_state("t")
    recorded_ids = {event.event_id for event in runtime.ledger.read_all()}
    assert state.basis_event_ids
    assert set(state.basis_event_ids) <= recorded_ids


def test_the_projection_states_what_it_could_not_establish(tmp_path):
    runtime = Runtime(SQLiteLedger(tmp_path / "ledger.sqlite"))
    state = runtime.process_state("missing-task")
    assert any("no task.created" in note for note in state.notes)


# ---------------- the decision is history, not a live recomputation ----------------


def test_the_recorded_decision_carries_its_facts_and_survives_a_changed_state(tmp_path):
    runtime = build(tmp_path)
    record_call(runtime)

    first = runtime.decide_next_for_task("t")
    assert first.operation == Operation.CHECK
    recorded = runtime.ledger.events_by_kind(("scheduler.decision_recorded",))[0]
    assert recorded.payload["operation"] == "CHECK"
    assert recorded.payload["policy_version"] == POLICY_VERSION
    assert recorded.payload["projection_version"] == PROCESS_STATE_V1
    assert recorded.payload["state"]["criteria_declared"] is True
    assert recorded.payload["state"]["check_required"] is True
    assert recorded.payload["state"]["check_satisfied"] is False
    assert recorded.payload["basis_event_ids"]

    # The world moves on: the check runs and passes.
    runtime.run_check(CheckRequest(check_id="k1", task_id="t"), verifier=PassingVerifier())
    second = runtime.decide_next_for_task("t")
    assert second.operation == Operation.ASK_HUMAN

    # The first decision still says what it said, on the facts it had.
    unchanged = runtime.ledger.events_by_kind(("scheduler.decision_recorded",))[0]
    assert unchanged.payload == recorded.payload


def test_a_recorded_decision_replays_to_the_same_answer(tmp_path):
    runtime = build(tmp_path)
    record_call(runtime)
    decision = runtime.decide_next_for_task("t")
    recorded = runtime.ledger.events_by_kind(("scheduler.decision_recorded",))[0].payload

    snapshot = recorded["state"]
    replayed = decide_next_step(
        SchedulerInput(
            has_required_verification=snapshot["check_required"] and not snapshot["check_satisfied"],
            requests_independent_proposals=snapshot["proposal_count"] == 0,
            requires_human_authority_for_next_effect=(
                snapshot["acceptance_required"] and not snapshot["acceptance_authority_available"]
            ),
            process_budget_exhausted=False,
            model_budget_exhausted=False,
            unresolved_effect=bool(snapshot["unresolved_effects"]),
        )
    )
    assert replayed.operation == decision.operation
    assert replayed.policy_version == recorded["policy_version"]


def test_the_state_hash_pins_the_facts(tmp_path):
    import hashlib

    runtime = build(tmp_path)
    record_call(runtime)
    runtime.decide_next_for_task("t")
    recorded = runtime.ledger.events_by_kind(("scheduler.decision_recorded",))[0].payload
    expected = hashlib.sha256(
        json.dumps(recorded["state"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert recorded["process_state_sha256"] == expected


# ---------------- a decision is not an effect ----------------


def test_deciding_call_does_not_start_one(tmp_path):
    runtime = build(tmp_path, criteria=())
    decision = runtime.decide_next_for_task("t")
    assert decision.operation == Operation.CALL
    kinds = {event.kind for event in runtime.ledger.read_all()}
    assert "call.manifest" not in kinds and "attempt.started" not in kinds
    # Re-deciding after a crash is free, because no effect boundary was crossed.
    assert runtime.decide_next_for_task("t").operation == Operation.CALL


def test_action_is_unreachable_from_the_scheduler(tmp_path):
    import itertools

    for flags in itertools.product((False, True), repeat=6):
        query = SchedulerInput(
            has_required_verification=flags[0],
            requests_independent_proposals=flags[1],
            requires_human_authority_for_next_effect=flags[2],
            process_budget_exhausted=flags[3],
            model_budget_exhausted=flags[4],
            unresolved_effect=flags[5],
        )
        assert decide_next_step(query).operation != Operation.ACTION


def test_state_snapshot_is_json_serializable(tmp_path):
    runtime = build(tmp_path)
    json.dumps(state_snapshot(project_process_state(runtime, "t")))
