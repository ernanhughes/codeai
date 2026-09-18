"""Chapter 28/29: a governed operation must carry the decision that selected it.

Composition audit gap 4. The scheduler decided correctly and the runtime executed
whatever it was asked; a CHECK decision and a WRITE action merely coexisted.

The rule is narrower than "every effect needs a decision":

    a process decision must never be silently bypassed while the resulting
    operation still appears to belong to that governed process

so two shapes are legitimate, and the record tells them apart:

    scheduler-governed   names a decision, which the runtime checks
    external or manual   names none, and cannot masquerade as governed

Governance is not authority. This layer answers *what should happen next*; the
directive chain answers *may this actor do it*. Both must pass.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from codeai.adapters import (
    ActionRequest,
    ActionResult,
    ActionStatus,
    CallSpec,
    CheckRequest,
    CheckResult,
    CheckVerdict,
)
from codeai.artifacts import FileArtifactStore
from codeai.context import ContextCompiler
from codeai.domain import ActorRef, Authority, Budget, Capability, Directive, Task
from codeai.governance import (
    DECISION_BINDING_V1,
    GovernanceRefused,
    GovernanceSource,
    GovernanceStatus,
)
from codeai.ledger import SQLiteLedger
from codeai.providers import HttpResponse, OpenCodeCognitionAdapter
from codeai.runtime import Runtime
from codeai.scheduler import Operation

CRITERIA = ("no percentage figure",)


class PassingVerifier:
    def run(self, request: CheckRequest) -> CheckResult:
        return CheckResult(check_id=request.check_id, verdict=CheckVerdict.PASS)


class CountingWriter:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ActionRequest) -> ActionResult:
        self.calls += 1
        return ActionResult(action_id=request.action_id, status=ActionStatus.SUCCEEDED)


def make_runtime(path: Path, *, criteria=CRITERIA, task_id="t1") -> Runtime:
    path.mkdir(parents=True, exist_ok=True)
    ledger = SQLiteLedger(path / "ledger.sqlite")
    runtime = Runtime(ledger, artifact_store=FileArtifactStore(path / "artifacts", ledger))
    runtime.open_directive(
        Directive(directive_id="d", objective="fixture", success_criteria=(), budget=Budget(),
                  authority=Authority(frozenset({Capability.WRITE, Capability.ACCEPT})))
    )
    runtime.create_task(Task(task_id, "d", "fixture", criteria, Budget(), Authority()))
    return runtime


def spec(runtime, task_id="t1"):
    actor = ActorRef("repairer", "model", provider="opencode", model="mimo-v2.5")
    context = ContextCompiler().compile(
        task_id=task_id, actor=actor, prompt="repair", prompt_version="v1"
    )
    return CallSpec(str(uuid.uuid4()), task_id, actor, context, str(uuid.uuid4()),
                    chamber="deep-review", parameters={"max_tokens": 64})


def model(text="repaired [S1]."):
    body = json.dumps(
        {"id": "s", "choices": [{"finish_reason": "stop", "message": {"content": text}}]}
    ).encode()

    def post(url, payload, headers, timeout):
        return HttpResponse(200, {"request-id": "s"}, body, "application/json")

    return OpenCodeCognitionAdapter(
        model="mimo-v2.5", protocol="chat_completions", gateway_plan="go",
        api_key="offline-decoy", http_post=post, timeout=60,
    )


def action(action_id="a1", task_id="t1"):
    return ActionRequest(
        action_id=action_id, task_id=task_id, directive_id="d", capability="write",
        instruction="write", precondition_hash=None, idempotency_key=f"k-{action_id}",
        requested_by="human", actor_id="worker", adapter="fixture",
    )


def decision_id_for(runtime, task_id="t1"):
    """Take a real decision and return its id."""
    runtime.decide_next_for_task(task_id)
    recorded = [
        event for event in runtime.ledger.events_by_kind(("scheduler.decision_recorded",))
        if event.payload["task_id"] == task_id
    ]
    return recorded[-1].payload["decision_id"], recorded[-1].payload["operation"]


def governance_events(runtime, subject_id):
    return [
        event
        for event in runtime.ledger.events_by_kind(
            ("operation.governance_recorded", "operation.governance_refused")
        )
        if event.payload["subject_id"] == subject_id
    ]


# ---------------- the matrix ----------------


def test_a_call_decision_permits_a_call(tmp_path):
    runtime = make_runtime(tmp_path, criteria=())
    decision_id, operation = decision_id_for(runtime)
    assert operation == "CALL"

    recorded = runtime.invoke_recorded_call(
        spec(runtime), adapter=model(), decision_id=decision_id,
        source=GovernanceSource.SCHEDULER,
    )
    assert recorded.call_id
    [event] = governance_events(runtime, recorded.call_id)
    assert event.kind == "operation.governance_recorded"
    assert event.payload["status"] == GovernanceStatus.GOVERNED
    assert event.payload["decided_operation"] == "CALL"
    assert event.payload["decision_basis_sha256"] == event.payload["current_basis_sha256"]
    assert event.payload["version"] == DECISION_BINDING_V1


def test_a_check_decision_permits_a_check(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.ledger.append(
        runtime.ledger.read_all()[0].__class__.create(
            stream_id="c1", kind="call.completed", actor_id="model",
            payload={"call_id": "c1", "task_id": "t1", "call_status": "succeeded"},
            correlation_id="t1",
        )
    )
    decision_id, operation = decision_id_for(runtime)
    assert operation == "CHECK"

    result = runtime.run_check(
        CheckRequest("k1", "t1"), verifier=PassingVerifier(),
        decision_id=decision_id, source=GovernanceSource.SCHEDULER,
    )
    assert result.verdict == CheckVerdict.PASS
    [event] = governance_events(runtime, "k1")
    assert event.payload["status"] == GovernanceStatus.GOVERNED


def test_a_check_decision_does_not_permit_an_action(tmp_path):
    """The headline case: a CHECK decision cannot silently become a WRITE."""
    runtime = make_runtime(tmp_path)
    runtime.ledger.append(
        runtime.ledger.read_all()[0].__class__.create(
            stream_id="c1", kind="call.completed", actor_id="model",
            payload={"call_id": "c1", "task_id": "t1", "call_status": "succeeded"},
            correlation_id="t1",
        )
    )
    decision_id, operation = decision_id_for(runtime)
    assert operation == "CHECK"

    writer = CountingWriter()
    result = runtime.execute_action(
        action(), adapter=writer, decision_id=decision_id, source=GovernanceSource.SCHEDULER
    )
    assert result.status == ActionStatus.FAILED
    assert writer.calls == 0
    [event] = governance_events(runtime, "a1")
    assert event.kind == "operation.governance_refused"
    assert event.payload["status"] == GovernanceStatus.NOT_SELECTABLE
    assert "effects are authorized on their own path" in event.payload["reason"]
    # It never reached authority, and no effect was possible.
    kinds = [e.kind for e in runtime.ledger.read_all() if e.stream_id == "a1"]
    assert kinds == ["action.requested", "operation.governance_refused", "action.completed"]
    assert runtime.action_state("a1").effect_state == "none"


def test_an_ask_human_decision_does_not_permit_a_call(tmp_path):
    runtime = make_runtime(tmp_path)
    for kind, payload in (
        ("call.completed", {"call_id": "c1", "task_id": "t1", "call_status": "succeeded"}),
        ("check.completed", {"check_id": "k0", "task_id": "t1", "verdict": "PASS"}),
    ):
        runtime.ledger.append(
            runtime.ledger.read_all()[0].__class__.create(
                stream_id=payload.get("call_id") or payload["check_id"], kind=kind,
                actor_id="fixture", payload=payload, correlation_id="t1",
            )
        )
    # No ACCEPT in this task's chain would make it ASK_HUMAN; here the grant
    # exists, so drive the gate directly through a task that lacks it.
    runtime.open_directive(
        Directive(directive_id="d-gated", objective="gated", success_criteria=(), budget=Budget(),
                  authority=Authority(frozenset({Capability.WRITE})))
    )
    runtime.create_task(Task("t2", "d-gated", "fixture", (), Budget(), Authority()))
    runtime.ledger.append(
        runtime.ledger.read_all()[0].__class__.create(
            stream_id="c2", kind="call.completed", actor_id="model",
            payload={"call_id": "c2", "task_id": "t2", "call_status": "succeeded"},
            correlation_id="t2",
        )
    )
    decision_id, operation = decision_id_for(runtime, "t2")
    assert operation == "ASK_HUMAN"

    with pytest.raises(GovernanceRefused) as refused:
        runtime.invoke_recorded_call(
            spec(runtime, "t2"), adapter=model(), decision_id=decision_id,
            source=GovernanceSource.SCHEDULER,
        )
    assert refused.value.standing.status == GovernanceStatus.WRONG_OPERATION
    assert "selected ASK_HUMAN, not CALL" in refused.value.standing.reason


def test_a_stop_decision_permits_nothing_governed(tmp_path):
    runtime = make_runtime(tmp_path, criteria=())
    runtime.ledger.append(
        runtime.ledger.read_all()[0].__class__.create(
            stream_id="c1", kind="call.completed", actor_id="model",
            payload={"call_id": "c1", "task_id": "t1", "call_status": "succeeded"},
            correlation_id="t1",
        )
    )
    decision_id, operation = decision_id_for(runtime)
    assert operation == "STOP"

    with pytest.raises(GovernanceRefused):
        runtime.invoke_recorded_call(
            spec(runtime), adapter=model(), decision_id=decision_id,
            source=GovernanceSource.SCHEDULER,
        )
    result = runtime.run_check(
        CheckRequest("k1", "t1"), verifier=PassingVerifier(),
        decision_id=decision_id, source=GovernanceSource.SCHEDULER,
    )
    assert result.verdict == CheckVerdict.ERROR
    assert "selected STOP" in result.error


def test_an_unknown_decision_is_refused(tmp_path):
    runtime = make_runtime(tmp_path, criteria=())
    writer = CountingWriter()
    result = runtime.execute_action(
        action(), adapter=writer, decision_id="no-such-decision",
        source=GovernanceSource.SCHEDULER,
    )
    assert result.status == ActionStatus.FAILED
    assert writer.calls == 0
    [event] = governance_events(runtime, "a1")
    assert event.payload["status"] == GovernanceStatus.UNKNOWN_DECISION


def test_a_decision_from_another_task_is_refused(tmp_path):
    runtime = make_runtime(tmp_path, criteria=())
    runtime.create_task(Task("t2", "d", "other", (), Budget(), Authority()))
    decision_id, _ = decision_id_for(runtime, "t1")

    result = runtime.run_check(
        CheckRequest("k1", "t2"), verifier=PassingVerifier(),
        decision_id=decision_id, source=GovernanceSource.SCHEDULER,
    )
    assert result.verdict == CheckVerdict.ERROR
    [event] = governance_events(runtime, "k1")
    assert event.payload["status"] == GovernanceStatus.WRONG_TASK


def test_a_stale_decision_is_refused(tmp_path):
    """The world moved after the decision, so it no longer describes it."""
    runtime = make_runtime(tmp_path)
    runtime.ledger.append(
        runtime.ledger.read_all()[0].__class__.create(
            stream_id="c1", kind="call.completed", actor_id="model",
            payload={"call_id": "c1", "task_id": "t1", "call_status": "succeeded"},
            correlation_id="t1",
        )
    )
    decision_id, operation = decision_id_for(runtime)
    assert operation == "CHECK"

    # A second proposal arrives: the process state is no longer what was decided on.
    runtime.ledger.append(
        runtime.ledger.read_all()[0].__class__.create(
            stream_id="c2", kind="call.completed", actor_id="model",
            payload={"call_id": "c2", "task_id": "t1", "call_status": "succeeded"},
            correlation_id="t1",
        )
    )
    result = runtime.run_check(
        CheckRequest("k1", "t1"), verifier=PassingVerifier(),
        decision_id=decision_id, source=GovernanceSource.SCHEDULER,
    )
    assert result.verdict == CheckVerdict.ERROR
    [event] = governance_events(runtime, "k1")
    assert event.payload["status"] == GovernanceStatus.STALE_BASIS
    assert event.payload["decision_basis_sha256"] != event.payload["current_basis_sha256"]


def test_claiming_scheduler_governance_without_a_decision_is_refused(tmp_path):
    """The masquerade this seam exists to prevent."""
    runtime = make_runtime(tmp_path, criteria=())
    writer = CountingWriter()
    result = runtime.execute_action(
        action(), adapter=writer, decision_id=None, source=GovernanceSource.SCHEDULER
    )
    assert result.status == ActionStatus.FAILED
    assert writer.calls == 0
    [event] = governance_events(runtime, "a1")
    assert event.payload["status"] == GovernanceStatus.UNKNOWN_DECISION
    assert "without naming a decision" in event.payload["reason"]


# ---------------- the external path stays open, and stays visible ----------------


def test_an_external_operation_runs_and_says_what_it_was(tmp_path):
    runtime = make_runtime(tmp_path, criteria=())
    writer = CountingWriter()
    result = runtime.execute_action(action(), adapter=writer)
    assert result.status == ActionStatus.SUCCEEDED
    assert writer.calls == 1

    [event] = governance_events(runtime, "a1")
    assert event.kind == "operation.governance_recorded"
    assert event.payload["status"] == GovernanceStatus.UNGOVERNED
    assert event.payload["source"] == GovernanceSource.EXTERNAL_REQUEST
    assert event.payload["decision_id"] is None


def test_a_human_override_is_recorded_as_one(tmp_path):
    runtime = make_runtime(tmp_path, criteria=())
    writer = CountingWriter()
    runtime.execute_action(
        action(), adapter=writer, source=GovernanceSource.HUMAN_OVERRIDE
    )
    [event] = governance_events(runtime, "a1")
    assert event.payload["source"] == GovernanceSource.HUMAN_OVERRIDE
    assert event.payload["status"] == GovernanceStatus.UNGOVERNED
    # Attribution, not authentication: nothing here says who the human was.
    assert event.payload["decision_id"] is None


def test_governance_is_not_authority(tmp_path):
    """Both gates run, in order, and they answer different questions."""
    runtime = make_runtime(tmp_path, criteria=())
    runtime.open_directive(
        Directive(directive_id="d-read", objective="read only", success_criteria=(),
                  budget=Budget(), authority=Authority(frozenset({Capability.READ})))
    )
    runtime.create_task(Task("t3", "d-read", "fixture", (), Budget(), Authority()))
    writer = CountingWriter()

    from dataclasses import replace as dc_replace

    request = dc_replace(action("a3", task_id="t3"), directive_id="d-read")
    result = runtime.execute_action(request, adapter=writer)
    # Governance permits it (external), authority refuses it.
    assert result.status == ActionStatus.DENIED
    assert writer.calls == 0
    [governance] = governance_events(runtime, "a3")
    assert governance.payload["status"] == GovernanceStatus.UNGOVERNED
    kinds = [e.kind for e in runtime.ledger.read_all() if e.stream_id == "a3"]
    assert kinds == [
        "action.requested", "operation.governance_recorded",
        "action.authorization_refused", "action.completed",
    ]


def test_the_projection_answers_without_appending(tmp_path):
    runtime = make_runtime(tmp_path, criteria=())
    decision_id, _ = decision_id_for(runtime)
    before = len(runtime.ledger.read_all())
    standing = runtime.operation_governance(
        task_id="t1", operation=Operation.CALL, decision_id=decision_id,
        source=GovernanceSource.SCHEDULER,
    )
    assert standing.governed is True
    assert len(runtime.ledger.read_all()) == before
