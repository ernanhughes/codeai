"""Chapter 20/29 extension: authority can change; it cannot be bypassed.

Probe N of the composition audit found a task whose directive never granted
ACCEPT reaching a state where completion is *unauthorized* -- the correct
result -- with no legal transition out of it. This is that transition.

    current durable authority has no ACCEPT
          -> the scheduler asks for human intervention
          -> the human cannot bypass the authority model
          -> authority itself changes, durably
          -> re-project
          -> acceptance is enforced the ordinary way

Two relationships, kept apart:

    delegation            parent -> child, may only narrow
    authority transition  old directive -> successor, may add or remove, because
                          it records a new external decision

Directives are immutable: a transition records a successor and leaves the
predecessor alone, so what was true at T1 is still readable at T3.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from codeai.acceptance import (
    TASK_ACCEPTED,
    AcceptanceRejected,
    AcceptanceRequest,
    artifact_target,
    criteria_sha256,
    text_sha256,
)
from codeai.adapters import CallSpec, CheckRequest, CheckResult, CheckVerdict
from codeai.artifacts import FileArtifactStore
from codeai.authority import AUTHORITY_TRANSITION_V1, AuthorityTransitionRefused, TransitionSource
from codeai.context import ContextCompiler
from codeai.domain import ActorRef, Authority, Budget, Capability, Directive, Task
from codeai.ledger import SQLiteLedger
from codeai.providers import HttpResponse, OpenCodeCognitionAdapter
from codeai.runtime import Runtime
from codeai.scheduler import Operation

CRITERIA = ("no percentage figure", "source marker [S1] retained exactly once")
REPAIRED = "The cache is intended to make page loads faster [S1]."


class PassingVerifier:
    def run(self, request: CheckRequest) -> CheckResult:
        return CheckResult(check_id=request.check_id, verdict=CheckVerdict.PASS)


def make_runtime(path: Path) -> Runtime:
    path.mkdir(parents=True, exist_ok=True)
    ledger = SQLiteLedger(path / "ledger.sqlite")
    return Runtime(ledger, artifact_store=FileArtifactStore(path / "artifacts", ledger))


def directive(directive_id, capabilities, parent=None):
    return Directive(
        directive_id=directive_id,
        parent_directive_id=parent,
        objective="fixture",
        success_criteria=(),
        budget=Budget(),
        authority=Authority(frozenset(capabilities)),
    )


def produce(runtime, *, task_id="t1", directive_id="d1"):
    """A recorded call and a passing check: everything but the grant."""
    runtime.create_task(
        Task(task_id, directive_id, "Repair the paragraph", CRITERIA, Budget(), Authority())
    )
    body = json.dumps(
        {"id": "s", "choices": [{"finish_reason": "stop", "message": {"content": REPAIRED}}]}
    ).encode()

    def post(url, payload, headers, timeout):
        return HttpResponse(200, {"request-id": "s"}, body, "application/json")

    actor = ActorRef("repairer", "model", provider="opencode", model="mimo-v2.5")
    context = ContextCompiler().compile(
        task_id=task_id, actor=actor, prompt="Repair: pages load 73% faster [S1].",
        prompt_version="repair-v1",
    )
    spec = CallSpec(str(uuid.uuid4()), task_id, actor, context, str(uuid.uuid4()),
                    chamber="deep-review", parameters={"max_tokens": 256})
    adapter = OpenCodeCognitionAdapter(
        model="mimo-v2.5", protocol="chat_completions", gateway_plan="go",
        api_key="offline-decoy", http_post=post, timeout=60,
    )
    call = runtime.invoke_recorded_call(spec, adapter=adapter, max_attempts=1)
    attempt = call.attempts[-1]
    interpretation = runtime.interpretations_for_attempt(attempt.attempt_id)[-1]
    output = json.loads(runtime.artifact_store.read_text(attempt.raw_artifact.artifact_id))[
        "output_text"
    ]
    runtime.artifact_store.store_text(output, artifact_type="candidate_output")
    sha = text_sha256(output)
    check_id = f"k-{uuid.uuid4()}"
    runtime.run_check(
        CheckRequest(check_id=check_id, task_id=task_id, target=artifact_target(sha)),
        verifier=PassingVerifier(),
    )
    return AcceptanceRequest(
        acceptance_id=str(uuid.uuid4()), task_id=task_id, actor_id="reviewer",
        criteria_sha256=criteria_sha256(CRITERIA), artifact_sha256=sha,
        source_call_id=call.call_id, source_attempt_id=attempt.attempt_id,
        source_interpretation_id=interpretation.interpretation_id, check_ids=(check_id,),
    )


def refused_reasons(runtime, request):
    with pytest.raises(AcceptanceRejected) as refused:
        runtime.accept_task(request)
    return refused.value.reasons


# ---------------- the path probe N could not take ----------------


def test_a_write_only_directive_still_refuses_acceptance(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.open_directive(directive("d1", (Capability.WRITE,)))
    request = produce(runtime)
    assert refused_reasons(runtime, request) == ("acceptance_not_granted:d1",)
    assert runtime.decide_next_for_task("t1").operation == Operation.ASK_HUMAN


def test_a_recorded_transition_opens_the_gate_the_ordinary_way(tmp_path):
    """ASK_HUMAN -> durable authority change -> re-project -> ordinary enforcement."""
    runtime = make_runtime(tmp_path)
    runtime.open_directive(directive("d1", (Capability.WRITE,)))
    request = produce(runtime)
    assert runtime.decide_next_for_task("t1").operation == Operation.ASK_HUMAN

    standing = runtime.transition_authority(
        "d1",
        directive("d1-accept", (Capability.WRITE, Capability.ACCEPT)),
        actor_id="a-named-reviewer",
        reason="the reviewer takes responsibility for accepting this work",
    )
    assert standing.effective_directive_id == "d1-accept"
    assert standing.supersession_chain == ("d1", "d1-accept")
    assert standing.allows("accept")

    # Nothing special about the acceptance itself: it is the ordinary rule.
    assert runtime.accept_task(request).status == "completed"
    [accepted] = runtime.ledger.events_by_kind((TASK_ACCEPTED,))
    basis = accepted.payload["authority_basis"]
    assert basis["status"] == "granted"
    assert basis["effective_directive_id"] == "d1-accept"
    assert basis["supersession_chain"] == ["d1", "d1-accept"]
    assert runtime.decide_next_for_task("t1").operation == Operation.STOP


def test_a_transition_can_take_authority_away(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.open_directive(directive("d1", (Capability.WRITE, Capability.ACCEPT)))
    request = produce(runtime)
    runtime.transition_authority(
        "d1", directive("d1-narrow", (Capability.WRITE,)),
        actor_id="reviewer", reason="acceptance moves to a different approver",
    )
    assert refused_reasons(runtime, request) == ("acceptance_not_granted:d1-narrow",)


def test_a_transition_does_not_rewrite_a_completed_acceptance(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.open_directive(directive("d1", (Capability.WRITE, Capability.ACCEPT)))
    request = produce(runtime)
    assert runtime.accept_task(request).status == "completed"

    runtime.transition_authority(
        "d1", directive("d1-narrow", (Capability.WRITE,)),
        actor_id="reviewer", reason="acceptance authority withdrawn after the fact",
    )
    # The history stays what it was; only what happens next is governed by the new grant.
    assert runtime.task_completion("t1").status == "completed"
    [accepted] = runtime.ledger.events_by_kind((TASK_ACCEPTED,))
    assert accepted.payload["authority_basis"]["effective_directive_id"] == "d1"
    assert runtime.acceptance_authority("t1").granted is False


# ---------------- delegation is still delegation ----------------


def test_a_child_still_cannot_widen_its_parent_after_transitions_exist(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.open_directive(directive("d1", (Capability.WRITE,)))
    with pytest.raises(ValueError):
        runtime.open_directive(
            directive("d1-child", (Capability.WRITE, Capability.ACCEPT), parent="d1")
        )
    assert runtime.ledger.events_by_kind(("directive.registration_refused",))


def test_a_superseding_directive_may_not_declare_a_parent(tmp_path):
    """A narrowing successor under a parent would be delegation wearing a new name."""
    runtime = make_runtime(tmp_path)
    runtime.open_directive(directive("d1", (Capability.WRITE,)))
    with pytest.raises(AuthorityTransitionRefused) as refused:
        runtime.transition_authority(
            "d1", directive("d2", (Capability.WRITE,), parent="d1"),
            actor_id="reviewer", reason="try to sneak a parent in",
        )
    assert "takes no parent" in refused.value.reason
    assert runtime.ledger.events_by_kind(("authority.transition_refused",))


# ---------------- refusals ----------------


def test_an_unknown_predecessor_is_refused(tmp_path):
    runtime = make_runtime(tmp_path)
    with pytest.raises(AuthorityTransitionRefused) as refused:
        runtime.transition_authority(
            "d-never-registered", directive("d2", (Capability.ACCEPT,)),
            actor_id="reviewer", reason="grant acceptance",
        )
    assert "cannot be resolved" in refused.value.reason
    [event] = runtime.ledger.events_by_kind(("authority.transition_refused",))
    assert event.payload["new_directive_id"] == "d2"


def test_a_transition_must_record_why(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.open_directive(directive("d1", (Capability.WRITE,)))
    with pytest.raises(AuthorityTransitionRefused) as refused:
        runtime.transition_authority(
            "d1", directive("d2", (Capability.ACCEPT,)), actor_id="reviewer", reason="",
        )
    assert "must record why" in refused.value.reason


def test_superseding_an_already_superseded_directive_is_refused(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.open_directive(directive("d1", (Capability.WRITE,)))
    runtime.transition_authority(
        "d1", directive("d2", (Capability.WRITE, Capability.ACCEPT)),
        actor_id="reviewer", reason="grant acceptance",
    )
    with pytest.raises(AuthorityTransitionRefused) as refused:
        runtime.transition_authority(
            "d1", directive("d3", (Capability.DESTRUCTIVE,)),
            actor_id="someone-else", reason="a competing successor",
        )
    assert "already been superseded by d2" in refused.value.reason
    # One epoch, one successor: the ambiguity never enters the record.
    assert len(runtime.ledger.events_by_kind(("authority.transitioned",))) == 1


def test_two_competing_successors_refuse_rather_than_choosing_one(tmp_path):
    """A hand-written ledger can still hold the ambiguity. It is not resolved."""
    runtime = make_runtime(tmp_path)
    runtime.open_directive(directive("d1", (Capability.WRITE,)))
    runtime.open_directive(directive("d2", (Capability.ACCEPT,)))
    runtime.open_directive(directive("d3", (Capability.DESTRUCTIVE,)))
    from codeai.ledger import Event

    for successor in ("d2", "d3"):
        runtime.ledger.append(
            Event.create(
                stream_id="d1", kind="authority.transitioned", actor_id="whoever",
                payload={"previous_directive_id": "d1", "new_directive_id": successor,
                         "version": AUTHORITY_TRANSITION_V1},
                correlation_id="d1",
            )
        )
    standing = runtime.directive_authority("d1")
    assert standing.resolvable is False
    assert "ambiguous" in standing.reason
    assert standing.effective_directive_id is None


# ---------------- the record afterwards ----------------


def test_the_history_reads_back_as_history(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.open_directive(directive("d1", (Capability.WRITE,)))
    runtime.transition_authority(
        "d1", directive("d2", (Capability.WRITE, Capability.ACCEPT)),
        actor_id="reviewer-1", reason="grant acceptance for this task",
    )
    runtime.transition_authority(
        "d2", directive("d3", (Capability.READ,)),
        actor_id="reviewer-2", reason="work is finished; withdraw everything but reading",
        source=TransitionSource.EXTERNAL_DECISION,
    )
    history = runtime.authority_history("d1")
    assert [(h["previous_directive_id"], h["new_directive_id"]) for h in history] == [
        ("d1", "d2"), ("d2", "d3")
    ]
    assert [h["previous_effective_capabilities"] for h in history] == [["write"], ["accept", "write"]]
    assert [h["new_effective_capabilities"] for h in history] == [["accept", "write"], ["read"]]
    assert [h["actor_id"] for h in history] == ["reviewer-1", "reviewer-2"]
    assert history[1]["source"] == "external_decision"

    now = runtime.directive_authority("d1")
    assert now.effective_capabilities == ("read",)
    assert now.supersession_chain == ("d1", "d2", "d3")
    # The predecessors are untouched: what was true then is still readable.
    opened = {e.stream_id: e for e in runtime.ledger.events_by_kind(("directive.opened",))}
    assert sorted(opened["d1"].payload["authority"]["capabilities"]) == ["write"]


def test_reopening_reconstructs_the_same_effective_authority(tmp_path):
    path = tmp_path / "run"
    runtime = make_runtime(path)
    runtime.open_directive(directive("d1", (Capability.WRITE,)))
    runtime.transition_authority(
        "d1", directive("d2", (Capability.WRITE, Capability.ACCEPT)),
        actor_id="reviewer", reason="grant acceptance",
    )
    before = runtime.directive_authority("d1")
    events = runtime.ledger.read_all()
    runtime.ledger.close()

    reopened = SQLiteLedger(path / "ledger.sqlite")
    second = Runtime(reopened, artifact_store=FileArtifactStore(path / "artifacts", reopened))
    after = second.directive_authority("d1")
    assert reopened.read_all() == events
    assert after.effective_capabilities == before.effective_capabilities
    assert after.supersession_chain == before.supersession_chain
    assert after.effective_directive_id == "d2"
    reopened.close()


def test_a_transition_invalidates_outstanding_scheduler_decisions(tmp_path):
    """The world the decision was about has changed, so the decision expires."""
    from codeai.governance import GovernanceSource, GovernanceStatus

    from codeai.ledger import Event

    runtime = make_runtime(tmp_path)
    runtime.open_directive(directive("d1", (Capability.WRITE,)))
    runtime.create_task(Task("t1", "d1", "fixture", CRITERIA, Budget(), Authority()))
    runtime.ledger.append(
        Event.create(stream_id="c1", kind="call.completed", actor_id="model",
                     payload={"call_id": "c1", "task_id": "t1", "call_status": "succeeded"},
                     correlation_id="t1")
    )
    assert runtime.decide_next_for_task("t1").operation == Operation.CHECK
    decision = runtime.ledger.events_by_kind(("scheduler.decision_recorded",))[-1].payload

    runtime.transition_authority(
        "d1", directive("d2", (Capability.WRITE, Capability.ACCEPT)),
        actor_id="reviewer", reason="grant acceptance",
    )
    standing = runtime.operation_governance(
        task_id="t1", operation="CHECK", decision_id=decision["decision_id"],
        source=GovernanceSource.SCHEDULER,
    )
    assert standing.status == GovernanceStatus.STALE_BASIS
    assert standing.decision_basis_sha256 != standing.current_basis_sha256


def test_the_actor_is_attribution_not_authentication(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.open_directive(directive("d1", (Capability.WRITE,)))
    runtime.transition_authority(
        "d1", directive("d2", (Capability.ACCEPT,)),
        actor_id="someone-claiming-to-be-the-director",
        reason="claims the authority to grant acceptance",
    )
    [event] = runtime.ledger.events_by_kind(("authority.transitioned",))
    # The record says an explicit transition attributed to this actor entered the
    # process here. It does not say the actor was entitled to make it.
    assert event.payload["actor_id"] == "someone-claiming-to-be-the-director"
    assert event.payload["source"] == "human_intervention"
    assert event.payload["reason"]
    assert "entitled" not in json.dumps(event.payload)
