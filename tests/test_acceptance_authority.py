"""Chapter 20/29: acceptance authority comes from the record, like everything else.

Composition audit gap 2. The action path resolved its grant from the recorded
directive chain; acceptance took an ``Authority`` object from whoever called it.
Two halves of one question, answered by two different standards.

    caller supplies Authority({ACCEPT})  ->  runtime trusts caller     (before)
    task.created names a directive       ->  runtime resolves it       (after)

The directive is read from the task record, never from the request: a caller who
could name the directive could name a permissive one.

This resolves authority from the durable record. It does not authenticate the
acceptor -- ``actor_id`` stays attribution, not identity.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from codeai.acceptance import (
    TASK_ACCEPTANCE_REJECTED,
    TASK_ACCEPTED,
    AcceptanceRejected,
    AcceptanceRequest,
    artifact_target,
    criteria_sha256,
    text_sha256,
)
from codeai.adapters import CallSpec, CheckRequest, CheckResult, CheckVerdict
from codeai.artifacts import FileArtifactStore
from codeai.context import ContextCompiler
from codeai.domain import ActorRef, Authority, Budget, Capability, Directive, Task
from codeai.ledger import Event, SQLiteLedger
from codeai.providers import HttpResponse, OpenCodeCognitionAdapter
from codeai.runtime import Runtime

CRITERIA = ("no percentage figure", "source marker [S1] retained exactly once")
REPAIRED = "The cache is intended to make page loads faster [S1]."
CLAIMS_ACCEPT = Authority(frozenset({Capability.ACCEPT}))


class PassingVerifier:
    def run(self, request: CheckRequest) -> CheckResult:
        return CheckResult(check_id=request.check_id, verdict=CheckVerdict.PASS)


def make_runtime(path: Path) -> Runtime:
    path.mkdir(parents=True, exist_ok=True)
    ledger = SQLiteLedger(path / "ledger.sqlite")
    return Runtime(ledger, artifact_store=FileArtifactStore(path / "artifacts", ledger))


def directive(runtime, directive_id, capabilities, parent=None):
    return runtime.open_directive(
        Directive(
            directive_id=directive_id,
            parent_directive_id=parent,
            objective="fixture",
            success_criteria=(),
            budget=Budget(),
            authority=Authority(frozenset(capabilities)),
        )
    )


def produce(runtime, *, task_id="t1", directive_id="d-root"):
    """A recorded call plus a passing check: everything acceptance needs but the grant."""
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
    artifact_sha = text_sha256(output)
    check_id = f"k-{uuid.uuid4()}"
    runtime.run_check(
        CheckRequest(check_id=check_id, task_id=task_id, target=artifact_target(artifact_sha)),
        verifier=PassingVerifier(),
    )
    return AcceptanceRequest(
        acceptance_id=str(uuid.uuid4()),
        task_id=task_id,
        actor_id="reviewer",
        criteria_sha256=criteria_sha256(CRITERIA),
        artifact_sha256=artifact_sha,
        source_call_id=call.call_id,
        source_attempt_id=attempt.attempt_id,
        source_interpretation_id=interpretation.interpretation_id,
        check_ids=(check_id,),
    )


def reasons_for(runtime, request, **kwargs):
    with pytest.raises(AcceptanceRejected) as refused:
        runtime.accept_task(request, **kwargs)
    return refused.value.reasons


# ---------------- the matrix ----------------


def test_a_directive_that_grants_accept_permits_acceptance(tmp_path):
    runtime = make_runtime(tmp_path)
    directive(runtime, "d-root", (Capability.WRITE, Capability.ACCEPT))
    completion = produce(runtime)
    assert runtime.accept_task(completion).status == "completed"


def test_a_directive_without_accept_refuses_and_the_caller_cannot_widen_it(tmp_path):
    """The frozen hostile case: recorded WRITE only, caller claims WRITE+ACCEPT."""
    runtime = make_runtime(tmp_path)
    directive(runtime, "d-root", (Capability.WRITE,))
    request = produce(runtime)

    claimed = Authority(frozenset({Capability.WRITE, Capability.ACCEPT}))
    assert reasons_for(runtime, request, authority=claimed) == ("acceptance_not_granted:d-root",)
    assert runtime.task_completion("t1").status != "completed"

    [rejected] = runtime.ledger.events_by_kind((TASK_ACCEPTANCE_REJECTED,))
    assert rejected.payload["directive_id"] == "d-root"
    assert rejected.payload["authority_basis"]["status"] == "denied"
    # What the caller claimed is recorded as a claim, and it changed nothing.
    assert rejected.payload["caller_claimed_capabilities"] == ["accept", "write"]


def test_a_child_directive_may_narrow_accept_away(tmp_path):
    runtime = make_runtime(tmp_path)
    directive(runtime, "d-root", (Capability.WRITE, Capability.ACCEPT))
    directive(runtime, "d-child", (Capability.WRITE,), parent="d-root")
    request = produce(runtime, directive_id="d-child")
    assert reasons_for(runtime, request, authority=CLAIMS_ACCEPT) == (
        "acceptance_not_granted:d-child",
    )


def test_a_child_directive_cannot_enlarge_its_parent(tmp_path):
    """Registration refuses the widening, so the chain never offers ACCEPT."""
    runtime = make_runtime(tmp_path)
    directive(runtime, "d-root", (Capability.WRITE,))
    with pytest.raises(ValueError):
        directive(runtime, "d-child", (Capability.WRITE, Capability.ACCEPT), parent="d-root")
    refusals = runtime.ledger.events_by_kind(("directive.registration_refused",))
    assert refusals, "the attempt to widen is durable"

    request = produce(runtime, directive_id="d-child")
    reasons = reasons_for(runtime, request, authority=CLAIMS_ACCEPT)
    assert reasons == ("acceptance_authority_unresolved:unknown_directive:d-child",)


def test_a_task_that_names_no_directive_is_refused_rather_than_falling_back(tmp_path):
    """A task.created with no directive -- the shape an older ledger can hold."""
    runtime = make_runtime(tmp_path)
    directive(runtime, "d-root", (Capability.WRITE, Capability.ACCEPT))
    request = produce(runtime)
    runtime.ledger.append(
        Event.create(stream_id="t2", kind="task.created", actor_id="human",
                     payload={"task_id": "t2", "success_criteria": list(CRITERIA)},
                     correlation_id="t2")
    )
    orphan = replace(request, task_id="t2", acceptance_id=str(uuid.uuid4()))
    assert reasons_for(runtime, orphan, authority=CLAIMS_ACCEPT) == (
        "acceptance_authority_unresolved:task_names_no_directive",
    )


def test_an_unrecorded_directive_is_refused_rather_than_falling_back(tmp_path):
    runtime = make_runtime(tmp_path)
    request = produce(runtime, directive_id="d-never-registered")
    assert reasons_for(runtime, request, authority=CLAIMS_ACCEPT) == (
        "acceptance_authority_unresolved:unknown_directive:d-never-registered",
    )


def test_an_ambiguous_chain_is_refused(tmp_path):
    runtime = make_runtime(tmp_path)
    directive(runtime, "d-root", (Capability.ACCEPT,))
    directive(runtime, "d-root", (Capability.ACCEPT,))  # a second root registration
    request = produce(runtime)
    reasons = reasons_for(runtime, request, authority=CLAIMS_ACCEPT)
    assert reasons[0].startswith("acceptance_authority_unresolved:")


# ---------------- the decision survives the record being reread ----------------


def test_the_acceptance_records_the_directive_and_chain_it_rested_on(tmp_path):
    runtime = make_runtime(tmp_path)
    directive(runtime, "d-root", (Capability.WRITE, Capability.ACCEPT))
    directive(runtime, "d-child", (Capability.ACCEPT,), parent="d-root")
    runtime.accept_task(produce(runtime, directive_id="d-child"))

    [accepted] = runtime.ledger.events_by_kind((TASK_ACCEPTED,))
    basis = accepted.payload["authority_basis"]
    assert accepted.payload["directive_id"] == "d-child"
    assert [link["directive_id"] for link in basis["grant_chain"]] == ["d-child", "d-root"]
    assert basis["effective_capabilities"] == ["accept"]
    assert len(basis["basis_event_ids"]) == 2


def test_reopening_the_ledger_resolves_the_same_acceptance_authority(tmp_path):
    path = tmp_path / "run"
    runtime = make_runtime(path)
    directive(runtime, "d-root", (Capability.WRITE, Capability.ACCEPT))
    runtime.accept_task(produce(runtime))
    before = runtime.acceptance_authority("t1")
    events = runtime.ledger.read_all()
    runtime.ledger.close()

    reopened = SQLiteLedger(path / "ledger.sqlite")
    second = Runtime(reopened, artifact_store=FileArtifactStore(path / "artifacts", reopened))
    after = second.acceptance_authority("t1")
    assert reopened.read_all() == events
    assert (after.status, after.granted) == (before.status, before.granted)
    assert after.standing.grant_chain == before.standing.grant_chain
    assert second.task_completion("t1").status == "completed"
    reopened.close()


def test_the_actor_label_is_attribution_not_authentication(tmp_path):
    """Anyone may be named as the acceptor; the grant is what is checked."""
    runtime = make_runtime(tmp_path)
    directive(runtime, "d-root", (Capability.WRITE, Capability.ACCEPT))
    request = produce(runtime)
    completion = runtime.accept_task(
        AcceptanceRequest(**{**request.identity(), "acceptance_id": request.acceptance_id,
                             "actor_id": "someone-who-says-they-are-a-reviewer"})
    )
    assert completion.status == "completed"
    [accepted] = runtime.ledger.events_by_kind((TASK_ACCEPTED,))
    assert accepted.actor_id == "someone-who-says-they-are-a-reviewer"
    # The record says who claimed to accept, and which grant permitted it. It
    # does not say that this actor is who they say they are.
    assert accepted.payload["authority_basis"]["grant_source"] == "recorded_directive"


def test_authority_is_decided_before_any_prior_acceptance_is_disclosed(tmp_path):
    """A caller without a current grant learns the refusal, not the history."""
    runtime = make_runtime(tmp_path)
    directive(runtime, "d-root", (Capability.WRITE, Capability.ACCEPT))
    request = produce(runtime)
    runtime.accept_task(request)

    # The same task, now asked for again through a chain that cannot be resolved.
    runtime.ledger.append(
        runtime.ledger.read_all()[0].__class__.create(
            stream_id="d-root", kind="directive.opened", actor_id="someone",
            payload={"directive_id": "d-root", "authority": {"capabilities": ["accept"]}},
            correlation_id="d-root",
        )
    )
    reasons = reasons_for(runtime, request, authority=CLAIMS_ACCEPT)
    assert reasons[0].startswith("acceptance_authority_unresolved:")
    assert not any("conflicting" in reason for reason in reasons)
