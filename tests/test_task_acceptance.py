"""Stage 14: explicit task acceptance is the only path to task completion.

Offline only: synthetic transport through the recorded call path, static
verifiers, no network, no credentials. The repaired paragraph below is a
fixture, not evidence of model quality.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from codeai import acceptance
from codeai.acceptance import (
    TASK_ACCEPTANCE_REJECTED,
    TASK_ACCEPTED,
    TASK_COMPLETED,
    AcceptanceRejected,
    AcceptanceRequest,
    artifact_target,
    criteria_sha256,
    text_sha256,
)
from codeai.adapters import CallSpec, CheckRequest, CheckResult, CheckVerdict
from codeai.artifacts import FileArtifactStore
from codeai.context import ContextCompiler
from codeai.domain import ActorRef, Authority, Budget, Capability, Task
from codeai.ledger import Event, SQLiteLedger
from codeai.providers import HttpResponse, OpenCodeCognitionAdapter
from codeai.runtime import Runtime

CRITERIA = ("no percentage figure", "source marker [S1] retained exactly once")
REPAIRED = "The cache is intended to make page loads faster [S1]."
ACCEPTOR = Authority(frozenset({Capability.ACCEPT}))


@dataclass
class Produced:
    runtime: Runtime
    task_id: str
    call_id: str
    attempt_id: str
    interpretation_id: str
    artifact_sha256: str


class StaticVerifier:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict

    def run(self, request: CheckRequest) -> CheckResult:
        return CheckResult(
            check_id=request.check_id,
            verdict=self.verdict,
            started_at="2026-09-13T00:00:00+00:00",
            completed_at="2026-09-13T00:00:00+00:00",
            exit_code=0 if self.verdict == CheckVerdict.PASS else 1,
            inconclusive_reason=(
                "the fixture cannot disambiguate this criterion"
                if self.verdict == CheckVerdict.INCONCLUSIVE
                else None
            ),
        )


def make_runtime(path: Path) -> Runtime:
    ledger = SQLiteLedger(path / "ledger.sqlite")
    return Runtime(ledger, artifact_store=FileArtifactStore(path / "artifacts", ledger))


def chat_post(text: str, finish_reason: str):
    body = json.dumps(
        {"id": "synthetic", "choices": [{"finish_reason": finish_reason, "message": {"content": text}}]}
    ).encode()

    def post(url, payload, headers, timeout):
        return HttpResponse(200, {"request-id": "synthetic"}, body, "application/json")

    return post


def produce(
    runtime: Runtime,
    *,
    task_id: str = "task-repair",
    text: str = REPAIRED,
    finish_reason: str = "stop",
    store: bool = True,
) -> Produced:
    runtime.create_task(
        Task(task_id, "run-repair", "Repair the paragraph", CRITERIA, Budget(), Authority())
    )
    actor = ActorRef("repairer", "model", provider="opencode", model="mimo-v2.5")
    context = ContextCompiler().compile(
        task_id=task_id, actor=actor, prompt="Repair: pages load 73% faster [S1].",
        prompt_version="repair-v1",
    )
    spec = CallSpec(
        str(uuid.uuid4()), task_id, actor, context, str(uuid.uuid4()),
        chamber="deep-review", parameters={"max_tokens": 256},
    )
    adapter = OpenCodeCognitionAdapter(
        model="mimo-v2.5", protocol="chat_completions", gateway_plan="go",
        api_key="offline-decoy", http_post=chat_post(text, finish_reason), timeout=60,
    )
    call = runtime.invoke_recorded_call(spec, adapter=adapter, max_attempts=1)
    attempt = call.attempts[-1]
    interpretation = runtime.interpretations_for_attempt(attempt.attempt_id)[-1]
    envelope = runtime.artifact_store.read_text(attempt.raw_artifact.artifact_id)
    output = json.loads(envelope)["output_text"]
    if store:
        runtime.artifact_store.store_text(output, artifact_type="candidate_output")
    return Produced(
        runtime, task_id, call.call_id, attempt.attempt_id,
        interpretation.interpretation_id, text_sha256(output),
    )


def run_check(p: Produced, verdict: str = CheckVerdict.PASS, *, task_id=None, sha=None) -> str:
    check_id = f"check-{uuid.uuid4()}"
    p.runtime.run_check(
        CheckRequest(
            check_id=check_id,
            task_id=task_id or p.task_id,
            target=artifact_target(sha or p.artifact_sha256),
        ),
        verifier=StaticVerifier(verdict),
    )
    return check_id


def request_for(p: Produced, check_ids, **overrides) -> AcceptanceRequest:
    fields = {
        "acceptance_id": str(uuid.uuid4()),
        "task_id": p.task_id,
        "actor_id": "reviewer",
        "criteria_sha256": criteria_sha256(CRITERIA),
        "artifact_sha256": p.artifact_sha256,
        "source_call_id": p.call_id,
        "source_attempt_id": p.attempt_id,
        "source_interpretation_id": p.interpretation_id,
        "check_ids": tuple(check_ids),
    }
    fields.update(overrides)
    return AcceptanceRequest(**fields)


def kinds(runtime: Runtime) -> list[str]:
    return [event.kind for event in runtime.ledger.read_all()]


def rejected_reasons(p: Produced, request: AcceptanceRequest, authority=ACCEPTOR) -> tuple[str, ...]:
    with pytest.raises(AcceptanceRejected) as excinfo:
        p.runtime.accept_task(request, authority=authority)
    assert TASK_COMPLETED not in kinds(p.runtime)
    assert TASK_ACCEPTED not in kinds(p.runtime)
    assert kinds(p.runtime).count(TASK_ACCEPTANCE_REJECTED) >= 1
    assert p.runtime.task_completion(p.task_id).status == "incomplete"
    return excinfo.value.reasons


# ---------------- the transition ----------------


def test_succeeded_call_and_passing_check_are_not_completion(tmp_path):
    p = produce(make_runtime(tmp_path))
    assert p.runtime.get_recorded_call(p.call_id).status == "succeeded"
    run_check(p)
    completion = p.runtime.task_completion(p.task_id)
    assert completion.status == "incomplete"
    assert completion.basis == "no acceptance recorded"
    assert TASK_COMPLETED not in kinds(p.runtime)


def test_valid_acceptance_completes_once_and_is_causally_linked(tmp_path):
    p = produce(make_runtime(tmp_path))
    check_id = run_check(p)
    completion = p.runtime.accept_task(request_for(p, [check_id]), authority=ACCEPTOR)
    assert completion.status == "completed"
    assert completion.artifact_sha256 == p.artifact_sha256
    assert completion.source_call_id == p.call_id
    assert completion.check_ids == (check_id,)
    [accepted] = p.runtime.ledger.events_by_kind((TASK_ACCEPTED,))
    [completed] = p.runtime.ledger.events_by_kind((TASK_COMPLETED,))
    assert completed.causation_id == accepted.event_id
    assert accepted.payload["authority_basis"] == ["accept"]
    assert completion.acceptance_event_id == accepted.event_id
    assert completion.completion_event_id == completed.event_id


def test_identical_repeat_is_idempotent(tmp_path):
    p = produce(make_runtime(tmp_path))
    check_id = run_check(p)
    request = request_for(p, [check_id])
    first = p.runtime.accept_task(request, authority=ACCEPTOR)
    before = len(p.runtime.ledger.read_all())
    again = p.runtime.accept_task(replace(request, acceptance_id="another-id"), authority=ACCEPTOR)
    assert len(p.runtime.ledger.read_all()) == before
    assert again == first


def test_conflicting_repeat_fails_visibly_and_keeps_one_completion(tmp_path):
    p = produce(make_runtime(tmp_path))
    first_check, second_check = run_check(p), run_check(p)
    p.runtime.accept_task(request_for(p, [first_check]), authority=ACCEPTOR)
    with pytest.raises(AcceptanceRejected) as excinfo:
        p.runtime.accept_task(request_for(p, [first_check, second_check]), authority=ACCEPTOR)
    assert excinfo.value.reasons == ("conflicting_acceptance",)
    assert kinds(p.runtime).count(TASK_COMPLETED) == 1
    completion = p.runtime.task_completion(p.task_id)
    assert completion.status == "completed"
    assert completion.rejected_acceptance_count == 1


def test_reopened_runtime_projects_the_same_completion(tmp_path):
    p = produce(make_runtime(tmp_path))
    completion = p.runtime.accept_task(request_for(p, [run_check(p)]), authority=ACCEPTOR)
    assert make_runtime(tmp_path).task_completion(p.task_id) == completion


# ---------------- negatives ----------------


def test_missing_check_cannot_complete(tmp_path):
    p = produce(make_runtime(tmp_path))
    assert rejected_reasons(p, request_for(p, [])) == ("missing_check",)


@pytest.mark.parametrize(
    "verdict", [CheckVerdict.FAIL, CheckVerdict.ERROR, CheckVerdict.INCONCLUSIVE]
)
def test_non_passing_check_cannot_complete(tmp_path, verdict):
    p = produce(make_runtime(tmp_path))
    check_id = run_check(p, verdict)
    assert rejected_reasons(p, request_for(p, [check_id])) == (
        f"check_not_passed:{check_id}:{verdict.value}",
    )


def test_unknown_check_cannot_complete(tmp_path):
    p = produce(make_runtime(tmp_path))
    assert rejected_reasons(p, request_for(p, ["check-nowhere"])) == (
        "check_not_found:check-nowhere",
    )


def test_check_recorded_for_another_task_cannot_complete(tmp_path):
    p = produce(make_runtime(tmp_path))
    produce(p.runtime, task_id="task-other")
    check_id = run_check(p, task_id="task-other")
    assert rejected_reasons(p, request_for(p, [check_id])) == (f"check_wrong_task:{check_id}",)


def test_check_of_different_bytes_cannot_complete(tmp_path):
    p = produce(make_runtime(tmp_path))
    check_id = run_check(p, sha=text_sha256("something else"))
    assert rejected_reasons(p, request_for(p, [check_id])) == (
        f"check_wrong_artifact:{check_id}",
    )


def test_changed_artifact_with_its_own_passing_check_cannot_complete(tmp_path):
    p = produce(make_runtime(tmp_path))
    edited = REPAIRED.replace("is intended to make", "should make")
    edited_ref = p.runtime.artifact_store.store_text(edited, artifact_type="candidate_output")
    check_id = run_check(p, sha=edited_ref.sha256)
    reasons = rejected_reasons(p, request_for(p, [check_id], artifact_sha256=edited_ref.sha256))
    assert reasons == ("artifact_not_source_output",)


def test_call_from_another_task_cannot_complete(tmp_path):
    p = produce(make_runtime(tmp_path))
    other = produce(p.runtime, task_id="task-other")
    check_id = run_check(p, sha=other.artifact_sha256)
    request = request_for(
        p, [check_id],
        artifact_sha256=other.artifact_sha256,
        source_call_id=other.call_id,
        source_attempt_id=other.attempt_id,
        source_interpretation_id=other.interpretation_id,
    )
    assert rejected_reasons(p, request) == ("source_call_wrong_task",)


def test_unauthorized_acceptor_cannot_complete(tmp_path):
    p = produce(make_runtime(tmp_path))
    everything_but_accept = Authority(frozenset(set(Capability) - {Capability.ACCEPT}))
    reasons = rejected_reasons(p, request_for(p, [run_check(p)]), authority=everything_but_accept)
    assert reasons == ("unauthorized",)


def test_producer_cannot_accept_its_own_output(tmp_path):
    p = produce(make_runtime(tmp_path))
    assert rejected_reasons(p, request_for(p, [run_check(p)], actor_id="repairer")) == (
        "self_acceptance",
    )


def test_truncated_generation_cannot_complete(tmp_path):
    p = produce(make_runtime(tmp_path), finish_reason="length")
    reasons = rejected_reasons(p, request_for(p, [run_check(p)]))
    assert "source_call_not_succeeded" in reasons
    assert "generation_not_complete" in reasons


def test_criteria_other_than_declared_cannot_complete(tmp_path):
    p = produce(make_runtime(tmp_path))
    request = request_for(p, [run_check(p)], criteria_sha256=criteria_sha256(("looks fine",)))
    assert rejected_reasons(p, request) == ("criteria_mismatch",)


def test_unpreserved_artifact_cannot_complete(tmp_path):
    p = produce(make_runtime(tmp_path), store=False)
    assert rejected_reasons(p, request_for(p, [run_check(p)])) == ("artifact_not_preserved",)


def test_completion_event_without_acceptance_projects_incomplete(tmp_path):
    p = produce(make_runtime(tmp_path))
    p.runtime.ledger.append(
        Event.create(
            stream_id=p.task_id, kind=TASK_COMPLETED, actor_id="repairer",
            payload={"task_id": p.task_id, "artifact_sha256": p.artifact_sha256},
        )
    )
    completion = p.runtime.task_completion(p.task_id)
    assert completion.status == "incomplete"
    assert completion.basis == "task.completed present without a causing task.accepted"


# ---------------- interruption window and compatibility ----------------


def test_acceptance_without_completion_is_repaired_by_identical_repeat(tmp_path, monkeypatch):
    p = produce(make_runtime(tmp_path))
    request = request_for(p, [run_check(p)])

    def interrupted(runtime, accepted):
        raise RuntimeError("simulated interruption between appends")

    monkeypatch.setattr(acceptance, "_append_completion", interrupted)
    with pytest.raises(RuntimeError):
        p.runtime.accept_task(request, authority=ACCEPTOR)
    monkeypatch.undo()

    reopened = make_runtime(tmp_path)
    pending = reopened.task_completion(p.task_id)
    assert pending.status == "incomplete" and pending.acceptance_pending_completion
    completion = reopened.accept_task(request, authority=ACCEPTOR)
    assert completion.status == "completed"
    assert kinds(reopened).count(TASK_ACCEPTED) == 1
    assert kinds(reopened).count(TASK_COMPLETED) == 1


def test_unknown_and_historical_tasks(tmp_path):
    runtime = make_runtime(tmp_path)
    assert runtime.task_completion("never-created").status == "unknown_task"
    runtime.create_task(Task("t-old", "run-old", "old", ("x",), Budget(), Authority()))
    before = len(runtime.ledger.read_all())
    assert runtime.task_completion("t-old").status == "incomplete"
    assert len(runtime.ledger.read_all()) == before
