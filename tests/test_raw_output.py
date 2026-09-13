"""Stage 17: preserve once, interpret many times.

Offline only. A synthetic provider counts every request it receives, so each
test can show that reinterpretation reads preserved bytes and never asks the
provider again. The truncated response is a fixture, not model evidence.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

from codeai.acceptance import (
    AcceptanceRejected,
    AcceptanceRequest,
    artifact_target,
    criteria_sha256,
    text_sha256,
)
from codeai.adapters import CallSpec, CheckRequest, CheckResult, CheckVerdict, FakeCognitionAdapter
from codeai.artifacts import FileArtifactStore
from codeai.context import ContextCompiler
from codeai.domain import ActorRef, Authority, Budget, Capability, Task
from codeai.interpretation import (
    ATTEMPT_POLICY_V1,
    ATTEMPT_POLICY_V2,
    INTERPRETER_V1,
    INTERPRETER_V2,
    ObservationUnavailable,
)
from codeai.ledger import SQLiteLedger
from codeai.providers import HttpResponse, OpenCodeCognitionAdapter
from codeai.runtime import Runtime

ACTOR = ActorRef("reviewer", "model", provider="opencode", model="mimo-v2.5")
CRITERIA = ("names the missing evidence",)
TRUNCATED_TEXT = "The cache claim is missing a"
TRUNCATED = {"id": "synthetic-17", "model": "mimo-v2.5",
             "choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": TRUNCATED_TEXT}}]}


class Crash(BaseException):
    pass


class Provider:
    def __init__(self) -> None:
        self.requests = 0

    def __call__(self, url, body, headers, timeout):
        self.requests += 1
        return HttpResponse(200, {}, json.dumps(TRUNCATED).encode(), "application/json")


def make_runtime(path: Path) -> Runtime:
    path.mkdir(parents=True, exist_ok=True)
    ledger = SQLiteLedger(path / "ledger.sqlite")
    return Runtime(ledger, artifact_store=FileArtifactStore(path / "artifacts", ledger))


def spec() -> CallSpec:
    package = ContextCompiler().compile(task_id="task-17", actor=ACTOR, prompt="Name the missing evidence.")
    return CallSpec(call_id="call-17", task_id="task-17", actor=ACTOR, context=package, idempotency_key="key-17",
                    chamber="deep-review", parameters={"max_tokens": 64})


def adapter(provider):
    return OpenCodeCognitionAdapter(model="mimo-v2.5", protocol="chat_completions", gateway_plan="go",
                                    api_key="offline-decoy", http_post=provider, timeout=5)


def observed_under_v1(path: Path):
    """Day one: the call runs under the historical v1 interpreter and policy."""
    runtime = make_runtime(path)
    runtime.create_task(Task("task-17", "run-17", "Review", CRITERIA, Budget(), Authority()))
    provider = Provider()
    recorded = runtime.invoke_recorded_call(spec(), adapter=adapter(provider),
                                            interpreter_version=INTERPRETER_V1, policy_version=ATTEMPT_POLICY_V1)
    return runtime, provider, recorded


def body_path(runtime: Runtime, attempt_id: str) -> Path:
    ref = runtime.get_attempt_observation(attempt_id)["response_body_artifact"]
    return Path(runtime.ledger.read_artifact(ref["artifact_id"]).uri)


def event_ids(runtime: Runtime) -> list[str]:
    return [e.event_id for e in runtime.ledger.read_all()]


def test_v1_accepted_a_truncated_response(tmp_path):
    runtime, provider, recorded = observed_under_v1(tmp_path)
    assert provider.requests == 1 and recorded.status == "succeeded"
    [interpretation] = runtime.interpretations_for_attempt(recorded.attempts[0].attempt_id)
    assert interpretation.interpreter_version == INTERPRETER_V1
    assert interpretation.generation_state == "complete"


def test_reinterpret_call_changes_the_conclusion_not_the_history(tmp_path):
    _day_one, provider, recorded = observed_under_v1(tmp_path)
    attempt_id = recorded.attempts[0].attempt_id
    runtime = make_runtime(tmp_path)  # day two: a different process, same files
    path = body_path(runtime, attempt_id)
    body_sha_before = hashlib.sha256(path.read_bytes()).hexdigest()
    history = event_ids(runtime)

    result = runtime.reinterpret_call("call-17", interpreter_version=INTERPRETER_V2, policy_version=ATTEMPT_POLICY_V2)

    assert result["prior_status"] == "succeeded" and result["status"] == "unresolved"
    assert provider.requests == 1
    assert hashlib.sha256(path.read_bytes()).hexdigest() == body_sha_before
    assert event_ids(runtime)[: len(history)] == history
    v1, v2 = runtime.interpretations_for_attempt(attempt_id)
    assert (v1.interpreter_version, v1.generation_state) == (INTERPRETER_V1, "complete")
    assert (v2.interpreter_version, v2.generation_state, v2.provider_reason) == (INTERPRETER_V2, "truncated", "length")
    [observed] = runtime.ledger.events_by_kind(("attempt.observed",))
    v2_event = next(e for e in runtime.ledger.events_by_kind(("attempt.interpreted",))
                    if e.payload["interpreter_version"] == INTERPRETER_V2)
    assert v2_event.causation_id == observed.event_id
    assert result["interpretation_ids"] == [v2.interpretation_id]
    assert result["observations"] == [{"attempt_id": attempt_id, "sha256": body_sha_before}]


def test_reinterpretation_is_idempotent_per_version(tmp_path):
    runtime, _provider, _recorded = observed_under_v1(tmp_path)
    first = runtime.reinterpret_call("call-17", interpreter_version=INTERPRETER_V2, policy_version=ATTEMPT_POLICY_V2)
    count = len(runtime.ledger.read_all())
    again = runtime.reinterpret_call("call-17", interpreter_version=INTERPRETER_V2, policy_version=ATTEMPT_POLICY_V2)
    assert again == first and len(runtime.ledger.read_all()) == count


def test_downstream_projections_follow_the_adopted_interpretation(tmp_path):
    runtime, _provider, recorded = observed_under_v1(tmp_path)
    attempt = recorded.attempts[0]
    before = runtime.work_state("task-17")
    assert before.calls[0].call_status == "succeeded" and before.task_next_operation == "check_and_accept"
    v1_id = runtime.interpretations_for_attempt(attempt.attempt_id)[0].interpretation_id

    runtime.reinterpret_call("call-17", interpreter_version=INTERPRETER_V2, policy_version=ATTEMPT_POLICY_V2)

    after = runtime.work_state("task-17")
    assert after.calls[0].call_status == "unresolved"
    assert INTERPRETER_V2 in (after.calls[0].status_basis or "")
    assert after.task_next_operation == "start_call"

    runtime.artifact_store.store_text(TRUNCATED_TEXT, artifact_type="candidate_output")

    class Pass:
        def run(self, request):
            return CheckResult(check_id=request.check_id, verdict=CheckVerdict.PASS)

    runtime.run_check(CheckRequest(check_id="check-17", task_id="task-17", target=artifact_target(
        text_sha256(TRUNCATED_TEXT))), verifier=Pass())
    request = AcceptanceRequest("acc-17", "task-17", "human-reviewer", criteria_sha256(CRITERIA),
                                text_sha256(TRUNCATED_TEXT), "call-17", attempt.attempt_id, v1_id, ("check-17",))
    with pytest.raises(AcceptanceRejected) as refused:
        runtime.accept_task(request, authority=Authority(frozenset({Capability.ACCEPT})))
    assert set(refused.value.reasons) == {"source_call_not_succeeded", "interpretation_not_decision_basis"}


@contextmanager
def crash_after(kind: str):
    original = SQLiteLedger.append

    def append(self, event):
        result = original(self, event)
        if event.kind == kind:
            raise Crash(kind)
        return result

    with patch.object(SQLiteLedger, "append", append):
        yield


def test_reinterpret_attempt_advances_an_observed_but_uninterpreted_attempt(tmp_path):
    runtime = make_runtime(tmp_path)
    provider = Provider()
    with crash_after("attempt.observed"), pytest.raises(Crash):
        runtime.invoke_recorded_call(spec(), adapter=adapter(provider))
    reopened = make_runtime(tmp_path)
    [attempt_started] = reopened.ledger.events_by_kind(("attempt.started",))
    attempt_id = attempt_started.payload["attempt_id"]
    assert reopened.work_state("task-17").calls[0].stage == "observed"
    assert reopened.interpret_attempt_as(attempt_id, version=INTERPRETER_V2) is None  # needs a completion record

    interpretation = reopened.reinterpret_attempt(attempt_id, interpreter_version=INTERPRETER_V2)

    assert interpretation.generation_state == "truncated" and provider.requests == 1
    state = reopened.work_state("task-17").calls[0]
    assert (state.stage, state.next_operation) == ("interpreted", "redecide")


@pytest.mark.parametrize("damage", ["corrupt", "delete"])
def test_missing_or_corrupt_bytes_refuse_and_append_nothing(tmp_path, damage):
    runtime, provider, recorded = observed_under_v1(tmp_path)
    path = body_path(runtime, recorded.attempts[0].attempt_id)
    if damage == "corrupt":
        path.write_bytes(path.read_bytes().replace(b"length", b"stop!!"))
    else:
        path.unlink()
    count = len(runtime.ledger.read_all())
    with pytest.raises(ObservationUnavailable):
        runtime.reinterpret_call("call-17", interpreter_version=INTERPRETER_V2, policy_version=ATTEMPT_POLICY_V2)
    with pytest.raises(ObservationUnavailable):
        runtime.reinterpret_attempt(recorded.attempts[0].attempt_id, interpreter_version=INTERPRETER_V2)
    assert len(runtime.ledger.read_all()) == count and provider.requests == 1


def test_attempt_without_a_preserved_observation_cannot_be_reinterpreted(tmp_path):
    runtime = make_runtime(tmp_path)
    recorded = runtime.invoke_recorded_call(spec(), adapter=FakeCognitionAdapter(responses=["text"]))
    with pytest.raises(ObservationUnavailable, match="attempt.observed"):
        runtime.reinterpret_attempt(recorded.attempts[0].attempt_id, interpreter_version=INTERPRETER_V2)


def test_unknown_versions_rejected_before_anything_is_appended(tmp_path):
    runtime, _provider, _recorded = observed_under_v1(tmp_path)
    count = len(runtime.ledger.read_all())
    with pytest.raises(ValueError):
        runtime.reinterpret_call("call-17", interpreter_version="attempt-interpretation-v9",
                                 policy_version=ATTEMPT_POLICY_V2)
    with pytest.raises(ValueError):
        runtime.reinterpret_call("call-17", interpreter_version=INTERPRETER_V2, policy_version="attempt-policy-v9")
    assert len(runtime.ledger.read_all()) == count
