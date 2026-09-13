"""Stage 16: restart is reopening; resume is knowing what is safe next.

Offline only. Interruptions are injected at ledger event boundaries (or inside
the provider transport) with a BaseException the runtime does not catch, so the
ledger holds exactly what a crashed process would have left.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

from codeai.adapters import CallSpec
from codeai.artifacts import FileArtifactStore
from codeai.context import RequiredContextMissing
from codeai.domain import ActorRef, Authority, Budget, Claim, Seal, Task
from codeai.ledger import Event, SQLiteLedger
from codeai.providers import HttpResponse, OpenCodeCognitionAdapter
from codeai.rendering import CONTEXT_RENDER_V1
from codeai.runtime import Runtime
from codeai.workstate import NextOperation, ResumeRefused

ACTOR = ActorRef("reviewer", "model", provider="opencode", model="mimo-v2.5")
INSTRUCTION = "You are reviewing one paragraph against its source."
QUERY = "Name the missing evidence in one sentence."
SOURCE_B = "SOURCE-B: The cache claim needs a benchmark citation [S1].\n"


class Crash(BaseException):
    """Simulated process death: not an Exception, so nothing in the runtime catches it."""


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


class Transport:
    def __init__(self, crash: bool = False) -> None:
        self.bodies: list[dict] = []
        self.crash = crash

    def __call__(self, url, body, headers, timeout):
        self.bodies.append(body)
        if self.crash:
            raise Crash("died while the provider was serving the request")
        payload = {"id": "synthetic", "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}
        return HttpResponse(200, {}, json.dumps(payload).encode(), "application/json")


def opencode(transport, model="mimo-v2.5"):
    return OpenCodeCognitionAdapter(model=model, protocol="chat_completions", gateway_plan="go",
                                    api_key="offline-decoy", http_post=transport, timeout=5)


def make_runtime(path: Path) -> Runtime:
    path.mkdir(parents=True, exist_ok=True)
    ledger = SQLiteLedger(path / "ledger.sqlite")
    return Runtime(ledger, artifact_store=FileArtifactStore(path / "artifacts", ledger))


def setup(path: Path):
    runtime = make_runtime(path)
    runtime.create_task(Task("task-w", "run-w", "Review", ("names missing evidence",), Budget(), Authority()))
    a = Event("A", "task-w", "fixture.objective", "human", {"text": "Review the paragraph."}, "2026-09-13T00:00:00Z")
    runtime.ledger.append(a)
    b = runtime.artifact_store.store_text(SOURCE_B, artifact_type="source").artifact_id
    runtime.record_claim(Claim("D", "task-w", "The cache claim lacks a benchmark.", "source-call"))
    compile_args = {
        "actor": ACTOR, "prompt": QUERY, "events": [a], "artifact_ids": [b], "claim_ids": ["D"],
        "required_event_ids": {"A"}, "required_artifact_ids": {b}, "prompt_version": "work-v1",
        "seal": Seal(forbidden_call_ids=frozenset({"sibling-call"})),
    }
    return runtime, compile_args


def spec_for(package, key="key-1", call_id="call-1"):
    return CallSpec(call_id=call_id, task_id="task-w", actor=ACTOR, context=package, idempotency_key=key,
                    instruction=INSTRUCTION, chamber="deep-review", parameters={"max_tokens": 64},
                    context_render=CONTEXT_RENDER_V1)


def call_state(runtime, call_id):
    return next(c for c in runtime.work_state("task-w").calls if c.call_id == call_id)


def interrupted_call(tmp_path, kind):
    runtime, args = setup(tmp_path)
    package, _ = runtime.compile_and_record_context(task_id="task-w", **args)
    transport = Transport()
    with crash_after(kind), pytest.raises(Crash):
        runtime.invoke_recorded_call(spec_for(package), adapter=opencode(transport))
    return make_runtime(tmp_path), transport  # a fresh runtime over the same files


# ---------------- compilation is recorded, including what was offered ----------------


def test_compilation_records_offered_inventory_and_outcome(tmp_path):
    runtime, args = setup(tmp_path)
    package, _ = runtime.compile_and_record_context(task_id="task-w", **args)
    [requested] = runtime.ledger.events_by_kind(("context.compilation_requested",))
    [compiled] = runtime.ledger.events_by_kind(("context.compiled",))
    assert compiled.payload["compilation_id"] == requested.payload["compilation_id"]
    assert compiled.causation_id == requested.event_id
    offered = {c["candidate_id"]: c for c in requested.payload["candidates"]}
    assert set(offered) == {"A", "D", *package.artifact_ids}
    assert offered["A"]["required"] is True and offered["D"]["required"] is False
    [state] = make_runtime(tmp_path).work_state("task-w").compilations
    assert state.stage == "compiled" and state.package_id == package.package_id


def test_failed_compilation_is_recorded_and_projected(tmp_path):
    runtime, args = setup(tmp_path)
    args["required_artifact_ids"] = set(args["required_artifact_ids"]) | {"never-offered"}
    with pytest.raises(RequiredContextMissing):
        runtime.compile_and_record_context(task_id="task-w", **args)
    [state] = make_runtime(tmp_path).work_state("task-w").compilations
    assert state.stage == "failed" and state.next_operation == NextOperation.FIX_INPUTS
    assert state.failure == "RequiredContextMissing"
    assert state.missing_required_ids == ("never-offered",)


def test_interrupted_compilation_projects_recompile(tmp_path):
    runtime, args = setup(tmp_path)
    with crash_after("context.compilation_requested"), pytest.raises(Crash):
        runtime.compile_and_record_context(task_id="task-w", **args)
    [state] = make_runtime(tmp_path).work_state("task-w").compilations
    assert state.stage == "interrupted" and state.next_operation == NextOperation.RECOMPILE


# ---------------- calls: which interruptions are safe to resume ----------------


@pytest.mark.parametrize(("kind", "stage"), [("call.requested", "requested"), ("call.manifest", "manifest_recorded")])
def test_no_effect_interruption_resumes_once(tmp_path, kind, stage):
    runtime, first = interrupted_call(tmp_path, kind)
    assert first.bodies == []
    state = call_state(runtime, "call-1")
    assert state.stage == stage and state.provider_effect == "none"
    assert state.next_operation == NextOperation.START_CALL and not state.duplicate_effect_risk
    transport = Transport()
    resumed = runtime.resume_call("call-1", adapter=opencode(transport))
    assert resumed.status == "succeeded" and len(transport.bodies) == 1
    after = {c.call_id: c for c in runtime.work_state("task-w").calls}
    assert after["call-1"].superseded_by == resumed.call_id
    assert after["call-1"].resumed_as == resumed.call_id
    assert after[resumed.call_id].stage == "completed"
    [link] = runtime.ledger.events_by_kind(("call.resumed",))
    assert link.payload["idempotency_key"] == "key-1" and link.payload["prior_stage"] == stage


def test_resume_reproduces_the_recorded_request(tmp_path):
    runtime, _ = interrupted_call(tmp_path, "call.manifest")
    [old_manifest] = runtime.ledger.events_by_kind(("call.manifest",))
    resumed = runtime.resume_call("call-1", adapter=opencode(Transport()))
    assert resumed.manifest.request_body_sha256 == old_manifest.payload["request_body_sha256"]
    assert resumed.manifest.rendered_context_sha256 == old_manifest.payload["rendered_context_sha256"]


def test_resume_refuses_when_recorded_intent_drifted(tmp_path):
    runtime, _ = interrupted_call(tmp_path, "call.manifest")
    before = len(runtime.ledger.read_all())
    transport = Transport()
    with pytest.raises(ResumeRefused, match="request_body_sha256"):
        runtime.resume_call("call-1", adapter=opencode(transport, model="a-different-model"))
    assert transport.bodies == [] and len(runtime.ledger.read_all()) == before


def test_effect_unknown_is_never_resumed(tmp_path):
    runtime, args = setup(tmp_path)
    package, _ = runtime.compile_and_record_context(task_id="task-w", **args)
    dying = Transport(crash=True)
    with pytest.raises(Crash):
        runtime.invoke_recorded_call(spec_for(package), adapter=opencode(dying))
    assert len(dying.bodies) == 1  # the provider received the request
    reopened = make_runtime(tmp_path)
    state = call_state(reopened, "call-1")
    assert state.stage == "effect_unknown" and state.provider_effect == "unknown"
    assert state.next_operation == NextOperation.RECONCILE_EFFECT and state.duplicate_effect_risk
    before = len(reopened.ledger.read_all())
    transport = Transport()
    with pytest.raises(ResumeRefused):
        reopened.resume_call("call-1", adapter=opencode(transport))
    assert transport.bodies == [] and len(reopened.ledger.read_all()) == before


@pytest.mark.parametrize(("kind", "stage", "operation"), [
    ("attempt.observed", "observed", NextOperation.REINTERPRET),
    ("attempt.interpreted", "interpreted", NextOperation.REDECIDE),
    ("attempt.retry_decided", "decided", NextOperation.FINALIZE_CALL),
])
def test_post_effect_interruptions_name_their_next_operation(tmp_path, kind, stage, operation):
    runtime, first = interrupted_call(tmp_path, kind)
    assert len(first.bodies) == 1
    state = call_state(runtime, "call-1")
    assert (state.stage, state.next_operation, state.provider_effect) == (stage, operation, "observed")
    with pytest.raises(ResumeRefused):
        runtime.resume_call("call-1", adapter=opencode(Transport()))


def test_preparation_failure_projects_fix_inputs(tmp_path):
    runtime, _args = setup(tmp_path)
    ghost, _ = runtime.compile_and_record_context(task_id="task-w", actor=ACTOR, prompt=QUERY,
                                                  artifact_ids=["ghost-artifact"],
                                                  required_artifact_ids={"ghost-artifact"})
    with pytest.raises(Exception, match="ghost-artifact"):
        runtime.invoke_recorded_call(spec_for(ghost), adapter=opencode(Transport()))
    state = call_state(make_runtime(tmp_path), "call-1")
    assert state.stage == "preparation_failed" and state.next_operation == NextOperation.FIX_INPUTS


# ---------------- task level, purity, reopen ----------------


def test_task_next_operation_moves_from_call_to_acceptance(tmp_path):
    runtime, _first = interrupted_call(tmp_path, "call.manifest")
    assert runtime.work_state("task-w").task_next_operation == NextOperation.START_CALL
    runtime.resume_call("call-1", adapter=opencode(Transport()))
    state = runtime.work_state("task-w")
    assert state.task_completion == "incomplete"
    assert state.task_next_operation == NextOperation.CHECK_AND_ACCEPT


def test_projection_is_pure_and_identical_after_reopen(tmp_path):
    runtime, _ = interrupted_call(tmp_path, "attempt.observed")
    before = len(runtime.ledger.read_all())
    one = runtime.work_state("task-w")
    two = make_runtime(tmp_path).work_state("task-w")
    assert len(runtime.ledger.read_all()) == before
    assert asdict(one) == asdict(two)
