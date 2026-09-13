"""Stage 15B: the recorded context selection determines the request bytes.

Offline only: synthetic transport through the recorded call path, no network,
no credentials. Fixture text is not evidence of model behaviour.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from codeai.adapters import CallSpec, FakeCognitionAdapter
from codeai.artifacts import FileArtifactStore
from codeai.context import ContextCompiler
from codeai.domain import ActorRef, Claim, RenderedContext, Seal
from codeai.ledger import Event, SQLiteLedger
from codeai.providers import HttpResponse, OpenCodeCognitionAdapter
from codeai.rendering import (
    CONTEXT_RENDER_V1,
    ContextResolutionError,
    compose_model_input,
    render_context,
)
from codeai.runtime import IdempotencyConflictError, Runtime

ACTOR = ActorRef("reviewer", "model", provider="opencode", model="mimo-v2.5")
INSTRUCTION = "You are reviewing one paragraph against its source."
QUERY = "Name the missing evidence in one sentence."
SOURCE_B = "SOURCE-B: The cache claim needs a benchmark citation [S1].\n"
SOURCE_E = "SOURCE-E: sibling-derived content that must stay out.\n"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class CountingTransport:
    def __init__(self) -> None:
        self.bodies: list[dict] = []

    def __call__(self, url, body, headers, timeout):
        self.bodies.append(body)
        payload = {"id": "synthetic", "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}
        return HttpResponse(200, {}, json.dumps(payload).encode(), "application/json")


def build(path: Path, *, objective="Review the paragraph using its source.",
          statement="The cache claim lacks a benchmark.", reverse=False, claims=("D",)):
    path.mkdir(parents=True, exist_ok=True)
    ledger = SQLiteLedger(path / "ledger.sqlite")
    store = FileArtifactStore(path / "artifacts", ledger)
    runtime = Runtime(ledger, artifact_store=store)
    a = Event("A", "task-r", "fixture.objective", "human", {"text": objective}, "2026-09-13T00:00:00Z")
    c = Event("C", "task-r", "fixture.prior", "human", {"text": "Earlier observation."}, "2026-09-13T00:00:00Z")
    ledger.append(a)
    ledger.append(c)
    b = store.store_text(SOURCE_B, artifact_type="source").artifact_id
    e = store.store_text(SOURCE_E, artifact_type="sibling_output").artifact_id
    runtime.record_claim(Claim("D", "task-r", statement, "source-call"))
    events, artifacts = [a, c], [b, e]
    if reverse:
        events.reverse()
        artifacts.reverse()
    package, _trace = ContextCompiler().compile_with_trace(
        task_id="task-r", actor=ACTOR, prompt=QUERY, events=events, artifact_ids=artifacts,
        claim_ids=list(claims), seal=Seal(forbidden_call_ids=frozenset({"sibling-call"})),
        required_event_ids={"A"}, required_artifact_ids={b},
        artifact_lineage={e: ["sibling-call"]}, claim_lineage={"D": ["source-call"]},
        prompt_version="render-fixture-v1", objective="Review",
    )
    return runtime, package, {"B": b, "E": e}


def spec_for(package, *, render=CONTEXT_RENDER_V1, key="key-1", call_id="call-1", rendered=None):
    return CallSpec(
        call_id=call_id, task_id="task-r", actor=ACTOR, context=package, idempotency_key=key,
        instruction=INSTRUCTION, chamber="deep-review", parameters={"max_tokens": 64},
        context_render=render, rendered_context=rendered,
    )


def opencode(transport):
    return OpenCodeCognitionAdapter(model="mimo-v2.5", protocol="chat_completions", gateway_plan="go",
                                    api_key="offline-decoy", http_post=transport, timeout=5)


def run(path, **build_args):
    runtime, package, ids = build(path, **build_args)
    transport = CountingTransport()
    recorded = runtime.invoke_recorded_call(spec_for(package), adapter=opencode(transport))
    return runtime, package, ids, transport, recorded


def items_by_id(manifest):
    return {item["item_id"]: item for item in manifest.rendered_items}


# ---------------- selected material reaches the request ----------------


def test_selected_bytes_reach_the_request_and_are_bound(tmp_path):
    runtime, package, ids, transport, recorded = run(tmp_path)
    assert recorded.status == "succeeded"
    [body] = transport.bodies
    text = body["messages"][0]["content"]
    assert SOURCE_B in text
    manifest = recorded.manifest
    assert manifest.context_render_version == CONTEXT_RENDER_V1
    assert manifest.context_package_id == package.package_id
    rendered = runtime.artifact_store.read_bytes(manifest.rendered_context_sha256)
    assert sha(rendered) == manifest.rendered_context_sha256
    assert [(i["kind"], i["item_id"]) for i in manifest.rendered_items] == [
        ("artifact", ids["B"]), ("claim", "D"), ("event", "A"), ("event", "C")]
    for item in manifest.rendered_items:
        assert sha(rendered[item["start"]:item["end"]]) == item["content_sha256"]
    assert rendered[items_by_id(manifest)[ids["B"]]["start"]:items_by_id(manifest)[ids["B"]]["end"]] == SOURCE_B.encode()
    assert items_by_id(manifest)[ids["B"]]["content_sha256"] == ids["B"]
    layout = {p["part"]: p for p in manifest.input_layout}
    raw = text.encode()
    assert raw[layout["instruction"]["start"]:layout["instruction"]["end"]] == INSTRUCTION.encode()
    assert raw[layout["context"]["start"]:layout["context"]["end"]] == rendered
    assert raw[layout["query"]["start"]:layout["query"]["end"]] == QUERY.encode()
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert sha(canonical) == manifest.request_body_sha256
    reopened = Runtime(SQLiteLedger(tmp_path / "ledger.sqlite")).get_recorded_call(recorded.call_id)
    assert reopened.manifest.rendered_context_sha256 == manifest.rendered_context_sha256
    assert reopened.manifest.input_layout == manifest.input_layout


def test_sealed_item_is_offered_excluded_and_absent(tmp_path):
    _runtime, package, ids, transport, recorded = run(tmp_path)
    assert ids["E"] not in package.artifact_ids
    assert ids["E"] not in items_by_id(recorded.manifest)
    assert "SOURCE-E" not in transport.bodies[0]["messages"][0]["content"]


# ---------------- identity under change ----------------


@pytest.mark.parametrize(("change", "changed_item"), [
    ({"objective": "Changed under the same event ID."}, "A"),
    ({"statement": "Changed under the same claim ID."}, "D"),
])
def test_content_change_under_same_selection_changes_rendered_identity(tmp_path, change, changed_item):
    _r1, p1, _i1, _t1, before = run(tmp_path / "before")
    _r2, p2, _i2, _t2, after = run(tmp_path / "after", **change)
    assert p1.package_id == p2.package_id
    assert before.manifest.rendered_context_sha256 != after.manifest.rendered_context_sha256
    assert before.manifest.request_body_sha256 != after.manifest.request_body_sha256
    b_items, a_items = items_by_id(before.manifest), items_by_id(after.manifest)
    changed = sorted(k for k in b_items if b_items[k]["content_sha256"] != a_items[k]["content_sha256"])
    assert changed == [changed_item]


def test_candidate_order_does_not_change_rendering(tmp_path):
    _r1, p1, _i1, _t1, forward = run(tmp_path / "forward")
    _r2, p2, _i2, _t2, backward = run(tmp_path / "backward", reverse=True)
    assert p1.package_id == p2.package_id
    assert forward.manifest.rendered_context_sha256 == backward.manifest.rendered_context_sha256
    assert forward.manifest.request_body_sha256 == backward.manifest.request_body_sha256


def test_render_is_pure_and_repeatable(tmp_path):
    runtime, package, _ids = build(tmp_path)
    before = len(runtime.ledger.read_all())
    one = render_context(package, ledger=runtime.ledger, artifact_store=runtime.artifact_store)
    two = render_context(package, ledger=runtime.ledger, artifact_store=runtime.artifact_store)
    assert one == two
    assert len(runtime.ledger.read_all()) == before


# ---------------- unresolvable selection fails before effect ----------------


def test_ghost_artifact_fails_before_any_provider_effect(tmp_path):
    runtime, _package, _ids = build(tmp_path)
    ghost, _ = ContextCompiler().compile_with_trace(
        task_id="task-r", actor=ACTOR, prompt=QUERY, artifact_ids=["ghost-artifact"],
        required_artifact_ids={"ghost-artifact"},
    )
    transport = CountingTransport()
    with pytest.raises(ContextResolutionError) as caught:
        runtime.invoke_recorded_call(spec_for(ghost), adapter=opencode(transport))
    assert caught.value.unresolved == ({"kind": "artifact", "id": "ghost-artifact", "reason": "artifact not found"},)
    assert transport.bodies == []
    kinds = [e.kind for e in runtime.ledger.read_all()]
    assert "call.preparation_failed" in kinds
    assert not {"call.manifest", "attempt.started", "call.completed"} & set(kinds)
    [failed] = runtime.ledger.events_by_kind(("call.preparation_failed",))
    assert failed.payload["stage"] == "context_render"
    assert failed.payload["provider_effect"] is False
    assert failed.payload["unresolved"][0]["id"] == "ghost-artifact"
    assert runtime.get_recorded_call("call-1") is None


def test_selected_event_missing_from_ledger_fails(tmp_path):
    runtime, _package, _ids = build(tmp_path)
    unrecorded = Event("never-appended", "task-r", "fixture.note", "human", {"text": "x"}, "2026-09-13T00:00:00Z")
    package, _ = ContextCompiler().compile_with_trace(task_id="task-r", actor=ACTOR, prompt=QUERY,
                                                     events=[unrecorded])
    transport = CountingTransport()
    with pytest.raises(ContextResolutionError):
        runtime.invoke_recorded_call(spec_for(package), adapter=opencode(transport))
    assert transport.bodies == []


# ---------------- compatibility and guards ----------------


def test_legacy_calls_render_prompt_only(tmp_path):
    runtime, package, _ids = build(tmp_path)
    transport = CountingTransport()
    recorded = runtime.invoke_recorded_call(spec_for(package, render=None), adapter=opencode(transport))
    assert transport.bodies[0]["messages"][0]["content"] == f"{INSTRUCTION}\n\n{QUERY}"
    assert recorded.manifest.rendered_context_sha256 is None
    assert recorded.manifest.rendered_items == ()
    assert recorded.manifest.input_layout == ()


def test_caller_supplied_rendering_is_ignored(tmp_path):
    runtime, package, _ids = build(tmp_path)
    forged = RenderedContext(CONTEXT_RENDER_V1, package.package_id, "FORGED CONTEXT", "0" * 64)
    transport = CountingTransport()
    runtime.invoke_recorded_call(spec_for(package, rendered=forged), adapter=opencode(transport))
    text = transport.bodies[0]["messages"][0]["content"]
    assert "FORGED" not in text and SOURCE_B in text


def test_same_key_with_different_rendered_context_conflicts(tmp_path):
    runtime, package, _ids = build(tmp_path)
    transport = CountingTransport()
    runtime.invoke_recorded_call(spec_for(package), adapter=opencode(transport))
    runtime.invoke_recorded_call(spec_for(package, call_id="call-2"), adapter=opencode(transport))
    assert len(transport.bodies) == 1  # identical request replays
    without_claim, _ = ContextCompiler().compile_with_trace(
        task_id="task-r", actor=ACTOR, prompt=QUERY, artifact_ids=list(package.artifact_ids),
        required_artifact_ids=set(package.artifact_ids),
    )
    with pytest.raises(IdempotencyConflictError, match="rendered_context_sha256"):
        runtime.invoke_recorded_call(spec_for(without_claim, call_id="call-3"), adapter=opencode(transport))
    assert len(transport.bodies) == 1


def test_rendering_requires_a_prepare_capable_adapter(tmp_path):
    runtime, package, _ids = build(tmp_path)
    before = len(runtime.ledger.read_all())
    with pytest.raises(ValueError, match="prepare"):
        runtime.invoke_recorded_call(spec_for(package), adapter=FakeCognitionAdapter(responses=["x"]))
    assert len(runtime.ledger.read_all()) == before


def test_unknown_renderer_version_rejected_before_events(tmp_path):
    runtime, package, _ids = build(tmp_path)
    before = len(runtime.ledger.read_all())
    with pytest.raises(ValueError, match="renderer"):
        runtime.invoke_recorded_call(spec_for(package, render="context-render-v0"),
                                     adapter=opencode(CountingTransport()))
    assert len(runtime.ledger.read_all()) == before


def test_compose_layout_omits_empty_parts():
    text, layout = compose_model_input("", None, "only query")
    assert text == "only query"
    assert layout == ({"part": "query", "sha256": sha(b"only query"), "start": 0, "end": 10},)
