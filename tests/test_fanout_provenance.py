"""Chapter 23 seam: blindness proved over the bytes, not over the selection.

`sealed_fanout` always sealed each branch from its siblings at compile time, so
the record established that no sibling was *chosen*. What it did not establish
was what each branch was *sent*: the branch spec carried no renderer, so the
binding that ties a prepared request to its rendered input was never invoked.

    selection record    no sibling was chosen        (before)
    rendered bytes      no sibling was in the bytes  (with context_render)

The second is what an experiment claiming independence actually needs. A branch
whose adapter cannot prepare a request fails as a branch rather than quietly
falling back: a fan-out that cannot prove its inputs must say so.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from codeai.artifacts import FileArtifactStore
from codeai.rendering import CONTEXT_RENDER_V1
from codeai.domain import ActorRef
from codeai.ledger import SQLiteLedger
from codeai.providers import HttpResponse, OpenCodeCognitionAdapter
from codeai.runtime import Runtime

BASE = "The cache configuration is under review."
SIBLING_A = "Branch A thinks the ttl should be 300."
SIBLING_B = "Branch B thinks the ttl should be 900."
QUERY = "Propose a ttl."


class CountingTransport:
    """Keeps every body actually handed to the provider."""

    def __init__(self, reply="proposal") -> None:
        self.bodies: list[dict] = []
        self.reply = reply

    def __call__(self, url, payload, headers, timeout):
        self.bodies.append(payload)
        body = json.dumps(
            {"id": "s", "choices": [{"finish_reason": "stop", "message": {"content": self.reply}}]}
        ).encode()
        return HttpResponse(200, {"request-id": "s"}, body, "application/json")


class LegacyAdapter:
    """An adapter from before prepare()/send(): it can invoke, and nothing else."""

    model = "legacy-model"

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, spec):
        from codeai.adapters import CallResult

        self.calls += 1
        return CallResult(call_id=spec.call_id, raw_output="legacy", status="succeeded")


def actor(name: str) -> ActorRef:
    return ActorRef(name, "model", provider="opencode", model="mimo-v2.5")


def opencode(transport):
    return OpenCodeCognitionAdapter(
        model="mimo-v2.5", protocol="chat_completions", gateway_plan="go",
        api_key="offline-decoy", http_post=transport, timeout=5,
    )


def make_runtime(path: Path) -> Runtime:
    path.mkdir(parents=True, exist_ok=True)
    ledger = SQLiteLedger(path / "ledger.sqlite")
    return Runtime(ledger, artifact_store=FileArtifactStore(path / "artifacts", ledger))


def with_sibling_material(runtime):
    """Base material both branches may see, plus one output per branch that the
    other must not see. Each sibling artifact is attributed to its own call."""
    store = runtime.artifact_store
    base = store.store_text(BASE, artifact_type="source")
    out_a = store.store_text(SIBLING_A, artifact_type="sibling_output")
    out_b = store.store_text(SIBLING_B, artifact_type="sibling_output")
    return base.artifact_id, out_a.artifact_id, out_b.artifact_id


def fanout(runtime, *, render=CONTEXT_RENDER_V1, transports=None, branches=None, base_ids=()):
    transports = transports or [CountingTransport("A"), CountingTransport("B")]
    branches = branches or [
        {"call_id": "call-a", "actor": actor("a"), "adapter": opencode(transports[0]),
         "idempotency_key": "k-a"},
        {"call_id": "call-b", "actor": actor("b"), "adapter": opencode(transports[1]),
         "idempotency_key": "k-b"},
    ]
    results = runtime.sealed_fanout(
        task_id="t1",
        base_prompt=QUERY,
        branches=branches,
        base_artifact_ids=tuple(base_ids),
        instruction="Answer from the material provided.",
        context_render=render,
    )
    return results, transports


def manifest_for(runtime, call_id):
    recorded = runtime.get_recorded_call(call_id)
    return recorded.manifest


# ---------------- the matrix ----------------


def test_each_branch_records_the_bytes_it_was_sent(tmp_path):
    runtime = make_runtime(tmp_path)
    base, *_ = with_sibling_material(runtime)
    results, transports = fanout(runtime, base_ids=(base,))

    assert [r.status for r in results] == ["succeeded", "succeeded"]
    a, b = manifest_for(runtime, "call-a"), manifest_for(runtime, "call-b")
    for m in (a, b):
        assert m.context_render_version == CONTEXT_RENDER_V1
        assert m.rendered_context_sha256 and m.request_body_sha256
        # The rendered bytes are retrievable and hash to what the manifest says.
        stored = runtime.artifact_store.read_bytes(m.rendered_context_sha256)
        assert sha256(stored).hexdigest() == m.rendered_context_sha256
    assert len(transports[0].bodies) == 1 and len(transports[1].bodies) == 1
    # Same material, same prompt, same model: the two prepared bodies are
    # byte-identical, and the record now proves it. Blind is not diverse —
    # these branches differ only by sampling, which is exactly what Chapter 23
    # says a same-model fan-out gets you.
    assert a.request_body_sha256 == b.request_body_sha256
    assert a.rendered_context_sha256 == b.rendered_context_sha256


def test_the_seal_holds_over_the_bytes_not_only_the_selection(tmp_path):
    """Each branch is offered its sibling's output; neither receives it."""
    runtime = make_runtime(tmp_path)
    base, out_a, out_b = with_sibling_material(runtime)
    transports = [CountingTransport("A"), CountingTransport("B")]
    branches = [
        {"call_id": "call-a", "actor": actor("a"), "adapter": opencode(transports[0]),
         "idempotency_key": "k-a"},
        {"call_id": "call-b", "actor": actor("b"), "adapter": opencode(transports[1]),
         "idempotency_key": "k-b"},
    ]
    # Both sibling outputs are offered to the compilation, attributed to the
    # calls that produced them. The seal must exclude each branch's sibling.
    results = runtime.sealed_fanout(
        task_id="t1", base_prompt=QUERY, branches=branches,
        base_artifact_ids=(base, out_a, out_b),
        base_artifact_lineage={out_a: ["call-a"], out_b: ["call-b"]},
        base_required_artifact_ids=(base,),   # the rest is offered, not demanded
        instruction="Answer from the material provided.",
        context_render=CONTEXT_RENDER_V1,
    )
    assert [r.status for r in results] == ["succeeded", "succeeded"]

    sent = {
        "call-a": transports[0].bodies[0]["messages"][0]["content"],
        "call-b": transports[1].bodies[0]["messages"][0]["content"],
    }
    rendered = {
        call_id: runtime.artifact_store.read_bytes(
            manifest_for(runtime, call_id).rendered_context_sha256
        ).decode("utf-8")
        for call_id in ("call-a", "call-b")
    }
    for call_id in ("call-a", "call-b"):
        # Shared base material is present, so the seal is not excluding everything.
        assert BASE in rendered[call_id]
        assert BASE in sent[call_id]
    # Neither branch was sent anything attributed to a sibling call.
    for call_id, own, foreign in (("call-a", SIBLING_A, SIBLING_B),
                                  ("call-b", SIBLING_B, SIBLING_A)):
        assert foreign not in rendered[call_id], f"{call_id} saw its sibling's bytes"
        assert foreign not in sent[call_id], f"{call_id} was sent its sibling's bytes"


def test_a_branch_that_cannot_prove_its_inputs_fails_and_its_siblings_do_not(tmp_path):
    runtime = make_runtime(tmp_path)
    base, *_ = with_sibling_material(runtime)
    transport = CountingTransport("A")
    legacy = LegacyAdapter()
    results = runtime.sealed_fanout(
        task_id="t1", base_prompt=QUERY,
        branches=[
            {"call_id": "call-a", "actor": actor("a"), "adapter": opencode(transport),
             "idempotency_key": "k-a"},
            {"call_id": "call-legacy", "actor": actor("legacy"), "adapter": legacy,
             "idempotency_key": "k-legacy"},
        ],
        base_artifact_ids=(base,),
        instruction="Answer from the material provided.",
        context_render=CONTEXT_RENDER_V1,
    )
    statuses = {r.call_id: r.status for r in results}
    assert statuses["call-a"] == "succeeded"
    assert statuses["call-legacy"] == "failed"
    # Never quietly downgraded: the adapter was not invoked on the unrendered path.
    assert legacy.calls == 0
    failure = [r for r in results if r.call_id == "call-legacy"][0]
    assert "prepare()" in (failure.error or "")
    # The healthy branch still proved its own inputs.
    assert manifest_for(runtime, "call-a").rendered_context_sha256


def test_without_a_renderer_the_path_is_exactly_what_it_was(tmp_path):
    runtime = make_runtime(tmp_path)
    base, *_ = with_sibling_material(runtime)
    results, _ = fanout(runtime, render=None, base_ids=(base,))
    assert [r.status for r in results] == ["succeeded", "succeeded"]
    for call_id in ("call-a", "call-b"):
        m = manifest_for(runtime, call_id)
        assert m.context_render_version is None
        assert m.rendered_context_sha256 is None
    requested = next(iter(runtime.ledger.events_by_kind(("fanout.requested",))))
    assert requested.payload["input_provenance"] == "selection_record"


def test_the_fanout_record_says_what_its_blindness_rests_on(tmp_path):
    runtime = make_runtime(tmp_path)
    base, *_ = with_sibling_material(runtime)
    fanout(runtime, base_ids=(base,))
    requested = next(iter(runtime.ledger.events_by_kind(("fanout.requested",))))
    assert requested.payload["context_render"] == CONTEXT_RENDER_V1
    assert requested.payload["input_provenance"] == "rendered_bytes"


def test_a_prepared_body_that_drops_the_rendered_input_is_refused(tmp_path):
    """The binding is what does the proving, so break it deliberately."""
    runtime = make_runtime(tmp_path)
    base, *_ = with_sibling_material(runtime)
    transport = CountingTransport("A")

    class DropsTheContext(OpenCodeCognitionAdapter):
        def prepare(self, spec):
            prepared = super().prepare(spec)
            body = json.loads(json.dumps(prepared.body))
            body["messages"][0]["content"] = "nothing the renderer produced"
            from dataclasses import replace as dc_replace

            return dc_replace(
                prepared, body=body,
                body_sha256=sha256(
                    json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            )

    tampered = DropsTheContext(
        model="mimo-v2.5", protocol="chat_completions", gateway_plan="go",
        api_key="offline-decoy", http_post=transport, timeout=5,
    )
    results = runtime.sealed_fanout(
        task_id="t1", base_prompt=QUERY,
        branches=[{"call_id": "call-x", "actor": actor("x"), "adapter": tampered,
                   "idempotency_key": "k-x"}],
        base_artifact_ids=(base,),
        instruction="Answer from the material provided.",
        context_render=CONTEXT_RENDER_V1,
    )
    assert results[0].status == "failed"
    assert transport.bodies == [], "a request that could not be bound was still sent"
    failures = runtime.ledger.events_by_kind(("call.preparation_failed",))
    assert [e.payload["stage"] for e in failures] == ["render_binding"]


def test_reopening_resolves_every_branch_input(tmp_path):
    path = tmp_path / "run"
    runtime = make_runtime(path)
    base, *_ = with_sibling_material(runtime)
    fanout(runtime, base_ids=(base,))
    before = {
        call_id: manifest_for(runtime, call_id).rendered_context_sha256
        for call_id in ("call-a", "call-b")
    }
    events = runtime.ledger.read_all()
    runtime.ledger.close()

    reopened = SQLiteLedger(path / "ledger.sqlite")
    second = Runtime(reopened, artifact_store=FileArtifactStore(path / "artifacts", reopened))
    assert reopened.read_all() == events
    for call_id, digest in before.items():
        m = second.get_recorded_call(call_id).manifest
        assert m.rendered_context_sha256 == digest
        stored = second.artifact_store.read_bytes(digest)
        assert sha256(stored).hexdigest() == digest
    reopened.close()


def test_rendering_does_not_make_the_branches_identical(tmp_path):
    """Different prompts still produce different bytes; the seal is not a leveller."""
    runtime = make_runtime(tmp_path)
    base, *_ = with_sibling_material(runtime)
    transports = [CountingTransport("A"), CountingTransport("B")]
    results = runtime.sealed_fanout(
        task_id="t1", base_prompt=QUERY,
        branches=[
            {"call_id": "call-a", "actor": actor("a"), "adapter": opencode(transports[0]),
             "idempotency_key": "k-a", "prompt": "Propose a ttl for a busy cache."},
            {"call_id": "call-b", "actor": actor("b"), "adapter": opencode(transports[1]),
             "idempotency_key": "k-b", "prompt": "Propose a ttl for a quiet cache."},
        ],
        base_artifact_ids=(base,),
        instruction="Answer from the material provided.",
        context_render=CONTEXT_RENDER_V1,
    )
    assert [r.status for r in results] == ["succeeded", "succeeded"]
    a, b = manifest_for(runtime, "call-a"), manifest_for(runtime, "call-b")
    # The selected material is the same, so the rendered context is the same.
    # The question differs, so the prompt and the bytes actually sent differ.
    assert a.rendered_context_sha256 == b.rendered_context_sha256
    assert a.prompt_hash != b.prompt_hash
    assert a.request_body_sha256 != b.request_body_sha256


def test_unattributed_sibling_material_is_selected_and_the_record_says_so(tmp_path):
    """The seal excludes by identity, so material with no identity is not excluded.

    This is Chapter 23's admitted provenance hole, on the fan-out path: the
    compiler cannot tell base material that never had lineage from base material
    whose lineage was stripped. The runtime does not guess. It records how much
    of the base was attributed, so a reader can see what the blindness claim
    rested on instead of assuming it was complete.
    """
    runtime = make_runtime(tmp_path)
    base, out_a, out_b = with_sibling_material(runtime)
    transports = [CountingTransport("A"), CountingTransport("B")]
    runtime.sealed_fanout(
        task_id="t1", base_prompt=QUERY,
        branches=[
            {"call_id": "call-a", "actor": actor("a"), "adapter": opencode(transports[0]),
             "idempotency_key": "k-a"},
            {"call_id": "call-b", "actor": actor("b"), "adapter": opencode(transports[1]),
             "idempotency_key": "k-b"},
        ],
        base_artifact_ids=(base, out_a, out_b),   # no lineage declared
        base_required_artifact_ids=(base,),
        instruction="Answer from the material provided.",
        context_render=CONTEXT_RENDER_V1,
    )
    sent_a = transports[0].bodies[0]["messages"][0]["content"]
    # Branch A was sent branch B's material, because nothing said it was B's.
    assert SIBLING_B in sent_a

    requested = next(iter(runtime.ledger.events_by_kind(("fanout.requested",))))
    assert requested.payload["base_offered"]["artifacts"] == 3
    assert requested.payload["base_lineage_declared"]["artifacts"] == 0
    # The record does not claim blindness it cannot support: three artifacts
    # were offered and none were attributed, which is visible without reading
    # the bytes.


def test_declared_lineage_is_what_makes_the_seal_bite(tmp_path):
    """The same fixture, attributed, and the sibling material disappears."""
    runtime = make_runtime(tmp_path)
    base, out_a, out_b = with_sibling_material(runtime)
    transports = [CountingTransport("A"), CountingTransport("B")]
    runtime.sealed_fanout(
        task_id="t1", base_prompt=QUERY,
        branches=[
            {"call_id": "call-a", "actor": actor("a"), "adapter": opencode(transports[0]),
             "idempotency_key": "k-a"},
            {"call_id": "call-b", "actor": actor("b"), "adapter": opencode(transports[1]),
             "idempotency_key": "k-b"},
        ],
        base_artifact_ids=(base, out_a, out_b),
        base_artifact_lineage={out_a: ["call-a"], out_b: ["call-b"]},
        base_required_artifact_ids=(base,),   # the rest is offered, not demanded
        instruction="Answer from the material provided.",
        context_render=CONTEXT_RENDER_V1,
    )
    sent_a = transports[0].bodies[0]["messages"][0]["content"]
    sent_b = transports[1].bodies[0]["messages"][0]["content"]
    assert SIBLING_B not in sent_a and SIBLING_A not in sent_b
    assert BASE in sent_a and BASE in sent_b

    requested = next(iter(runtime.ledger.events_by_kind(("fanout.requested",))))
    assert requested.payload["base_lineage_declared"]["artifacts"] == 2


def test_required_material_a_branch_may_not_see_refuses_that_branch(tmp_path):
    """Loud by default: the seal does not quietly drop what was demanded.

    Offering a branch material its seal forbids, and demanding that material,
    is a contradiction. The compiler refuses rather than compiling a package
    that silently lacks what the caller said was required, and the fan-out
    records that branch as failed while its sibling completes.
    """
    runtime = make_runtime(tmp_path)
    base, out_a, out_b = with_sibling_material(runtime)
    transports = [CountingTransport("A"), CountingTransport("B")]
    results = runtime.sealed_fanout(
        task_id="t1", base_prompt=QUERY,
        branches=[
            {"call_id": "call-a", "actor": actor("a"), "adapter": opencode(transports[0]),
             "idempotency_key": "k-a"},
            {"call_id": "call-b", "actor": actor("b"), "adapter": opencode(transports[1]),
             "idempotency_key": "k-b"},
        ],
        base_artifact_ids=(base, out_b),
        base_artifact_lineage={out_b: ["call-b"]},
        # no base_required_artifact_ids: everything offered is demanded
        instruction="Answer from the material provided.",
        context_render=CONTEXT_RENDER_V1,
    )
    statuses = {r.call_id: r.status for r in results}
    assert statuses["call-a"] == "failed"
    assert "context compilation refused" in (
        [r for r in results if r.call_id == "call-a"][0].error or ""
    )
    assert "seal forbids call lineage" in (
        [r for r in results if r.call_id == "call-a"][0].error or ""
    )
    # Branch B may see its own output, so it runs.
    assert statuses["call-b"] == "succeeded"
    assert transports[0].bodies == []
