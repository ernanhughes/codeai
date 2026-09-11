from __future__ import annotations

import sys
from pathlib import Path

import pytest

from codeai.adapters import CallSpec, CheckRequest, FakeCognitionAdapter
from codeai.artifacts import FileArtifactStore
from codeai.claims import disagreement_report, project_claims
from codeai.context import (
    ContextBudgetUnsatisfiable,
    ContextCandidate,
    ContextCompiler,
    ContextSealViolation,
    RequiredContextMissing,
)
from codeai.domain import (
    ActorRef,
    Authority,
    Capability,
    Claim,
    ClaimRelationship,
    ClaimRelationshipType,
    ClaimStatus,
    EvidenceClass,
    Seal,
    Variant,
)
from codeai.ledger import Event, SQLiteLedger
from codeai.runtime import Runtime, default_repository_state_hash
from codeai.scheduler import Operation, SchedulerInput, decide_next_step
from codeai.verifier import LocalCommandVerifier


def make_actor(actor_id="m1", model="fake-model", provider="fake-provider"):
    return ActorRef(actor_id=actor_id, kind="model", provider=provider, model=model, version="v1")


def make_runtime(tmp_path: Path, ledger_path: str | Path | None = None):
    ledger = SQLiteLedger(str(ledger_path) if ledger_path else ":memory:")
    store = FileArtifactStore(tmp_path / "artifacts", ledger)
    return Runtime(ledger, artifact_store=store)


def make_spec(task_id="t1", call_id="c1", prompt="solve", adapter_id="fake"):
    actor = make_actor()
    package = ContextCompiler().compile(task_id=task_id, actor=actor, prompt=prompt, events=())
    return CallSpec(
        call_id=call_id,
        task_id=task_id,
        actor=actor,
        context=package,
        idempotency_key=f"key-{call_id}",
        adapter_id=adapter_id,
        instruction=prompt,
    )


# ---------------- Cognition ----------------


def test_successful_call_records_lifecycle_and_artifact(tmp_path):
    runtime = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(responses=["hello world"], input_tokens=7, output_tokens=3, cost_usd=0.01)
    result = runtime.invoke_call(make_spec(), adapter=adapter)
    assert result.status == "succeeded"
    assert result.raw_output == "hello world"
    assert result.raw_artifact is not None
    assert runtime.artifact_store.read_text(result.raw_artifact.artifact_id) == "hello world"
    kinds = [e.kind for e in runtime.ledger.read_all()]
    assert kinds == ["call.requested", "call.completed"]


def test_failed_call_preserves_error_and_metadata(tmp_path):
    runtime = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(fail_call_ids={"c1"})
    result = runtime.invoke_call(make_spec(call_id="c1"), adapter=adapter)
    assert result.status == "failed"
    assert result.error
    # failure retains what was attempted: ledger has both events
    assert [e.kind for e in runtime.ledger.read_all()] == ["call.requested", "call.completed"]


def test_usage_attribution_per_call(tmp_path):
    runtime = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(input_tokens=11, output_tokens=22, cost_usd=0.05, latency_ms=9)
    result = runtime.invoke_call(make_spec(), adapter=adapter)
    assert (result.input_tokens, result.output_tokens, result.cost_usd, result.latency_ms) == (11, 22, 0.05, 9)
    assert result.provider == "fake-provider"
    assert result.model == "fake-model"


def test_parser_failure_must_not_destroy_raw_source(tmp_path):
    """Raw output before structure: raw artifact is canonical."""
    runtime = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(responses=["not valid json {{{"])
    result = runtime.invoke_call(make_spec(), adapter=adapter)
    assert result.raw_artifact is not None
    assert runtime.artifact_store.read_text(result.raw_artifact.artifact_id) == "not valid json {{{"


# ---------------- Context ----------------


def test_context_package_hash_deterministic_and_ordered():
    compiler = ContextCompiler()
    actor = make_actor()
    e1 = Event.create(stream_id="s", kind="k", actor_id="a", payload={"n": 1})
    e2 = Event.create(stream_id="s", kind="k", actor_id="a", payload={"n": 2})
    a = compiler.compile(task_id="t", actor=actor, prompt="p", events=[e1, e2])
    b = compiler.compile(task_id="t", actor=actor, prompt="p", events=[e2, e1])
    assert a.package_id == b.package_id
    assert tuple(sorted(a.event_ids)) == tuple(a.event_ids)


def test_required_context_missing_fails_loudly():
    compiler = ContextCompiler()
    with pytest.raises(RequiredContextMissing):
        compiler.compile_with_trace(
            task_id="t",
            actor=make_actor(),
            prompt="p",
            events=(),
            required_event_ids=["missing-event"],
        )


def test_budget_unsatisfiable_fails_loudly():
    compiler = ContextCompiler()
    candidates = [
        ContextCandidate(candidate_id="a", kind="event", size_tokens=100, required=True),
        ContextCandidate(candidate_id="b", kind="event", size_tokens=100, required=True),
    ]
    with pytest.raises(ContextBudgetUnsatisfiable):
        compiler.compile_candidates(
            task_id="t", actor=make_actor(), prompt="p", candidates=candidates, budget_tokens=50
        )


def test_compilation_trace_explains_inclusion_and_budget():
    compiler = ContextCompiler()
    candidates = [
        ContextCandidate(candidate_id="req", kind="event", size_tokens=10, required=True),
        ContextCandidate(candidate_id="opt-big", kind="event", size_tokens=1000, required=False),
        ContextCandidate(candidate_id="opt-small", kind="event", size_tokens=5, required=False),
    ]
    _, trace = compiler.compile_candidates(
        task_id="t", actor=make_actor(), prompt="p", candidates=candidates, budget_tokens=20
    )
    reasons = {e.candidate_id: e.reason for e in trace.entries}
    assert "required" in reasons["req"]
    assert "budget" in reasons["opt-big"]
    assert trace.required_tokens == 10


# ---------------- Sealing ----------------


def test_sibling_outputs_excluded_through_compilation():
    compiler = ContextCompiler()
    sibling = Event.create(
        stream_id="call-a", kind="call.completed", actor_id="m1", payload={"call_id": "call-a"}
    )
    seal = Seal(forbidden_call_ids=frozenset({"call-a"}))
    candidates = [
        ContextCandidate(
            candidate_id=sibling.event_id, kind="event", lineage_ids=("call-a",), required=False
        )
    ]
    package, trace = compiler.compile_candidates(
        task_id="t", actor=make_actor("m2"), prompt="p", candidates=candidates, seal=seal
    )
    assert sibling.event_id not in package.event_ids
    assert trace.reason_for(sibling.event_id) is not None
    assert "seal" in trace.reason_for(sibling.event_id)


def test_derived_sibling_claims_excluded_via_lineage():
    """Call B -> Claim B1 -> Summary S: sealing B must exclude B1 and S."""
    compiler = ContextCompiler()
    seal = Seal(forbidden_call_ids=frozenset({"call-b"}))
    candidates = [
        ContextCandidate(candidate_id="claim-b1", kind="claim", lineage_ids=("call-b", "claim-b1")),
        ContextCandidate(
            candidate_id="summary-s", kind="event", lineage_ids=("call-b", "claim-b1", "summary-s")
        ),
        ContextCandidate(candidate_id="independent", kind="event", lineage_ids=("other",)),
    ]
    package, _ = compiler.compile_candidates(
        task_id="t", actor=make_actor(), prompt="p", candidates=candidates, seal=seal
    )
    assert "claim-b1" not in package.claim_ids
    assert "summary-s" not in package.event_ids
    assert "independent" in package.event_ids


def test_independently_compiled_packages_reproducible():
    compiler = ContextCompiler()
    actor = make_actor()
    events = [Event.create(stream_id="s", kind="k", actor_id="a", payload={"n": i}) for i in range(3)]
    a, ta = compiler.compile_with_trace(task_id="t", actor=actor, prompt="p", events=events)
    b, tb = compiler.compile_with_trace(task_id="t", actor=actor, prompt="p", events=events)
    assert a.package_id == b.package_id
    assert ta.trace_id == tb.trace_id


def test_legacy_seal_violation_still_loud():
    with pytest.raises(ContextSealViolation):
        ContextCompiler().compile(
            task_id="t",
            actor=make_actor(),
            prompt="p",
            events=[
                Event.create(
                    stream_id="d", kind="call.completed", actor_id="m", payload={"call_id": "x"}
                )
            ],
            seal=Seal(forbidden_call_ids=frozenset({"x"})),
        )


# ---------------- Fanout ----------------


def test_fanout_creates_n_independent_calls(tmp_path):
    runtime = make_runtime(tmp_path)
    branches = [
        {
            "actor": make_actor(f"m{i}", model="fake-model"),
            "adapter": FakeCognitionAdapter(responses=[f"out-{i}"]),
            "variant": Variant(model="fake-model", experiment="C1"),
        }
        for i in range(3)
    ]
    results = runtime.sealed_fanout(task_id="t1", base_prompt="solve", branches=branches)
    assert len(results) == 3
    assert {r.raw_output for r in results} == {"out-0", "out-1", "out-2"}
    assert len({r.call_id for r in results}) == 3


def test_fanout_one_failure_does_not_destroy_siblings(tmp_path):
    runtime = make_runtime(tmp_path)
    good = FakeCognitionAdapter(responses=["good"])
    bad = FakeCognitionAdapter(fail_call_ids={"bad-call"})
    branches = [
        {"actor": make_actor("g"), "adapter": good, "call_id": "good-call",
         "variant": Variant(model="m", experiment="H1")},
        {"actor": make_actor("b"), "adapter": bad, "call_id": "bad-call",
         "variant": Variant(model="m2", experiment="H1")},
    ]
    results = runtime.sealed_fanout(task_id="t1", base_prompt="solve", branches=branches)
    by_id = {r.call_id: r for r in results}
    assert by_id["good-call"].status == "succeeded"
    assert by_id["bad-call"].status == "failed"
    assert by_id["good-call"].raw_output == "good"


def test_homogeneous_and_heterogeneous_share_primitive(tmp_path):
    runtime = make_runtime(tmp_path)
    homo = [
        {"actor": make_actor(f"h{i}", model="same"), "adapter": FakeCognitionAdapter(responses=[f"h{i}"]),
         "variant": Variant(model="same", experiment="C1")}
        for i in range(2)
    ]
    hetero = [
        {"actor": make_actor("a", model="model-a"), "adapter": FakeCognitionAdapter(responses=["A"], model="model-a"),
         "variant": Variant(model="model-a", experiment="H1")},
        {"actor": make_actor("b", model="model-b"), "adapter": FakeCognitionAdapter(responses=["B"], model="model-b"),
         "variant": Variant(model="model-b", experiment="H1")},
    ]
    homo_results = runtime.sealed_fanout(task_id="t1", base_prompt="p", branches=homo)
    hetero_results = runtime.sealed_fanout(task_id="t1", base_prompt="p", branches=hetero)
    assert len(homo_results) == len(hetero_results) == 2
    # diversity-source metadata preserved on specs
    specs = [e.payload for e in runtime.ledger.events_by_kind(("call.requested",))]
    assert any("fanout" in str(s.get("pattern", "")) for s in specs)


def test_fanout_branches_sealed_from_siblings(tmp_path):
    runtime = make_runtime(tmp_path)
    branches = [
        {"actor": make_actor("m0"), "adapter": FakeCognitionAdapter(responses=["o0"]),
         "call_id": "call-0", "variant": Variant(model="m")},
        {"actor": make_actor("m1"), "adapter": FakeCognitionAdapter(responses=["o1"]),
         "call_id": "call-1", "variant": Variant(model="m")},
    ]
    runtime.sealed_fanout(task_id="t1", base_prompt="p", branches=branches)
    contexts = list(runtime.ledger.events_by_kind(("context.compiled",)))
    assert len(contexts) == 2
    seals = [e.payload.get("seal", {}) for e in contexts]
    assert any("call-1" in s.get("forbidden_call_ids", []) for s in seals)
    assert any("call-0" in s.get("forbidden_call_ids", []) for s in seals)


# ---------------- Claims ----------------


def test_claim_anchors_to_source_and_raw_canonical(tmp_path):
    runtime = make_runtime(tmp_path)
    result = runtime.invoke_call(make_spec(call_id="c1"), adapter=FakeCognitionAdapter(responses=["raw text here"]))
    assert result.raw_artifact is not None
    claim = Claim(
        claim_id="claim-1",
        task_id="t1",
        statement="the sky is blue",
        source_call_id="c1",
        source_artifact_id=result.raw_artifact.artifact_id,
        source_span="raw text",
        run_id="r1",
    )
    runtime.record_claim(claim)
    stored = runtime.claims_for_task("t1")["claim-1"]
    assert stored.source_artifact_id == result.raw_artifact.artifact_id
    assert stored.source_span == "raw text"
    # raw artifact remains canonical
    assert runtime.artifact_store.read_text(stored.source_artifact_id) == "raw text here"


def test_concurrence_does_not_upgrade_evidence(tmp_path):
    events = (
        Event.create(stream_id="a", kind="claim.recorded", actor_id="x", payload={
            "claim_id": "a", "task_id": "t", "statement": "Same Statement",
            "source_call_id": "call-1", "run_id": "r"}),
        Event.create(stream_id="b", kind="claim.recorded", actor_id="x", payload={
            "claim_id": "b", "task_id": "t", "statement": "same statement ",
            "source_call_id": "call-2", "run_id": "r"}),
    )
    claims = project_claims(events)
    report = disagreement_report(claims)
    assert len(report["concurrence"]) == 1
    assert report["concurrence"][0]["concurrence"] == 2
    # evidence stays ASSERTED
    assert all(c.evidence_class == EvidenceClass.ASSERTED for c in claims.values())


def test_contradictions_representable(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.record_claim(Claim(claim_id="c1", task_id="t1", statement="X is true", source_call_id="a", run_id="r1"))
    runtime.record_claim(Claim(claim_id="c2", task_id="t1", statement="X is false", source_call_id="b", run_id="r1"))
    runtime.link_claims(
        ClaimRelationship(
            relationship_id="rel-1", from_claim_id="c1", to_claim_id="c2",
            relationship_type=ClaimRelationshipType.CONTRADICTS, run_id="r1",
        )
    )
    report = runtime.disagreement_for_run("r1")
    assert len(report["contradicted"]) == 2


# ---------------- Verification -> Claims ----------------


def test_passing_check_promotes_targeted_claim_to_reproduced(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.record_claim(Claim(claim_id="c1", task_id="t1", statement="cmd works", source_call_id="m", run_id="r"))
    result = runtime.run_check(
        CheckRequest(
            check_id="chk-1", task_id="t1", claim_ids=("c1",),
            command=(sys.executable, "-c", "print('ok')"), cwd=str(tmp_path),
        ),
        verifier=LocalCommandVerifier(),
    )
    assert result.verdict == "PASS"
    assert runtime.claims_for_task("t1")["c1"].evidence_class == EvidenceClass.REPRODUCED


def test_failing_check_refutes_targeted_claim(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.record_claim(Claim(claim_id="c1", task_id="t1", statement="cmd works", source_call_id="m", run_id="r"))
    result = runtime.run_check(
        CheckRequest(
            check_id="chk-2", task_id="t1", claim_ids=("c1",),
            command=(sys.executable, "-c", "import sys; sys.exit(1)"), cwd=str(tmp_path),
        ),
        verifier=LocalCommandVerifier(),
    )
    assert result.verdict == "FAIL"
    assert runtime.claims_for_task("t1")["c1"].status == ClaimStatus.REFUTED


def test_successful_action_does_not_blanket_promote_claims(tmp_path):
    """Evidence must be scoped: unrelated claims stay ASSERTED after a passing check."""
    runtime = make_runtime(tmp_path)
    runtime.record_claim(Claim(claim_id="target", task_id="t1", statement="targeted", source_call_id="m", run_id="r"))
    runtime.record_claim(Claim(claim_id="other", task_id="t1", statement="unrelated", source_call_id="m", run_id="r"))
    runtime.run_check(
        CheckRequest(
            check_id="chk-3", task_id="t1", claim_ids=("target",),
            command=(sys.executable, "-c", "print('ok')"), cwd=str(tmp_path),
        ),
        verifier=LocalCommandVerifier(),
    )
    claims = runtime.claims_for_task("t1")
    assert claims["target"].evidence_class == EvidenceClass.REPRODUCED
    assert claims["other"].evidence_class == EvidenceClass.ASSERTED


# ---------------- Scheduler ----------------


def test_scheduler_separates_epistemic_choice_from_routing():
    assert decide_next_step(SchedulerInput(has_required_verification=True)).operation == Operation.CHECK
    assert decide_next_step(SchedulerInput(requests_independent_proposals=True)).operation == Operation.CALL
    assert decide_next_step(SchedulerInput(requires_destructive_capability=True)).operation == Operation.ASK_HUMAN
    assert decide_next_step(SchedulerInput(budget_exhausted=True)).operation == Operation.STOP
    decision = decide_next_step(SchedulerInput(has_required_verification=True))
    assert decision.policy_version


# ---------------- Seams ----------------


def test_action_request_distinguishes_requester_executor_adapter(tmp_path):
    from codeai.adapters import ActionRequest, ActionResult, ActionStatus

    runtime = make_runtime(tmp_path)

    class Echo:
        def execute(self, request):
            return ActionResult(action_id=request.action_id, status=ActionStatus.SUCCEEDED)

    result = runtime.execute_action(
        ActionRequest(
            action_id="a1", task_id="t1", capability="execute", instruction="do",
            precondition_hash=None, idempotency_key="k1",
            requested_by="human", actor_id="opencode", adapter="opencode", adapter_id="opencode",
        ),
        authority=Authority(frozenset({Capability.EXECUTE})),
        adapter=Echo(),
    )
    assert result.status == ActionStatus.SUCCEEDED
    requested = runtime.ledger.events_by_kind(("action.requested",))[0]
    completed = runtime.ledger.events_by_kind(("action.completed",))[0]
    assert requested.actor_id == "human"
    assert completed.actor_id == "opencode"
    assert requested.payload["adapter_id"] == "opencode"


def test_repository_state_hash_is_deterministic_and_sensitive(tmp_path):
    h1 = default_repository_state_hash(tmp_path)
    h2 = default_repository_state_hash(tmp_path)
    assert h1 == h2
    (tmp_path / "file.txt").write_text("content")
    h3 = default_repository_state_hash(tmp_path)
    assert h3 != h1


# ---------------- Persistence ----------------


def test_restart_preserves_calls_claims_traces_artifacts(tmp_path):
    ledger_path = tmp_path / "ledger.sqlite"
    runtime = make_runtime(tmp_path, ledger_path=ledger_path)
    result = runtime.invoke_call(make_spec(call_id="persist-1"), adapter=FakeCognitionAdapter(responses=["p"]))
    runtime.record_claim(
        Claim(claim_id="pc-1", task_id="t1", statement="persisted", source_call_id="persist-1",
              source_artifact_id=result.raw_artifact.artifact_id, run_id="r1")
    )
    runtime.compile_and_record_context(task_id="t1", actor=make_actor(), prompt="p", events=())
    artifact_id = result.raw_artifact.artifact_id

    # Simulate restart: new Runtime over the same sqlite file + artifact dir.
    runtime2 = make_runtime(tmp_path, ledger_path=ledger_path)
    calls = runtime2.ledger.events_by_kind(("call.completed",))
    assert any(e.payload["call_id"] == "persist-1" for e in calls)
    assert "pc-1" in runtime2.claims_for_task("t1")
    assert len(runtime2.ledger.events_by_kind(("context.compiled",))) == 1
    assert runtime2.artifact_store.read_text(artifact_id) == "p"


def test_first_experiment_representable_at_matched_context(tmp_path):
    """C1 (same model x N) vs H1 (heterogeneous x N) at matched budget/context."""
    runtime = make_runtime(tmp_path)
    homo_branches = [
        {"actor": make_actor(f"h{i}", model="same-m"), "adapter": FakeCognitionAdapter(responses=[f"h{i}"], model="same-m"),
         "variant": Variant(model="same-m", experiment="C1")}
        for i in range(2)
    ]
    hetero_branches = [
        {"actor": make_actor("a", model="model-a"), "adapter": FakeCognitionAdapter(responses=["A"], model="model-a"),
         "variant": Variant(model="model-a", experiment="H1")},
        {"actor": make_actor("b", model="model-b"), "adapter": FakeCognitionAdapter(responses=["B"], model="model-b"),
         "variant": Variant(model="model-b", experiment="H1")},
    ]
    homo = runtime.sealed_fanout(task_id="t1", base_prompt="same problem", branches=homo_branches)
    hetero = runtime.sealed_fanout(task_id="t1", base_prompt="same problem", branches=hetero_branches)
    # ledger makes cost/latency/success/concurrence computable later
    assert len(homo) == len(hetero) == 2
    total_cost = sum(r.cost_usd or 0 for r in (*homo, *hetero))
    assert total_cost >= 0
    assert all(r.status == "succeeded" for r in (*homo, *hetero))
