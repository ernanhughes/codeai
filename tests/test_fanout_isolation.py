"""Chapter 23: blind != independent != diverse for sealed fan-out.

Pins the actual isolation guarantees: declared sibling lineage excluded from
packages, omitted provenance included (boundary), prompt contamination not
filtered, per-branch failure containment, and the durable lineage chain.
"""

from codeai.adapters import FakeCognitionAdapter
from codeai.context import ContextCompiler
from codeai.domain import ActorRef, Seal
from codeai.ledger import Event, SQLiteLedger
from codeai.runtime import Runtime

SENTINEL = "sibling-secret-S3NT1N3L"


def actor(name):
    return ActorRef(actor_id=name, kind="model", provider="fake", model="fake")


def make_runtime(path=None):
    ledger = SQLiteLedger(str(path) if path else ":memory:")
    return Runtime(ledger)


def base_event():
    return Event.create(stream_id="base", kind="task.created", actor_id="runtime",
                        payload={"note": "shared source pack"})


def sibling_event():
    return Event.create(stream_id="call-B", kind="call.completed", actor_id="reviewer-B",
                        payload={"call_id": "call-B", "text": SENTINEL})


def seal_a():
    return Seal(forbidden_call_ids=frozenset({"call-B", "call-C"}),
                forbidden_lineage_ids=frozenset({"call-B", "call-C"}))


class FailingTransportAdapter:
    """A branch whose transport fails outside (ValueError, RuntimeError)."""

    def __init__(self):
        self.calls = 0

    def invoke(self, spec):
        self.calls += 1
        raise OSError("simulated transport failure")


def test_declared_sibling_lineage_excluded_base_included():
    base = base_event()
    sib = sibling_event()
    package, trace = ContextCompiler().compile_with_trace(
        task_id="t", actor=actor("reviewer-A"), prompt="review",
        events=(base, sib),
        artifact_ids=("artifact-sib-B",), claim_ids=("claim-sib-B",),
        seal=seal_a(), required_event_ids=(base.event_id,),
        artifact_lineage={"artifact-sib-B": ("call-B",)},
        claim_lineage={"claim-sib-B": ("call-B",)},
    )
    entries = {e.candidate_id: e for e in trace.entries}
    assert entries[base.event_id].decision == "included"
    for candidate in (sib.event_id, "artifact-sib-B", "claim-sib-B"):
        assert entries[candidate].decision == "excluded"
        assert "seal forbids" in entries[candidate].reason
    assert SENTINEL not in package.prompt
    assert sib.event_id not in package.event_ids


def test_omitted_lineage_is_included_boundary():
    # The same artifact id with no declared lineage is included. This pins the
    # limitation: seals exclude declared identifiers, not unknown provenance.
    base = base_event()
    _, trace = ContextCompiler().compile_with_trace(
        task_id="t", actor=actor("reviewer-A"), prompt="review",
        events=(base,),
        artifact_ids=("artifact-sib-B",),
        seal=seal_a(), required_event_ids=(base.event_id,),
    )
    entries = {e.candidate_id: e for e in trace.entries}
    assert entries["artifact-sib-B"].decision == "included"


def test_shared_prompt_contamination_not_filtered():
    # Sibling text copied into the shared prompt is not removed by any seal:
    # seals match declared identifiers, never arbitrary semantics.
    base = base_event()
    package, _ = ContextCompiler().compile_with_trace(
        task_id="t", actor=actor("reviewer-A"),
        prompt=f"review. A sibling once wrote: {SENTINEL}",
        events=(base,), seal=seal_a(),
        required_event_ids=(base.event_id,),
    )
    assert SENTINEL in package.prompt


def test_transport_failure_in_one_branch_kills_no_sibling(tmp_path):
    runtime = make_runtime()
    failing = FailingTransportAdapter()
    results = runtime.sealed_fanout(
        task_id="t", base_prompt="review",
        branches=[
            {"call_id": "call-A", "actor": actor("A"),
             "adapter": FakeCognitionAdapter(responses=["proposal A"])},
            {"call_id": "call-B", "actor": actor("B"), "adapter": failing},
            {"call_id": "call-C", "actor": actor("C"),
             "adapter": FakeCognitionAdapter(responses=["proposal C"])},
        ],
    )
    by_id = {r.call_id: r for r in results}
    assert by_id["call-A"].status == "succeeded"
    assert by_id["call-C"].status == "succeeded"
    assert by_id["call-B"].status == "failed"
    assert "simulated transport failure" in (by_id["call-B"].error or "")
    completed = runtime.ledger.events_by_kind(("fanout.completed",))
    assert [r.status for r in results] == completed[0].payload["statuses"]
    # The interrupted attempt is inspectable, not resolved: started with no
    # observation, then the synthesized failed completion.
    kinds = [e.kind for e in runtime.ledger.read_all()
             if e.stream_id in ("call-B",) or str(e.payload.get("call_id")) == "call-B"]
    assert "attempt.started" in kinds
    assert "attempt.observed" not in kinds


def test_fanout_packages_contain_only_base_inputs():
    runtime = make_runtime()
    base = base_event()
    runtime.sealed_fanout(
        task_id="t", base_prompt="review",
        base_events=(base,),
        branches=[
            {"call_id": "call-A", "actor": actor("A"),
             "adapter": FakeCognitionAdapter(responses=["proposal A"])},
            {"call_id": "call-B", "actor": actor("B"),
             "adapter": FakeCognitionAdapter(responses=["proposal B"])},
        ],
    )
    requested = {e.stream_id: e for e in runtime.ledger.events_by_kind(("call.requested",))}
    for call_id in ("call-A", "call-B"):
        payload = requested[call_id].payload
        assert payload["context"]["event_ids"] == [base.event_id]
        assert payload["context"]["prompt"] == "review"
        assert SENTINEL not in payload["context"]["prompt"]
        seal = payload["context"]["seal"]
        assert "call-A" in seal["forbidden_call_ids"] or "call-B" in seal["forbidden_call_ids"]


def test_isolation_chain_survives_reopen(tmp_path):
    path = tmp_path / "ledger.sqlite"
    runtime = Runtime(SQLiteLedger(str(path)))
    runtime.sealed_fanout(
        task_id="t", base_prompt="review", base_events=(base_event(),),
        branches=[
            {"call_id": "call-A", "actor": actor("A"),
             "adapter": FakeCognitionAdapter(responses=["proposal A"])},
            {"call_id": "call-B", "actor": actor("B"),
             "adapter": FakeCognitionAdapter(responses=["proposal B"])},
        ],
    )
    reopened = Runtime(SQLiteLedger(str(path)))
    events = reopened.ledger.read_all()
    assert [e.kind for e in events].count("fanout.requested") == 1
    assert [e.kind for e in events].count("fanout.completed") == 1
    for call_id in ("call-A", "call-B"):
        requested = next(e for e in events
                         if e.kind == "call.requested" and e.stream_id == call_id)
        package_id = requested.payload["context"]["package_id"]
        compiled = next(e for e in events
                        if e.kind == "context.compiled"
                        and e.payload["package_id"] == package_id)
        assert compiled.payload["trace_id"]
        assert compiled.payload["seal"]["forbidden_call_ids"]
        manifest = next(e for e in events
                        if e.kind == "call.manifest" and e.stream_id == call_id)
        assert manifest.payload["context_package_id"] == package_id
    reopened.ledger._conn.close()
