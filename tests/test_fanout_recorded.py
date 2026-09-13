"""Stage 11.5d-B: sealed fanout executes through recorded cognition.

Offline only. Fresh branches gain the full recorded evidence chain; reruns
replay with zero new provider effects and stable experiment metrics.
"""

from __future__ import annotations

from pathlib import Path

from codeai.adapters import FakeCognitionAdapter
from codeai.analysis import build_report
from codeai.artifacts import FileArtifactStore
from codeai.context import ContextCandidate, ContextCompiler, event_lineage
from codeai.corpus import get_task
from codeai.domain import ActorRef, Seal, Variant
from codeai.experiments import (
    ArmDef,
    ExperimentBudget,
    build_config,
    create_experiment,
    run_arm,
)
from codeai.ledger import SQLiteLedger
from codeai.modelconfig import ModelMapping
from codeai.runtime import Runtime

# Full recorded chain for transport adapters. Pure fakes perform no transport,
# so attempt.observed is absent for them by design (unavailable, never empty).
FULL_BRANCH_KINDS = (
    "call.requested",
    "call.manifest",
    "attempt.started",
    "attempt.interpreted",
    "attempt.retry_decided",
    "attempt.completed",
    "call.status_decided",
    "call.completed",
)
TRANSPORT_KINDS = ("attempt.observed",)


def make_runtime(tmp_path: Path, ledger_path=None) -> Runtime:
    ledger = SQLiteLedger(str(ledger_path) if ledger_path else ":memory:")
    return Runtime(ledger, artifact_store=FileArtifactStore(tmp_path / "artifacts", ledger))


def make_actor(name: str, model: str = "fake-model") -> ActorRef:
    return ActorRef(actor_id=name, kind="model", provider="fake-provider", model=model)


def branch(name: str, call_id: str, key: str, output: str, **extra) -> dict:
    spec = {
        "actor": make_actor(name),
        "adapter": FakeCognitionAdapter(responses=[output]),
        "adapter_id": "fake-provider",
        "variant": Variant(model="fake-model", provider="fake-provider"),
        "prompt": "solve it",
        "call_id": call_id,
        "idempotency_key": key,
    }
    spec.update(extra)
    return spec


def kinds_for(runtime: Runtime, call_id: str) -> list[str]:
    return [
        e.kind
        for e in runtime.ledger.read_all()
        if str(e.payload.get("call_id", e.stream_id)) == call_id or e.stream_id == call_id
    ]


# Fresh branches gain the full recorded chain ------------------------------------------


def test_fresh_fanout_branches_are_recorded_calls(tmp_path):
    runtime = make_runtime(tmp_path)
    results = runtime.sealed_fanout(
        task_id="t1",
        base_prompt="solve it",
        branches=[
            branch("a", "call-a", "key-a", "out-a"),
            branch("b", "call-b", "key-b", "out-b"),
        ],
    )
    assert {r.raw_output for r in results} == {"out-a", "out-b"}
    assert all(r.status == "succeeded" and not r.replayed for r in results)
    for call_id in ("call-a", "call-b"):
        for kind in FULL_BRANCH_KINDS:
            assert kind in kinds_for(runtime, call_id), (call_id, kind)
        recorded = runtime.get_recorded_call(call_id)
        assert recorded is not None and len(recorded.attempts) == 1


def test_fanout_failure_isolation_with_recorded_evidence(tmp_path):
    runtime = make_runtime(tmp_path)
    results = runtime.sealed_fanout(
        task_id="t1",
        base_prompt="p",
        branches=[
            {
                "actor": make_actor("bad"),
                "adapter": FakeCognitionAdapter(fail_call_ids={"call-bad"}),
                "call_id": "call-bad",
                "idempotency_key": "key-bad",
            },
            branch("good1", "call-g1", "key-g1", "ok-1"),
            branch("good2", "call-g2", "key-g2", "ok-2"),
        ],
    )
    by_id = {r.call_id: r for r in results}
    assert by_id["call-bad"].status == "failed"
    assert by_id["call-g1"].raw_output == "ok-1"
    assert by_id["call-g2"].raw_output == "ok-2"
    for call_id in ("call-bad", "call-g1", "call-g2"):
        assert "attempt.completed" in kinds_for(runtime, call_id)
    completed = runtime.ledger.events_by_kind(("fanout.completed",))
    assert [r.status for r in results] == completed[0].payload["statuses"]


# Rerun replays ------------------------------------------------------------------------------


def test_fanout_rerun_replays_with_zero_new_effects(tmp_path):
    runtime = make_runtime(tmp_path)
    adapters = [FakeCognitionAdapter(responses=[f"out-{i}"]) for i in range(2)]

    def run():
        return runtime.sealed_fanout(
            task_id="t1",
            base_prompt="p",
            branches=[
                {
                    "actor": make_actor(f"m{i}"),
                    "adapter": adapters[i],
                    "call_id": f"call-{i}",
                    "idempotency_key": f"key-{i}",
                }
                for i in range(2)
            ],
        )

    first = run()
    assert [len(a.calls) for a in adapters] == [1, 1]
    second = run()
    assert [len(a.calls) for a in adapters] == [1, 1]  # zero new effects
    assert all(r.replayed for r in second)
    assert [r.raw_output for r in second] == [r.raw_output for r in first]
    assert len(runtime.ledger.events_by_kind(("call.replayed",))) == 2
    assert len(runtime.ledger.events_by_kind(("attempt.started",))) == 2
    assert len(runtime.ledger.events_by_kind(("call.completed",))) == 2


def test_partial_fanout_replay(tmp_path):
    runtime = make_runtime(tmp_path)
    adapters = {
        "a": FakeCognitionAdapter(responses=["out-a"]),
        "b": FakeCognitionAdapter(responses=["out-b"]),
        "c": FakeCognitionAdapter(responses=["out-c"]),
    }
    runtime.sealed_fanout(
        task_id="t1",
        base_prompt="p",
        branches=[
            {
                "actor": make_actor("a"),
                "adapter": adapters["a"],
                "call_id": "c-a",
                "idempotency_key": "k-a",
            },
            {
                "actor": make_actor("b"),
                "adapter": adapters["b"],
                "call_id": "c-b",
                "idempotency_key": "k-b",
            },
        ],
    )
    results = runtime.sealed_fanout(
        task_id="t1",
        base_prompt="p",
        branches=[
            {
                "actor": make_actor("a"),
                "adapter": adapters["a"],
                "call_id": "c-a",
                "idempotency_key": "k-a",
            },
            {
                "actor": make_actor("b"),
                "adapter": adapters["b"],
                "call_id": "c-b",
                "idempotency_key": "k-b",
            },
            {
                "actor": make_actor("c"),
                "adapter": adapters["c"],
                "call_id": "c-c",
                "idempotency_key": "k-c",
            },
        ],
    )
    by_id = {r.call_id: r for r in results}
    assert by_id["c-a"].replayed and by_id["c-b"].replayed
    assert not by_id["c-c"].replayed and by_id["c-c"].raw_output == "out-c"
    assert len(adapters["a"].calls) == len(adapters["b"].calls) == 1
    assert len(adapters["c"].calls) == 1


def test_changed_branch_new_key_executes(tmp_path):
    runtime = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(responses=["v1"])
    runtime.sealed_fanout(
        task_id="t1",
        base_prompt="p",
        branches=[
            {
                "actor": make_actor("a"),
                "adapter": adapter,
                "call_id": "c-a",
                "idempotency_key": "k-a",
            }
        ],
    )
    adapter2 = FakeCognitionAdapter(responses=["v2"])
    (result,) = runtime.sealed_fanout(
        task_id="t1",
        base_prompt="p",
        branches=[
            {
                "actor": make_actor("a"),
                "adapter": adapter2,
                "call_id": "c-a2",
                "idempotency_key": "k-a2",
            }
        ],
    )
    assert not result.replayed and result.raw_output == "v2"


# Seal across new surfaces --------------------------------------------------------------------------


def test_sibling_seal_excludes_all_new_event_kinds(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.sealed_fanout(
        task_id="t1",
        base_prompt="p",
        branches=[
            branch("a", "call-a", "key-a", "out-a"),
            branch("b", "call-b", "key-b", "out-b"),
        ],
    )
    events = list(runtime.ledger.read_all())
    sibling_b_lineage = {
        line
        for e in events
        if str(e.payload.get("call_id", e.stream_id)) == "call-b"
        for line in event_lineage(e)
    } | {"call-b"}
    candidates = [
        ContextCandidate(
            candidate_id=e.event_id,
            kind="event",
            size_tokens=1,
            required=False,
            lineage_ids=tuple(sorted(event_lineage(e))),
        )
        for e in events
    ]
    package, _ = ContextCompiler().compile_candidates(
        task_id="t1",
        actor=make_actor("a"),
        prompt="p",
        candidates=candidates,
        seal=Seal(forbidden_call_ids=frozenset({"call-b"})),
    )
    for event_id in package.event_ids:
        event = next(e for e in events if e.event_id == event_id)
        assert str(event.payload.get("call_id", event.stream_id)) != "call-b", event.kind
        assert not (set(event_lineage(event)) & sibling_b_lineage), event.kind
    # every recorded kind for the sibling was excludable: none leaked in
    kinds_included = {
        next(e for e in events if e.event_id == event_id).kind for event_id in package.event_ids
    }
    assert kinds_included  # own call evidence usable; sibling excluded above


def test_sibling_own_call_still_usable(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.sealed_fanout(
        task_id="t1",
        base_prompt="p",
        branches=[
            branch("a", "call-a", "key-a", "out-a"),
            branch("b", "call-b", "key-b", "out-b"),
        ],
    )
    events = list(runtime.ledger.read_all())
    candidates = [
        ContextCandidate(
            candidate_id=e.event_id,
            kind="event",
            size_tokens=1,
            required=False,
            lineage_ids=tuple(sorted(event_lineage(e))),
        )
        for e in events
    ]
    package, _ = ContextCompiler().compile_candidates(
        task_id="t1",
        actor=make_actor("a"),
        prompt="p",
        candidates=candidates,
        seal=Seal(forbidden_call_ids=frozenset({"call-b"})),
    )
    own = [
        e
        for e in events
        if e.event_id in set(package.event_ids)
        and str(e.payload.get("call_id", e.stream_id)) == "call-a"
    ]
    assert {e.kind for e in own} >= {"call.requested", "call.manifest", "call.completed"}


# run_arm integration ----------------------------------------------------------------------------------


class StubModelConfig:
    def __init__(self, adapters):
        self._adapters = adapters

    def resolve(self, logical_name):
        return ModelMapping(logical_name=logical_name, adapter="fake", model=logical_name)

    def build_adapter(self, logical_name):
        return self._adapters[logical_name]


def _code_response(code: str) -> str:
    return f"Here is the fix:\n```python\n{code}\n```\n"


def test_run_arm_rerun_replays_with_stable_metrics(tmp_path):
    runtime = make_runtime(tmp_path)
    task = get_task("off-by-one-sum")
    config = build_config(
        name="t",
        hypothesis="h",
        task_ids=(task.task_id,),
        arms=(ArmDef(name="C0", models=("m",), samples=1),),
        budget=ExperimentBudget(),
    )
    create_experiment(runtime, config)
    adapter = FakeCognitionAdapter(responses=[_code_response(task.reference_solution)])
    stub = StubModelConfig({"m": adapter})
    first = run_arm(
        runtime, config.experiment_id, "C0", (task,), stub, candidates_root=tmp_path / "cand"
    )
    assert first["completed_tasks"] == 1
    calls_after_first = len(adapter.calls)
    report_before = build_report(runtime, config.experiment_id)
    usage_before = runtime.ledger.events_by_kind(("call.completed",))

    second = run_arm(
        runtime, config.experiment_id, "C0", (task,), stub, candidates_root=tmp_path / "cand"
    )
    assert len(adapter.calls) == calls_after_first  # zero new provider effects
    assert len(runtime.ledger.events_by_kind(("call.completed",))) == len(usage_before)
    assert build_report(runtime, config.experiment_id) == report_before
    assert second["completed_tasks"] == 1
    assert runtime.ledger.events_by_kind(("call.replayed",))


def test_effects_equal_attempt_started_events(tmp_path):
    runtime = make_runtime(tmp_path)
    task = get_task("off-by-one-sum")
    config = build_config(
        name="t",
        hypothesis="h",
        task_ids=(task.task_id,),
        arms=(ArmDef(name="C1", models=("m",), samples=2),),
        budget=ExperimentBudget(),
    )
    create_experiment(runtime, config)
    adapter = FakeCognitionAdapter(responses=[_code_response(task.reference_solution)])
    run_arm(
        runtime,
        config.experiment_id,
        "C1",
        (task,),
        StubModelConfig({"m": adapter}),
        candidates_root=tmp_path / "cand",
    )
    started = runtime.ledger.events_by_kind(("attempt.started",))
    assert len(started) == len(adapter.calls) == 2


def test_integration_three_branches_retry_then_restart_rerun(tmp_path):
    """Acceptance fixture: experiment, three sealed branches, one branch
    retries once; rerun after restart replays everything with zero effects,
    stable usage totals, metrics, and seals."""
    ledger_path = tmp_path / "ledger.sqlite"
    runtime = make_runtime(tmp_path, ledger_path=ledger_path)
    flaky = FakeCognitionAdapter(
        behaviors=[
            {
                "output": "",
                "status": "failed",
                "error": "blip",
                "error_kind": "transient_failure",
                "input_tokens": 5,
                "output_tokens": 5,
            },
            {"output": "recovered", "status": "succeeded", "input_tokens": 10, "output_tokens": 5},
        ]
    )
    steady_a = FakeCognitionAdapter(responses=["out-a"], input_tokens=10, output_tokens=5)
    steady_b = FakeCognitionAdapter(responses=["out-b"], input_tokens=10, output_tokens=5)
    branches = [
        {"actor": make_actor("a"), "adapter": steady_a, "call_id": "c-a", "idempotency_key": "k-a"},
        {
            "actor": make_actor("f"),
            "adapter": flaky,
            "call_id": "c-f",
            "idempotency_key": "k-f",
            "max_attempts": 2,
        },
        {"actor": make_actor("b"), "adapter": steady_b, "call_id": "c-b", "idempotency_key": "k-b"},
    ]
    first = runtime.sealed_fanout(task_id="t1", base_prompt="p", branches=branches)
    assert [r.raw_output for r in first] == ["out-a", "recovered", "out-b"]
    assert len(flaky.calls) == 2  # one retry happened
    assert sum(e.kind == "attempt.started" for e in runtime.ledger.read_all()) == 4

    runtime2 = make_runtime(tmp_path, ledger_path=ledger_path)
    steady_a2 = FakeCognitionAdapter(responses=["changed"])
    flaky2 = FakeCognitionAdapter(responses=["changed"])
    steady_b2 = FakeCognitionAdapter(responses=["changed"])
    rerun_branches = [
        {
            "actor": make_actor("a"),
            "adapter": steady_a2,
            "call_id": "c-a",
            "idempotency_key": "k-a",
        },
        {
            "actor": make_actor("f"),
            "adapter": flaky2,
            "call_id": "c-f",
            "idempotency_key": "k-f",
            "max_attempts": 2,
        },
        {
            "actor": make_actor("b"),
            "adapter": steady_b2,
            "call_id": "c-b",
            "idempotency_key": "k-b",
        },
    ]
    second = runtime2.sealed_fanout(task_id="t1", base_prompt="p", branches=rerun_branches)
    assert [r.raw_output for r in second] == ["out-a", "recovered", "out-b"]
    assert all(r.replayed for r in second)
    assert len(steady_a2.calls) == len(flaky2.calls) == len(steady_b2.calls) == 0
    assert sum(e.kind == "attempt.started" for e in runtime2.ledger.read_all()) == 4
    assert len(runtime2.ledger.events_by_kind(("call.replayed",))) == 3
    # usage accounting sees the original three calls exactly once
    completed = [
        e
        for e in runtime2.ledger.events_by_kind(("call.completed",))
        if e.payload.get("idempotency_key") in ("k-a", "k-f", "k-b")
    ]
    assert len(completed) == 3


def test_changed_branch_conflicts_sibling_replays(tmp_path):
    runtime = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(responses=["v1"])
    runtime.sealed_fanout(
        task_id="t1",
        base_prompt="same prompt",
        branches=[
            {
                "actor": make_actor("a"),
                "adapter": adapter,
                "call_id": "c-a",
                "idempotency_key": "k-a",
            },
            {
                "actor": make_actor("b"),
                "adapter": adapter,
                "call_id": "c-b",
                "idempotency_key": "k-b",
            },
        ],
    )
    # Same keys but a materially changed prompt on branch A only: replay
    # lookup hits, fingerprint mismatches, and only that branch fails.
    results = runtime.sealed_fanout(
        task_id="t1",
        base_prompt="same prompt",
        branches=[
            {
                "actor": make_actor("a"),
                "adapter": adapter,
                "call_id": "c-a",
                "idempotency_key": "k-a",
                "prompt": "CHANGED prompt",
            },
            {
                "actor": make_actor("b"),
                "adapter": adapter,
                "call_id": "c-b",
                "idempotency_key": "k-b",
            },
        ],
    )
    by_id = {r.call_id: r for r in results}
    assert by_id["c-a"].status == "failed"
    assert isinstance(by_id["c-a"].error, str) and "prompt_hash" in by_id["c-a"].error
    assert by_id["c-b"].replayed and by_id["c-b"].raw_output == "v1"
    assert len(adapter.calls) == 2  # only the two original effects


def test_restart_fanout_replay(tmp_path):
    ledger_path = tmp_path / "ledger.sqlite"
    runtime = make_runtime(tmp_path, ledger_path=ledger_path)
    adapter = FakeCognitionAdapter(responses=["out-a"])
    runtime.sealed_fanout(
        task_id="t1",
        base_prompt="p",
        branches=[
            {
                "actor": make_actor("a"),
                "adapter": adapter,
                "call_id": "c-a",
                "idempotency_key": "k-a",
            }
        ],
    )
    runtime2 = make_runtime(tmp_path, ledger_path=ledger_path)
    adapter2 = FakeCognitionAdapter(responses=["other"])
    (result,) = runtime2.sealed_fanout(
        task_id="t1",
        base_prompt="p",
        branches=[
            {
                "actor": make_actor("a"),
                "adapter": adapter2,
                "call_id": "c-a",
                "idempotency_key": "k-a",
            }
        ],
    )
    assert result.replayed and result.raw_output == "out-a"
    assert adapter2.calls == []
    assert runtime2.ledger.events_by_kind(("call.replayed",))
