"""Stage 0.5: experiment budgets fail closed under unknown cost/usage.

Offline only. A legacy call.completed fragment below mirrors the frozen
P-series shape (p1-export.json: int tokens, null cost) with provenance noted;
the pinned exports themselves are never mutated.
"""

from __future__ import annotations

from pathlib import Path

from codeai.adapters import CallSpec, FakeCognitionAdapter
from codeai.artifacts import FileArtifactStore
from codeai.context import ContextCompiler
from codeai.corpus import CorpusTask
from codeai.domain import ActorRef
from codeai.experiments import (
    ExperimentBudget,
    ExperimentUsage,
    build_config,
    call_consumption,
    combine_usage,
    create_experiment,
    experiment_usage,
    run_arm,
)
from codeai.experiments import (
    _budget_allows as budget_allows,
)
from codeai.ledger import SQLiteLedger
from codeai.modelconfig import ModelMapping
from codeai.runtime import Runtime


def make_runtime(tmp_path: Path) -> Runtime:
    ledger = SQLiteLedger()
    return Runtime(ledger, artifact_store=FileArtifactStore(tmp_path / "artifacts", ledger))


def make_actor(model: str = "fake-model") -> ActorRef:
    return ActorRef(actor_id="m", kind="model", provider="fake-provider", model=model)


def recorded_spec(
    call_id: str,
    experiment_id: str,
    arm: str = "C0",
    task_id: str = "t1",
    model: str = "fake-model",
) -> CallSpec:
    actor = make_actor(model)
    package = ContextCompiler().compile(task_id=task_id, actor=actor, prompt="p", events=())
    return CallSpec(
        call_id=call_id,
        task_id=task_id,
        actor=actor,
        context=package,
        idempotency_key=f"key-{call_id}",
        experiment_id=experiment_id,
        arm=arm,
    )


def stub_config(**overrides):
    from codeai.experiments import ArmDef, ExperimentBudget

    arms = overrides.pop("arms", (ArmDef(name="C0", models=("m",), samples=1),))
    budget = overrides.pop("budget", ExperimentBudget())
    return build_config(
        name="t",
        hypothesis="h",
        task_ids=("task-a", "task-b"),
        arms=arms,
        budget=budget,
        **overrides,
    )


class StubModelConfig:
    def __init__(self, adapter):
        self._adapter = adapter

    def resolve(self, logical_name: str) -> ModelMapping:
        return ModelMapping(logical_name=logical_name, adapter="fake", model=logical_name)

    def build_adapter(self, logical_name: str):
        return self._adapter


def corpus_tasks(ids=("off-by-one-sum", "inverted-comparison-adult")) -> tuple[CorpusTask, ...]:
    from codeai.corpus import get_task

    return tuple(get_task(i) for i in ids)


# A. one known single-attempt call ---------------------------------------------


def test_known_single_attempt_reports_complete_totals(tmp_path):
    import pytest

    runtime = make_runtime(tmp_path)
    # gpt-4o-mini is priced, so the recorded path derives a known cost. Note
    # the pricing table matches by prefix in order, so the gpt-4o rate wins:
    # 100*2.50/1e6 + 20*10.00/1e6 = 0.00045 (pre-existing table semantics).
    spec = recorded_spec("c1", "exp-a", model="gpt-4o-mini")
    runtime.invoke_recorded_call(
        spec,
        adapter=FakeCognitionAdapter(
            responses=["out"],
            input_tokens=100,
            output_tokens=20,
            cost_usd=0.01,
            model="gpt-4o-mini",
        ),
    )
    usage = experiment_usage(runtime, "exp-a")
    assert usage.calls == 1
    assert (usage.known_input_tokens, usage.known_output_tokens) == (100, 20)
    assert usage.known_cost_usd == pytest.approx(0.00045)
    assert usage.tokens_complete and usage.cost_complete


# B. multi-attempt logical total, not final attempt ------------------------------


def test_two_attempt_known_usage_counts_logical_total(tmp_path):
    runtime = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(
        behaviors=[
            {
                "output": "",
                "status": "failed",
                "error": "blip",
                "error_kind": "transient_failure",
                "input_tokens": 100,
                "output_tokens": 20,
                "usage_source": "measured",
            },
            {
                "output": "done",
                "status": "succeeded",
                "input_tokens": 110,
                "output_tokens": 25,
                "usage_source": "measured",
            },
        ]
    )
    recorded = runtime.invoke_recorded_call(
        recorded_spec("c1", "exp-b"), adapter=adapter, max_attempts=2
    )
    assert len(recorded.attempts) == 2
    usage = experiment_usage(runtime, "exp-b")
    assert (usage.known_input_tokens, usage.known_output_tokens) == (210, 45)
    assert usage.tokens_complete
    # fake-model is absent from the pricing table: cost stays unknown even
    # though tokens are fully known (mixed completeness is representable).
    assert not usage.cost_complete and usage.unknown_cost_calls == 1


# C. partial usage keeps lower bound, not complete -------------------------------


def test_partial_usage_not_reported_complete(tmp_path):
    runtime = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(
        behaviors=[
            {
                "output": "",
                "status": "failed",
                "error": "blip",
                "error_kind": "transient_failure",
                "input_tokens": None,
                "output_tokens": None,
                "usage_source": "unavailable",
            },
            {
                "output": "done",
                "status": "succeeded",
                "input_tokens": 100,
                "output_tokens": 20,
                "usage_source": "measured",
            },
        ]
    )
    runtime.invoke_recorded_call(recorded_spec("c1", "exp-c"), adapter=adapter, max_attempts=2)
    usage = experiment_usage(runtime, "exp-c")
    assert (usage.known_input_tokens, usage.known_output_tokens) == (100, 20)
    assert not usage.tokens_complete


# D. unknown cost, no cost budget -> may continue, still UNKNOWN ------------------


def test_unknown_cost_without_budget_permits_continue(tmp_path):
    config = stub_config()
    usage = ExperimentUsage(calls=1, cost_complete=False, unknown_cost_calls=1)
    ok, _reason, state = budget_allows(config, usage)
    assert ok and state == "ok"
    assert usage.known_cost_usd == 0.0  # lower bound kept, never claimed


# E. unknown cost + max_cost -> STOP ----------------------------------------------


def test_unknown_cost_with_budget_stops_unknown(tmp_path):
    config = stub_config(budget=ExperimentBudget(max_cost_usd=1.0))
    usage = ExperimentUsage(calls=1, cost_complete=False, unknown_cost_calls=1)
    ok, reason, state = budget_allows(config, usage)
    assert not ok and state == "unknown"
    assert "cannot enforce max_cost_usd" in reason


# F. override permits continuation, cost stays UNKNOWN -----------------------------


def test_override_permits_continue_without_changing_unknown(tmp_path):
    config = stub_config(
        budget=ExperimentBudget(
            max_cost_usd=1.0, allow_unknown_cost=True, unknown_cost_reason="lab run"
        )
    )
    usage = ExperimentUsage(calls=1, known_cost_usd=0.0, cost_complete=False, unknown_cost_calls=1)
    ok, _reason, state = budget_allows(config, usage)
    assert ok and state == "override"
    assert not usage.cost_complete and usage.known_cost_usd == 0.0


# G/H. known-cost threshold semantics preserved -------------------------------------


def test_known_cost_threshold_uses_ge(tmp_path):
    below = stub_config(budget=ExperimentBudget(max_cost_usd=1.0))
    usage = ExperimentUsage(calls=2, known_cost_usd=0.5, cost_complete=True)
    assert budget_allows(below, usage)[0] is True
    at_limit = stub_config(budget=ExperimentBudget(max_cost_usd=0.5))
    ok, _reason, state = budget_allows(at_limit, usage)
    assert not ok and state == "exhausted"  # >= stops, as before


def test_max_calls_threshold_uses_gt(tmp_path):
    config = stub_config(budget=ExperimentBudget(max_calls=2))
    assert budget_allows(config, ExperimentUsage(calls=1), planned_calls=1)[0] is True
    assert budget_allows(config, ExperimentUsage(calls=2), planned_calls=1)[0] is False


# I. legacy P-series shape stays readable -------------------------------------------


def test_legacy_p_series_fragment_readable():
    # Shape mirrors frozen p1-export.json call.completed entries: int tokens,
    # null cost, no total_* fields. Pinned exports themselves are untouched.
    fragment = {
        "call_id": "ce2a6ac0-907a-4e1b-b355-ccb774aaab8e",
        "input_tokens": 94,
        "output_tokens": 20,
        "cost_usd": None,
        "status": "succeeded",
        "model": "qwen2.5-coder:latest",
        "provider": "openai-compatible",
        "experiment_id": "0ea37186-bada-4e1b-b355-ccb774aaab8e",
        "arm": "C0",
    }
    usage = call_consumption(fragment)
    assert (usage.known_input_tokens, usage.known_output_tokens) == (94, 20)
    assert usage.tokens_complete  # legacy ints are measurements under the old contract
    assert not usage.cost_complete and usage.known_cost_usd == 0.0


def test_legacy_known_cost_figure_unchanged():
    fragment = {"call_id": "x", "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.01}
    usage = call_consumption(fragment)
    assert usage.cost_complete and usage.known_cost_usd == 0.01
    combined = combine_usage([usage, usage])
    assert combined.calls == 2 and combined.known_cost_usd == 0.02


# J. recorded totals preferred over final-attempt fields -------------------------------


def test_recorded_totals_preferred():
    payload = {
        "call_id": "c",
        "input_tokens": 110,
        "output_tokens": 25,
        "cost_usd": 0.002,
        "total_input_tokens": 210,
        "total_output_tokens": 45,
        "total_cost_usd": None,
        "attempt_count": 2,
    }
    usage = call_consumption(payload)
    assert (usage.known_input_tokens, usage.known_output_tokens) == (210, 45)
    assert usage.tokens_complete
    assert not usage.cost_complete


# K. pre-effect guard: unsafe budget stops before next invocation -----------------------


def test_unsafe_budget_stops_before_next_provider_effect(tmp_path):
    runtime = make_runtime(tmp_path)
    config = stub_config(budget=ExperimentBudget(max_cost_usd=0.5))
    create_experiment(runtime, config)
    adapter = FakeCognitionAdapter(responses=["code"], cost_usd=None)
    stub = StubModelConfig(adapter)
    tasks = corpus_tasks()
    summary = run_arm(
        runtime, config.experiment_id, "C0", tasks, stub, candidates_root=tmp_path / "cand"
    )
    assert summary["completed_tasks"] == 1
    assert summary["stopped"] and summary["stopped"]["reason"] == "BUDGET_STOPPED"
    assert summary["stopped"]["budget_state"] == "unknown"
    first_task_invocations = adapter.calls.__len__()
    assert first_task_invocations == 1  # C0 single branch
    # The guard fired before task 2: no further provider effects occurred.
    assert adapter.calls.__len__() == first_task_invocations
    stops = runtime.ledger.events_by_kind(("experiment.arm_stopped",))
    assert stops and stops[0].payload["budget_state"] == "unknown"


def test_override_allows_arm_to_continue_and_records_evidence(tmp_path):
    runtime = make_runtime(tmp_path)
    config = stub_config(
        budget=ExperimentBudget(
            max_cost_usd=0.5, allow_unknown_cost=True, unknown_cost_reason="metered separately"
        )
    )
    create_experiment(runtime, config)
    adapter = FakeCognitionAdapter(responses=["code"], cost_usd=None)
    summary = run_arm(
        runtime,
        config.experiment_id,
        "C0",
        corpus_tasks(),
        StubModelConfig(adapter),
        candidates_root=tmp_path / "cand",
    )
    assert summary["completed_tasks"] == 2
    assert summary["stopped"] is None
    overrides = runtime.ledger.events_by_kind(("budget.override",))
    assert len(overrides) == 1
    assert overrides[0].payload["dimension"] == "cost"
    assert overrides[0].payload["reason"] == "metered separately"
    usage = experiment_usage(runtime, config.experiment_id)
    assert not usage.cost_complete  # override never converts unknown to known
