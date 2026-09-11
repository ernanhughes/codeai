from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from codeai.adapters import FakeCognitionAdapter
from codeai.analysis import (
    arm_metrics,
    build_report,
    conditional_failure,
    export_experiment,
    render_report,
    unique_rescues,
)
from codeai.artifacts import FileArtifactStore
from codeai.corpus import get_task, seeded_corpus, visible_prompt
from codeai.domain import EvidenceClass
from codeai.experiments import (
    ARM_C0,
    ARM_C1,
    ARM_H1,
    ArmDef,
    ExperimentBudget,
    build_config,
    create_experiment,
    get_experiment,
    plan_experiment,
    run_arm,
)
from codeai.ledger import SQLiteLedger
from codeai.modelconfig import ModelConfig, ModelMapping, load_model_config
from codeai.providers import (
    AnthropicAdapter,
    OpenAIAdapter,
    OpenAICompatibleAdapter,
    estimate_cost_usd,
)
from codeai.runtime import Runtime


class StubModelConfig:
    """Duck-typed model config: scripted adapters per logical name."""

    def __init__(self, adapters: dict[str, object], mappings: dict[str, ModelMapping] | None = None):
        self._adapters = adapters
        self._mappings = mappings or {
            name: ModelMapping(logical_name=name, adapter="fake", model=name)
            for name in adapters
        }

    def resolve(self, logical_name: str) -> ModelMapping:
        if logical_name not in self._mappings:
            raise KeyError(logical_name)
        return self._mappings[logical_name]

    def build_adapter(self, logical_name: str) -> object:
        return self._adapters[logical_name]


def make_runtime(tmp_path: Path) -> Runtime:
    ledger = SQLiteLedger()
    return Runtime(ledger, artifact_store=FileArtifactStore(tmp_path / "artifacts", ledger))


def code_response(code: str) -> str:
    return f"Here is the fix:\n```python\n{code}\n```\n"


def exp_config(task_ids: tuple[str, ...] = ("off-by-one-sum",), **kwargs) -> object:
    arms = kwargs.pop("arms", (ArmDef(name=ARM_C1, models=("m",), samples=2),))
    return build_config(
        name="t", hypothesis="h", task_ids=task_ids, arms=arms,
        budget=kwargs.pop("budget", ExperimentBudget()), **kwargs)


# ---------------- Providers ----------------


def test_openai_adapter_maps_request_and_result():
    seen: dict = {}

    def fake_post(url, payload, headers, timeout):
        seen.update({"url": url, "payload": payload, "headers": headers})
        return {
            "id": "chatcmpl-1", "model": "gpt-4o-mini",
            "choices": [{"message": {"content": "fixed code"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

    adapter = OpenAIAdapter(model="gpt-4o-mini", api_key="k", http_post=fake_post)
    from codeai.context import ContextCompiler
    from codeai.domain import ActorRef, CallSpec

    actor = ActorRef(actor_id="m", kind="model")
    package = ContextCompiler().compile(task_id="t", actor=actor, prompt="fix it", events=())
    spec = CallSpec(call_id="c", task_id="t", actor=actor, context=package,
                    idempotency_key="k", parameters={"temperature": 0.2, "seed": "7"})
    result = adapter.invoke(spec)
    assert result.raw_output == "fixed code"
    assert (result.input_tokens, result.output_tokens) == (100, 50)
    assert result.provider_call_id == "chatcmpl-1"
    assert result.provider == "openai" and result.model == "gpt-4o-mini"
    assert result.cost_usd is not None and result.cost_usd > 0
    assert seen["payload"]["temperature"] == 0.2
    assert seen["url"].endswith("/chat/completions")


def test_openai_missing_credentials_is_failed_result_not_exception(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    adapter = OpenAIAdapter(model="gpt-4o-mini", api_key="")
    from codeai.context import ContextCompiler
    from codeai.domain import ActorRef, CallSpec

    actor = ActorRef(actor_id="m", kind="model")
    package = ContextCompiler().compile(task_id="t", actor=actor, prompt="p", events=())
    result = adapter.invoke(CallSpec(call_id="c", task_id="t", actor=actor,
                                     context=package, idempotency_key="k"))
    assert result.status == "failed"
    assert "OPENAI_API_KEY" in (result.error or "")


def test_anthropic_adapter_maps_result_and_usage():
    def fake_post(url, payload, headers, timeout):
        assert url.endswith("/messages")
        assert headers["x-api-key"] == "sk-ant"
        return {
            "id": "msg-1", "model": "claude-haiku", "type": "message",
            "content": [{"type": "text", "text": "answer"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    adapter = AnthropicAdapter(model="claude-haiku", api_key="sk-ant", http_post=fake_post)
    from codeai.context import ContextCompiler
    from codeai.domain import ActorRef, CallSpec

    actor = ActorRef(actor_id="m", kind="model")
    package = ContextCompiler().compile(task_id="t", actor=actor, prompt="p", events=())
    result = adapter.invoke(CallSpec(call_id="c", task_id="t", actor=actor,
                                     context=package, idempotency_key="k"))
    assert result.raw_output == "answer"
    assert (result.input_tokens, result.output_tokens) == (10, 5)
    assert result.provider == "anthropic"


def test_provider_failure_and_unknown_cost():
    def boom(url, payload, headers, timeout):
        raise RuntimeError("unreachable")

    from codeai.providers import ProviderError

    adapter = OpenAICompatibleAdapter(model="unknown-local-model", base_url="http://x/v1",
                                      http_post=lambda *a: (_ for _ in ()).throw(ProviderError("down")))
    from codeai.context import ContextCompiler
    from codeai.domain import ActorRef, CallSpec

    actor = ActorRef(actor_id="m", kind="model")
    package = ContextCompiler().compile(task_id="t", actor=actor, prompt="p", events=())
    result = adapter.invoke(CallSpec(call_id="c", task_id="t", actor=actor,
                                     context=package, idempotency_key="k"))
    assert result.status == "failed"
    assert estimate_cost_usd("unknown-local-model", 100, 100) is None


def test_raw_output_captured_before_extraction(tmp_path):
    runtime = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(responses=["```python\ndef total(n):\n    return 1\n```"])
    from codeai.context import ContextCompiler
    from codeai.domain import ActorRef, CallSpec

    actor = ActorRef(actor_id="m", kind="model")
    package = ContextCompiler().compile(task_id="t", actor=actor, prompt="p", events=())
    result = runtime.invoke_call(
        CallSpec(call_id="c", task_id="t", actor=actor, context=package, idempotency_key="k"),
        adapter=adapter)
    assert result.raw_artifact is not None
    assert "def total" in runtime.artifact_store.read_text(result.raw_artifact.artifact_id)


# ---------------- Corpus ----------------


def test_corpus_has_enough_tasks_and_hides_secrets():
    corpus = seeded_corpus()
    assert len(corpus) >= 10
    for task in corpus:
        prompt = visible_prompt(task)
        assert task.hidden_tests not in prompt
        assert task.reference_solution not in prompt
        assert task.starter_code in prompt


def test_corpus_hidden_tests_fail_on_starter_and_pass_on_reference():
    for task in seeded_corpus():
        for label, code, expect_ok in (
            ("starter", task.starter_code, False),
            ("reference", task.reference_solution, True),
        ):
            with tempfile.TemporaryDirectory() as d:
                Path(d, "candidate.py").write_text(code)
                Path(d, "hidden_test.py").write_text(task.hidden_tests)
                completed = subprocess.run(
                    [sys.executable, "hidden_test.py"], cwd=d, check=False,
                    capture_output=True, text=True)
                assert (completed.returncode == 0) == expect_ok, (task.task_id, label)


# ---------------- Experiments ----------------


def test_experiment_config_immutable():
    runtime = make_runtime(Path(tempfile.mkdtemp()))
    config = exp_config()
    create_experiment(runtime, config)
    with pytest.raises(ValueError):
        create_experiment(runtime, config)
    assert get_experiment(runtime, config.experiment_id) is not None


def test_c0_generates_single_call(tmp_path):
    runtime = make_runtime(tmp_path)
    task = get_task("off-by-one-sum")
    config = exp_config(arms=(ArmDef(name=ARM_C0, models=("m",), samples=1),))
    create_experiment(runtime, config)
    stub = StubModelConfig({"m": FakeCognitionAdapter(responses=[code_response(task.reference_solution)])})
    summary = run_arm(runtime, config.experiment_id, ARM_C0, (task,), stub,
                      candidates_root=tmp_path / "cand")
    assert summary["completed_tasks"] == 1
    calls = [e for e in runtime.ledger.events_by_kind(("call.completed",))]
    assert len(calls) == 1
    verified = [e for e in runtime.ledger.events_by_kind(("experiment.candidate_verified",))]
    assert verified[0].payload["outcome"] == "VERIFIER_PASS"


def test_c1_homogeneous_fanout_uses_same_primitive_and_matched_context(tmp_path):
    runtime = make_runtime(tmp_path)
    task = get_task("off-by-one-sum")
    config = exp_config(arms=(ArmDef(name=ARM_C1, models=("m",), samples=3),))
    create_experiment(runtime, config)
    stub = StubModelConfig({"m": FakeCognitionAdapter(responses=[code_response(task.reference_solution)])})
    run_arm(runtime, config.experiment_id, ARM_C1, (task,), stub, candidates_root=tmp_path / "cand")
    fanouts = runtime.ledger.events_by_kind(("fanout.requested",))
    assert len(fanouts) == 1 and len(fanouts[0].payload["call_ids"]) == 3
    contexts = runtime.ledger.events_by_kind(("context.compiled",))
    assert len(contexts) == 3
    seals = [c.payload["seal"]["forbidden_call_ids"] for c in contexts]
    for seal, ctx in zip(seals, contexts):
        assert len(seal) == 2  # sealed from the two siblings


def test_h1_heterogeneous_models_same_primitive(tmp_path):
    runtime = make_runtime(tmp_path)
    task = get_task("off-by-one-sum")
    config = exp_config(arms=(ArmDef(name=ARM_H1, models=("a", "b"), samples=2),))
    create_experiment(runtime, config)
    stub = StubModelConfig({
        "a": FakeCognitionAdapter(responses=[code_response(task.reference_solution)], model="a"),
        "b": FakeCognitionAdapter(responses=["no code here"], model="b"),
    })
    run_arm(runtime, config.experiment_id, ARM_H1, (task,), stub, candidates_root=tmp_path / "cand")
    outcomes = sorted(e.payload["outcome"]
                      for e in runtime.ledger.events_by_kind(("experiment.candidate_verified",)))
    assert outcomes == ["GENERATED", "VERIFIER_PASS"]


def test_budget_stop_records_incomplete_arm(tmp_path):
    runtime = make_runtime(tmp_path)
    task = get_task("off-by-one-sum")
    config = exp_config(
        arms=(ArmDef(name=ARM_C1, models=("m",), samples=3),),
        budget=ExperimentBudget(max_calls=1))
    create_experiment(runtime, config)
    stub = StubModelConfig({"m": FakeCognitionAdapter()})
    summary = run_arm(runtime, config.experiment_id, ARM_C1, (task,), stub,
                      candidates_root=tmp_path / "cand")
    assert summary["stopped"] and summary["stopped"]["reason"] == "BUDGET_STOPPED"
    stops = runtime.ledger.events_by_kind(("experiment.arm_stopped",))
    assert stops and stops[0].payload["remaining"] == ["off-by-one-sum"]
    assert summary["completed_tasks"] == 0


def test_isolated_candidate_state_same_start(tmp_path):
    runtime = make_runtime(tmp_path)
    task = get_task("inverted-comparison-adult")
    config = exp_config(arms=(ArmDef(name=ARM_C1, models=("m",), samples=2),))
    create_experiment(runtime, config)
    stub = StubModelConfig({"m": FakeCognitionAdapter(responses=[code_response(task.reference_solution)])})
    root = tmp_path / "cand"
    run_arm(runtime, config.experiment_id, ARM_C1, (task,), stub, candidates_root=root)
    verified = runtime.ledger.events_by_kind(("experiment.candidate_verified",))
    assert len(verified) == 2
    assert verified[0].payload["starting_state_hash"] == verified[1].payload["starting_state_hash"]
    dirs = sorted((root / config.experiment_id / ARM_C1 / task.task_id).iterdir())
    assert len(dirs) == 2 and dirs[0] != dirs[1]


def test_verifier_result_attribution_and_claim_evidence(tmp_path):
    runtime = make_runtime(tmp_path)
    task = get_task("off-by-one-sum")
    config = exp_config()
    create_experiment(runtime, config)
    stub = StubModelConfig({"m": FakeCognitionAdapter(responses=[code_response(task.reference_solution)])})
    run_arm(runtime, config.experiment_id, ARM_C1, (task,), stub, candidates_root=tmp_path / "cand")
    verified = runtime.ledger.events_by_kind(("experiment.candidate_verified",))[0].payload
    assert verified["check_id"]
    check = next(e for e in runtime.ledger.events_by_kind(("check.completed",))
                 if e.payload["check_id"] == verified["check_id"])
    assert check.payload["verdict"] == "PASS"
    claims = runtime.claims_for_task(verified["task_id"])
    assert claims[f"claim-{verified['call_id']}"].evidence_class == EvidenceClass.REPRODUCED


def test_dry_run_causes_zero_calls_and_no_events(tmp_path):
    runtime = make_runtime(tmp_path)
    config = exp_config(arms=(
        ArmDef(name=ARM_C1, models=("m",), samples=3),
        ArmDef(name=ARM_H1, models=("a", "b"), samples=2)))
    model_config = ModelConfig(models={
        "m": ModelMapping(logical_name="m", adapter="fake", model="m"),
        "a": ModelMapping(logical_name="a", adapter="openai", model="gpt-4o-mini"),
        "b": ModelMapping(logical_name="b", adapter="anthropic", model="claude-haiku"),
    })
    before = len(runtime.ledger.read_all())
    plan = plan_experiment(config, model_config, corpus_tasks=seeded_corpus()[:2])
    assert len(runtime.ledger.read_all()) == before
    assert plan["total_calls_planned"] == 1 * 3 + 1 * 2
    assert plan["arms"][0]["calls_planned"] == 3
    assert any(m["missing"] for arm in plan["arms"] for m in arm["models"])


def test_export_round_trip_has_no_secrets(tmp_path):
    runtime = make_runtime(tmp_path)
    task = get_task("off-by-one-sum")
    config = exp_config(arms=(ArmDef(name=ARM_C0, models=("m",), samples=1),))
    create_experiment(runtime, config)
    stub = StubModelConfig({"m": FakeCognitionAdapter(responses=[code_response(task.reference_solution)])})
    run_arm(runtime, config.experiment_id, ARM_C0, (task,), stub, candidates_root=tmp_path / "cand")
    payload = export_experiment(runtime, config.experiment_id)
    text = json.dumps(payload, sort_keys=True, default=str)
    assert json.loads(text) == json.loads(json.dumps(payload, sort_keys=True, default=str))
    assert "OPENAI_API_KEY" not in text and "ANTHROPIC_API_KEY" not in text
    assert payload["format"] == "codeai-experiment-export-v1"
    assert payload["experiment"]["config_hash"] == config.config_hash


# ---------------- Metrics ----------------


def test_oracle_solve_rate_cost_per_solve():
    metrics = arm_metrics("C1", ["t1", "t2"], [
        {"arm": "C1", "corpus_task_id": "t1", "outcome": "VERIFIER_PASS"},
        {"arm": "C1", "corpus_task_id": "t1", "outcome": "VERIFIER_FAIL"},
        {"arm": "C1", "corpus_task_id": "t2", "outcome": "VERIFIER_FAIL"},
    ], [
        {"arm": "C1", "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.01, "latency_ms": 10},
        {"arm": "C1", "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.01, "latency_ms": 30},
        {"arm": "C1", "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.01, "latency_ms": 20},
    ])
    assert metrics["oracle_at_2"] == 0.5
    assert metrics["verified_solve_rate"] == 0.5
    assert metrics["cost_per_verified_solve"] == pytest.approx(0.03)
    assert metrics["median_latency_ms"] == 20


def test_unique_rescue_and_conditional_failure():
    assert unique_rescues({"t1", "t2"}, {"t2", "t3"}) == ["t1"]
    assert conditional_failure({"t1"}, {"t1", "t2"}, {"t1", "t2", "t3"}) == pytest.approx(0.5)
    assert conditional_failure({"t1", "t2"}, {"t1"}, {"t1"}) is None


def test_heterogeneity_premium_report(tmp_path):
    runtime = make_runtime(tmp_path)
    tasks = (get_task("off-by-one-sum"), get_task("inverted-comparison-adult"))
    refs = {t.task_id: t.reference_solution for t in tasks}
    config = build_config(
        name="premium", hypothesis="h", task_ids=tuple(t.task_id for t in tasks),
        arms=(ArmDef(name=ARM_C1, models=("m",), samples=1),
              ArmDef(name=ARM_H1, models=("a", "b"), samples=1)))
    create_experiment(runtime, config)
    from codeai.adapters import CallSpec

    class PromptRouter:
        """Deterministic stand-in: C1 solves task 1 only, H1 solves both via model b."""

        def __init__(self, solve: set):
            self.solve = solve

        def invoke(self, spec: CallSpec):
            task_hit = next((t for t in tasks if t.problem_statement[:30] in spec.context.prompt), None)
            good = task_hit is not None and (task_hit.task_id in self.solve)
            code = refs[task_hit.task_id] if (good and task_hit) else "no code"
            return FakeCognitionAdapter(responses=[code_response(code)]).invoke(spec)

    stub = StubModelConfig({"m": PromptRouter({"off-by-one-sum"}),
                            "a": PromptRouter(set()),
                            "b": PromptRouter({"off-by-one-sum", "inverted-comparison-adult"})})
    run_arm(runtime, config.experiment_id, ARM_C1, tasks, stub, candidates_root=tmp_path / "c1")
    run_arm(runtime, config.experiment_id, ARM_H1, tasks, stub, candidates_root=tmp_path / "h1")
    report = build_report(runtime, config.experiment_id)
    assert report["arms"]["C1"]["verified_solve_rate"] == 0.5
    assert report["arms"]["H1"]["verified_solve_rate"] == 1.0
    assert report["heterogeneity_premium"] == 0.5
    assert report["unique_rescues"]["H1"] == ["inverted-comparison-adult"]
    text = render_report(report)
    assert "Heterogeneity premium" in text and "tiny-N" in text or "WARNING" in text


def test_model_config_file_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".codeai").mkdir()
    (tmp_path / ".codeai" / "config.toml").write_text(
        '[models.qwen]\nadapter = "openai-compatible"\nbase_url = "http://localhost:11434/v1"\nmodel = "qwen"\n')
    config = load_model_config()
    assert config.resolve("qwen").model == "qwen"
    with pytest.raises(KeyError):
        config.resolve("nope")
