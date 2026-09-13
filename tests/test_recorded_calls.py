"""Recorded cognition calls: chamber resolution + explicit attempt accounting.

Offline only: deterministic fakes, no network, no credentials. The fixture
model output below ("The one-second latency claim requires measurement.") is
NOT evidence of model intelligence; it is evidence that the runtime correctly
records execution provenance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeai.adapters import (
    NORMALIZER_VERSION,
    CallResult,
    CallSpec,
    FakeCognitionAdapter,
    sanitize_effective_params,
)
from codeai.artifacts import ArtifactCorruptionError, FileArtifactStore
from codeai.context import ContextCompiler
from codeai.domain import ActorRef, Task
from codeai.ledger import SQLiteLedger
from codeai.modelconfig import ModelConfig, ModelMapping
from codeai.providers import PRICING_VERSION, estimate_cost_usd
from codeai.runtime import Runtime

FIXTURE_OUTPUT = "The one-second latency claim requires measurement."


def make_actor(actor_id="reviewer-1", provider="opencode", model="mimo-v2.5"):
    return ActorRef(actor_id=actor_id, kind="model", provider=provider, model=model)


def make_task(task_id="task-001"):
    from codeai.domain import Authority, Budget

    return Task(
        task_id=task_id,
        directive_id="run-001",
        objective="Review this paragraph for unsupported factual claims.",
        success_criteria=("unsupported claims identified",),
        budget=Budget(),
        authority=Authority(),
    )


def make_spec(
    task_id="task-001",
    call_id="call-001",
    chamber="deep-review",
    logical_model="review",
    parameters=None,
):
    actor = make_actor()
    package = ContextCompiler().compile(
        task_id=task_id,
        actor=actor,
        prompt="The service processes every request within one second.",
        events=(),
    )
    return CallSpec(
        call_id=call_id,
        task_id=task_id,
        actor=actor,
        context=package,
        idempotency_key=f"key-{call_id}",
        adapter_id="opencode",
        instruction="Review this paragraph for unsupported factual claims.",
        chamber=chamber,
        logical_model=logical_model,
        parameters=parameters or {"temperature": 0.2, "max_tokens": 512},
    )


def make_runtime(tmp_path: Path, ledger_path=None):
    ledger = SQLiteLedger(str(ledger_path) if ledger_path else ":memory:")
    store = FileArtifactStore(tmp_path / "artifacts", ledger)
    runtime = Runtime(ledger, artifact_store=store)
    return runtime, ledger, store


def make_model_config():
    return ModelConfig(
        models={
            "review": ModelMapping(logical_name="review", adapter="opencode", model="mimo-v2.5"),
            "deep-review": ModelMapping(
                logical_name="deep-review", adapter="opencode", model="mimo-v2.5"
            ),
        }
    )


# 1. one task -> one call -> one successful attempt ---------------------------


def test_single_attempt_recorded_call(tmp_path):
    runtime, _ledger, store = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(
        responses=[FIXTURE_OUTPUT],
        provider="opencode",
        model="mimo-v2.5",
        model_version=None,  # unknown revision
        provider_request_ids=["prov-req-1"],
    )
    recorded = runtime.invoke_recorded_call(
        make_spec(), adapter=adapter, model_config=make_model_config()
    )
    assert recorded.task_id == "task-001"
    assert recorded.call_id == "call-001"
    assert recorded.chamber == "deep-review"
    assert recorded.manifest.requested_model == "review"
    assert recorded.manifest.provider == "opencode"
    assert recorded.manifest.resolved_model_id == "mimo-v2.5"
    assert recorded.manifest.pricing_version == PRICING_VERSION
    assert len(recorded.attempts) == 1
    attempt = recorded.attempts[0]
    assert attempt.attempt_index == 1
    assert attempt.status == "succeeded"
    assert attempt.provider_request_id == "prov-req-1"
    assert attempt.raw_artifact is not None
    assert attempt.normalizer_version == NORMALIZER_VERSION
    assert recorded.status == "succeeded"
    # raw observation preserved as JSON, readable through the artifact store
    raw = json.loads(store.read_text(attempt.raw_artifact.artifact_id))
    assert raw["output_text"] == FIXTURE_OUTPUT


# 2./3. one call -> two attempts, stable identities ---------------------------


def test_two_attempts_share_call_and_task_ids(tmp_path):
    runtime, _ledger, _store = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(
        behaviors=[
            {
                "output": "",
                "status": "failed",
                "error": "transient blip",
                "error_kind": "transient_failure",
                "usage_source": "unavailable",
                "input_tokens": None,
                "output_tokens": None,
                "provider_call_id": "prov-req-1",
            },
            {
                "output": FIXTURE_OUTPUT,
                "status": "succeeded",
                "input_tokens": 100,
                "output_tokens": 50,
                "usage_source": "measured",
                "provider_call_id": "prov-req-2",
            },
        ],
        provider="opencode",
        model="mimo-v2.5",
    )
    recorded = runtime.invoke_recorded_call(
        make_spec(), adapter=adapter, max_attempts=2, model_config=make_model_config()
    )
    assert len(recorded.attempts) == 2
    first, second = recorded.attempts
    assert first.task_id == second.task_id == "task-001"
    assert first.call_id == second.call_id == "call-001"
    assert first.task_id != first.call_id != first.attempt_id
    assert first.attempt_id != second.attempt_id
    assert (first.attempt_index, second.attempt_index) == (1, 2)
    assert first.status == "transient_failure"
    assert second.status == "succeeded"
    assert recorded.status == "succeeded"
    # per-attempt usage preserved, not collapsed
    assert first.usage.input_tokens is None
    assert (second.usage.input_tokens, second.usage.output_tokens) == (100, 50)
    # call totals derived from attempts (all-known here except first -> unknown)
    assert recorded.total_input_tokens is None  # unknown propagates, not partial sum


# 4. chamber / logical-model resolution ---------------------------------------


def test_chamber_resolution_uses_model_config(tmp_path):
    runtime, _ledger, _store = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(responses=[FIXTURE_OUTPUT])
    spec = make_spec(chamber="deep-review", logical_model="review")
    recorded = runtime.invoke_recorded_call(spec, adapter=adapter, model_config=make_model_config())
    assert recorded.manifest.requested_model == "review"
    assert recorded.manifest.resolved_model_id == "mimo-v2.5"
    config = make_model_config()
    assert config.resolve_chamber("deep-review").model == "mimo-v2.5"
    with pytest.raises(KeyError):
        config.resolve_chamber("no-such-chamber")


# 5. requested vs effective ----------------------------------------------------


def test_requested_vs_effective_parameters(tmp_path):
    runtime, _ledger, _store = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(
        responses=[FIXTURE_OUTPUT],
        unsupported_params=("reasoning_effort",),
    )
    spec = make_spec(parameters={"temperature": 0.2, "reasoning_effort": "low", "max_tokens": 512})
    recorded = runtime.invoke_recorded_call(spec, adapter=adapter)
    assert recorded.manifest.requested_parameters["reasoning_effort"] == "low"
    assert "reasoning_effort" not in recorded.manifest.effective_parameters
    assert recorded.manifest.effective_parameters["temperature"] == 0.2
    assert recorded.attempts[0].effective_parameters == recorded.manifest.effective_parameters


# 6. unknown model revision ----------------------------------------------------


def test_unknown_model_revision_stays_unknown(tmp_path):
    runtime, _ledger, _store = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(
        responses=[FIXTURE_OUTPUT],
        provider="opencode",
        model="mimo-v2.5",
        model_version=None,
    )
    recorded = runtime.invoke_recorded_call(make_spec(), adapter=adapter)
    attempt = recorded.attempts[0]
    assert attempt.resolved_model_id == "mimo-v2.5"
    assert attempt.provider_revision is None
    assert attempt.revision_source is None


def test_reported_revision_recorded_verbatim(tmp_path):
    runtime, _ledger, _store = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(responses=[FIXTURE_OUTPUT], model_version="mimo-v2.5-0314")
    recorded = runtime.invoke_recorded_call(make_spec(), adapter=adapter)
    attempt = recorded.attempts[0]
    assert attempt.provider_revision == "mimo-v2.5-0314"
    assert attempt.revision_source == "reported"


# 7. unknown pricing ------------------------------------------------------------


def test_unknown_pricing_stays_unknown_not_zero(tmp_path):
    assert estimate_cost_usd("no-such-model-xyz", 100, 100) is None
    runtime, _ledger, _store = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(
        responses=[FIXTURE_OUTPUT],
        provider="opencode",
        model="no-such-model-xyz",
        input_tokens=100,
        output_tokens=50,
        cost_usd=None,
    )
    recorded = runtime.invoke_recorded_call(make_spec(), adapter=adapter)
    attempt = recorded.attempts[0]
    assert attempt.cost_usd is None
    assert attempt.cost_source == "unknown"
    assert attempt.pricing_version == PRICING_VERSION


def test_known_pricing_records_version_and_estimate(tmp_path):
    runtime, _ledger, _store = make_runtime(tmp_path)
    from codeai.providers import OpenAIAdapter

    seen = {}

    def fake_post(url, payload, headers, timeout):
        seen["headers"] = headers
        return {
            "id": "chatcmpl-9",
            "model": "gpt-4o-mini",
            "choices": [{"message": {"content": FIXTURE_OUTPUT}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

    adapter = OpenAIAdapter(model="gpt-4o-mini", api_key="k", http_post=fake_post)
    actor = ActorRef(actor_id="a", kind="model", provider="openai", model="gpt-4o-mini")
    package = ContextCompiler().compile(task_id="task-001", actor=actor, prompt="p", events=())
    spec = CallSpec(
        call_id="call-001", task_id="task-001", actor=actor, context=package, idempotency_key="k1"
    )
    recorded = runtime.invoke_recorded_call(spec, adapter=adapter)
    attempt = recorded.attempts[0]
    assert attempt.cost_usd is not None and attempt.cost_usd > 0
    assert attempt.cost_source == "estimated"
    assert attempt.pricing_version == PRICING_VERSION
    assert "Authorization" not in json.dumps(dict(attempt.effective_parameters))


# 8. unavailable usage distinct from zero ---------------------------------------


def test_unavailable_usage_is_none_not_zero(tmp_path):
    runtime, _ledger, _store = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(
        responses=[FIXTURE_OUTPUT],
        input_tokens=None,
        output_tokens=None,
        usage_source="unavailable",
    )
    recorded = runtime.invoke_recorded_call(make_spec(), adapter=adapter)
    usage = recorded.attempts[0].usage
    assert usage.source.value == "unavailable"
    assert usage.input_tokens is None and usage.output_tokens is None


def test_measured_zero_stays_zero(tmp_path):
    runtime, _ledger, _store = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(
        responses=[FIXTURE_OUTPUT],
        input_tokens=0,
        output_tokens=0,
        usage_source="measured",
    )
    recorded = runtime.invoke_recorded_call(make_spec(), adapter=adapter)
    usage = recorded.attempts[0].usage
    assert (usage.input_tokens, usage.output_tokens) == (0, 0)
    assert usage.source.value == "measured"


# 9./10. raw artifact persisted, readable, linked --------------------------------


def test_raw_artifact_linked_and_stable(tmp_path):
    runtime, _ledger, store = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(responses=[FIXTURE_OUTPUT])
    recorded = runtime.invoke_recorded_call(make_spec(), adapter=adapter)
    attempt = recorded.attempts[0]
    assert attempt.raw_artifact is not None
    assert attempt.raw_observation_kind == "adapter-sanitized-observation-v1"
    body = store.read_text(attempt.raw_artifact.artifact_id)
    assert FIXTURE_OUTPUT in body
    # normalized interpretation links back to the raw observation
    assert attempt.normalizer_version == NORMALIZER_VERSION


def test_normalizer_change_does_not_mutate_raw(tmp_path):
    runtime, _ledger, store = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(responses=[FIXTURE_OUTPUT])
    recorded = runtime.invoke_recorded_call(make_spec(), adapter=adapter)
    attempt = recorded.attempts[0]
    before = store.read_bytes(attempt.raw_artifact.artifact_id)
    # simulate a normalizer fix producing a new interpretation of the same raw
    reparsed = json.loads(before.decode("utf-8"))
    assert reparsed["output_text"] == FIXTURE_OUTPUT
    after = store.read_bytes(attempt.raw_artifact.artifact_id)
    assert before == after


# 13. clean reopen preserving identities ----------------------------------------


def test_clean_restart_preserves_call_and_attempts(tmp_path):
    ledger_path = tmp_path / "ledger.sqlite"
    runtime, _ledger, _store = make_runtime(tmp_path, ledger_path=ledger_path)
    adapter = FakeCognitionAdapter(
        behaviors=[
            {
                "output": "",
                "status": "failed",
                "error": "blip",
                "error_kind": "transient_failure",
                "usage_source": "unavailable",
                "input_tokens": None,
                "output_tokens": None,
            },
            {
                "output": FIXTURE_OUTPUT,
                "status": "succeeded",
                "input_tokens": 100,
                "output_tokens": 50,
                "usage_source": "measured",
            },
        ],
        provider="opencode",
        model="mimo-v2.5",
    )
    recorded = runtime.invoke_recorded_call(
        make_spec(), adapter=adapter, max_attempts=2, model_config=make_model_config()
    )
    artifact_id = recorded.attempts[1].raw_artifact.artifact_id
    call_id = recorded.call_id
    attempt_ids = [a.attempt_id for a in recorded.attempts]

    # clean restart: new objects over the same sqlite file + artifact dir
    runtime2, _ledger2, store2 = make_runtime(tmp_path, ledger_path=ledger_path)
    reopened = runtime2.get_recorded_call(call_id)
    assert reopened is not None
    assert reopened.call_id == call_id
    assert reopened.task_id == "task-001"
    assert [a.attempt_id for a in reopened.attempts] == attempt_ids
    assert [a.attempt_index for a in reopened.attempts] == [1, 2]
    assert reopened.attempts[1].raw_artifact is not None
    assert reopened.attempts[1].raw_artifact.artifact_id == artifact_id
    body = store2.read_text(artifact_id)
    assert FIXTURE_OUTPUT in body


# artifact integrity --------------------------------------------------------------


def test_artifact_corruption_detected(tmp_path):
    _runtime, ledger, store = make_runtime(tmp_path)
    ref = store.store_bytes(
        b'{"output_text": "x"}',
        media_type="application/json",
        artifact_type="raw_provider_observation",
    )
    assert store.read_bytes(ref.artifact_id) == b'{"output_text": "x"}'
    from pathlib import Path as _Path

    path = _Path(ref.uri)
    path.write_bytes(b"tampered")
    with pytest.raises(ArtifactCorruptionError):
        store.read_bytes(ref.artifact_id)
    assert ledger.read_artifact(ref.artifact_id) is not None


# task completion stays separate ----------------------------------------------------


def test_cognition_success_is_not_task_completion(tmp_path):
    runtime, _ledger, _store = make_runtime(tmp_path)
    runtime.create_task(make_task())
    adapter = FakeCognitionAdapter(responses=[FIXTURE_OUTPUT])
    recorded = runtime.invoke_recorded_call(make_spec(), adapter=adapter)
    assert recorded.status == "succeeded"
    kinds = [e.kind for e in runtime.ledger.read_all()]
    assert "task.completed" not in kinds
    assert "call.manifest" in kinds and "attempt.completed" in kinds


# sanitizer ---------------------------------------------------------------------------


def test_sanitizer_drops_credential_like_keys():
    cleaned = sanitize_effective_params(
        {
            "temperature": 0.2,
            "api_key": "sekret",
            "Authorization": "bearer x",
            "model": "m",
            "max_tokens": 512,
        }
    )
    assert cleaned == {"temperature": 0.2, "model": "m", "max_tokens": 512}


# query surface ---------------------------------------------------------------------------


def test_query_surface_answers_provenance_questions(tmp_path):
    runtime, _ledger, _store = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(responses=[FIXTURE_OUTPUT], model_version=None)
    runtime.invoke_recorded_call(make_spec(), adapter=adapter, model_config=make_model_config())
    fetched = runtime.get_recorded_call("call-001")
    assert fetched is not None
    assert fetched.manifest.chamber == "deep-review"
    assert fetched.manifest.resolved_model_id == "mimo-v2.5"
    assert fetched.manifest.pricing_version == PRICING_VERSION
    assert len(runtime.list_call_attempts("call-001")) == 1
    assert runtime.get_recorded_call("missing") is None


def test_legacy_invoke_call_still_works(tmp_path):
    """Compatibility: the pre-existing single-result path is unchanged."""
    runtime, _ledger, _store = make_runtime(tmp_path)
    actor = make_actor()
    package = ContextCompiler().compile(task_id="t1", actor=actor, prompt="p", events=())
    spec = CallSpec(
        call_id="c1", task_id="t1", actor=actor, context=package, idempotency_key="k-c1"
    )
    result = runtime.invoke_call(spec, adapter=FakeCognitionAdapter(responses=["hi"]))
    assert isinstance(result, CallResult)
    assert result.status == "succeeded" and result.raw_output == "hi"
