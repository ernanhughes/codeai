"""Stage 11.5d-A: durable idempotent replay for recorded calls.

Offline only. Every test pins the same invariant: same idempotency key +
completed recorded call -> zero new provider effects, explicit call.replayed
provenance, original identities preserved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeai.adapters import CallSpec, FakeCognitionAdapter
from codeai.artifacts import FileArtifactStore
from codeai.context import ContextCompiler
from codeai.domain import ActorRef
from codeai.experiments import experiment_usage
from codeai.interpretation import ATTEMPT_POLICY_V1, INTERPRETER_V1
from codeai.ledger import SQLiteLedger
from codeai.runtime import IdempotencyConflictError, Runtime

TEXT = "replay fixture output"


def make_spec(
    call_id: str = "call-r1",
    task_id: str = "task-r1",
    key: str = "key-r1",
    prompt: str = "do work",
    experiment_id=None,
) -> CallSpec:
    actor = ActorRef(actor_id="m", kind="model", provider="fake-provider", model="fake-model")
    package = ContextCompiler().compile(task_id=task_id, actor=actor, prompt=prompt, events=())
    return CallSpec(
        call_id=call_id,
        task_id=task_id,
        actor=actor,
        context=package,
        idempotency_key=key,
        instruction=prompt,
        experiment_id=experiment_id,
    )


def make_runtime(tmp_path: Path, ledger_path=None) -> Runtime:
    ledger = SQLiteLedger(str(ledger_path) if ledger_path else ":memory:")
    return Runtime(ledger, artifact_store=FileArtifactStore(tmp_path / "artifacts", ledger))


def kinds(runtime: Runtime) -> list[str]:
    return [e.kind for e in runtime.ledger.read_all()]


# Fresh success then same-process replay --------------------------------------------


def test_replay_returns_original_with_zero_new_effects(tmp_path):
    runtime = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(responses=[TEXT])
    first = runtime.invoke_recorded_call(make_spec(), adapter=adapter)
    assert not first.replayed
    assert len(adapter.calls) == 1

    second = runtime.invoke_recorded_call(make_spec(), adapter=adapter)
    assert second.replayed
    assert second.call_id == first.call_id  # Design A: original identity preserved
    assert [a.attempt_id for a in second.attempts] == [a.attempt_id for a in first.attempts]
    assert len(adapter.calls) == 1  # zero new provider effects

    replayed = runtime.ledger.events_by_kind(("call.replayed",))
    assert len(replayed) == 1
    payload = replayed[0].payload
    assert payload["requested_call_id"] == "call-r1"
    assert payload["original_call_id"] == "call-r1"
    assert payload["idempotency_key"] == "key-r1"
    assert payload["original_attempt_ids"] == [a.attempt_id for a in first.attempts]
    assert payload["original_call_status"] == first.status
    assert payload["replayed_at"] and payload["replay_reason"]

    # No new attempt/observation/interpretation/decision/completion evidence.
    assert kinds(runtime).count("attempt.started") == 1
    assert kinds(runtime).count("attempt.completed") == 1
    assert kinds(runtime).count("call.completed") == 1
    assert (
        "attempt.interpreted" not in kinds(runtime)
        or kinds(runtime).count("attempt.interpreted") == 1
    )


def test_replay_emits_only_requested_and_replayed(tmp_path):
    runtime = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(responses=[TEXT])
    runtime.invoke_recorded_call(make_spec(), adapter=adapter)
    before = len(runtime.ledger.read_all())
    runtime.invoke_recorded_call(make_spec(), adapter=adapter)
    new_kinds = kinds(runtime)[before:]
    assert new_kinds == ["call.requested", "call.replayed"]


def test_replay_with_new_call_id_resolves_to_original(tmp_path):
    runtime = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(responses=[TEXT])
    runtime.invoke_recorded_call(make_spec(call_id="call-A"), adapter=adapter)
    second = runtime.invoke_recorded_call(make_spec(call_id="call-B"), adapter=adapter)
    assert second.replayed and second.call_id == "call-A"
    assert len(adapter.calls) == 1
    # requested id resolves through call.replayed to the original
    resolved = runtime.get_recorded_call("call-B")
    assert resolved is not None and resolved.call_id == "call-A"


# Restart replay -----------------------------------------------------------------------


def test_restart_replay_has_zero_effects(tmp_path):
    ledger_path = tmp_path / "ledger.sqlite"
    runtime = make_runtime(tmp_path, ledger_path=ledger_path)
    adapter = FakeCognitionAdapter(responses=[TEXT], input_tokens=10, output_tokens=5)
    first = runtime.invoke_recorded_call(make_spec(), adapter=adapter)
    artifact_id = first.attempts[0].raw_artifact.artifact_id

    runtime2 = make_runtime(tmp_path, ledger_path=ledger_path)
    adapter2 = FakeCognitionAdapter(responses=["different"])
    second = runtime2.invoke_recorded_call(make_spec(), adapter=adapter2)
    assert second.replayed and second.call_id == first.call_id
    assert adapter2.calls == []
    assert second.attempts[0].raw_artifact.artifact_id == artifact_id
    assert runtime2.ledger.events_by_kind(("call.replayed",))


# Multi-attempt + failed + unknown-cost replay ----------------------------------------------


def test_multi_attempt_replay_references_original_attempts(tmp_path):
    runtime = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(
        behaviors=[
            {
                "output": "",
                "status": "failed",
                "error": "blip",
                "error_kind": "transient_failure",
                "input_tokens": 5,
                "output_tokens": 5,
            },
            {"output": TEXT, "status": "succeeded", "input_tokens": 10, "output_tokens": 5},
        ]
    )
    first = runtime.invoke_recorded_call(make_spec(), adapter=adapter, max_attempts=2)
    assert len(first.attempts) == 2
    second = runtime.invoke_recorded_call(make_spec(), adapter=adapter, max_attempts=2)
    assert second.replayed and len(second.attempts) == 2  # no attempt 3
    assert len(adapter.calls) == 2
    payload = runtime.ledger.events_by_kind(("call.replayed",))[0].payload
    assert payload["original_attempt_ids"] == [a.attempt_id for a in first.attempts]


def test_failed_terminal_call_replays_without_new_effect(tmp_path):
    runtime = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(fail_call_ids={"call-r1"})
    first = runtime.invoke_recorded_call(make_spec(), adapter=adapter)
    assert first.status == "failed"
    second = runtime.invoke_recorded_call(make_spec(), adapter=adapter)
    assert second.replayed and second.status == "failed"
    assert len(adapter.calls) == 1


def test_replay_does_not_duplicate_accounting(tmp_path):
    runtime = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(
        responses=[TEXT], input_tokens=100, output_tokens=20, cost_usd=None
    )
    runtime.invoke_recorded_call(make_spec(experiment_id="exp-r"), adapter=adapter)
    before = experiment_usage(runtime, "exp-r")
    runtime.invoke_recorded_call(make_spec(experiment_id="exp-r"), adapter=adapter)
    after = experiment_usage(runtime, "exp-r")
    assert (after.known_input_tokens, after.known_output_tokens) == (
        before.known_input_tokens,
        before.known_output_tokens,
    )
    assert after.calls == before.calls == 1
    assert after.known_cost_usd == before.known_cost_usd


def test_historical_interpretation_preserved_on_replay(tmp_path):
    runtime = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(fail_call_ids={"call-r1"})
    first = runtime.invoke_recorded_call(
        make_spec(),
        adapter=adapter,
        interpreter_version=INTERPRETER_V1,
        policy_version=ATTEMPT_POLICY_V1,
    )
    v1_ids = [
        a.interpretation_id
        for a in runtime.interpretations_for_attempt(first.attempts[0].attempt_id)
    ]
    assert len(v1_ids) == 1
    second = runtime.invoke_recorded_call(make_spec(), adapter=adapter)
    assert second.replayed and second.status == "failed"
    # No reinterpretation appended; the v1 record still explains the outcome.
    assert [
        a.interpretation_id
        for a in runtime.interpretations_for_attempt(first.attempts[0].attempt_id)
    ] == v1_ids
    # Counterfactual v2 remains separately queryable, not history.
    v2 = runtime.interpret_attempt_as(
        first.attempts[0].attempt_id, version="attempt-interpretation-v2"
    )
    assert v2 is not None and v2.interpretation_id not in v1_ids


# Mismatch + incomplete + legacy-key contracts -----------------------------------------------


def test_same_key_different_request_conflicts_before_effect(tmp_path):
    runtime = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(responses=[TEXT])
    runtime.invoke_recorded_call(make_spec(), adapter=adapter)
    with pytest.raises(IdempotencyConflictError) as caught:
        runtime.invoke_recorded_call(make_spec(prompt="different work"), adapter=adapter)
    assert "prompt_hash" in str(caught.value)
    assert "SHOULD" not in str(caught.value)
    assert len(adapter.calls) == 1
    assert runtime.ledger.events_by_kind(("call.replayed",)) == ()


def test_incomplete_call_is_not_replayed_as_success(tmp_path):
    from codeai.ledger import Event
    from codeai.runtime import _prompt_hash_for_spec

    runtime = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(responses=[TEXT])
    spec = make_spec()
    # Simulate a crash: requested + manifest + started, but no terminal completion.
    runtime.ledger.append(
        Event.create(
            stream_id=spec.call_id,
            kind="call.requested",
            actor_id="m",
            payload=Runtime._call_spec_payload(spec),
            correlation_id=spec.task_id,
        )
    )
    runtime.ledger.append(
        Event.create(
            stream_id=spec.call_id,
            kind="call.manifest",
            actor_id="m",
            payload={
                "call_id": spec.call_id,
                "task_id": spec.task_id,
                "chamber": None,
                "requested_model": "fake-model",
                "provider": "fake-provider",
                "resolved_model_id": "fake-model",
                "provider_revision": None,
                "revision_source": None,
                "pricing_version": "2026-09-01",
                "context_package_id": spec.context.package_id,
                "prompt_hash": _prompt_hash_for_spec(spec),
                "requested_parameters": {},
                "effective_parameters": {},
                "created_at": "2026-09-13T00:00:00+00:00",
            },
            correlation_id=spec.task_id,
        )
    )
    runtime.ledger.append(
        Event.create(
            stream_id="att-crash",
            kind="attempt.started",
            actor_id="m",
            payload={
                "attempt_id": "att-crash",
                "call_id": spec.call_id,
                "task_id": spec.task_id,
                "attempt_index": 1,
            },
            correlation_id=spec.task_id,
        )
    )
    # No call.completed exists: rerunning executes fresh (legacy parity),
    # never a silent successful replay. Crash-after-effect ambiguity documented.
    recorded = runtime.invoke_recorded_call(spec, adapter=adapter)
    assert not recorded.replayed
    assert len(adapter.calls) == 1
    assert recorded.status == "succeeded"


def test_legacy_key_does_not_replay_recorded_path(tmp_path):
    runtime = make_runtime(tmp_path)
    adapter = FakeCognitionAdapter(responses=["legacy"])
    legacy_spec = make_spec(call_id="legacy-1", key="shared-key")
    runtime.invoke_call(legacy_spec, adapter=adapter)
    recorded = runtime.invoke_recorded_call(
        make_spec(call_id="rec-1", key="shared-key"), adapter=adapter
    )
    assert not recorded.replayed  # different execution contract: fresh effect
    assert len(adapter.calls) == 2
