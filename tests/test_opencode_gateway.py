"""OpenCode Zen gateway: offline transport + recorded-call integration tests.

No network, no credentials. http_post is patched with fixtures; credential
handling is verified through env manipulation only (values never persisted).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeai.artifacts import FileArtifactStore
from codeai.context import ContextCompiler
from codeai.domain import ActorRef, Authority, Budget, CallSpec, Task
from codeai.ledger import SQLiteLedger
from codeai.modelconfig import ModelConfig, ModelMapping, load_model_config, missing_credentials
from codeai.providers import (
    OPENCODE_ZEN_API_KEY_ENV,
    OPENCODE_ZEN_BASE_URL,
    OpenCodeCognitionAdapter,
    ProviderHttpError,
)
from codeai.runtime import Runtime

FIXTURE_TEXT = "The one-second latency claim requires measurement."


def responses_fixture(text=FIXTURE_TEXT, **usage):
    payload = {
        "id": "resp-123",
        "model": "mimo-v2.5",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }
    if usage:
        payload["usage"] = usage
    return payload


def make_spec(call_id="call-oc-1", task_id="task-oc-1", parameters=None):
    actor = ActorRef(actor_id="reviewer-1", kind="model", provider="opencode", model="mimo-v2.5")
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
        chamber="deep-review",
        logical_model="review",
        instruction="Review this paragraph for unsupported factual claims.",
        parameters=parameters or {"reasoning_effort": "low", "max_tokens": 512},
    )


def make_runtime(tmp_path: Path):
    ledger = SQLiteLedger()
    return Runtime(ledger, artifact_store=FileArtifactStore(tmp_path / "artifacts", ledger))


# --- request shape ------------------------------------------------------------


def test_targets_configured_zen_base_url_with_responses_path():
    seen = {}

    def fake_post(url, payload, headers, timeout):
        seen.update(url=url, payload=payload, headers=headers)
        return responses_fixture(input_tokens=10, output_tokens=5)

    adapter = OpenCodeCognitionAdapter(
        model="mimo-v2.5",
        api_key="zen-test-key",
        base_url="https://zen.example/v9",
        http_post=fake_post,
    )
    result = adapter.invoke(make_spec())
    assert seen["url"] == "https://zen.example/v9/v1/responses"
    assert seen["headers"]["User-Agent"].startswith("codeai/")
    assert "Bearer" in seen["headers"]["Authorization"]
    assert seen["payload"]["model"] == "mimo-v2.5"
    assert seen["payload"]["reasoning"] == {"effort": "low"}
    assert seen["payload"]["max_output_tokens"] == 512
    assert "The service processes every request" in seen["payload"]["input"]
    assert result.status == "succeeded"


def test_default_base_url_is_zen_gateway():
    assert OPENCODE_ZEN_BASE_URL == "https://opencode.ai/zen/go"
    adapter = OpenCodeCognitionAdapter(model="mimo-v2.5", api_key="k")
    assert adapter.base_url == OPENCODE_ZEN_BASE_URL


# --- authentication ------------------------------------------------------------


def test_uses_zen_credential_not_openai_key(monkeypatch):
    seen = {}

    def fake_post(url, payload, headers, timeout):
        seen["headers"] = headers
        return responses_fixture()

    monkeypatch.setenv(OPENCODE_ZEN_API_KEY_ENV, "zen-secret-value")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-decoy-value")
    adapter = OpenCodeCognitionAdapter(model="mimo-v2.5", http_post=fake_post)
    adapter.invoke(make_spec())
    assert seen["headers"]["Authorization"] == "Bearer zen-secret-value"
    assert "openai-decoy-value" not in json.dumps(seen["headers"])


def test_missing_zen_credential_is_failed_result(monkeypatch):
    monkeypatch.delenv(OPENCODE_ZEN_API_KEY_ENV, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-decoy-value")
    adapter = OpenCodeCognitionAdapter(model="mimo-v2.5", api_key="")
    result = adapter.invoke(make_spec())
    assert result.status == "failed"
    assert result.error_kind == "missing_credentials"
    assert OPENCODE_ZEN_API_KEY_ENV in (result.error or "")
    assert "openai-decoy-value" not in (result.error or "")


def test_unsupported_protocol_rejected_explicitly():
    with pytest.raises(ValueError, match="unsupported OpenCode protocol"):
        OpenCodeCognitionAdapter(model="mimo-v2.5", api_key="k", protocol="smoke-signals")


# --- extraction -----------------------------------------------------------------


def test_fixture_output_becomes_canonical_text():
    adapter = OpenCodeCognitionAdapter(
        model="mimo-v2.5",
        api_key="k",
        http_post=lambda *a: responses_fixture(input_tokens=10, output_tokens=5),
    )
    result = adapter.invoke(make_spec())
    assert result.raw_output == FIXTURE_TEXT
    assert result.provider == "opencode"  # gateway, never "openai"
    assert result.protocol == "responses"
    assert result.provider_call_id == "resp-123"
    assert result.raw_observation_kind == "decoded_json"
    assert result.raw_payload["id"] == "resp-123"


def test_reasoning_trace_skipped_malformed_raises():
    adapter = OpenCodeCognitionAdapter(
        model="mimo-v2.5",
        api_key="k",
        http_post=lambda *a: {
            "id": "resp-1",
            "output": [
                {"type": "reasoning", "content": [{"type": "thinking", "text": "hmm"}]},
                {"type": "message", "content": [{"type": "output_text", "text": "answer"}]},
            ],
        },
    )
    assert adapter.invoke(make_spec()).raw_output == "answer"

    empty = OpenCodeCognitionAdapter(
        model="mimo-v2.5", api_key="k", http_post=lambda *a: {"id": "resp-2", "output": []}
    )
    result = empty.invoke(make_spec())
    assert result.status == "failed"
    assert result.error_kind == "provider_error"  # malformed: no output text


# --- usage -----------------------------------------------------------------------


def test_known_usage_is_measured():
    adapter = OpenCodeCognitionAdapter(
        model="mimo-v2.5",
        api_key="k",
        http_post=lambda *a: responses_fixture(input_tokens=100, output_tokens=50),
    )
    result = adapter.invoke(make_spec())
    assert (result.input_tokens, result.output_tokens) == (100, 50)
    assert result.usage_source == "measured"


def test_missing_usage_is_unavailable_not_zero():
    adapter = OpenCodeCognitionAdapter(
        model="mimo-v2.5", api_key="k", http_post=lambda *a: responses_fixture()
    )
    result = adapter.invoke(make_spec())
    assert result.input_tokens is None and result.output_tokens is None
    assert result.usage_source == "unavailable"


# --- errors stay attempts, not text -----------------------------------------------


def test_http_401_is_authentication_failure():
    def fake_post(url, payload, headers, timeout):
        raise ProviderHttpError(401, "provider HTTP 401: unauthorized")

    adapter = OpenCodeCognitionAdapter(model="mimo-v2.5", api_key="k", http_post=fake_post)
    result = adapter.invoke(make_spec())
    assert result.status == "failed"
    assert result.error_kind == "authentication_error"
    assert result.raw_output == ""


def test_http_429_is_rate_limited():
    def fake_post(url, payload, headers, timeout):
        raise ProviderHttpError(429, "provider HTTP 429: slow down")

    adapter = OpenCodeCognitionAdapter(model="mimo-v2.5", api_key="k", http_post=fake_post)
    result = adapter.invoke(make_spec())
    assert result.status == "failed"
    assert result.error_kind == "rate_limited"


def test_transport_failure_is_provider_error_not_text():
    def fake_post(url, payload, headers, timeout):
        raise ProviderHttpError(None, "provider request failed: boom")

    adapter = OpenCodeCognitionAdapter(model="mimo-v2.5", api_key="k", http_post=fake_post)
    result = adapter.invoke(make_spec())
    assert result.status == "failed"
    assert "unavailable" not in result.raw_output
    assert result.raw_output == ""


# --- effective request ------------------------------------------------------------


def test_effective_request_answers_protocol_questions():
    adapter = OpenCodeCognitionAdapter(model="mimo-v2.5", api_key="k")
    effective = adapter.effective_request(
        make_spec(parameters={"reasoning_effort": "low", "max_tokens": 512})
    )
    assert effective["gateway"] == "opencode"
    assert effective["protocol"] == "responses"
    assert effective["model"] == "mimo-v2.5"
    assert effective["reasoning_effort"] == "low"
    assert effective["max_output_tokens"] == 512
    assert "temperature" not in effective  # not requested -> not sent
    assert "Authorization" not in json.dumps(effective)


# --- model config ------------------------------------------------------------------


def test_config_round_trip_with_protocol(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".codeai").mkdir()
    (tmp_path / ".codeai" / "config.toml").write_text(
        '[models.deep-review]\nadapter = "opencode"\nmodel = "mimo-v2.5"\nprotocol = "responses"\n'
    )
    config = load_model_config()
    mapping = config.resolve_chamber("deep-review")
    assert mapping.adapter == "opencode"
    assert mapping.model == "mimo-v2.5"
    assert mapping.protocol == "responses"
    adapter = config.build_adapter("deep-review")
    assert isinstance(adapter, OpenCodeCognitionAdapter)
    assert adapter.protocol == "responses"


def test_missing_zen_credential_reported(monkeypatch):
    monkeypatch.delenv(OPENCODE_ZEN_API_KEY_ENV, raising=False)
    mapping = ModelMapping(logical_name="deep-review", adapter="opencode", model="mimo-v2.5")
    assert missing_credentials(mapping) == f"set {OPENCODE_ZEN_API_KEY_ENV}"
    monkeypatch.setenv(OPENCODE_ZEN_API_KEY_ENV, "k")
    assert missing_credentials(mapping) is None


def test_model_config_stub_builds_opencode_adapter():
    config = ModelConfig(
        models={
            "review": ModelMapping(
                logical_name="review", adapter="opencode", model="mimo-v2.5", protocol="responses"
            )
        }
    )
    adapter = config.build_adapter("review")
    assert isinstance(adapter, OpenCodeCognitionAdapter)
    assert adapter.base_url == OPENCODE_ZEN_BASE_URL


# --- recorded-call integration --------------------------------------------------------


def test_recorded_call_through_zen_gateway(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.create_task(
        Task(
            task_id="task-oc-1",
            directive_id="run-oc-1",
            objective="Review this paragraph for unsupported factual claims.",
            success_criteria=("unsupported claims identified",),
            budget=Budget(),
            authority=Authority(),
        )
    )
    config = ModelConfig(
        models={
            "review": ModelMapping(
                logical_name="review", adapter="opencode", model="mimo-v2.5", protocol="responses"
            ),
            "deep-review": ModelMapping(
                logical_name="deep-review",
                adapter="opencode",
                model="mimo-v2.5",
                protocol="responses",
            ),
        }
    )
    mapping = config.resolve_chamber("deep-review")
    adapter = OpenCodeCognitionAdapter(
        model=mapping.model,
        api_key="zen-test-key",
        http_post=lambda *a: responses_fixture(input_tokens=100, output_tokens=50),
    )
    recorded = runtime.invoke_recorded_call(make_spec(), adapter=adapter, model_config=config)
    assert recorded.chamber == "deep-review"
    assert recorded.manifest.requested_model == "review"
    assert recorded.manifest.provider == "opencode"
    assert recorded.manifest.resolved_model_id == "mimo-v2.5"
    attempt = recorded.attempts[0]
    assert attempt.provider == "opencode"  # gateway, not "openai"
    assert attempt.protocol == "responses"
    assert attempt.raw_observation_kind == "decoded_json"
    assert attempt.raw_artifact is not None
    raw = json.loads(runtime.artifact_store.read_text(attempt.raw_artifact.artifact_id))
    assert raw["provider_response"]["id"] == "resp-123"
    assert raw["output_text"] == FIXTURE_TEXT
    assert "zen-test-key" not in json.dumps(raw)
    assert (attempt.usage.input_tokens, attempt.usage.output_tokens) == (100, 50)
    assert attempt.cost_usd is None and attempt.cost_source == "unknown"  # mimo absent from pricing
    assert recorded.status == "succeeded"
    # reopen: gateway/protocol survive
    fetched = runtime.get_recorded_call(recorded.call_id)
    assert fetched is not None
    assert fetched.attempts[0].provider == "opencode"
    assert fetched.attempts[0].protocol == "responses"


def test_recorded_gateway_failure_is_attempt_not_task_completion(tmp_path):
    runtime = make_runtime(tmp_path)

    def fake_post(url, payload, headers, timeout):
        raise ProviderHttpError(401, "provider HTTP 401: unauthorized")

    adapter = OpenCodeCognitionAdapter(model="mimo-v2.5", api_key="k", http_post=fake_post)
    recorded = runtime.invoke_recorded_call(make_spec(), adapter=adapter)
    assert recorded.status == "failed"
    assert recorded.attempts[0].error_kind == "authentication_error"
    assert recorded.attempts[0].provider == "opencode"
    kinds = [e.kind for e in runtime.ledger.read_all()]
    assert "task.completed" not in kinds
