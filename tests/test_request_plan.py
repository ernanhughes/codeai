"""Stage 11.5c: the sent request and the recorded effective request are one fact.

Offline only. A counting transport double proves prepare-once/send-exact;
decoy credentials prove transport-use without persistence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeai.adapters import (
    InvalidControlError,
    UnknownControlError,
)
from codeai.artifacts import FileArtifactStore
from codeai.context import ContextCompiler
from codeai.domain import ActorRef, CallSpec
from codeai.ledger import SQLiteLedger
from codeai.providers import (
    OPENCODE_REQUEST_PLAN,
    HttpResponse,
    OpenCodeCognitionAdapter,
)
from codeai.runtime import Runtime

TEXT = "The one-second latency claim requires measurement."


def chat_payload(text: str = TEXT) -> dict:
    return {
        "id": "chatcmpl-1",
        "model": "mimo-v2.5",
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def responses_payload(text: str = TEXT) -> dict:
    return {
        "id": "resp-1",
        "model": "mimo-v2.5",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def make_spec(call_id: str = "call-p1", parameters=None) -> CallSpec:
    actor = ActorRef(actor_id="r1", kind="model", provider="opencode", model="mimo-v2.5")
    package = ContextCompiler().compile(task_id="task-p1", actor=actor, prompt="p", events=())
    return CallSpec(
        call_id=call_id,
        task_id="task-p1",
        actor=actor,
        context=package,
        idempotency_key=f"key-{call_id}",
        chamber="deep-review",
        logical_model="review",
        parameters=parameters or {},
    )


def make_runtime(tmp_path: Path, ledger_path=None) -> Runtime:
    ledger = SQLiteLedger(str(ledger_path) if ledger_path else ":memory:")
    return Runtime(ledger, artifact_store=FileArtifactStore(tmp_path / "artifacts", ledger))


class CountingTransport:
    """Intercepts the HTTP layer: counts sends, captures exact bodies/headers."""

    def __init__(self, reply):
        self.reply = reply
        self.bodies: list[dict] = []
        self.headers: list[dict] = []
        self.urls: list[str] = []

    def __call__(self, url, payload, headers, timeout):
        self.urls.append(url)
        self.bodies.append(json.loads(json.dumps(payload)))
        self.headers.append(dict(headers))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class CountingAdapter(OpenCodeCognitionAdapter):
    prepares = 0

    def prepare(self, spec):
        type(self).prepares += 1
        return super().prepare(spec)


# Responses matrix ------------------------------------------------------------------


def test_responses_requested_effective_body_agree():
    adapter = OpenCodeCognitionAdapter(model="mimo-v2.5", api_key="k", protocol="responses")
    spec = make_spec(parameters={"temperature": 0.2, "max_tokens": 256, "reasoning_effort": "low"})
    prepared = adapter.prepare(spec)
    assert prepared.plan_version == OPENCODE_REQUEST_PLAN == "opencode-request-plan-v1"
    # requested: logical names and raw caller values
    assert prepared.requested_controls == {
        "temperature": 0.2,
        "max_tokens": 256,
        "reasoning_effort": "low",
    }
    # wire mapping: alias resolved, nested structure preserved
    assert prepared.body["max_output_tokens"] == 256
    assert "max_tokens" not in prepared.body
    assert prepared.body["reasoning"] == {"effort": "low"}
    assert prepared.body["temperature"] == 0.2
    # effective is wire-faithful and omits nothing sent
    assert prepared.effective_controls == {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "reasoning": {"effort": "low"},
    }
    assert prepared.omitted_unsupported == ()
    assert prepared.defaulted_controls == {}
    assert prepared.body_sha256 is not None


def test_chat_matrix_and_omitted_reasoning():
    adapter = OpenCodeCognitionAdapter(model="mimo-v2.5", api_key="k", protocol="chat_completions")
    spec = make_spec(parameters={"temperature": 0.2, "max_tokens": 256, "reasoning_effort": "low"})
    prepared = adapter.prepare(spec)
    assert prepared.body["max_tokens"] == 256
    assert "reasoning" not in prepared.body
    assert prepared.effective_controls == {"temperature": 0.2, "max_tokens": 256}
    assert prepared.omitted_unsupported == ("reasoning_effort",)
    assert prepared.requested_controls["reasoning_effort"] == "low"


def test_seed_is_known_but_unsupported_everywhere():
    for protocol in ("responses", "chat_completions"):
        adapter = OpenCodeCognitionAdapter(model="m", api_key="k", protocol=protocol)
        prepared = adapter.prepare(make_spec(parameters={"seed": "7"}))
        assert prepared.omitted_unsupported == ("seed",)
        assert "seed" not in prepared.body
        assert "seed" not in prepared.effective_controls


def test_alias_conflict_rejected():
    adapter = OpenCodeCognitionAdapter(model="m", api_key="k", protocol="chat_completions")
    with pytest.raises(InvalidControlError):
        adapter.prepare(make_spec(parameters={"max_tokens": 256, "max_output_tokens": 128}))
    # equal spellings agree: accepted deterministically
    prepared = adapter.prepare(make_spec(parameters={"max_tokens": 256, "max_output_tokens": 256}))
    assert prepared.body["max_tokens"] == 256


def test_invalid_values_rejected_before_effect():
    adapter = OpenCodeCognitionAdapter(model="m", api_key="k", protocol="responses")
    with pytest.raises(InvalidControlError):
        adapter.prepare(make_spec(parameters={"temperature": True}))
    with pytest.raises(InvalidControlError):
        adapter.prepare(make_spec(parameters={"temperature": "hot"}))
    with pytest.raises(InvalidControlError):
        adapter.prepare(make_spec(parameters={"max_tokens": "abc"}))
    with pytest.raises(InvalidControlError):
        adapter.prepare(make_spec(parameters={"max_tokens": True}))
    with pytest.raises(InvalidControlError):
        adapter.prepare(make_spec(parameters={"reasoning_effort": 42}))


def test_unknown_control_rejected_pre_effect_no_invocation(tmp_path):
    transport = CountingTransport(HttpResponse(200, {}, b"{}", None))
    adapter = OpenCodeCognitionAdapter(
        model="m", api_key="k", protocol="chat_completions", http_post=transport
    )
    runtime = make_runtime(tmp_path)
    with pytest.raises(UnknownControlError) as caught:
        runtime.invoke_recorded_call(
            make_spec(parameters={"future_magic_control": 123}), adapter=adapter
        )
    assert caught.value.control == "future_magic_control"
    assert "123" not in str(caught.value)
    assert transport.bodies == []  # zero provider effects
    assert runtime.ledger.events_by_kind(("call.completed",)) == ()


def test_structural_model_override_rejected():
    adapter = OpenCodeCognitionAdapter(model="m", api_key="k", protocol="chat_completions")
    with pytest.raises(UnknownControlError):
        adapter.prepare(make_spec(parameters={"model": "other-model"}))


def test_secret_parameter_rejected_never_persisted(tmp_path):
    transport = CountingTransport(HttpResponse(200, {}, b"{}", None))
    adapter = OpenCodeCognitionAdapter(
        model="m", api_key="k", protocol="chat_completions", http_post=transport
    )
    runtime = make_runtime(tmp_path)
    with pytest.raises(UnknownControlError):
        runtime.invoke_recorded_call(
            make_spec(parameters={"api_key": "SHOULD-NOT-SEND"}), adapter=adapter
        )
    assert transport.bodies == []
    blob = json.dumps([e.payload for e in runtime.ledger.read_all()], default=str)
    assert "SHOULD-NOT-SEND" not in blob


# Body equality + prepare-once ----------------------------------------------------------


def test_sent_body_equals_prepared_body_exactly(tmp_path):
    body_bytes = json.dumps(chat_payload()).encode()
    transport = CountingTransport(
        HttpResponse(200, {"Content-Type": "application/json"}, body_bytes, "application/json")
    )
    adapter = OpenCodeCognitionAdapter(
        model="mimo-v2.5", api_key="k", protocol="chat_completions", http_post=transport
    )
    spec = make_spec(parameters={"temperature": 0.2, "max_tokens": 256})
    prepared = adapter.prepare(spec)
    runtime = make_runtime(tmp_path)
    # send the same prepared object the manifest will record
    result = adapter.send(prepared)
    assert transport.urls == ["https://opencode.ai/zen/go/v1/chat/completions"]
    assert transport.bodies == [prepared.body]
    assert result.effective_parameters == prepared.recorded_effective()
    assert runtime is not None


def test_prepare_once_send_once_per_attempt(tmp_path):
    CountingAdapter.prepares = 0
    transport = CountingTransport(
        HttpResponse(200, {}, json.dumps(chat_payload()).encode(), "application/json")
    )
    adapter = CountingAdapter(
        model="mimo-v2.5", api_key="k", protocol="chat_completions", http_post=transport
    )
    runtime = make_runtime(tmp_path)
    runtime.invoke_recorded_call(make_spec(parameters={"max_tokens": 64}), adapter=adapter)
    assert CountingAdapter.prepares == 1
    assert len(transport.bodies) == 1


def test_effective_controls_match_body_projection():
    adapter = OpenCodeCognitionAdapter(model="m", api_key="k", protocol="responses")
    prepared = adapter.prepare(
        make_spec(
            parameters={"temperature": 0.5, "max_output_tokens": 128, "reasoning_effort": "low"}
        )
    )
    body = prepared.body
    projected = {}
    if "temperature" in body:
        projected["temperature"] = body["temperature"]
    if "max_output_tokens" in body:
        projected["max_output_tokens"] = body["max_output_tokens"]
    if "reasoning" in body:
        projected["reasoning"] = body["reasoning"]
    assert projected == prepared.effective_controls


# Session identity ---------------------------------------------------------------------------


def test_generated_session_recorded_is_sent(tmp_path):
    transport = CountingTransport(
        HttpResponse(200, {}, json.dumps(chat_payload()).encode(), "application/json")
    )
    adapter = OpenCodeCognitionAdapter(
        model="m", api_key="k", protocol="chat_completions", http_post=transport
    )
    runtime = make_runtime(tmp_path)
    recorded = runtime.invoke_recorded_call(make_spec(), adapter=adapter)
    manifest_session = recorded.manifest.effective_parameters.get("session_id")
    attempt_session = recorded.attempts[0].effective_parameters.get("session_id")
    assert manifest_session is not None
    assert manifest_session == attempt_session == transport.headers[0]["x-opencode-session"]


def test_explicit_session_preserved_not_regenerated():
    adapter = OpenCodeCognitionAdapter(
        model="m", api_key="k", protocol="chat_completions", session_id="sess-explicit"
    )
    prepared = adapter.prepare(make_spec())
    assert prepared.routing["session_id"] == "sess-explicit"
    assert prepared.public_headers["x-opencode-session"] == "sess-explicit"
    assert prepared.recorded_effective()["session_id"] == "sess-explicit"


def test_retry_resends_same_prepared_request(tmp_path):
    bodies = [
        HttpResponse(429, {"Retry-After": "0"}, b'{"error":"slow"}', "application/json"),
        HttpResponse(200, {}, json.dumps(chat_payload()).encode(), "application/json"),
    ]
    transport = CountingTransport(bodies[0])

    def sequenced(url, payload, headers, timeout):
        transport(url, payload, headers, timeout)
        return bodies[len(transport.bodies) - 1]

    adapter = OpenCodeCognitionAdapter(
        model="m", api_key="k", protocol="chat_completions", http_post=sequenced
    )
    runtime = make_runtime(tmp_path)
    recorded = runtime.invoke_recorded_call(make_spec(), adapter=adapter, max_attempts=2)
    assert len(recorded.attempts) == 2
    assert transport.bodies[0] == transport.bodies[1]  # identical resend
    sessions = {h["x-opencode-session"] for h in transport.headers}
    assert len(sessions) == 1  # one session for the logical call
    assert recorded.manifest.effective_parameters["session_id"] == next(iter(sessions))
    assert recorded.manifest.request_body_sha256 is not None


# Credentials -------------------------------------------------------------------------------------


def test_credentials_used_not_persisted(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "zen-live-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-decoy")
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "compat-decoy")
    transport = CountingTransport(
        HttpResponse(200, {}, json.dumps(chat_payload()).encode(), "application/json")
    )
    adapter = OpenCodeCognitionAdapter(model="m", protocol="chat_completions", http_post=transport)
    runtime = make_runtime(tmp_path)
    recorded = runtime.invoke_recorded_call(
        make_spec(parameters={"temperature": 0.1}), adapter=adapter
    )
    assert recorded.status == "succeeded"
    # transport received the credential...
    assert transport.headers[0]["Authorization"] == "Bearer zen-live-secret"
    # ...but persistence contains none of the decoys or the live secret
    blob = ""
    for event in runtime.ledger.read_all():
        blob += json.dumps(event.payload, sort_keys=True, default=str)
    for artifact in [a for a in [recorded.attempts[0].raw_artifact] if a is not None]:
        blob += runtime.artifact_store.read_text(artifact.artifact_id)
    assert "zen-live-secret" not in blob
    assert "openai-decoy" not in blob
    assert "compat-decoy" not in blob
    assert "Authorization" not in blob


# Manifest durability + legacy readability --------------------------------------------------------------


def test_manifest_plan_fields_survive_restart(tmp_path):
    ledger_path = tmp_path / "ledger.sqlite"
    runtime = make_runtime(tmp_path, ledger_path=ledger_path)
    adapter = OpenCodeCognitionAdapter(model="m", api_key="k", protocol="chat_completions")
    recorded = runtime.invoke_recorded_call(
        make_spec("call-r", parameters={"max_tokens": 32, "seed": "7"}), adapter=adapter
    )
    assert recorded.manifest.request_plan_version == OPENCODE_REQUEST_PLAN
    assert recorded.manifest.omitted_unsupported == ("seed",)
    assert recorded.manifest.defaulted_parameters == {}
    assert recorded.manifest.requested_controls == {"max_tokens": 32, "seed": "7"}

    runtime2 = make_runtime(tmp_path, ledger_path=ledger_path)
    reopened = runtime2.get_recorded_call("call-r")
    assert reopened is not None
    assert reopened.manifest.request_plan_version == OPENCODE_REQUEST_PLAN
    assert reopened.manifest.omitted_unsupported == ("seed",)
    assert reopened.manifest.request_body_sha256 == recorded.manifest.request_body_sha256


def test_legacy_manifest_without_plan_fields_readable():
    from codeai.runtime import _manifest_from_payload

    manifest = _manifest_from_payload(
        {"call_id": "c", "task_id": "t", "requested_parameters": {"a": 1}}
    )
    assert manifest.request_plan_version is None
    assert manifest.omitted_unsupported == ()
    assert manifest.defaulted_parameters == {}
    assert manifest.request_body_sha256 is None
