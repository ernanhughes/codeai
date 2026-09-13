"""Stage 11.5a: complete transport observations without changing interpretation.

Offline only. Fixtures are synthetic provider bytes delivered through the
HttpResponse test seam; no network occurs. Classifications asserted here are
the pre-existing labels, deliberately unchanged by this increment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from codeai.adapters import FakeCognitionAdapter
from codeai.artifacts import FileArtifactStore
from codeai.context import ContextCompiler
from codeai.domain import ActorRef, CallSpec
from codeai.ledger import SQLiteLedger
from codeai.providers import HttpResponse, OpenCodeCognitionAdapter, TransportFailure
from codeai.runtime import Runtime

TEXT = "The one-second latency claim requires measurement."


def chat_body(text: str = TEXT, in_tokens: int = 100, out_tokens: int = 50) -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl-zen-1",
            "model": "mimo-v2.5",
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": in_tokens, "completion_tokens": out_tokens},
        },
        separators=(",", ":"),
    ).encode("utf-8")


def make_spec(call_id: str = "call-t1", task_id: str = "task-t1") -> CallSpec:
    actor = ActorRef(actor_id="r1", kind="model", provider="opencode", model="mimo-v2.5")
    package = ContextCompiler().compile(task_id=task_id, actor=actor, prompt="p", events=())
    return CallSpec(
        call_id=call_id,
        task_id=task_id,
        actor=actor,
        context=package,
        idempotency_key=f"key-{call_id}",
        chamber="deep-review",
        logical_model="review",
    )


def make_runtime(tmp_path: Path, ledger_path=None) -> Runtime:
    ledger = SQLiteLedger(str(ledger_path) if ledger_path else ":memory:")
    return Runtime(ledger, artifact_store=FileArtifactStore(tmp_path / "artifacts", ledger))


def zen_adapter(reply, **kwargs) -> OpenCodeCognitionAdapter:
    def fake_post(url, payload, headers, timeout):
        if isinstance(reply, Exception):
            raise reply
        return reply

    return OpenCodeCognitionAdapter(
        model="mimo-v2.5",
        protocol="chat_completions",
        api_key="zen-test-key",
        http_post=fake_post,
        **kwargs,
    )


def observed_for(runtime: Runtime, attempt_id: str) -> dict:
    payload = runtime.get_attempt_observation(attempt_id)
    assert payload is not None
    return payload


# A. HTTP 200 valid JSON ---------------------------------------------------------


def test_success_preserves_exact_bytes_and_old_result(tmp_path):
    runtime = make_runtime(tmp_path)
    body = chat_body()
    headers = {"Content-Type": "application/json", "X-Request-Id": "req-1"}
    recorded = runtime.invoke_recorded_call(
        make_spec(), adapter=zen_adapter(HttpResponse(200, headers, body, "application/json"))
    )
    attempt = recorded.attempts[0]
    assert recorded.status == "succeeded"
    assert attempt.provider_request_id == "chatcmpl-zen-1"
    assert (attempt.usage.input_tokens, attempt.usage.output_tokens) == (100, 50)

    obs = observed_for(runtime, attempt.attempt_id)
    assert obs["transport_outcome"] == "response_received"
    assert obs["http_status"] == 200
    assert obs["call_id"] == recorded.call_id
    ref = obs["response_body_artifact"]
    assert ref["sha256"] == hashlib.sha256(body).hexdigest()
    assert runtime.artifact_store.read_bytes(ref["artifact_id"]) == body
    assert obs["content_type"] == "application/json"
    assert obs["headers"]["x-request-id"] == "req-1"

    # Legacy mixed envelope still present with its old shape.
    legacy = json.loads(runtime.artifact_store.read_text(attempt.raw_artifact.artifact_id))
    assert legacy["output_text"] == TEXT
    assert legacy["kind"] == "decoded_json"


# B. HTTP 200 malformed JSON ------------------------------------------------------


def test_malformed_body_preserved_before_parser_failure(tmp_path):
    runtime = make_runtime(tmp_path)
    body = b'{"choices": [not json'
    recorded = runtime.invoke_recorded_call(
        make_spec(), adapter=zen_adapter(HttpResponse(200, {}, body, "application/json"))
    )
    assert recorded.status == "failed"  # existing classification unchanged
    attempt = recorded.attempts[0]
    assert attempt.error_kind == "malformed_response"
    obs = observed_for(runtime, attempt.attempt_id)
    assert obs["transport_outcome"] == "response_received"
    assert obs["http_status"] == 200
    assert runtime.artifact_store.read_bytes(obs["response_body_artifact"]["artifact_id"]) == body


# C. HTTP 200 empty body is a response, not no-response ------------------------------


def test_empty_body_is_response_received_with_empty_hash(tmp_path):
    runtime = make_runtime(tmp_path)
    recorded = runtime.invoke_recorded_call(
        make_spec(), adapter=zen_adapter(HttpResponse(200, {}, b"", "application/json"))
    )
    attempt = recorded.attempts[0]
    obs = observed_for(runtime, attempt.attempt_id)
    assert obs["transport_outcome"] == "response_received"
    ref = obs["response_body_artifact"]
    assert ref["sha256"] == hashlib.sha256(b"").hexdigest()
    assert runtime.artifact_store.read_bytes(ref["artifact_id"]) == b""


# D/E/F. HTTP errors keep full bodies, old labels -------------------------------------


def test_http_400_full_body_and_old_label(tmp_path):
    runtime = make_runtime(tmp_path)
    body = json.dumps({"type": "error", "error": {"type": "MissingSessionID"}}).encode()
    recorded = runtime.invoke_recorded_call(
        make_spec(), adapter=zen_adapter(HttpResponse(400, {}, body, "application/json"))
    )
    attempt = recorded.attempts[0]
    assert recorded.status == "failed" and attempt.error_kind == "provider_error"
    obs = observed_for(runtime, attempt.attempt_id)
    assert obs["transport_outcome"] == "http_error" and obs["http_status"] == 400
    assert runtime.artifact_store.read_bytes(obs["response_body_artifact"]["artifact_id"]) == body


def test_http_401_old_label_unchanged(tmp_path):
    runtime = make_runtime(tmp_path)
    body = b'{"error": "unauthorized"}'
    recorded = runtime.invoke_recorded_call(
        make_spec(), adapter=zen_adapter(HttpResponse(401, {}, body, "application/json"))
    )
    assert recorded.attempts[0].error_kind == "authentication_error"
    obs = observed_for(runtime, recorded.attempts[0].attempt_id)
    assert (obs["transport_outcome"], obs["http_status"]) == ("http_error", 401)


def test_http_403_body_beyond_byte_500_survives(tmp_path):
    runtime = make_runtime(tmp_path)
    marker = b"EVIDENCE-AFTER-BYTE-500"
    body = b'{"error":"forbidden","detail":"' + b"x" * 600 + marker + b'"}'
    assert len(body) > 500 and body.index(marker) > 500
    recorded = runtime.invoke_recorded_call(
        make_spec(), adapter=zen_adapter(HttpResponse(403, {}, body, "application/json"))
    )
    assert recorded.attempts[0].error_kind == "authentication_error"  # old label kept
    stored = runtime.artifact_store.read_bytes(
        observed_for(runtime, recorded.attempts[0].attempt_id)["response_body_artifact"][
            "artifact_id"
        ]
    )
    assert stored == body and marker in stored and len(stored) == len(body)


# G. HTTP 429 keeps body + retry headers, old label --------------------------------------


def test_http_429_retry_headers_preserved_old_label_kept(tmp_path):
    runtime = make_runtime(tmp_path)
    body = b'{"error": "slow down"}'
    headers = {
        "Retry-After": "4",
        "X-RateLimit-Remaining": "0",
        "X-RATELIMIT-RESET": "99",
        "X-Request-ID": "req-429",
    }
    recorded = runtime.invoke_recorded_call(
        make_spec(), adapter=zen_adapter(HttpResponse(429, headers, body, "application/json"))
    )
    assert recorded.attempts[0].error_kind == "rate_limited"
    obs = observed_for(runtime, recorded.attempts[0].attempt_id)
    assert obs["headers"]["retry-after"] == "4"
    assert obs["headers"]["x-ratelimit-remaining"] == "0"
    assert obs["headers"]["x-ratelimit-reset"] == "99"
    assert runtime.artifact_store.read_bytes(obs["response_body_artifact"]["artifact_id"]) == body


# H. HTTP 500 ---------------------------------------------------------------------------


def test_http_500_body_preserved_old_label_kept(tmp_path):
    runtime = make_runtime(tmp_path)
    body = b'{"type":"error","error":{"type":"error","message":"Internal server error"}}'
    recorded = runtime.invoke_recorded_call(
        make_spec(), adapter=zen_adapter(HttpResponse(500, {}, body, "application/json"))
    )
    assert recorded.attempts[0].error_kind == "provider_error"
    obs = observed_for(runtime, recorded.attempts[0].attempt_id)
    assert obs["http_status"] == 500
    assert runtime.artifact_store.read_bytes(obs["response_body_artifact"]["artifact_id"]) == body


# I. no response ---------------------------------------------------------------------------


def test_timeout_is_no_response_with_exception_type(tmp_path):
    runtime = make_runtime(tmp_path)
    recorded = runtime.invoke_recorded_call(
        make_spec(),
        adapter=zen_adapter(TransportFailure("TimeoutError", "provider request failed: timed out")),
    )
    attempt = recorded.attempts[0]
    assert recorded.status == "failed"
    obs = observed_for(runtime, attempt.attempt_id)
    assert obs["transport_outcome"] == "no_response"
    assert obs["http_status"] is None
    assert obs["response_body_artifact"] is None
    assert obs["exception_type"] == "TimeoutError"
    # Existing interpretation path unchanged for transport failures.
    assert attempt.error_kind == "timeout"


def test_request_bytes_no_response_carries_exception_type(tmp_path, monkeypatch):
    import urllib.request

    from codeai.providers import TransportFailure, _request_bytes

    def boom(request, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(TransportFailure) as captured:
        _request_bytes("http://x/v1/chat/completions", {}, {}, 1.0)
    assert captured.value.exception_type == "TimeoutError"


# J/K. header allowlist + secret rejection ---------------------------------------------------


def test_header_allowlist_case_insensitive_and_secrets_dropped(tmp_path):
    runtime = make_runtime(tmp_path)
    headers = {
        "Authorization": "Bearer SHOULD-NOT-PERSIST",
        "Set-Cookie": "session=SHOULD-NOT-PERSIST",
        "X-Api-Key": "SHOULD-NOT-PERSIST",
        "Proxy-Authorization": "Basic SHOULD-NOT-PERSIST",
        "X-Custom-Unknown": "drop-me",
        "RETRY-AFTER": "5",
        "X-Request-Id": "req-safe-123",
        "X-RateLimit-Limit": "100",
    }
    recorded = runtime.invoke_recorded_call(
        make_spec(), adapter=zen_adapter(HttpResponse(200, headers, chat_body(), None))
    )
    obs = observed_for(runtime, recorded.attempts[0].attempt_id)
    assert obs["headers"] == {
        "retry-after": "5",
        "x-request-id": "req-safe-123",
        "x-ratelimit-limit": "100",
    }
    blob = json.dumps(obs)
    assert "SHOULD-NOT-PERSIST" not in blob
    assert "set-cookie" not in blob and "authorization" not in blob


# L. byte identity, incl. non-UTF8 --------------------------------------------------------------


def test_non_ascii_bytes_survive_verbatim(tmp_path):
    runtime = make_runtime(tmp_path)
    text = "Prüfung — ünïcodé ✓"
    body = json.dumps(
        {
            "id": "chatcmpl-zen-1",
            "model": "mimo-v2.5",
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    recorded = runtime.invoke_recorded_call(
        make_spec(), adapter=zen_adapter(HttpResponse(200, {}, body, "application/json"))
    )
    assert recorded.attempts[0].status == "succeeded"
    stored = runtime.artifact_store.read_bytes(
        observed_for(runtime, recorded.attempts[0].attempt_id)["response_body_artifact"][
            "artifact_id"
        ]
    )
    assert stored == body


def test_non_utf8_error_body_stored_without_decoding(tmp_path):
    runtime = make_runtime(tmp_path)
    body = b"\xff\x00\xfe binary \x80 evidence"
    recorded = runtime.invoke_recorded_call(
        make_spec(), adapter=zen_adapter(HttpResponse(500, {}, body, None))
    )
    stored = runtime.artifact_store.read_bytes(
        observed_for(runtime, recorded.attempts[0].attempt_id)["response_body_artifact"][
            "artifact_id"
        ]
    )
    assert stored == body


# M/N. dedup across attempts, distinct events ---------------------------------------------------


def test_identical_bodies_share_hash_across_distinct_attempts(tmp_path):
    runtime = make_runtime(tmp_path)
    body = chat_body()
    reply = HttpResponse(200, {}, body, "application/json")
    first = runtime.invoke_recorded_call(make_spec("call-a"), adapter=zen_adapter(reply))
    second = runtime.invoke_recorded_call(make_spec("call-b"), adapter=zen_adapter(reply))
    ref_a = observed_for(runtime, first.attempts[0].attempt_id)["response_body_artifact"]
    ref_b = observed_for(runtime, second.attempts[0].attempt_id)["response_body_artifact"]
    assert ref_a["sha256"] == ref_b["sha256"] == hashlib.sha256(body).hexdigest()
    assert first.attempts[0].attempt_id != second.attempts[0].attempt_id
    events = runtime.ledger.events_by_kind(("attempt.observed",))
    assert len(events) == 2 and events[0].event_id != events[1].event_id


# Event order ------------------------------------------------------------------------------------


def test_event_order_success_error_noresponse(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.invoke_recorded_call(
        make_spec("call-ok"), adapter=zen_adapter(HttpResponse(200, {}, chat_body(), None))
    )
    runtime.invoke_recorded_call(
        make_spec("call-err"), adapter=zen_adapter(HttpResponse(500, {}, b"boom", None))
    )
    runtime.invoke_recorded_call(
        make_spec("call-nr"),
        adapter=zen_adapter(TransportFailure("TimeoutError", "provider request failed: t/o")),
    )
    per_call = ["call.requested", "call.manifest", "attempt.started", "attempt.observed"]
    # attempt.completed is appended right after the legacy envelope, then call.completed
    per_call += ["attempt.completed", "call.completed"]
    assert [e.kind for e in runtime.ledger.read_all()] == per_call * 3


def test_observed_precedes_completed_in_sequence(tmp_path):
    runtime = make_runtime(tmp_path)
    runtime.invoke_recorded_call(
        make_spec(), adapter=zen_adapter(HttpResponse(200, {}, chat_body(), None))
    )
    kinds = [e.kind for e in runtime.ledger.read_all()]
    assert kinds.index("attempt.observed") < kinds.index("attempt.completed")
    assert kinds.index("attempt.started") < kinds.index("attempt.observed")
    assert kinds.index("attempt.completed") < kinds.index("call.completed")


# O. restart ----------------------------------------------------------------------------------------


def test_observation_survives_clean_restart(tmp_path):
    ledger_path = tmp_path / "ledger.sqlite"
    runtime = make_runtime(tmp_path, ledger_path=ledger_path)
    body = chat_body()
    recorded = runtime.invoke_recorded_call(
        make_spec(), adapter=zen_adapter(HttpResponse(200, {"X-Request-Id": "r1"}, body, None))
    )
    attempt_id = recorded.attempts[0].attempt_id

    runtime2 = make_runtime(tmp_path, ledger_path=ledger_path)
    obs = runtime2.get_attempt_observation(attempt_id)
    assert obs is not None and obs["http_status"] == 200
    assert runtime2.artifact_store.read_bytes(obs["response_body_artifact"]["artifact_id"]) == body
    assert runtime2.get_attempt_observation("missing") is None


# Legacy adapters produce no observation ---------------------------------------------------------------


def test_fake_adapter_call_has_no_observed_event(tmp_path):
    runtime = make_runtime(tmp_path)
    recorded = runtime.invoke_recorded_call(
        make_spec(), adapter=FakeCognitionAdapter(responses=["hi"])
    )
    assert runtime.get_attempt_observation(recorded.attempts[0].attempt_id) is None
    assert runtime.ledger.events_by_kind(("attempt.observed",)) == ()


# R + serializers ------------------------------------------------------------------------------------------


def test_transport_never_enters_ledger_or_completed_payloads(tmp_path):
    runtime = make_runtime(tmp_path)
    body = chat_body()
    runtime.invoke_recorded_call(
        make_spec(), adapter=zen_adapter(HttpResponse(200, {}, body, "application/json"))
    )
    for event in runtime.ledger.read_all():
        assert "transport" not in event.payload
        text = json.dumps(event.payload, sort_keys=True, default=str)
        assert "b'{" not in text  # no coerced-bytes corruption
    # full JSON round-trip, as an export would perform
    exported = json.loads(json.dumps([e.payload for e in runtime.ledger.read_all()], default=str))
    assert len(exported) == len(runtime.ledger.read_all())
    for payload in exported:
        assert "transport" not in payload


def test_call_completed_compatible_for_existing_readers(tmp_path):
    import dataclasses

    from codeai.experiments import experiment_usage

    runtime = make_runtime(tmp_path)
    spec = dataclasses.replace(make_spec("call-e"), experiment_id="exp-t")
    runtime.invoke_recorded_call(
        spec, adapter=zen_adapter(HttpResponse(200, {}, chat_body(in_tokens=10, out_tokens=5), None))
    )
    completed = runtime.ledger.events_by_kind(("call.completed",))
    assert len(completed) == 1
    payload = completed[0].payload
    for key in (
        "call_id",
        "raw_output",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "status",
        "total_input_tokens",
        "total_cost_usd",
    ):
        assert key in payload
    usage = experiment_usage(runtime, "exp-t")
    assert usage.calls == 1 and (usage.known_input_tokens, usage.known_output_tokens) == (10, 5)
