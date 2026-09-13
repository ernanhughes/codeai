"""Stage 11.5b: versioned interpretation, completion, decision provenance.

Pure-interpreter cases run offline with synthetic evidence (marked as such);
runtime cases use deterministic HTTP fixtures. v1 must reproduce historical
labels; v2 must be more precise only where evidence supports it.
"""

from __future__ import annotations

import json
from pathlib import Path

from codeai.artifacts import FileArtifactStore
from codeai.context import ContextCompiler
from codeai.domain import ActorRef, CallSpec
from codeai.interpretation import (
    ATTEMPT_POLICY_V1,
    ATTEMPT_POLICY_V2,
    INTERPRETER_V1,
    INTERPRETER_V2,
    InterpretationInput,
    decide_attempt,
    interpret_attempt,
)
from codeai.ledger import SQLiteLedger
from codeai.providers import HttpResponse, OpenCodeCognitionAdapter
from codeai.runtime import Runtime

TEXT = "The one-second latency claim requires measurement."


def entry(**kwargs) -> InterpretationInput:
    base = {
        "attempt_id": "att-1",
        "call_id": "call-1",
        "task_id": "task-1",
        "protocol": "chat_completions",
    }
    base.update(kwargs)
    return InterpretationInput(**base)


def chat_parsed(finish_reason="stop", text=TEXT, in_tokens=100, out_tokens=50):
    return {
        "id": "chatcmpl-1",
        "model": "mimo-v2.5",
        "choices": [
            {"message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}
        ],
        "usage": {"prompt_tokens": in_tokens, "completion_tokens": out_tokens},
    }


def chat_body(payload: dict) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode()


# Synthetic corpus: (name, input, expected v1, expected v2) ---------------------------
# expected entries: (error_kind, generation_state, classification_basis)


def transport_case(status, body, headers=None, protocol="chat_completions", parsed=None):
    return entry(
        transport_outcome="http_error" if status != 200 else "response_received",
        http_status=status,
        body_bytes=body,
        parsed=parsed if parsed is not None else {},
        protocol=protocol,
        output_text="",
        adapter_status="failed" if status != 200 else "succeeded",
        failure_message=None if status == 200 else f"provider HTTP {status}: ...",
    )


CORPUS = [
    (
        "chat complete",
        entry(
            transport_outcome="response_received",
            http_status=200,
            body_bytes=chat_body(chat_parsed("stop")),
            parsed=chat_parsed("stop"),
            output_text=TEXT,
            adapter_status="succeeded",
        ),
        (None, "complete", None),
        (None, "complete", None),
    ),
    (
        "chat truncated",
        entry(
            transport_outcome="response_received",
            http_status=200,
            body_bytes=chat_body(chat_parsed("length")),
            parsed=chat_parsed("length"),
            output_text=TEXT,
            adapter_status="succeeded",
        ),
        (None, "complete", None),
        (None, "truncated", None),
    ),
    (
        "chat filtered",
        entry(
            transport_outcome="response_received",
            http_status=200,
            body_bytes=chat_body(chat_parsed("content_filter", text="")),
            parsed=chat_parsed("content_filter", text=""),
            output_text="",
            adapter_status="failed",
            failure_message="OpenCode Chat Completions payload contained no output text",
        ),
        ("empty_output", "empty", "generation_state"),
        (None, "filtered", None),
    ),
    (
        "responses complete",
        entry(
            protocol="responses",
            transport_outcome="response_received",
            http_status=200,
            body_bytes=b'{"status":"completed"}',
            parsed={"status": "completed"},
            output_text=TEXT,
            adapter_status="succeeded",
        ),
        (None, "complete", None),
        (None, "complete", None),
    ),
    (
        "responses truncated",
        entry(
            protocol="responses",
            transport_outcome="response_received",
            http_status=200,
            body_bytes=b'{"incomplete_details":{"reason":"max_output_tokens"}}',
            parsed={"incomplete_details": {"reason": "max_output_tokens"}},
            output_text=TEXT,
            adapter_status="succeeded",
        ),
        (None, "complete", None),
        (None, "truncated", None),
    ),
    (
        "messages complete",
        entry(
            protocol="messages",
            transport_outcome="response_received",
            http_status=200,
            body_bytes=b'{"stop_reason":"end_turn"}',
            parsed={"stop_reason": "end_turn"},
            output_text=TEXT,
            adapter_status="succeeded",
        ),
        (None, "complete", None),
        (None, "complete", None),
    ),
    (
        "messages truncated",
        entry(
            protocol="messages",
            transport_outcome="response_received",
            http_status=200,
            body_bytes=b'{"stop_reason":"max_tokens"}',
            parsed={"stop_reason": "max_tokens"},
            output_text=TEXT,
            adapter_status="succeeded",
        ),
        (None, "complete", None),
        (None, "truncated", None),
    ),
    (
        "401",
        transport_case(401, b'{"error":"unauthorized"}'),
        ("authentication_error", "unknown", "status_only"),
        ("authentication_error", "unknown", "status_only"),
    ),
    (
        "403 generic",
        transport_case(403, b'{"error":"forbidden"}'),
        ("authentication_error", "unknown", "status_only"),
        ("provider_error", "unknown", "status_only"),
    ),
    (
        "403 cloudflare",
        transport_case(403, b"error code: 1010 forbidden"),
        ("authentication_error", "unknown", "status_only"),
        ("edge_rejected", "unknown", "body_signature"),
    ),
    (
        "429",
        transport_case(429, b'{"error":"slow"}'),
        ("rate_limited", "unknown", "status_only"),
        ("rate_limited", "unknown", "status_only"),
    ),
    (
        "500",
        transport_case(500, b"boom"),
        ("provider_error", "unknown", "status_only"),
        ("provider_error", "unknown", "status_only"),
    ),
    (
        "400 generic",
        transport_case(400, b"bad"),
        ("provider_error", "unknown", "status_only"),
        ("invalid_request", "unknown", "status_only"),
    ),
    (
        "400 missing session",
        transport_case(400, b'{"error":{"type":"MissingSessionID"}}'),
        ("provider_error", "unknown", "status_only"),
        ("invalid_request", "unknown", "body_signature"),
    ),
    (
        "timeout genuine",
        entry(
            transport_outcome="no_response",
            exception_type="TimeoutError",
            failure_message="provider request failed: timed out",
            adapter_status="failed",
        ),
        ("timeout", "unknown", "exception_type"),
        ("timeout", "unknown", "exception_type"),
    ),
    (
        "reset is not timeout",
        entry(
            transport_outcome="no_response",
            exception_type="ConnectionResetError",
            failure_message="provider request failed: connection timed out",
            adapter_status="failed",
        ),
        ("timeout", "unknown", "exception_type"),
        ("provider_error", "unknown", "exception_type"),
    ),
    (
        "malformed",
        entry(
            transport_outcome="response_received",
            http_status=200,
            body_bytes=b"not json{",
            parsed={},
            output_text="",
            adapter_status="failed",
            failure_message="provider returned non-JSON response: ...",
        ),
        ("malformed_response", "unknown", "parser_failure"),
        ("malformed_response", "unknown", "parser_failure"),
    ),
    (
        "empty",
        entry(
            transport_outcome="response_received",
            http_status=200,
            body_bytes=chat_body(chat_parsed("stop", text="")),
            parsed=chat_parsed("stop", text=""),
            output_text="",
            adapter_status="failed",
            failure_message="OpenCode Chat Completions payload contained no output text",
        ),
        ("empty_output", "empty", "generation_state"),
        ("empty_output", "empty", "generation_state"),
    ),
    (
        "missing credentials",
        entry(
            transport_outcome=None,
            adapter_status="failed",
            adapter_error_kind="missing_credentials",
            failure_message="OpenCode credentials absent: set OPENCODE_ZEN_API_KEY",
        ),
        ("missing_credentials", "unknown", "configuration"),
        ("missing_credentials", "unknown", "configuration"),
    ),
    (
        "fake success text",
        entry(
            transport_outcome=None,
            adapter_status="succeeded",
            output_text="hi",
            protocol=None,
        ),
        (None, "complete", None),
        (None, "unknown", None),
    ),
    (
        "fake failure",
        entry(
            transport_outcome=None,
            adapter_status="failed",
            adapter_error_kind="provider_error",
            failure_message="boom",
            protocol=None,
        ),
        ("provider_error", "unknown", "adapter_reported"),
        ("provider_error", "unknown", "adapter_reported"),
    ),
]


def test_synthetic_corpus_v1_reproduces_history():
    for name, data, expected_v1, _expected_v2 in CORPUS:
        interp = interpret_attempt(data, version=INTERPRETER_V1)
        assert (interp.error_kind, interp.generation_state, interp.classification_basis) == (
            expected_v1
        ), name
        assert interp.interpreter_version == INTERPRETER_V1


def test_synthetic_corpus_v2_rules():
    for name, data, _expected_v1, expected_v2 in CORPUS:
        interp = interpret_attempt(data, version=INTERPRETER_V2)
        assert (interp.error_kind, interp.generation_state, interp.classification_basis) == (
            expected_v2
        ), name
        assert interp.interpreter_version == INTERPRETER_V2
        assert "retry" not in (interp.error_kind or "")
        assert interp.transport_state == (data.transport_outcome or "unknown")


def test_unknown_interpreter_version_fails_loudly():
    import pytest

    with pytest.raises(ValueError, match="unknown interpreter version"):
        interpret_attempt(entry(), version="v99")
    with pytest.raises(ValueError, match="unknown policy version"):
        decide_attempt(interpret_attempt(entry(), version=INTERPRETER_V1), policy_version="p99")


def test_overclaiming_guards():
    # generic 403 is not Cloudflare, 500 is not a route diagnosis, reset is not timeout
    by_name = {name: v2 for name, _data, _v1, v2 in CORPUS}
    assert by_name["403 generic"][0] == "provider_error"
    assert by_name["500"][0] == "provider_error"
    assert by_name["reset is not timeout"][0] == "provider_error"
    assert by_name["400 generic"][0] == "invalid_request"


def test_provider_reason_preserved_verbatim():
    interp = interpret_attempt(CORPUS[1][1], version=INTERPRETER_V2)  # chat truncated
    assert interp.provider_reason == "length"
    assert interp.provider_reason_source == "chat_completions:choices[0].finish_reason"
    interp = interpret_attempt(CORPUS[6][1], version=INTERPRETER_V2)  # messages truncated
    assert interp.provider_reason == "max_tokens"
    assert interp.provider_reason_source == "messages:stop_reason"


# Runtime: causality ------------------------------------------------------------------


def make_runtime(tmp_path: Path) -> Runtime:
    ledger = SQLiteLedger()
    return Runtime(ledger, artifact_store=FileArtifactStore(tmp_path / "artifacts", ledger))


def make_spec(call_id="call-1", task_id="task-1") -> CallSpec:
    actor = ActorRef(actor_id="r1", kind="model", provider="opencode", model="mimo-v2.5")
    package = ContextCompiler().compile(task_id=task_id, actor=actor, prompt="p", events=())
    return CallSpec(
        call_id=call_id,
        task_id=task_id,
        actor=actor,
        context=package,
        idempotency_key=f"key-{call_id}",
    )


def zen_chat(reply, **kwargs):
    def fake_post(url, payload, headers, timeout):
        if isinstance(reply, Exception):
            raise reply
        return reply

    return OpenCodeCognitionAdapter(
        model="mimo-v2.5", protocol="chat_completions", api_key="k", http_post=fake_post, **kwargs
    )


def test_causality_survives_reinterpretation(tmp_path):
    from codeai.providers import TransportFailure

    runtime = make_runtime(tmp_path)
    reset_then_ok = [
        TransportFailure("ConnectionResetError", "provider request failed: connection timed out"),
        None,
    ]
    calls = {"n": 0}

    def flaky_post(url, payload, headers, timeout):
        calls["n"] += 1
        reply = reset_then_ok[calls["n"] - 1]
        if isinstance(reply, Exception):
            raise reply
        return HttpResponse(200, {}, json.dumps(chat_parsed()).encode(), "application/json")

    adapter = OpenCodeCognitionAdapter(
        model="mimo-v2.5", protocol="chat_completions", api_key="k", http_post=flaky_post
    )
    # Execute history under the v1 rules (the historical policy).
    recorded = runtime.invoke_recorded_call(
        make_spec(),
        adapter=adapter,
        max_attempts=2,
        interpreter_version=INTERPRETER_V1,
        policy_version=ATTEMPT_POLICY_V1,
    )
    assert calls["n"] == 2
    assert len(recorded.attempts) == 2
    first_id = recorded.attempts[0].attempt_id
    retry_events = runtime.ledger.events_by_kind(("attempt.retry_decided",))
    assert len(retry_events) == 2
    first_decision = retry_events[0].payload
    assert first_decision["decision"] == "retry" and first_decision["executed"] is True
    v1_id = first_decision["interpretation_id"]
    assert runtime.interpretations_for_attempt(first_id)[0].interpretation_id == v1_id

    # Later v2 reading of the same preserved evidence: would NOT have retried.
    v2 = runtime.interpret_attempt_as(first_id, version=INTERPRETER_V2)
    assert v2 is not None and v2.error_kind == "provider_error"
    decision, _reason = decide_attempt(v2, policy_version=ATTEMPT_POLICY_V2)
    assert decision == "terminal"
    # History is untouched: attempt 2 and the original v1 decision remain.
    assert len(runtime.ledger.events_by_kind(("attempt.completed",))) == 2
    assert (
        runtime.ledger.events_by_kind(("attempt.retry_decided",))[0].payload["interpretation_id"]
        == v1_id
    )


def test_truncated_call_is_unresolved_not_success(tmp_path):
    runtime = make_runtime(tmp_path)
    body = json.dumps(chat_parsed("length")).encode()
    recorded = runtime.invoke_recorded_call(
        make_spec(), adapter=zen_chat(HttpResponse(200, {}, body, "application/json"))
    )
    assert recorded.status == "unresolved"
    attempt = recorded.attempts[0]
    assert attempt.status == "failed" and attempt.error_kind is None
    interps = runtime.interpretations_for_attempt(attempt.attempt_id)
    assert len(interps) == 1 and interps[0].generation_state == "truncated"
    assert interps[0].provider_reason == "length"
    decided = runtime.ledger.events_by_kind(("call.status_decided",))
    assert len(decided) == 1
    assert decided[0].payload["status"] == "unresolved"
    assert decided[0].payload["interpretation_ids"] == [interps[0].interpretation_id]
    completed = runtime.ledger.events_by_kind(("call.completed",))
    assert completed[0].payload["call_status"] == "unresolved"
    assert completed[0].payload["decision_basis_interpretation_id"] == interps[0].interpretation_id
    # Counterfactual v1 projection: history called it success; payloads unmutated.
    projected = runtime.project_call_as(
        recorded.call_id,
        interpreter_version=INTERPRETER_V1,
        policy_version=ATTEMPT_POLICY_V1,
    )
    assert projected is not None and projected["status"] == "succeeded"
    assert runtime.ledger.events_by_kind(("call.completed",))[0].payload["call_status"] == (
        "unresolved"
    )


def test_interpretations_for_attempt_empty_for_unknown(tmp_path):
    assert make_runtime(tmp_path).interpretations_for_attempt("missing") == []
    assert make_runtime(tmp_path).interpret_attempt_as("missing", version=INTERPRETER_V2) is None


# Legacy derivation (§31) -----------------------------------------------------------------


def test_legacy_derived_inputs_replay_table():
    """Derived (not observed) inputs mirroring the ch11 live evidence classes."""
    run4 = entry(
        transport_outcome=None,
        protocol="chat_completions",
        parsed=chat_parsed("length"),
        output_text=TEXT,
        adapter_status="succeeded",
    )
    v1 = interpret_attempt(run4, version=INTERPRETER_V1)
    v2 = interpret_attempt(run4, version=INTERPRETER_V2)
    assert (v1.error_kind, v1.generation_state) == (None, "complete")
    assert (v2.error_kind, v2.generation_state) == (None, "truncated")
    assert v2.provider_reason == "length"

    run1 = entry(
        transport_outcome=None,
        protocol="chat_completions",
        adapter_status="failed",
        adapter_error_kind="authentication_error",
        failure_message="provider HTTP 403: error code: 1010",
    )
    assert interpret_attempt(run1, version=INTERPRETER_V1).error_kind == "authentication_error"
    v2 = interpret_attempt(run1, version=INTERPRETER_V2)
    assert v2.error_kind == "edge_rejected"
    assert v2.classification_basis == "body_signature"

    run2 = entry(
        transport_outcome=None,
        protocol="chat_completions",
        adapter_status="failed",
        adapter_error_kind="provider_error",
        failure_message="provider HTTP 500: Internal server error",
    )
    assert interpret_attempt(run2, version=INTERPRETER_V1).error_kind == "provider_error"
    assert interpret_attempt(run2, version=INTERPRETER_V2).error_kind == "provider_error"

    run3 = entry(
        transport_outcome=None,
        protocol="chat_completions",
        adapter_status="failed",
        adapter_error_kind="provider_error",
        failure_message="provider HTTP 400: MissingSessionID",
    )
    assert interpret_attempt(run3, version=INTERPRETER_V1).error_kind == "provider_error"
    v2 = interpret_attempt(run3, version=INTERPRETER_V2)
    assert v2.error_kind == "invalid_request"


def test_policy_versions_reproduce_and_diverge(tmp_path):
    v1_timeout = interpret_attempt(
        entry(
            transport_outcome="no_response",
            exception_type="TimeoutError",
            failure_message="provider request failed: timed out",
            adapter_status="failed",
        ),
        version=INTERPRETER_V1,
    )
    assert decide_attempt(v1_timeout, policy_version=ATTEMPT_POLICY_V1)[0] == "retry"
    v2_ok = interpret_attempt(
        entry(transport_outcome=None, adapter_status="succeeded", output_text="hi"),
        version=INTERPRETER_V2,
    )
    assert decide_attempt(v2_ok, policy_version=ATTEMPT_POLICY_V2)[0] == "accept"
    v2_trunc = interpret_attempt(CORPUS[1][1], version=INTERPRETER_V2)
    assert decide_attempt(v2_trunc, policy_version=ATTEMPT_POLICY_V2)[0] == "terminal"
