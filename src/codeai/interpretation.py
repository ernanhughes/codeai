"""Versioned interpretation of preserved transport evidence.

Observation (attempt.observed / derived input) is evidence. An
AttemptInterpretation is a conclusion under a named interpreter version.
Policy answers (retry / accept / terminal) are derived from interpretations
by a separately versioned decision policy. Interpreters are deterministic
and pure: no models, no network, no wall-clock pricing, no diagnoses.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .domain import (
    AttemptInterpretation,
    ClassificationBasis,
    GenerationState,
)

INTERPRETER_V1 = "attempt-interpretation-v1"
INTERPRETER_V2 = "attempt-interpretation-v2"
COMPLETION_MAP_LEGACY = "completion-legacy-acceptance-v1"
COMPLETION_MAP_V1 = "completion-map-v1"
ERROR_CLASSIFIER_V1 = "error-classifier-v1"
ERROR_CLASSIFIER_V2 = "error-classifier-v2"
ATTEMPT_POLICY_V1 = "attempt-policy-v1"
ATTEMPT_POLICY_V2 = "attempt-policy-v2"

# Retryable error modes. v1 reproduces the pre-11.5b retry set exactly;
# policy-v2 retries generation-empty plus this same set.
POLICY_V1_RETRYABLE = frozenset({"transient_failure", "empty_output", "timeout", "rate_limited"})
POLICY_V2_RETRYABLE = frozenset({"transient_failure", "timeout", "rate_limited"})

# Exception types that genuinely support a timeout conclusion (v2).
TIMEOUT_EXCEPTIONS = frozenset(
    {"TimeoutError", "TimeoutExpired", "ConnectTimeout", "ReadTimeout", "Timeout"}
)

# Body-signature registry: distinctive provider/edge substrings. Matching is
# over preserved evidence (response body preferred, adapter error text when
# the body was never captured); the matched source is always recorded.
SIGNATURE_CLOUDFLARE_1010 = "cloudflare_1010_v1"
SIGNATURE_CLOUDFLARE_1010_MARK = "error code: 1010"
SIGNATURE_MISSING_SESSION = "missing_session_id_v1"
SIGNATURE_MISSING_SESSION_MARK = "MissingSessionID"

# Completion-reason mapping per dialect: (protocol, provider reason) -> state.
# Only explicit provider signals map; anything else stays unknown.
_COMPLETION_REASONS = {
    ("chat_completions", "stop"): GenerationState.COMPLETE.value,
    ("chat_completions", "length"): GenerationState.TRUNCATED.value,
    ("chat_completions", "content_filter"): GenerationState.FILTERED.value,
    ("messages", "end_turn"): GenerationState.COMPLETE.value,
    ("messages", "max_tokens"): GenerationState.TRUNCATED.value,
    ("responses", "completed"): GenerationState.COMPLETE.value,
    ("responses", "max_output_tokens"): GenerationState.TRUNCATED.value,
    ("responses", "content_filter"): GenerationState.FILTERED.value,
}

# JSON paths naming where a provider completion reason came from.
REASON_SOURCES = {
    "chat_completions": "chat_completions:choices[0].finish_reason",
    "messages": "messages:stop_reason",
    "responses": "responses:incomplete_details.reason",
}


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class InterpretationInput:
    """Evidence assembled for one interpretation. All fields are preserved
    evidence (transport record, decoded payload, adapter facts); nothing here
    is a conclusion."""

    attempt_id: str
    call_id: str
    task_id: str
    protocol: str | None = None
    transport_outcome: str | None = None  # response_received | http_error | no_response
    http_status: int | None = None
    exception_type: str | None = None
    failure_message: str | None = None
    body_bytes: bytes | None = None
    parsed: Mapping[str, Any] = field(default_factory=dict)
    output_text: str = ""
    adapter_status: str = "failed"  # succeeded | failed (adapter's own claim)
    adapter_error_kind: str | None = None


def _signature_corpus(entry: InterpretationInput) -> tuple[str, str]:
    """Searchable text plus where it came from (response body preferred)."""
    if entry.body_bytes is not None:
        return entry.body_bytes.decode("utf-8", errors="replace"), "response_body"
    return entry.failure_message or "", "adapter_error_text"


def _provider_reason(
    protocol: str | None, parsed: Mapping[str, Any]
) -> tuple[str | None, str | None]:
    """Extract the verbatim provider completion reason and its JSON path."""
    if not isinstance(parsed, dict):
        return None, None
    if protocol == "chat_completions":
        choices = parsed.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            reason = choices[0].get("finish_reason")
            if isinstance(reason, str):
                return reason, REASON_SOURCES["chat_completions"]
    elif protocol == "messages":
        reason = parsed.get("stop_reason")
        if isinstance(reason, str):
            return reason, REASON_SOURCES["messages"]
    elif protocol == "responses":
        details = parsed.get("incomplete_details")
        if isinstance(details, dict) and isinstance(details.get("reason"), str):
            return str(details["reason"]), REASON_SOURCES["responses"]
        status = parsed.get("status")
        if isinstance(status, str) and status != "completed":
            return status, "responses:status"
        if status == "completed":
            return status, "responses:status"
    return None, None


def _parse_outcome(entry: InterpretationInput) -> str:
    """Classify the preserved payload: ok | malformed | provider_error_field | unknown."""
    if entry.body_bytes is not None:
        try:
            parsed = json.loads(entry.body_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return "malformed"
        if isinstance(parsed, dict) and parsed.get("error"):
            return "provider_error_field"
        return "ok" if isinstance(parsed, dict) else "malformed"
    if isinstance(entry.parsed, dict) and entry.parsed:
        if entry.parsed.get("error"):
            return "provider_error_field"
        return "ok"
    message = entry.failure_message or ""
    if "non-JSON response" in message or "unexpected response shape" in message:
        return "malformed"
    if message.startswith("provider error:"):
        return "provider_error_field"
    return "unknown"


def _new_id() -> str:
    return str(uuid.uuid4())


def _base(
    entry: InterpretationInput,
    *,
    interpreter_version: str,
    completion_map_version: str | None,
    classifier_version: str,
    observation_event_id: str | None,
    response_body_artifact: Any,
    transport_state: str,
    generation_state: str,
    provider_reason: str | None,
    provider_reason_source: str | None,
    error_kind: str | None,
    classification_basis: str | None,
    basis_detail: dict[str, Any],
) -> AttemptInterpretation:
    return AttemptInterpretation(
        interpretation_id=_new_id(),
        attempt_id=entry.attempt_id,
        call_id=entry.call_id,
        task_id=entry.task_id,
        observation_event_id=observation_event_id,
        response_body_artifact=response_body_artifact,
        interpreter_version=interpreter_version,
        completion_map_version=completion_map_version,
        classifier_version=classifier_version,
        created_at=now_utc(),
        transport_state=transport_state,
        generation_state=generation_state,
        provider_reason=provider_reason,
        provider_reason_source=provider_reason_source,
        error_kind=error_kind,
        classification_basis=classification_basis,
        basis_detail=basis_detail,
    )


# --------------------------------------------------------------------------
# v1: historical rules, encoded for replay. Not an endorsement.
# --------------------------------------------------------------------------


def interpret_v1(
    entry: InterpretationInput,
    *,
    observation_event_id: str | None = None,
    response_body_artifact: Any = None,
) -> AttemptInterpretation:
    """Reproduce the pre-11.5b labels: status-keyed HTTP mapping, message-keyed
    timeout rule, legacy acceptance (text == complete output)."""
    transport = entry.transport_outcome or "unknown"
    message = entry.failure_message or ""

    if entry.transport_outcome is None:
        # No transport: legacy adapters/fakes. Reproduces _classify_attempt.
        if entry.adapter_status == "succeeded":
            if entry.output_text:
                return _base(
                    entry,
                    interpreter_version=INTERPRETER_V1,
                    completion_map_version=COMPLETION_MAP_LEGACY,
                    classifier_version=ERROR_CLASSIFIER_V1,
                    observation_event_id=observation_event_id,
                    response_body_artifact=response_body_artifact,
                    transport_state=transport,
                    generation_state=GenerationState.COMPLETE.value,
                    provider_reason=None,
                    provider_reason_source=None,
                    error_kind=None,
                    classification_basis=None,
                    basis_detail={},
                )
            return _base(
                entry,
                interpreter_version=INTERPRETER_V1,
                completion_map_version=COMPLETION_MAP_LEGACY,
                classifier_version=ERROR_CLASSIFIER_V1,
                observation_event_id=observation_event_id,
                response_body_artifact=response_body_artifact,
                transport_state=transport,
                generation_state=GenerationState.EMPTY.value,
                provider_reason=None,
                provider_reason_source=None,
                error_kind="empty_output",
                classification_basis=ClassificationBasis.GENERATION_STATE.value,
                basis_detail={"generated_text_empty": True},
            )
        error_kind = entry.adapter_error_kind or "provider_error"
        basis = (
            ClassificationBasis.CONFIGURATION.value
            if error_kind == "missing_credentials"
            else ClassificationBasis.ADAPTER_REPORTED.value
        )
        return _base(
            entry,
            interpreter_version=INTERPRETER_V1,
            completion_map_version=COMPLETION_MAP_LEGACY,
            classifier_version=ERROR_CLASSIFIER_V1,
            observation_event_id=observation_event_id,
            response_body_artifact=response_body_artifact,
            transport_state=transport,
            generation_state=GenerationState.UNKNOWN.value,
            provider_reason=None,
            provider_reason_source=None,
            error_kind=error_kind,
            classification_basis=basis,
            basis_detail={"adapter_error_kind": entry.adapter_error_kind},
        )

    if entry.transport_outcome == "no_response":
        # Legacy message rule reproduced verbatim.
        if "timed out" in message.lower() or "timeout" in message.lower():
            error_kind: str | None = "timeout"
        else:
            error_kind = "provider_error"
        return _base(
            entry,
            interpreter_version=INTERPRETER_V1,
            completion_map_version=COMPLETION_MAP_LEGACY,
            classifier_version=ERROR_CLASSIFIER_V1,
            observation_event_id=observation_event_id,
            response_body_artifact=response_body_artifact,
            transport_state=transport,
            generation_state=GenerationState.UNKNOWN.value,
            provider_reason=None,
            provider_reason_source=None,
            error_kind=error_kind,
            classification_basis=ClassificationBasis.EXCEPTION_TYPE.value,
            basis_detail={
                "exception_type": entry.exception_type,
                "rule": "legacy-message-match",
            },
        )

    if entry.transport_outcome == "http_error":
        status = entry.http_status
        if status == 401 or status == 403:
            # v1 keyed status only, including the Cloudflare edge case.
            error_kind = "authentication_error"
        elif status == 429:
            error_kind = "rate_limited"
        else:
            error_kind = "provider_error"
        return _base(
            entry,
            interpreter_version=INTERPRETER_V1,
            completion_map_version=COMPLETION_MAP_LEGACY,
            classifier_version=ERROR_CLASSIFIER_V1,
            observation_event_id=observation_event_id,
            response_body_artifact=response_body_artifact,
            transport_state=transport,
            generation_state=GenerationState.UNKNOWN.value,
            provider_reason=None,
            provider_reason_source=None,
            error_kind=error_kind,
            classification_basis=ClassificationBasis.STATUS_ONLY.value,
            basis_detail={"http_status": status},
        )

    # response_received: reproduce adapter parse outcomes.
    outcome = _parse_outcome(entry)
    if outcome == "malformed":
        return _base(
            entry,
            interpreter_version=INTERPRETER_V1,
            completion_map_version=COMPLETION_MAP_LEGACY,
            classifier_version=ERROR_CLASSIFIER_V1,
            observation_event_id=observation_event_id,
            response_body_artifact=response_body_artifact,
            transport_state=transport,
            generation_state=GenerationState.UNKNOWN.value,
            provider_reason=None,
            provider_reason_source=None,
            error_kind="malformed_response",
            classification_basis=ClassificationBasis.PARSER_FAILURE.value,
            basis_detail={},
        )
    if outcome == "provider_error_field":
        return _base(
            entry,
            interpreter_version=INTERPRETER_V1,
            completion_map_version=COMPLETION_MAP_LEGACY,
            classifier_version=ERROR_CLASSIFIER_V1,
            observation_event_id=observation_event_id,
            response_body_artifact=response_body_artifact,
            transport_state=transport,
            generation_state=GenerationState.UNKNOWN.value,
            provider_reason=None,
            provider_reason_source=None,
            error_kind="provider_error",
            classification_basis=ClassificationBasis.BODY_SIGNATURE.value,
            basis_detail={"signature": "error_field", "source": "parsed_payload"},
        )
    if entry.output_text:
        generation = GenerationState.COMPLETE.value
        error = None
    else:
        generation = GenerationState.EMPTY.value
        error = "empty_output"
    return _base(
        entry,
        interpreter_version=INTERPRETER_V1,
        completion_map_version=COMPLETION_MAP_LEGACY,
        classifier_version=ERROR_CLASSIFIER_V1,
        observation_event_id=observation_event_id,
        response_body_artifact=response_body_artifact,
        transport_state=transport,
        generation_state=generation,
        provider_reason=None,
        provider_reason_source=None,
        error_kind=error,
        classification_basis=(
            None if error is None else ClassificationBasis.GENERATION_STATE.value
        ),
        basis_detail={} if error is None else {"generated_text_empty": True},
    )


# --------------------------------------------------------------------------
# v2: evidence-graded rules.
# --------------------------------------------------------------------------


def interpret_v2(
    entry: InterpretationInput,
    *,
    observation_event_id: str | None = None,
    response_body_artifact: Any = None,
) -> AttemptInterpretation:
    """Current rules: explicit provider reasons drive completion; error labels
    carry how they were derived; nothing is inferred from prose; HTTP 500 is
    never promoted to a route diagnosis."""
    transport = entry.transport_outcome or "unknown"
    corpus, corpus_source = _signature_corpus(entry)

    if entry.transport_outcome is None:
        if entry.adapter_error_kind == "missing_credentials":
            return _base(
                entry,
                interpreter_version=INTERPRETER_V2,
                completion_map_version=COMPLETION_MAP_V1,
                classifier_version=ERROR_CLASSIFIER_V2,
                observation_event_id=observation_event_id,
                response_body_artifact=response_body_artifact,
                transport_state=transport,
                generation_state=GenerationState.UNKNOWN.value,
                provider_reason=None,
                provider_reason_source=None,
                error_kind="missing_credentials",
                classification_basis=ClassificationBasis.CONFIGURATION.value,
                basis_detail={},
            )
        if entry.adapter_status == "succeeded":
            if entry.output_text:
                # A derived decoded payload may still carry an explicit
                # provider reason (legacy run-4 class); map it. With no
                # reason, compatibility accepts the text but completion stays
                # unknown (§24) — never inferred from prose.
                reason, reason_source = _provider_reason(entry.protocol, entry.parsed)
                state = _COMPLETION_REASONS.get(
                    (entry.protocol or "", reason or ""), GenerationState.UNKNOWN.value
                )
                return _base(
                    entry,
                    interpreter_version=INTERPRETER_V2,
                    completion_map_version=COMPLETION_MAP_V1,
                    classifier_version=ERROR_CLASSIFIER_V2,
                    observation_event_id=observation_event_id,
                    response_body_artifact=response_body_artifact,
                    transport_state=transport,
                    generation_state=state,
                    provider_reason=reason,
                    provider_reason_source=reason_source,
                    error_kind=None,
                    classification_basis=None,
                    basis_detail={},
                )
            return _base(
                entry,
                interpreter_version=INTERPRETER_V2,
                completion_map_version=COMPLETION_MAP_V1,
                classifier_version=ERROR_CLASSIFIER_V2,
                observation_event_id=observation_event_id,
                response_body_artifact=response_body_artifact,
                transport_state=transport,
                generation_state=GenerationState.EMPTY.value,
                provider_reason=None,
                provider_reason_source=None,
                error_kind="empty_output",
                classification_basis=ClassificationBasis.GENERATION_STATE.value,
                basis_detail={"generated_text_empty": True},
            )
        # Failed without transport: recognise preserved body signatures even
        # in adapter error text (legacy-derived evidence), else carry the
        # adapter's label forward without strengthening it.
        if SIGNATURE_CLOUDFLARE_1010_MARK in corpus:
            return _v2_signed(
                entry, observation_event_id, response_body_artifact, transport,
                "edge_rejected", SIGNATURE_CLOUDFLARE_1010, corpus_source, entry.http_status,
            )
        if SIGNATURE_MISSING_SESSION_MARK in corpus:
            return _v2_signed(
                entry, observation_event_id, response_body_artifact, transport,
                "invalid_request", SIGNATURE_MISSING_SESSION, corpus_source, entry.http_status,
            )
        return _base(
            entry,
            interpreter_version=INTERPRETER_V2,
            completion_map_version=COMPLETION_MAP_V1,
            classifier_version=ERROR_CLASSIFIER_V2,
            observation_event_id=observation_event_id,
            response_body_artifact=response_body_artifact,
            transport_state=transport,
            generation_state=GenerationState.UNKNOWN.value,
            provider_reason=None,
            provider_reason_source=None,
            error_kind=entry.adapter_error_kind or "provider_error",
            classification_basis=ClassificationBasis.ADAPTER_REPORTED.value,
            basis_detail={"adapter_error_kind": entry.adapter_error_kind},
        )

    if entry.transport_outcome == "no_response":
        if (entry.exception_type or "") in TIMEOUT_EXCEPTIONS:
            error_kind = "timeout"
        else:
            # A reset/refused connection is not a timeout, whatever the text says.
            error_kind = "provider_error"
        return _base(
            entry,
            interpreter_version=INTERPRETER_V2,
            completion_map_version=COMPLETION_MAP_V1,
            classifier_version=ERROR_CLASSIFIER_V2,
            observation_event_id=observation_event_id,
            response_body_artifact=response_body_artifact,
            transport_state=transport,
            generation_state=GenerationState.UNKNOWN.value,
            provider_reason=None,
            provider_reason_source=None,
            error_kind=error_kind,
            classification_basis=ClassificationBasis.EXCEPTION_TYPE.value,
            basis_detail={"exception_type": entry.exception_type},
        )

    if entry.transport_outcome == "http_error":
        status = entry.http_status
        if status == 400:
            if SIGNATURE_MISSING_SESSION_MARK in corpus:
                return _v2_signed(
                    entry, observation_event_id, response_body_artifact, transport,
                    "invalid_request", SIGNATURE_MISSING_SESSION, corpus_source, status,
                )
            return _v2_status(entry, observation_event_id, response_body_artifact,
                              transport, "invalid_request", status)
        if status == 401:
            return _v2_status(entry, observation_event_id, response_body_artifact,
                              transport, "authentication_error", status)
        if status == 403:
            if SIGNATURE_CLOUDFLARE_1010_MARK in corpus:
                return _v2_signed(
                    entry, observation_event_id, response_body_artifact, transport,
                    "edge_rejected", SIGNATURE_CLOUDFLARE_1010, corpus_source, status,
                )
            # A bare 403 proves refusal, not invalid credentials.
            return _v2_status(entry, observation_event_id, response_body_artifact,
                              transport, "provider_error", status)
        if status == 429:
            return _v2_status(entry, observation_event_id, response_body_artifact,
                              transport, "rate_limited", status)
        return _v2_status(entry, observation_event_id, response_body_artifact,
                          transport, "provider_error", status)

    # response_received
    outcome = _parse_outcome(entry)
    if outcome == "malformed":
        return _base(
            entry,
            interpreter_version=INTERPRETER_V2,
            completion_map_version=COMPLETION_MAP_V1,
            classifier_version=ERROR_CLASSIFIER_V2,
            observation_event_id=observation_event_id,
            response_body_artifact=response_body_artifact,
            transport_state=transport,
            generation_state=GenerationState.UNKNOWN.value,
            provider_reason=None,
            provider_reason_source=None,
            error_kind="malformed_response",
            classification_basis=ClassificationBasis.PARSER_FAILURE.value,
            basis_detail={"evidence_source": "response_body"
                          if entry.body_bytes is not None else "adapter_error_text"},
        )
    if outcome == "provider_error_field":
        return _base(
            entry,
            interpreter_version=INTERPRETER_V2,
            completion_map_version=COMPLETION_MAP_V1,
            classifier_version=ERROR_CLASSIFIER_V2,
            observation_event_id=observation_event_id,
            response_body_artifact=response_body_artifact,
            transport_state=transport,
            generation_state=GenerationState.UNKNOWN.value,
            provider_reason=None,
            provider_reason_source=None,
            error_kind="provider_error",
            classification_basis=ClassificationBasis.BODY_SIGNATURE.value,
            basis_detail={"signature": "error_field", "source": "parsed_payload"},
        )
    reason, reason_source = _provider_reason(entry.protocol, entry.parsed)
    if not entry.output_text:
        # Empty output stays empty even when a benign reason is present; an
        # explicit truncated/filtered reason is preserved alongside it, and
        # then the generation fact (not an error label) carries the meaning.
        state = _COMPLETION_REASONS.get(
            (entry.protocol or "", reason or ""), GenerationState.EMPTY.value
        )
        if state == GenerationState.COMPLETE.value:
            state = GenerationState.EMPTY.value
        if state in (GenerationState.TRUNCATED.value, GenerationState.FILTERED.value):
            error_kind = None
            basis = None
            detail: dict[str, Any] = {}
        else:
            error_kind = "empty_output"
            basis = ClassificationBasis.GENERATION_STATE.value
            detail = {"generated_text_empty": True}
        return _base(
            entry,
            interpreter_version=INTERPRETER_V2,
            completion_map_version=COMPLETION_MAP_V1,
            classifier_version=ERROR_CLASSIFIER_V2,
            observation_event_id=observation_event_id,
            response_body_artifact=response_body_artifact,
            transport_state=transport,
            generation_state=state,
            provider_reason=reason,
            provider_reason_source=reason_source,
            error_kind=error_kind,
            classification_basis=basis,
            basis_detail=detail,
        )
    if reason is None:
        return _base(
            entry,
            interpreter_version=INTERPRETER_V2,
            completion_map_version=COMPLETION_MAP_V1,
            classifier_version=ERROR_CLASSIFIER_V2,
            observation_event_id=observation_event_id,
            response_body_artifact=response_body_artifact,
            transport_state=transport,
            generation_state=GenerationState.UNKNOWN.value,
            provider_reason=None,
            provider_reason_source=None,
            error_kind=None,
            classification_basis=None,
            basis_detail={},
        )
    state = _COMPLETION_REASONS.get((entry.protocol or "", reason), GenerationState.UNKNOWN.value)
    return _base(
        entry,
        interpreter_version=INTERPRETER_V2,
        completion_map_version=COMPLETION_MAP_V1,
        classifier_version=ERROR_CLASSIFIER_V2,
        observation_event_id=observation_event_id,
        response_body_artifact=response_body_artifact,
        transport_state=transport,
        generation_state=state,
        provider_reason=reason,
        provider_reason_source=reason_source,
        error_kind=None,
        classification_basis=None,
        basis_detail={},
    )


def _v2_status(
    entry: InterpretationInput,
    observation_event_id: str | None,
    response_body_artifact: Any,
    transport: str,
    error_kind: str,
    status: int | None,
) -> AttemptInterpretation:
    return _base(
        entry,
        interpreter_version=INTERPRETER_V2,
        completion_map_version=COMPLETION_MAP_V1,
        classifier_version=ERROR_CLASSIFIER_V2,
        observation_event_id=observation_event_id,
        response_body_artifact=response_body_artifact,
        transport_state=transport,
        generation_state=GenerationState.UNKNOWN.value,
        provider_reason=None,
        provider_reason_source=None,
        error_kind=error_kind,
        classification_basis=ClassificationBasis.STATUS_ONLY.value,
        basis_detail={"http_status": status},
    )


def _v2_signed(
    entry: InterpretationInput,
    observation_event_id: str | None,
    response_body_artifact: Any,
    transport: str,
    error_kind: str,
    signature: str,
    corpus_source: str,
    status: int | None,
) -> AttemptInterpretation:
    return _base(
        entry,
        interpreter_version=INTERPRETER_V2,
        completion_map_version=COMPLETION_MAP_V1,
        classifier_version=ERROR_CLASSIFIER_V2,
        observation_event_id=observation_event_id,
        response_body_artifact=response_body_artifact,
        transport_state=transport,
        generation_state=GenerationState.UNKNOWN.value,
        provider_reason=None,
        provider_reason_source=None,
        error_kind=error_kind,
        classification_basis=ClassificationBasis.BODY_SIGNATURE.value,
        basis_detail={
            "signature": signature,
            "signature_source": corpus_source,
            "http_status": status,
        },
    )


def interpret_attempt(
    entry: InterpretationInput,
    *,
    version: str = INTERPRETER_V2,
    observation_event_id: str | None = None,
    response_body_artifact: Any = None,
) -> AttemptInterpretation:
    """Deterministic dispatch over preserved evidence. Unknown versions fail
    loudly rather than falling back silently."""
    if version == INTERPRETER_V1:
        return interpret_v1(
            entry,
            observation_event_id=observation_event_id,
            response_body_artifact=response_body_artifact,
        )
    if version == INTERPRETER_V2:
        return interpret_v2(
            entry,
            observation_event_id=observation_event_id,
            response_body_artifact=response_body_artifact,
        )
    raise ValueError(f"unknown interpreter version: {version!r}")


# --------------------------------------------------------------------------
# Policy: interpretations in, decisions out. No new evidence is consulted.
# --------------------------------------------------------------------------


def decide_attempt(
    interpretation: AttemptInterpretation,
    *,
    policy_version: str = ATTEMPT_POLICY_V2,
    has_output_text: bool = True,
) -> tuple[str, str]:
    """Return (decision, reason) with decision in accept | retry | terminal.

    v1 reproduces the pre-11.5b retry rule exactly. v2 retries empty
    generation and transient/rate-limit modes, accepts complete or
    reason-less output, and terminates truncated/filtered/error output
    without claiming a cause beyond the interpretation.
    """
    if policy_version == ATTEMPT_POLICY_V1:
        error_kind = interpretation.error_kind
        if error_kind is None:
            return "accept", "v1: no error recorded"
        if error_kind in POLICY_V1_RETRYABLE:
            return "retry", f"v1: retryable error mode {error_kind}"
        return "terminal", f"v1: non-retryable error mode {error_kind}"
    if policy_version == ATTEMPT_POLICY_V2:
        if interpretation.generation_state == GenerationState.EMPTY.value:
            return "retry", "v2: empty generation is retryable"
        if (interpretation.error_kind or "") in POLICY_V2_RETRYABLE:
            return "retry", f"v2: retryable error mode {interpretation.error_kind}"
        if interpretation.error_kind is None and interpretation.generation_state in (
            GenerationState.COMPLETE.value,
            GenerationState.UNKNOWN.value,
        ):
            if has_output_text:
                return "accept", "v2: output accepted"
            return "terminal", "v2: no error but no output text"
        if interpretation.generation_state in (
            GenerationState.TRUNCATED.value,
            GenerationState.FILTERED.value,
        ):
            return (
                "terminal",
                (
                    f"v2: generation {interpretation.generation_state}, "
                    "not treated as completed cognition"
                ),
            )
        return "terminal", f"v2: non-retryable error mode {interpretation.error_kind}"
    raise ValueError(f"unknown policy version: {policy_version!r}")
