from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from .domain import ArtifactRef, CallSpec

# Version of the observation -> canonical-result interpretation applied by
# CodeAI's own normalization (usage mapping, cost derivation, status mapping).
# Stored on every attempt so future normalizer fixes need not rewrite history.
NORMALIZER_VERSION = "codeai-normalizer-v1"

# What "raw" means for observations preserved through this layer: the
# sanitized adapter-level observation (decoded text + reported usage +
# provider call identity), NOT exact HTTP bytes/headers. Provider adapters do
# not expose transport bytes, so CodeAI records the strongest observation
# actually available and never persists credentials.
RAW_OBSERVATION_KIND = "adapter-sanitized-observation-v1"

# Tokenized credential hints matched against key parts split on _ - . and
# case. Deliberately token-based (not substring): "max_tokens" splits to
# ["max", "tokens"] and is a legitimate generation control that must be kept,
# while "api_key" -> ["api", "key"] and "Authorization" are dropped.
_SECRET_KEY_TOKENS = frozenset(
    {
        "key",
        "keys",
        "token",
        "secret",
        "secrets",
        "auth",
        "authorization",
        "password",
        "passwd",
        "credential",
        "credentials",
        "bearer",
    }
)


def _looks_credential_like(key: str) -> bool:
    import re as _re

    parts = [part for part in _re.split(r"[_\-.]+", key.lower()) if part]
    return any(part in _SECRET_KEY_TOKENS for part in parts)


def is_credential_key(key: str) -> bool:
    """Public predicate: does this key name look credential-like? Deterministic."""
    return _looks_credential_like(key)


def sanitize_effective_params(params: Mapping[str, object]) -> dict[str, object]:
    """Return a sanitized copy of effective provider controls.

    Drops anything whose key looks credential-like. Deterministic; no LLM.
    """
    return {
        str(key): value for key, value in params.items() if not _looks_credential_like(str(key))
    }


class TransportOutcome(StrEnum):
    """What the HTTP transport itself produced, before any interpretation."""

    RESPONSE_RECEIVED = "response_received"
    HTTP_ERROR = "http_error"
    NO_RESPONSE = "no_response"


@dataclass(frozen=True, slots=True)
class TransportObservation:
    """Ephemeral transport evidence carried from adapter to runtime.

    Provider-neutral: exact response bytes (or the no-response fact), never
    interpretations (no error_kind, usage, cost, completion). The runtime
    persists the bytes to the content-addressed artifact store and the
    metadata to an attempt.observed event; the bytes must never be serialized
    into CallResult payloads, ledger events, or exports.
    """

    outcome: str = TransportOutcome.NO_RESPONSE.value
    status_code: int | None = None
    body: bytes | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    content_type: str | None = None
    endpoint: str | None = None
    exception_type: str | None = None
    observed_at: str | None = None


@dataclass(frozen=True, slots=True)
class CallResult:
    call_id: str
    raw_output: str
    # Legacy integer fields kept for backward compatibility. Going forward,
    # None means unknown/not-reported; usage_source disambiguates. Old rows
    # with 0 + source "measured" keep their historical meaning.
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    cost_usd: float | None = None
    latency_ms: int | None = None
    # Epistemic-collaboration attribution (all optional for backward compatibility).
    provider: str | None = None
    model: str | None = None
    model_version: str | None = None
    fingerprint: str | None = None
    provider_call_id: str | None = None
    request_id: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    status: str = "succeeded"
    error: str | None = None
    raw_artifact: ArtifactRef | None = None
    # Recorded-cognition provenance (all optional for backward compatibility).
    # usage_source: "measured" | "estimated" | "unavailable".
    usage_source: str = "measured"
    # error_kind: machine-readable classification, e.g. "transient_failure" |
    # "empty_output" | "provider_error" | "missing_credentials" | None.
    error_kind: str | None = None
    pricing_version: str | None = None
    # cost_source: "estimated" | "unknown". Unknown cost stays None, never 0.
    cost_source: str | None = None
    normalizer_version: str | None = None
    attempt_id: str | None = None
    attempt_index: int | None = None
    effective_parameters: Mapping[str, object] = field(default_factory=dict)
    # Gateway/protocol separation: which wire dialect produced this result,
    # e.g. "responses" | "chat_completions" | "messages". None when the
    # adapter has no protocol distinction (e.g. fakes).
    protocol: str | None = None
    # Raw-observation provenance: how to read raw_payload, e.g.
    # "decoded_json" when the adapter received decoded JSON (never raw HTTP
    # bytes — adapters do not expose transport bytes).
    raw_observation_kind: str | None = None
    # Sanitized decoded provider payload preserved into the attempt's raw
    # artifact by the runtime. Must never contain credentials.
    raw_payload: Mapping[str, object] = field(default_factory=dict)
    # Ephemeral transport evidence for the runtime only. Excluded from every
    # ledger/export serialization: the artifact store carries the bytes, the
    # ledger carries a reference. Never put response bytes here by value into
    # a persisted payload.
    transport: TransportObservation | None = None
    # Replay marker: True when this result reuses a prior completed effect
    # without a new provider invocation. Operational provenance only, never
    # persisted into call.completed (history is not rewritten).
    replayed: bool = False


class CognitionAdapter(Protocol):
    """Pure cognitive call. It receives a compiled package and produces raw output."""

    def invoke(self, spec: CallSpec) -> CallResult: ...


class RequestPlanError(ValueError):
    """A cognition request cannot be prepared exactly as specified.

    Raised before any provider effect. Carries the offending control name,
    never a secret value.
    """


class UnknownControlError(RequestPlanError):
    """Caller supplied a control the codec does not declare. Reject, don't guess."""

    def __init__(self, control: str) -> None:
        super().__init__(
            f"unknown cognition control: {control!r}; "
            "only codec-declared controls may affect a provider request"
        )
        self.control = control


class InvalidControlError(RequestPlanError):
    """A declared control carries a value the codec cannot place on the wire."""

    def __init__(self, control: str, reason: str) -> None:
        super().__init__(f"invalid value for cognition control {control!r}: {reason}")
        self.control = control


@dataclass(frozen=True, slots=True)
class PreparedCognitionRequest:
    """One prepared provider request: the single fact that is both recorded
    and sent. The outbound HTTP body is produced from this representation,
    never rebuilt independently.

    - requested_controls: logical control names -> merged caller values
      (spec.parameters over variant), including known-but-unsupported ones.
    - effective_controls: wire-shaped controls actually present in the body.
    - omitted_unsupported: declared controls the route does not send.
    - defaulted_controls: values CodeAI itself supplied (provider omissions
      are not invented here).
    - routing: public non-secret routing metadata (e.g. session id).
    - public_headers: exact non-secret headers to send (credentials are
      applied structurally by transport and never appear here).
    - body: the semantic JSON object to submit. body_sha256 hashes canonical JSON,
      not outbound HTTP bytes;
      prompt content stays referenced via context provenance, not duplicated.
    """

    gateway: str
    protocol: str
    endpoint: str
    model: str
    body: Mapping[str, object] = field(default_factory=dict)
    requested_controls: Mapping[str, object] = field(default_factory=dict)
    effective_controls: Mapping[str, object] = field(default_factory=dict)
    omitted_unsupported: tuple[str, ...] = ()
    defaulted_controls: Mapping[str, object] = field(default_factory=dict)
    routing: Mapping[str, object] = field(default_factory=dict)
    public_headers: Mapping[str, str] = field(default_factory=dict)
    body_sha256: str | None = None
    plan_version: str = ""

    def recorded_effective(self) -> dict[str, object]:
        """Durable effective view: route identity plus the wire controls.

        Single definition used both for persistence (manifest/attempt) and
        for the result returned by send(), so recorded and sent provenance
        cannot drift apart.
        """
        effective: dict[str, object] = {
            "gateway": self.gateway,
            "protocol": self.protocol,
            "endpoint": self.endpoint,
            "model": self.model,
        }
        if "gateway_plan" in self.routing:
            effective["gateway_plan"] = self.routing["gateway_plan"]
        session_id = self.routing.get("session_id")
        if session_id is not None:
            effective["session_id"] = session_id
        effective.update(dict(self.effective_controls))
        return effective


class ActionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"


@dataclass(frozen=True, slots=True)
class ActionRequest:
    action_id: str
    task_id: str
    capability: str
    instruction: str
    precondition_hash: str | None
    idempotency_key: str
    directive_id: str | None = None
    actor_id: str = "runtime"
    adapter: str = "opencode"
    payload: Mapping[str, object] = field(default_factory=dict)
    # Authority seam: who requested/authorized vs who executes vs via what transport.
    # - requested_by: authorizing principal (e.g. "human")
    # - actor_id:    executing actor (e.g. "opencode")
    # - adapter/adapter_id: transport that performed it (e.g. "opencode")
    requested_by: str | None = None
    adapter_id: str | None = None
    # Optional: the Chapter 18 decision this operation claims as its
    # justification. Citing one invites the runtime to check that the claims it
    # rested on still stand; citing none claims no evidentiary basis at all.
    decision_id: str | None = None

    def effective_requester(self) -> str:
        return self.requested_by or self.actor_id

    def effective_adapter_id(self) -> str:
        return self.adapter_id or self.adapter


@dataclass(frozen=True, slots=True)
class ActionResult:
    action_id: str
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    artifacts: tuple[ArtifactRef, ...] = ()
    state_hash: str | None = None
    resulting_state_hash: str | None = None
    transcript: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    exit_code: int | None = None
    cost_usd: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    reused_from_action_id: str | None = None
    # Runtime-owned reading from state_resolver after execution (or refusal).
    # Never taken from the adapter. None means no runtime reading is available.
    # On reuse this remains the original reading, not a fresh observation.
    # resulting_state_hash retains its legacy adapter-first/fallback semantics.
    observed_state_hash: str | None = None


class ExecutionAdapter(Protocol):
    """Side-effecting worker such as OpenCode, Codex, or Claude Code."""

    def execute(self, request: ActionRequest) -> ActionResult: ...


class CheckVerdict(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class CheckRequest:
    check_id: str
    task_id: str
    claim_ids: tuple[str, ...] = ()
    directive_id: str | None = None
    command: tuple[str, ...] = ()
    cwd: str = "."
    timeout_seconds: float = 300.0
    environment: Mapping[str, str] = field(default_factory=dict)
    target: str | None = None
    target_state_hash: str | None = None
    environment_hash: str | None = None
    # How outcomes map to verdicts, declared before execution so the mapping is
    # recorded rather than inferred from what came back (see codeai.verification).
    verdict_policy: Mapping[str, object] | None = None
    # Runtime-owned: where the runtime wrote the verified artifact bytes for this
    # check. A caller setting it is overwritten; ARTIFACT_PLACEHOLDER in the
    # command is substituted with this path.
    materialized_artifact_path: str | None = None


@dataclass(frozen=True, slots=True)
class CheckResult:
    check_id: str
    verdict: str
    started_at: str | None = None
    completed_at: str | None = None
    exit_code: int | None = None
    artifacts: tuple[ArtifactRef, ...] = ()
    stdout: str | None = None
    stderr: str | None = None
    details: str | None = None
    error: str | None = None
    # Why a legitimate check could not settle the question. Required when the
    # verdict is INCONCLUSIVE: an unexplained "I do not know" is not evidence.
    inconclusive_reason: str | None = None
    # Which declared mapping the verifier applied, when it applied one.
    verdict_policy_id: str | None = None
    # Runtime-owned reading before verifier invocation, when target_state_hash
    # was requested. None means no reading was obtained (or none requested).
    # This is not a reading of what the verifier actually consumed, nor a lock.
    observed_target_state_hash: str | None = None
    # Runtime-owned: what the runtime could establish about the checked subject.
    # A verifier cannot set this; codeai.verification overwrites it.
    binding_status: str | None = None
    # Runtime-owned: what the runtime could establish about the bytes the check
    # was given (see ArtifactBinding).
    artifact_binding_status: str | None = None


class VerificationAdapter(Protocol):
    def run(self, request: CheckRequest) -> CheckResult: ...


class FakeCognitionAdapter(CognitionAdapter):
    """Deterministic fake model adapter for tests.

    No network, no credentials. Supports predefined responses, simulated
    failure, simulated usage/cost, call recording, and inspection of the
    context package received.

    Recorded-cognition extensions (all optional, backward compatible):
    - ``behaviors``: per-invocation script. Each entry may hold
      ``output`` | ``error`` | ``status`` | ``error_kind`` | ``input_tokens`` |
      ``output_tokens`` | ``usage_source`` | ``provider_call_id`` |
      ``model_version``. Entry ``i`` drives invocation ``i``; when the script
      is exhausted the adapter falls back to the legacy response resolution.
      ``input_tokens``/``output_tokens`` may be None to mean unknown.
    - ``unsupported_params``: requested parameter names the fake provider
      "does not support"; they are omitted from ``effective_request()`` so
      requested != effective stays inspectable.
    - ``usage_source``: default provenance label for non-scripted calls.
    - ``provider_request_ids``: cycled provider request identities.
    ``model_version`` may be None to represent an unknown revision honestly.
    """

    def __init__(
        self,
        responses: list[str] | dict[str, str] | None = None,
        *,
        fail_call_ids: set[str] | frozenset[str] | None = None,
        error_message: str = "fake model failure",
        input_tokens: int | None = 10,
        output_tokens: int | None = 20,
        cost_usd: float | None = 0.001,
        latency_ms: int | None = 5,
        provider: str = "fake-provider",
        model: str = "fake-model",
        model_version: str | None = "fake-1.0",
        behaviors: list[dict[str, object]] | None = None,
        unsupported_params: tuple[str, ...] | list[str] = (),
        usage_source: str = "measured",
        provider_request_ids: list[str] | None = None,
        error_kind: str | None = None,
    ) -> None:
        self._responses = responses
        self._fail_call_ids = frozenset(fail_call_ids or ())
        self._error_message = error_message
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._cost_usd = cost_usd
        self._latency_ms = latency_ms
        self._provider = provider
        self._model = model
        self._model_version = model_version
        self._behaviors = list(behaviors) if behaviors else []
        self._unsupported_params = frozenset(unsupported_params)
        self._usage_source = usage_source
        self._provider_request_ids = list(provider_request_ids) if provider_request_ids else []
        self._error_kind = error_kind
        self.calls: list[CallSpec] = []
        self.received_contexts: list[object] = []

    def effective_request(self, spec: CallSpec) -> dict[str, object]:
        """Sanitized effective controls this fake would "send".

        Omits ``unsupported_params`` and credential-like keys. Never includes
        secrets: the fake holds none.
        """
        effective = {
            str(key): value
            for key, value in dict(spec.parameters).items()
            if key not in self._unsupported_params
        }
        effective.setdefault("model", self._model)
        return sanitize_effective_params(effective)

    def invoke(self, spec: CallSpec) -> CallResult:
        self.calls.append(spec)
        self.received_contexts.append(spec.context)
        invocation_index = len(self.calls) - 1
        if invocation_index < len(self._behaviors):
            return self._result_from_behavior(
                spec, self._behaviors[invocation_index], invocation_index
            )
        if spec.call_id in self._fail_call_ids:
            return CallResult(
                call_id=spec.call_id,
                raw_output="",
                input_tokens=self._input_tokens,
                output_tokens=0,
                cost_usd=self._cost_usd,
                latency_ms=self._latency_ms,
                provider=self._provider,
                model=self._model,
                model_version=self._model_version,
                status="failed",
                error=self._error_message,
                error_kind=self._error_kind or "provider_error",
                usage_source=self._usage_source,
                normalizer_version=NORMALIZER_VERSION,
                effective_parameters=self.effective_request(spec),
            )
        output = self._resolve_output(spec)
        provider_call_id = (
            self._provider_request_ids[invocation_index % len(self._provider_request_ids)]
            if self._provider_request_ids
            else None
        )
        return CallResult(
            call_id=spec.call_id,
            raw_output=output,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            cost_usd=self._cost_usd,
            latency_ms=self._latency_ms,
            provider=self._provider,
            model=self._model,
            model_version=self._model_version,
            provider_call_id=provider_call_id,
            status="succeeded",
            usage_source=self._usage_source,
            normalizer_version=NORMALIZER_VERSION,
            effective_parameters=self.effective_request(spec),
        )

    def _result_from_behavior(
        self, spec: CallSpec, behavior: dict[str, object], invocation_index: int
    ) -> CallResult:
        status = str(behavior.get("status", "failed" if behavior.get("error") else "succeeded"))
        provider_call_id = behavior.get("provider_call_id")
        if provider_call_id is None and self._provider_request_ids:
            provider_call_id = self._provider_request_ids[
                invocation_index % len(self._provider_request_ids)
            ]
        in_tokens = behavior.get("input_tokens", self._input_tokens)
        out_tokens = behavior.get("output_tokens", self._output_tokens)
        return CallResult(
            call_id=spec.call_id,
            raw_output=str(behavior.get("output", "")),
            input_tokens=in_tokens
            if in_tokens is None or isinstance(in_tokens, int)
            else int(in_tokens),  # type: ignore[arg-type]
            output_tokens=out_tokens
            if out_tokens is None or isinstance(out_tokens, int)
            else int(out_tokens),  # type: ignore[arg-type]
            cost_usd=behavior.get("cost_usd", self._cost_usd),  # type: ignore[arg-type]
            latency_ms=self._latency_ms,
            provider=self._provider,
            model=self._model,
            model_version=behavior.get("model_version", self._model_version),  # type: ignore[arg-type]
            provider_call_id=str(provider_call_id) if provider_call_id is not None else None,
            status=status,
            error=str(behavior["error"]) if behavior.get("error") is not None else None,
            error_kind=str(
                behavior.get(
                    "error_kind",
                    self._error_kind or ("transient_failure" if status != "succeeded" else None),
                )
            )
            if status != "succeeded" or behavior.get("error_kind")
            else None,
            usage_source=str(behavior.get("usage_source", self._usage_source)),
            normalizer_version=NORMALIZER_VERSION,
            effective_parameters=self.effective_request(spec),
        )

    def _resolve_output(self, spec: CallSpec) -> str:
        if isinstance(self._responses, dict):
            return self._responses.get(spec.call_id, f"fake output for {spec.call_id}")
        if isinstance(self._responses, list) and self._responses:
            index = len(self.calls) - 1
            return self._responses[index % len(self._responses)]
        return f"fake output for {spec.call_id}"
