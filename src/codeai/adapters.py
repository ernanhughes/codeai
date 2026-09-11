from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from .domain import ArtifactRef, CallSpec


@dataclass(frozen=True, slots=True)
class CallResult:
    call_id: str
    raw_output: str
    input_tokens: int = 0
    output_tokens: int = 0
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


class CognitionAdapter(Protocol):
    """Pure cognitive call. It receives a compiled package and produces raw output."""

    def invoke(self, spec: CallSpec) -> CallResult: ...


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


class VerificationAdapter(Protocol):
    def run(self, request: CheckRequest) -> CheckResult: ...


class FakeCognitionAdapter(CognitionAdapter):
    """Deterministic fake model adapter for tests.

    No network, no credentials. Supports predefined responses, simulated
    failure, simulated usage/cost, call recording, and inspection of the
    context package received.
    """

    def __init__(
        self,
        responses: list[str] | dict[str, str] | None = None,
        *,
        fail_call_ids: set[str] | frozenset[str] | None = None,
        error_message: str = "fake model failure",
        input_tokens: int = 10,
        output_tokens: int = 20,
        cost_usd: float | None = 0.001,
        latency_ms: int | None = 5,
        provider: str = "fake-provider",
        model: str = "fake-model",
        model_version: str = "fake-1.0",
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
        self.calls: list[CallSpec] = []
        self.received_contexts: list[object] = []

    def invoke(self, spec: CallSpec) -> CallResult:
        self.calls.append(spec)
        self.received_contexts.append(spec.context)
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
            )
        output = self._resolve_output(spec)
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
        )

    def _resolve_output(self, spec: CallSpec) -> str:
        if isinstance(self._responses, dict):
            return self._responses.get(spec.call_id, f"fake output for {spec.call_id}")
        if isinstance(self._responses, list) and self._responses:
            index = len(self.calls) - 1
            return self._responses[index % len(self._responses)]
        return f"fake output for {spec.call_id}"
