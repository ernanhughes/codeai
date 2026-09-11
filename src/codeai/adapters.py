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
