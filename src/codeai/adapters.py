from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class ActionRequest:
    action_id: str
    task_id: str
    capability: str
    instruction: str
    precondition_hash: str | None
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class ActionResult:
    action_id: str
    status: str
    artifacts: tuple[ArtifactRef, ...] = ()
    state_hash: str | None = None
    transcript: str | None = None


class ExecutionAdapter(Protocol):
    """Side-effecting worker such as OpenCode, Codex, or Claude Code."""

    def execute(self, request: ActionRequest) -> ActionResult: ...


@dataclass(frozen=True, slots=True)
class CheckRequest:
    check_id: str
    task_id: str
    claim_ids: tuple[str, ...]
    procedure: str
    environment_hash: str | None = None


@dataclass(frozen=True, slots=True)
class CheckResult:
    check_id: str
    verdict: str
    artifact_ids: tuple[str, ...] = ()
    details: str | None = None


class VerificationAdapter(Protocol):
    def run(self, request: CheckRequest) -> CheckResult: ...
