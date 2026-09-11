from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EvidenceClass(StrEnum):
    ASSERTED = "E0_ASSERTED"
    ATTRIBUTED = "E1_ATTRIBUTED"
    SOURCE_CHECKED = "E2_SOURCE_CHECKED"
    REPRODUCED = "E3_REPRODUCED"
    ROBUST = "E4_ROBUST"


class ClaimStatus(StrEnum):
    OPEN = "open"
    CONTESTED = "contested"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    UNVERIFIABLE = "unverifiable"
    SUPERSEDED = "superseded"


class Capability(StrEnum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    VERSION_CONTROL = "version_control"
    REMOTE = "remote"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True, slots=True)
class ActorRef:
    actor_id: str
    kind: str
    provider: str | None = None
    model: str | None = None
    version: str | None = None


@dataclass(frozen=True, slots=True)
class Budget:
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_seconds: int | None = None
    max_turns: int | None = None
    max_human_minutes: int | None = None

    def narrows(self, parent: Budget) -> bool:
        pairs = (
            (self.max_tokens, parent.max_tokens),
            (self.max_cost_usd, parent.max_cost_usd),
            (self.max_seconds, parent.max_seconds),
            (self.max_turns, parent.max_turns),
            (self.max_human_minutes, parent.max_human_minutes),
        )
        return all(p is None or (c is not None and c <= p) for c, p in pairs)


@dataclass(frozen=True, slots=True)
class Authority:
    capabilities: frozenset[Capability] = frozenset()

    def allows(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def narrows(self, parent: Authority) -> bool:
        return self.capabilities.issubset(parent.capabilities)


@dataclass(frozen=True, slots=True)
class Directive:
    directive_id: str
    objective: str
    success_criteria: tuple[str, ...]
    budget: Budget
    authority: Authority
    parent_directive_id: str | None = None

    def validate_child(self, child: Directive) -> None:
        if not child.budget.narrows(self.budget):
            raise ValueError("child directive budget must narrow the parent budget")
        if not child.authority.narrows(self.authority):
            raise ValueError("child directive authority must narrow the parent authority")


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    directive_id: str
    objective: str
    success_criteria: tuple[str, ...]
    budget: Budget
    authority: Authority
    parent_task_id: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    artifact_id: str
    sha256: str
    media_type: str
    uri: str | None = None


@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str
    task_id: str
    statement: str
    source_call_id: str
    evidence_class: EvidenceClass = EvidenceClass.ASSERTED
    status: ClaimStatus = ClaimStatus.OPEN
    scope: str | None = None
    anchors: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Decision:
    decision_id: str
    directive_id: str
    statement: str
    relied_on_claim_ids: tuple[str, ...]
    unresolved_claim_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Seal:
    forbidden_event_ids: frozenset[str] = frozenset()
    forbidden_call_ids: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class ContextPackage:
    package_id: str
    task_id: str
    actor: ActorRef
    prompt: str
    event_ids: tuple[str, ...]
    artifact_ids: tuple[str, ...] = ()
    seal: Seal = field(default_factory=Seal)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CallSpec:
    call_id: str
    task_id: str
    actor: ActorRef
    context: ContextPackage
    idempotency_key: str
    pattern: str = "single"
    parameters: Mapping[str, Any] = field(default_factory=dict)
