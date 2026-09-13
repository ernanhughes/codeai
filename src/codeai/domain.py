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
    ASSERTED = "asserted"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    CONTESTED = "contested"
    UNRESOLVED = "unresolved"
    UNVERIFIABLE = "unverifiable"
    SUPERSEDED = "superseded"
    # Legacy alias: early v0 used OPEN for newly asserted claims.
    OPEN = "open"


class ClaimRelationshipType(StrEnum):
    SAME_AS = "SAME_AS"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    DEPENDS_ON = "DEPENDS_ON"
    DERIVED_FROM = "DERIVED_FROM"
    TESTED_BY = "TESTED_BY"


class Capability(StrEnum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    VERSION_CONTROL = "version_control"
    REMOTE = "remote"
    DESTRUCTIVE = "destructive"


class UsageSource(StrEnum):
    """Provenance of a token-usage measurement.

    MEASURED: the provider reported usage for this attempt.
    ESTIMATED: usage was derived locally (e.g. token counting), not measured.
    UNAVAILABLE: the provider supplied no usage; token fields must be None,
        never zero. Zero means measured zero.
    """

    MEASURED = "measured"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


class AttemptStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TRANSIENT_FAILURE = "transient_failure"


class LogicalCallStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNRESOLVED = "unresolved"


class CostSource(StrEnum):
    """How an attempt/call cost estimate was derived.

    ESTIMATED: computed from measured/estimated usage via a pricing version.
    UNKNOWN: model absent from the pricing table, or usage unavailable.
        Unknown is persisted as None, never 0.
    """

    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


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
    status: ClaimStatus = ClaimStatus.ASSERTED
    scope: str | None = None
    anchors: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    # Epistemic-collaboration extensions (all optional for backward compatibility).
    run_id: str | None = None
    source_artifact_id: str | None = None
    source_span: str | None = None
    conditions: str | None = None


@dataclass(frozen=True, slots=True)
class ClaimRelationship:
    relationship_id: str
    from_claim_id: str
    to_claim_id: str
    relationship_type: ClaimRelationshipType
    run_id: str | None = None
    created_by: str | None = None


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
    forbidden_artifact_ids: frozenset[str] = frozenset()
    forbidden_lineage_ids: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class Variant:
    """Diversity-source metadata: why two calls differ.

    Small tag structure sufficient to distinguish the first experiments:
    same model x N vs heterogeneous models x N, prompt/temperature/context variants.
    """

    model: str | None = None
    provider: str | None = None
    prompt_variant: str | None = None
    temperature: float | None = None
    seed: str | None = None
    context_variant: str | None = None
    tool_variant: str | None = None
    evidence_partition: str | None = None
    experiment: str | None = None
    tags: Mapping[str, str] = field(default_factory=dict)


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
    # Epistemic extensions (optional, backward compatible).
    objective: str | None = None
    claim_ids: tuple[str, ...] = ()
    budget_tokens: int | None = None
    prompt_version: str | None = None
    trace_hash: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CallSpec:
    call_id: str
    task_id: str
    actor: ActorRef
    context: ContextPackage
    idempotency_key: str
    pattern: str = "single"
    parameters: Mapping[str, Any] = field(default_factory=dict)
    # Epistemic-collaboration extensions.
    directive_id: str | None = None
    run_id: str | None = None
    adapter_id: str | None = None
    instruction: str = ""
    prompt_version: str | None = None
    budget: Budget | None = None
    variant: Variant = field(default_factory=Variant)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Experiment grouping (optional; ledger-recoverable analysis keys).
    experiment_id: str | None = None
    arm: str | None = None
    # Recorded-cognition extensions (all optional for backward compatibility).
    # chamber: logical job name (e.g. "deep-review"), resolved through
    # ModelConfig to a concrete provider/model occupant. When None, the
    # logical model key (if any) or actor.model is used as the request label.
    chamber: str | None = None
    logical_model: str | None = None


@dataclass(frozen=True, slots=True)
class Usage:
    """Token usage for one attempt. Unknowns are None, never zero."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    source: UsageSource = UsageSource.UNAVAILABLE


@dataclass(frozen=True, slots=True)
class ResolvedOccupant:
    """Concrete model occupant selected for a chamber/logical name."""

    provider: str | None = None
    model_id: str | None = None
    # Epistemic rule: unknown != absent != zero != inferred. A missing
    # revision is None; never synthesize one from timestamps.
    provider_revision: str | None = None
    revision_source: str | None = None  # e.g. "reported" | "config" | None


@dataclass(frozen=True, slots=True)
class CallManifest:
    """Immutable record of what CodeAI intended/resolved before any attempt.

    The manifest records intent; attempts record observation. Persisted as a
    ``call.manifest`` ledger event before execution.
    """

    call_id: str
    task_id: str
    chamber: str | None = None
    requested_model: str | None = None
    provider: str | None = None
    resolved_model_id: str | None = None
    provider_revision: str | None = None
    revision_source: str | None = None
    pricing_version: str | None = None
    context_package_id: str | None = None
    prompt_hash: str | None = None
    requested_parameters: Mapping[str, Any] = field(default_factory=dict)
    effective_parameters: Mapping[str, Any] = field(default_factory=dict)
    # Request-plan provenance (Stage 11.5c, additive; historically unavailable
    # means unavailable, never an asserted empty fact):
    # - requested_controls: logical declared-control view (parameters over variant).
    # - omitted_unsupported: declared controls the route does not send.
    # - defaulted_parameters: values CodeAI itself supplied.
    # - request_plan_version: preparation contract identifier.
    # - request_body_sha256: identity of the exact body to be sent (content
    #   itself is not duplicated; prompt provenance stays with the context).
    requested_controls: Mapping[str, Any] = field(default_factory=dict)
    omitted_unsupported: tuple[str, ...] = ()
    defaulted_parameters: Mapping[str, Any] = field(default_factory=dict)
    request_plan_version: str | None = None
    request_body_sha256: str | None = None
    created_at: str | None = None


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """One provider attempt beneath a logical call.

    task_id != call_id != attempt_id. attempt_index is 1-based within the call.
    """

    attempt_id: str
    call_id: str
    task_id: str
    attempt_index: int
    started_at: str | None = None
    finished_at: str | None = None
    latency_ms: int | None = None
    provider: str | None = None
    resolved_model_id: str | None = None
    provider_revision: str | None = None
    revision_source: str | None = None
    # Wire dialect that produced this attempt, e.g. "responses" |
    # "chat_completions" | "messages". Gateway stays in `provider`; the two
    # are never conflated.
    protocol: str | None = None
    provider_request_id: str | None = None
    status: str = AttemptStatus.FAILED.value
    error_kind: str | None = None
    error: str | None = None
    usage: Usage = field(default_factory=Usage)
    pricing_version: str | None = None
    cost_usd: float | None = None
    cost_source: str = CostSource.UNKNOWN.value
    currency: str = "USD"
    raw_artifact: ArtifactRef | None = None
    raw_observation_kind: str | None = None
    normalizer_version: str | None = None
    effective_parameters: Mapping[str, Any] = field(default_factory=dict)
    # Compatibility projection provenance (Stage 11.5b): status/error_kind
    # above project the execution-time interpretation + policy decision named
    # here. They are not timeless facts; see attempt.interpreted and
    # attempt.retry_decided. None for pre-11.5b records.
    interpretation_id: str | None = None
    policy_version: str | None = None


@dataclass(frozen=True, slots=True)
class RecordedCall:
    """Logical call with its manifest, attempts, and final interpretation.

    Cognition completion (status) is distinct from task completion: a
    succeeded call only means the cognition operation produced an output.
    """

    call_id: str
    task_id: str
    chamber: str | None
    manifest: CallManifest
    attempts: tuple[AttemptRecord, ...]
    status: str = LogicalCallStatus.UNRESOLVED.value
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_cost_usd: float | None = None


class GenerationState(StrEnum):
    """Provider-evidenced generation completion, independent of transport."""

    COMPLETE = "complete"
    TRUNCATED = "truncated"
    FILTERED = "filtered"
    EMPTY = "empty"
    UNKNOWN = "unknown"


class ClassificationBasis(StrEnum):
    """How an error classification was reached. The basis is representation;
    the label taxonomy it justifies is versioned separately."""

    STATUS_ONLY = "status_only"
    BODY_SIGNATURE = "body_signature"
    EXCEPTION_TYPE = "exception_type"
    PARSER_FAILURE = "parser_failure"
    GENERATION_STATE = "generation_state"
    CONFIGURATION = "configuration"
    ADAPTER_REPORTED = "adapter_reported"


@dataclass(frozen=True, slots=True)
class AttemptInterpretation:
    """One immutable interpretation of one attempt's preserved evidence.

    Observation (attempt.observed / derived input) is evidence; this record
    is a conclusion under a named interpreter version. Same observation +
    different version -> different records; older records are never
    overwritten. Contains no policy answers (no retry/call_success).
    """

    interpretation_id: str
    attempt_id: str
    call_id: str
    task_id: str
    observation_event_id: str | None = None
    response_body_artifact: ArtifactRef | None = None
    interpreter_version: str = ""
    completion_map_version: str | None = None
    classifier_version: str | None = None
    created_at: str | None = None
    transport_state: str = ""
    generation_state: str = GenerationState.UNKNOWN.value
    provider_reason: str | None = None
    provider_reason_source: str | None = None
    error_kind: str | None = None
    classification_basis: str | None = None
    basis_detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AttemptDecision:
    """Policy answer derived from one interpretation. Persisted as
    attempt.retry_decided / call.status_decided; never part of the
    interpretation itself."""

    decision: str = "terminal"  # accept | retry | terminal
    reason: str = ""
    executed: bool = True
