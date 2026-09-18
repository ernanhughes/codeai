"""Explicit task acceptance and a derived completion projection (Stage 14).

A succeeded cognition call is an observation that output exists; it is not
completed work. Authority to accept is resolved from the task own recorded directive chain, not
from an ``Authority`` object the caller passes in (composition audit gap 2). The
caller argument is recorded as a claim and never consulted:

    task.created names a directive
          -> resolve that chain, intersected the same way actions resolve it
          -> ACCEPT in the effective grant, or the acceptance is refused

The directive is read from the task record rather than the request, because a
caller who could name the directive could name a permissive one. A task whose
chain does not grant ACCEPT cannot be accepted by anyone through this API until
the record says otherwise.

This resolves authority from the durable record. It does not authenticate the
acceptor: ``actor_id`` remains attribution, not identity.

A task projects as completed only when:

1. an acceptor whose authority grants ``Capability.ACCEPT`` submits an
   acceptance naming the task, the task's declared criteria (by hash), the
   exact artifact bytes (by SHA-256), the recorded call and final attempt that
   produced them, the interpretation that call's status decision rested on,
   and at least one deterministic check;
2. every reference validates against the ledger: same task, succeeded call,
   complete generation, artifact equal to the call's output and preserved
   intact, each check targeted at those bytes and recorded as PASS, and the
   acceptor is not the actor that produced the artifact;
3. ``task.accepted`` is appended, then ``task.completed`` with
   ``causation_id`` pointing at that acceptance.

The projection trusts only that causal pair. A ``task.completed`` event with
no acceptance behind it completes nothing. Refused acceptances are recorded as
``task.acceptance_rejected`` and change no task state.

Limits, stated rather than solved: actor ids are in-process labels, not
authenticated identities; ledger events are not signed, so the ledger is the
trust boundary; the ledger is single-writer, so no concurrent-acceptance
guarantee is made; a check verifies declared mechanical criteria, not the
truth of the prose.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from .adapters import CheckVerdict
from .authority import AuthorizationStatus, authorize, resolve_directive_authority
from .artifacts import ArtifactCorruptionError
from .domain import Authority, Capability, GenerationState, LogicalCallStatus
from .ledger import Event

if TYPE_CHECKING:
    from .runtime import Runtime

TASK_ACCEPTANCE_V1 = "task-acceptance-v1"
KNOWN_ACCEPTANCE_POLICIES = frozenset({TASK_ACCEPTANCE_V1})

TASK_ACCEPTED = "task.accepted"
TASK_COMPLETED = "task.completed"
TASK_ACCEPTANCE_REJECTED = "task.acceptance_rejected"


class TaskCompletionStatus(StrEnum):
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    UNKNOWN_TASK = "unknown_task"


def criteria_sha256(success_criteria: Iterable[str]) -> str:
    """Identity of a task's declared criteria, as the acceptor saw them."""
    canonical = json.dumps(
        [str(item) for item in success_criteria], ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def artifact_target(artifact_sha256: str) -> str:
    """CheckRequest.target naming the exact bytes a check examined."""
    return f"artifact:sha256:{artifact_sha256}"


class AcceptanceRejected(RuntimeError):
    def __init__(self, reasons: Iterable[str], *, event_id: str | None = None) -> None:
        self.reasons = tuple(reasons)
        self.event_id = event_id
        super().__init__("task acceptance rejected: " + ", ".join(self.reasons))


@dataclass(frozen=True, slots=True)
class AcceptanceRequest:
    acceptance_id: str
    task_id: str
    actor_id: str
    criteria_sha256: str
    artifact_sha256: str
    source_call_id: str
    source_attempt_id: str
    source_interpretation_id: str
    check_ids: tuple[str, ...] = ()
    policy_version: str = TASK_ACCEPTANCE_V1

    def identity(self) -> dict[str, Any]:
        """What makes two acceptances the same decision; the request id is excluded."""
        payload = asdict(self)
        payload.pop("acceptance_id")
        payload["check_ids"] = sorted(self.check_ids)
        return payload


_IDENTITY_FIELDS = tuple(f.name for f in fields(AcceptanceRequest) if f.name != "acceptance_id")


@dataclass(frozen=True, slots=True)
class TaskCompletion:
    task_id: str
    status: str
    basis: str
    acceptance_event_id: str | None = None
    completion_event_id: str | None = None
    artifact_sha256: str | None = None
    source_call_id: str | None = None
    check_ids: tuple[str, ...] = ()
    acceptance_pending_completion: bool = False
    rejected_acceptance_count: int = 0
    completion_event_count: int = 0


def accept_task(
    runtime: Runtime, request: AcceptanceRequest, *, authority: Authority
) -> TaskCompletion:
    """Validate and record an acceptance, then the completion it causes.

    Identical repeats are idempotent (and append a missing completion left by
    an interruption between the two appends); a different acceptance for an
    already-accepted task is refused.
    """
    # Resolved from the record before anything else is considered, and before
    # any prior acceptance is disclosed: a caller with no current grant learns
    # only the refusal.
    decision, directive_id = resolve_acceptance_authority(runtime, request.task_id)
    reasons = _authority_reasons(decision, directive_id)
    if reasons:
        _reject(runtime, request, authority, reasons, decision=decision,
                directive_id=directive_id)

    prior = _task_events(runtime, TASK_ACCEPTED, request.task_id)
    if prior:
        accepted = prior[0]
        if _identity_from_payload(accepted.payload) != request.identity():
            _reject(runtime, request, authority, ("conflicting_acceptance",))
        if not _completions_caused_by(runtime, accepted):
            _append_completion(runtime, accepted)
        return project_task_completion(runtime, request.task_id)

    reasons, check_event_ids = _validate(runtime, request)
    if reasons:
        _reject(runtime, request, authority, reasons, decision=decision,
                directive_id=directive_id)

    accepted = Event.create(
        stream_id=request.task_id,
        kind=TASK_ACCEPTED,
        actor_id=request.actor_id,
        payload={
            **request.identity(),
            "acceptance_id": request.acceptance_id,
            "directive_id": directive_id,
            "authority_basis": decision.basis_payload(),
            "caller_claimed_capabilities": _claimed(authority),
            "check_completed_event_ids": check_event_ids,
        },
        correlation_id=request.task_id,
    )
    runtime.ledger.append(accepted)
    _append_completion(runtime, accepted)
    return project_task_completion(runtime, request.task_id)


def project_task_completion(runtime: Runtime, task_id: str) -> TaskCompletion:
    """Derive task completion from the ledger; appends nothing."""
    created = _task_events(runtime, "task.created", task_id)
    accepted = _task_events(runtime, TASK_ACCEPTED, task_id)
    completed = _task_events(runtime, TASK_COMPLETED, task_id)
    counts = {
        "rejected_acceptance_count": len(_task_events(runtime, TASK_ACCEPTANCE_REJECTED, task_id)),
        "completion_event_count": len(completed),
    }
    if not created:
        return TaskCompletion(
            task_id, TaskCompletionStatus.UNKNOWN_TASK.value, "no task.created event", **counts
        )
    for acceptance in accepted:
        caused = [event for event in completed if _completes(event, acceptance)]
        if caused:
            return TaskCompletion(
                task_id,
                TaskCompletionStatus.COMPLETED.value,
                "task.completed caused by a validated task.accepted",
                acceptance_event_id=acceptance.event_id,
                completion_event_id=caused[0].event_id,
                artifact_sha256=str(acceptance.payload.get("artifact_sha256")),
                source_call_id=str(acceptance.payload.get("source_call_id")),
                check_ids=tuple(str(c) for c in acceptance.payload.get("check_ids") or ()),
                **counts,
            )
    if accepted:
        basis = "task.accepted recorded; task.completed not yet appended"
    elif completed:
        basis = "task.completed present without a causing task.accepted"
    else:
        basis = "no acceptance recorded"
    return TaskCompletion(
        task_id,
        TaskCompletionStatus.INCOMPLETE.value,
        basis,
        acceptance_pending_completion=bool(accepted),
        **counts,
    )


# ---------------- validation ----------------


def resolve_acceptance_authority(runtime: Runtime, task_id: str):
    """Resolve ACCEPT for a task from its own recorded directive. Appends nothing.

    Returns (decision, directive_id). The directive comes from ``task.created``:
    a caller who could name it could name a permissive one.
    """
    events = runtime.ledger.read_all()
    created = _task_events(runtime, "task.created", task_id)
    directive_id = None
    if len(created) == 1:
        raw = created[0].payload.get("directive_id")
        directive_id = str(raw) if raw else None
    standing = resolve_directive_authority(events, directive_id)
    return authorize(standing, Capability.ACCEPT), directive_id


def _authority_reasons(decision, directive_id: str | None) -> list[str]:
    """Why this task may not be accepted, in the vocabulary of the record."""
    if decision.granted:
        return []
    if directive_id is None:
        return ["acceptance_authority_unresolved:task_names_no_directive"]
    if decision.status == AuthorizationStatus.DENIED:
        return [f"acceptance_not_granted:{directive_id}"]
    return [f"acceptance_authority_unresolved:{decision.status}:{directive_id}"]


def _validate(runtime: Runtime, request: AcceptanceRequest) -> tuple[list[str], list[str]]:
    reasons: list[str] = []
    if request.policy_version not in KNOWN_ACCEPTANCE_POLICIES:
        reasons.append("unknown_policy")
    if not request.actor_id:
        reasons.append("missing_actor")

    definitions = {
        criteria_sha256(event.payload.get("success_criteria") or ())
        for event in _task_events(runtime, "task.created", request.task_id)
    }
    if not definitions:
        reasons.append("unknown_task")
    elif len(definitions) > 1:
        reasons.append("ambiguous_task_definition")
    elif request.criteria_sha256 not in definitions:
        reasons.append("criteria_mismatch")

    reasons.extend(_validate_source(runtime, request))
    reasons.extend(_validate_artifact(runtime, request))
    check_reasons, check_event_ids = _validate_checks(runtime, request)
    reasons.extend(check_reasons)
    return reasons, check_event_ids


def _validate_source(runtime: Runtime, request: AcceptanceRequest) -> list[str]:
    # Built from call.status_decided (stream = call id), the attempt records
    # and the preserved attempt envelope. call.completed is deliberately not
    # used: ledgers written before the Stage 14 fix carry an empty call_id there.
    decided = [
        event
        for event in runtime.ledger.events_by_kind(("call.status_decided",))
        if event.stream_id == request.source_call_id
    ]
    recorded = runtime.get_recorded_call(request.source_call_id)
    if not decided or recorded is None or recorded.call_id != request.source_call_id:
        return ["source_call_not_found"]
    # The adopted status is the latest of the execution-time decision and any
    # later reinterpretation of the preserved observations (Stage 17).
    adopted = [
        event
        for event in runtime.ledger.events_by_kind(("call.status_decided", "call.reinterpreted"))
        if event.stream_id == request.source_call_id
    ]
    status = adopted[-1]

    reasons: list[str] = []
    if recorded.task_id != request.task_id or str(status.payload.get("task_id")) != request.task_id:
        reasons.append("source_call_wrong_task")
    if status.payload.get("status") != LogicalCallStatus.SUCCEEDED.value:
        reasons.append("source_call_not_succeeded")
    named = next((a for a in recorded.attempts if a.attempt_id == request.source_attempt_id), None)
    if named is None:
        reasons.append("source_attempt_not_in_call")
    elif recorded.attempts[-1].attempt_id != request.source_attempt_id:
        reasons.append("source_attempt_not_final")
    basis = [str(item) for item in status.payload.get("interpretation_ids") or ()]
    if not basis or basis[-1] != request.source_interpretation_id:
        reasons.append("interpretation_not_decision_basis")
    matches = [
        interpretation
        for interpretation in runtime.interpretations_for_attempt(request.source_attempt_id)
        if interpretation.interpretation_id == request.source_interpretation_id
    ]
    if not matches:
        reasons.append("interpretation_not_found")
    else:
        if matches[0].call_id != request.source_call_id:
            reasons.append("interpretation_wrong_call")
        if matches[0].generation_state != GenerationState.COMPLETE.value:
            reasons.append("generation_not_complete")
    if request.actor_id == decided[-1].actor_id:  # the producing actor, not a reinterpreter
        reasons.append("self_acceptance")
    if named is not None:
        output = _preserved_output_text(runtime, named.raw_artifact)
        if output is None:
            reasons.append("source_output_not_preserved")
        elif text_sha256(output) != request.artifact_sha256:
            reasons.append("artifact_not_source_output")
    return reasons


def _preserved_output_text(runtime: Runtime, envelope_ref: Any) -> str | None:
    """Output text from an attempt's preserved envelope, read with integrity checks."""
    if envelope_ref is None or runtime.artifact_store is None:
        return None
    try:
        envelope = json.loads(runtime.artifact_store.read_text(envelope_ref.artifact_id))
    except (FileNotFoundError, ArtifactCorruptionError, ValueError):
        return None
    if not isinstance(envelope, dict):
        return None
    return str(envelope.get("output_text") or "")


def _validate_artifact(runtime: Runtime, request: AcceptanceRequest) -> list[str]:
    store = runtime.artifact_store
    if store is None:
        return ["artifact_store_unavailable"]
    try:
        store.read_bytes(request.artifact_sha256)
    except FileNotFoundError:
        return ["artifact_not_preserved"]
    except ArtifactCorruptionError:
        return ["artifact_corrupted"]
    return []


def _validate_checks(
    runtime: Runtime, request: AcceptanceRequest
) -> tuple[list[str], list[str]]:
    if not request.check_ids:
        return ["missing_check"], []
    requested = runtime.ledger.events_by_kind(("check.requested",))
    finished = runtime.ledger.events_by_kind(("check.completed",))
    passing = str(getattr(CheckVerdict.PASS, "value", CheckVerdict.PASS))
    reasons: list[str] = []
    event_ids: list[str] = []
    for check_id in sorted(set(request.check_ids)):
        req = [event for event in requested if event.stream_id == check_id]
        done = [event for event in finished if event.stream_id == check_id]
        if not req or not done:
            reasons.append(f"check_not_found:{check_id}")
            continue
        if len(req) > 1 or len(done) > 1:
            reasons.append(f"check_ambiguous:{check_id}")
            continue
        if done[0].causation_id != req[0].event_id:
            reasons.append(f"check_unlinked:{check_id}")
        if str(req[0].payload.get("task_id")) != request.task_id:
            reasons.append(f"check_wrong_task:{check_id}")
        binding = done[0].payload.get("artifact_binding")
        if binding is None:
            # A record written before artifact binding existed cannot establish
            # which bytes the check examined, and is not read as if it could.
            reasons.append(f"check_artifact_unestablished:{check_id}")
        elif (
            binding.get("status") != "bound"
            or binding.get("resolved_artifact_sha256") != request.artifact_sha256
        ):
            reasons.append(f"check_wrong_artifact:{check_id}")
        verdict = str(done[0].payload.get("verdict"))
        if verdict != passing:
            reasons.append(f"check_not_passed:{check_id}:{verdict}")
        event_ids.append(done[0].event_id)
    return reasons, event_ids


# ---------------- ledger helpers ----------------


def _task_events(runtime: Runtime, kind: str, task_id: str) -> list[Event]:
    return [
        event
        for event in runtime.ledger.events_by_kind((kind,))
        if str(event.payload.get("task_id")) == task_id
    ]


def _identity_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    identity = {name: payload.get(name) for name in _IDENTITY_FIELDS}
    identity["check_ids"] = sorted(str(item) for item in identity["check_ids"] or ())
    return identity


def _completes(completion: Event, acceptance: Event) -> bool:
    return (
        completion.causation_id == acceptance.event_id
        and completion.payload.get("acceptance_event_id") == acceptance.event_id
        and completion.payload.get("artifact_sha256") == acceptance.payload.get("artifact_sha256")
    )


def _completions_caused_by(runtime: Runtime, acceptance: Event) -> list[Event]:
    task_id = str(acceptance.payload.get("task_id"))
    return [
        event
        for event in _task_events(runtime, TASK_COMPLETED, task_id)
        if _completes(event, acceptance)
    ]


def _append_completion(runtime: Runtime, acceptance: Event) -> Event:
    event = Event.create(
        stream_id=str(acceptance.payload.get("task_id")),
        kind=TASK_COMPLETED,
        actor_id="runtime",
        payload={
            "task_id": acceptance.payload.get("task_id"),
            "acceptance_id": acceptance.payload.get("acceptance_id"),
            "acceptance_event_id": acceptance.event_id,
            "artifact_sha256": acceptance.payload.get("artifact_sha256"),
            "policy_version": acceptance.payload.get("policy_version"),
        },
        causation_id=acceptance.event_id,
        correlation_id=str(acceptance.payload.get("task_id")),
    )
    runtime.ledger.append(event)
    return event


def _claimed(authority: Authority | None) -> list[str]:
    """What the caller passed. Recorded, never consulted."""
    return sorted(str(c) for c in (authority.capabilities if authority else ()))


def _reject(
    runtime: Runtime,
    request: AcceptanceRequest,
    authority: Authority | None,
    reasons: Iterable[str],
    *,
    decision=None,
    directive_id: str | None = None,
) -> None:
    reasons = tuple(reasons)
    event = Event.create(
        stream_id=request.task_id,
        kind=TASK_ACCEPTANCE_REJECTED,
        actor_id=request.actor_id or "unknown",
        payload={
            **request.identity(),
            "acceptance_id": request.acceptance_id,
            "reasons": list(reasons),
            "directive_id": directive_id,
            "authority_basis": decision.basis_payload() if decision is not None else None,
            "caller_claimed_capabilities": _claimed(authority),
        },
        correlation_id=request.task_id,
    )
    runtime.ledger.append(event)
    raise AcceptanceRejected(reasons, event_id=event.event_id)
