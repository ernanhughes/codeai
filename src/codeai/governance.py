"""Which decision an operation is acting under (the Chapter 28/29 seam).

The composition audit found the scheduler deciding correctly and the runtime
executing whatever it was asked, with nothing connecting the two (gap 4). A
recorded CHECK decision and a WRITE action simply coexisted in the ledger.

The rule this module enforces is deliberately narrower than "every effect needs a
scheduler decision":

    a process decision must never be silently bypassed while the resulting
    operation still appears to belong to that governed process

So there are two legitimate shapes, and they are told apart in the record:

    scheduler-governed     names a decision; the runtime checks it and refuses a
                           mismatch
    external or manual     names no decision; the source is recorded, and it
                           cannot masquerade as governed

An operation that claims a decision must satisfy all four of:

    the decision is recorded
    it belongs to this task
    it selected this operation class
    it still refers to the state it was derived from

The last one is freshness, checked by re-projecting the task's process state and
comparing its digest with the one the decision recorded. Nothing here is a
transaction: between the check and the effect the world may move again. What the
record establishes is that the operation was launched under a decision that still
described the world at launch.

Governance is not authority, and the two run in order:

    decision permits this operation class?      (here)
    authority permits this capability?          (codeai.authority)
    precondition still holds?                   (the operation itself)
    execute

A scheduler decision can never select ACTION -- the policy has no such output, by
design, because effects are gated by authority rather than by the scheduler. So
no action can be scheduler-governed, and an action that claims a decision is
refused. That is the rule working, not a gap in it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from .ledger import Event

if TYPE_CHECKING:
    from .runtime import Runtime

DECISION_BINDING_V1 = "decision-binding-v1"

GOVERNANCE_RECORDED = "operation.governance_recorded"
GOVERNANCE_REFUSED = "operation.governance_refused"

SCHEDULER_SELECTABLE = frozenset({"CALL", "CHECK"})


class GovernanceSource(StrEnum):
    """Where the instruction to perform this operation came from."""

    SCHEDULER = "scheduler"
    HUMAN_OVERRIDE = "human_override"
    EXTERNAL_REQUEST = "external_request"


class GovernanceStatus(StrEnum):
    GOVERNED = "governed"                    # a recorded decision selected this operation
    UNGOVERNED = "ungoverned"                # no decision claimed; the source is recorded
    UNKNOWN_DECISION = "unknown_decision"    # no such decision, or governance claimed without one
    WRONG_TASK = "wrong_task"                # the decision belongs to another task
    WRONG_OPERATION = "wrong_operation"      # the decision selected something else
    STALE_BASIS = "stale_basis"              # the state it was decided on has moved
    NOT_SELECTABLE = "not_selectable"        # the scheduler cannot select this operation at all


@dataclass(frozen=True, slots=True)
class GovernanceStanding:
    status: str
    source: str
    requested_operation: str
    task_id: str | None
    decision_id: str | None = None
    decided_operation: str | None = None
    decision_event_id: str | None = None
    decision_basis_sha256: str | None = None
    current_basis_sha256: str | None = None
    reason: str = ""
    version: str = DECISION_BINDING_V1

    @property
    def permits_execution(self) -> bool:
        return self.status in (GovernanceStatus.GOVERNED, GovernanceStatus.UNGOVERNED)

    @property
    def governed(self) -> bool:
        return self.status == GovernanceStatus.GOVERNED

    def as_payload(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "source": str(self.source),
            "requested_operation": self.requested_operation,
            "task_id": self.task_id,
            "decision_id": self.decision_id,
            "decided_operation": self.decided_operation,
            "decision_event_id": self.decision_event_id,
            "decision_basis_sha256": self.decision_basis_sha256,
            "current_basis_sha256": self.current_basis_sha256,
            "reason": self.reason,
            "version": self.version,
        }


def _digest(snapshot: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def current_basis_sha256(runtime: Runtime, task_id: str | None) -> str | None:
    """The digest of the task's process state as it stands now. Appends nothing."""
    if not task_id:
        return None
    from .process_state import project_process_state, state_snapshot

    try:
        return _digest(state_snapshot(project_process_state(runtime, task_id)))
    except Exception:  # noqa: BLE001 - an unprojectable task cannot establish freshness
        return None


def resolve_governance(
    runtime: Runtime,
    *,
    task_id: str | None,
    operation: str,
    decision_id: str | None,
    source: str = GovernanceSource.EXTERNAL_REQUEST.value,
) -> GovernanceStanding:
    """Decide whether this operation may be performed under the decision it claims."""
    requested = str(operation).upper()
    source = str(source)

    if decision_id is None:
        if source == GovernanceSource.SCHEDULER:
            # Claiming the scheduler's name without its decision is the exact
            # masquerade this seam exists to prevent.
            return GovernanceStanding(
                status=GovernanceStatus.UNKNOWN_DECISION.value,
                source=source,
                requested_operation=requested,
                task_id=task_id,
                reason="scheduler governance was claimed without naming a decision",
            )
        return GovernanceStanding(
            status=GovernanceStatus.UNGOVERNED.value,
            source=source,
            requested_operation=requested,
            task_id=task_id,
            reason=f"no scheduler decision was named; recorded as {source}",
        )

    recorded = [
        event
        for event in runtime.ledger.events_by_kind(("scheduler.decision_recorded",))
        if str(event.payload.get("decision_id")) == str(decision_id)
    ]
    base = {
        "source": GovernanceSource.SCHEDULER.value,
        "requested_operation": requested,
        "task_id": task_id,
        "decision_id": str(decision_id),
    }
    if len(recorded) != 1:
        return GovernanceStanding(
            status=GovernanceStatus.UNKNOWN_DECISION.value,
            reason=(
                f"decision {decision_id} is not recorded"
                if not recorded
                else f"decision {decision_id} has more than one record"
            ),
            **base,
        )

    event = recorded[0]
    payload = event.payload
    decided = str(payload.get("operation") or "")
    base = {
        **base,
        "decided_operation": decided,
        "decision_event_id": event.event_id,
        "decision_basis_sha256": payload.get("process_state_sha256"),
    }

    if str(payload.get("task_id")) != str(task_id):
        return GovernanceStanding(
            status=GovernanceStatus.WRONG_TASK.value,
            reason=(
                f"decision {decision_id} belongs to task {payload.get('task_id')}, "
                f"not {task_id}"
            ),
            **base,
        )
    if requested not in SCHEDULER_SELECTABLE:
        return GovernanceStanding(
            status=GovernanceStatus.NOT_SELECTABLE.value,
            reason=(
                f"the scheduler cannot select {requested}: effects are authorized on their own "
                f"path, so this operation cannot be scheduler-governed"
            ),
            **base,
        )
    if decided != requested:
        return GovernanceStanding(
            status=GovernanceStatus.WRONG_OPERATION.value,
            reason=f"decision {decision_id} selected {decided}, not {requested}",
            **base,
        )

    current = current_basis_sha256(runtime, task_id)
    if current is None or current != payload.get("process_state_sha256"):
        return GovernanceStanding(
            status=GovernanceStatus.STALE_BASIS.value,
            current_basis_sha256=current,
            reason=(
                "the process state has moved since this decision was taken, so it no longer "
                "describes the world it was about"
            ),
            **base,
        )

    return GovernanceStanding(
        status=GovernanceStatus.GOVERNED.value,
        current_basis_sha256=current,
        reason=f"decision {decision_id} selected {decided} for this task on this state",
        **base,
    )


def record_governance(
    runtime: Runtime,
    standing: GovernanceStanding,
    *,
    subject_kind: str,
    subject_id: str,
    actor_id: str,
    causation_id: str | None = None,
) -> Event:
    """Append what this operation was launched under, refusal or not."""
    event = Event.create(
        stream_id=subject_id,
        kind=GOVERNANCE_RECORDED if standing.permits_execution else GOVERNANCE_REFUSED,
        actor_id=actor_id,
        payload={
            "subject_kind": subject_kind,
            "subject_id": subject_id,
            # Name the subject in the key the seal filters on, so a governance
            # event is excludable exactly like the operation it describes. A
            # blind branch must not see a sibling's governance record either
            # (Chapter 23).
            **({"call_id": subject_id} if subject_kind == "call" else {}),
            "lineage_ids": [subject_id],
            **standing.as_payload(),
        },
        causation_id=causation_id,
        correlation_id=standing.task_id,
    )
    runtime.ledger.append(event)
    return event


def govern(
    runtime: Runtime,
    *,
    task_id: str | None,
    operation: str,
    decision_id: str | None,
    source: str,
    subject_kind: str,
    subject_id: str,
    actor_id: str,
    causation_id: str | None = None,
) -> GovernanceStanding:
    """Resolve and record in one step; the caller decides what to do about it."""
    standing = resolve_governance(
        runtime, task_id=task_id, operation=operation, decision_id=decision_id, source=source
    )
    record_governance(
        runtime,
        standing,
        subject_kind=subject_kind,
        subject_id=subject_id,
        actor_id=actor_id,
        causation_id=causation_id,
    )
    return standing


class GovernanceRefused(RuntimeError):
    def __init__(self, standing: GovernanceStanding) -> None:
        self.standing = standing
        super().__init__(f"operation refused by decision binding: {standing.reason}")
