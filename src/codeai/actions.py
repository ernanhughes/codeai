"""Effect recovery for actions (the Chapter 19/22/29 seam).

Chapter 16 gave model calls a recovery model: intent is recorded before the
effect, so a later process can tell "no effect was possible" from "the effect
may have happened". Actions had no equivalent. A request was recorded, the
adapter was invoked, and only the completion was written, so an interrupted
action left a record that could not answer the one question recovery needs.

This module gives actions the same ordering argument, and keeps two things
apart that the single ``ActionStatus`` enum conflates:

    result status   what the operation reported: SUCCEEDED, FAILED, DENIED
    effect state    what the record establishes about the world:
                    NONE, UNKNOWN, OBSERVED

``FAILED`` is a result status. It is not an answer to *could the effect have
happened?* — an adapter can fail after writing. The ordering that makes the
distinction decidable is:

    action.requested
          -> authority and precondition checks
    action.execution_started      (committed before adapter.execute)
          -> the adapter acts
    action.completed              (result plus the runtime's own observation)

From that, for actions recorded by a runtime that emits ``execution_started``:

    requested, no execution_started      effect NONE      safe to start
    execution_started, no completion     effect UNKNOWN   reconcile, never retry
    completed SUCCEEDED                  effect OBSERVED  terminal
    completed DENIED                     effect NONE      terminal
    completed FAILED after start         effect UNKNOWN   reconcile
    completed FAILED before start        effect NONE      fix the inputs

Actions recorded before this event existed cannot be read that way: a request
without a completion could have been interrupted on either side of the adapter
call. Those stay UNKNOWN. The projection refuses to rewrite history it cannot
see.

Reconciliation is append-only evidence about an unknown effect, never an
erasure of it. ``STILL_UNKNOWN`` is a valid final answer.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from .ledger import Event

if TYPE_CHECKING:
    from .runtime import Runtime

ACTION_RECOVERY_V1 = "action-recovery-v1"


class EffectState(StrEnum):
    """What the record establishes about the world, not what was reported."""

    NONE = "none"
    UNKNOWN = "unknown"
    OBSERVED = "observed"


class ActionNextOperation(StrEnum):
    NONE = "none"
    START_ACTION = "start_action"
    RECONCILE_EFFECT = "reconcile_effect"
    FIX_INPUTS = "fix_inputs"


class ReconciliationVerdict(StrEnum):
    EFFECT_CONFIRMED = "effect_confirmed"
    NO_EFFECT_CONFIRMED = "no_effect_confirmed"
    STILL_UNKNOWN = "still_unknown"


class ReconciliationRefused(RuntimeError):
    def __init__(self, action_id: str, reason: str) -> None:
        self.action_id = action_id
        self.reason = reason
        super().__init__(f"reconciliation refused for {action_id}: {reason}")


@dataclass(frozen=True, slots=True)
class Reconciliation:
    action_id: str
    verdict: str
    actor_id: str
    basis: str | None = None
    evidence_refs: tuple[str, ...] = ()
    recorded_at: str | None = None


@dataclass(frozen=True, slots=True)
class ActionWorkState:
    action_id: str
    task_id: str | None
    idempotency_key: str | None
    stage: str
    result_status: str | None
    effect_state: str
    next_operation: str
    duplicate_effect_risk: bool
    reason: str
    basis: str
    version: str = ACTION_RECOVERY_V1
    reused_from_action_id: str | None = None
    observed_state_hash: str | None = None
    reconciliations: tuple[Reconciliation, ...] = ()


# ---------------- projection ----------------


def _events_by_action(events: tuple[Event, ...]) -> dict[str, list[Event]]:
    by_action: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        if event.kind.startswith("action."):
            by_action[event.stream_id].append(event)
    return by_action


def _reconciliations(action_events: list[Event]) -> tuple[Reconciliation, ...]:
    out = []
    for event in action_events:
        if event.kind != "action.reconciled":
            continue
        payload = event.payload
        out.append(
            Reconciliation(
                action_id=event.stream_id,
                verdict=str(payload.get("verdict")),
                actor_id=event.actor_id,
                basis=payload.get("basis"),
                evidence_refs=tuple(str(ref) for ref in payload.get("evidence_refs") or ()),
                recorded_at=event.created_at,
            )
        )
    return tuple(out)


def _apply_reconciliation(state: ActionWorkState) -> ActionWorkState:
    """Later evidence about an unknown effect. It never erases the record."""
    if not state.reconciliations:
        return state
    latest = state.reconciliations[-1]
    basis = f"reconciliation:{latest.verdict}"
    if latest.verdict == ReconciliationVerdict.EFFECT_CONFIRMED:
        return _replace(
            state,
            effect_state=EffectState.OBSERVED.value,
            next_operation=ActionNextOperation.NONE.value,
            duplicate_effect_risk=False,
            reason="reconciled: the effect is confirmed to have happened",
            basis=basis,
        )
    if latest.verdict == ReconciliationVerdict.NO_EFFECT_CONFIRMED:
        return _replace(
            state,
            effect_state=EffectState.NONE.value,
            next_operation=ActionNextOperation.START_ACTION.value,
            duplicate_effect_risk=False,
            reason="reconciled: no effect happened, so the operation may be started again",
            basis=basis,
        )
    return _replace(
        state,
        reason="reconciled: the effect remains unknown, which is a valid final answer",
        basis=basis,
    )


def _replace(state: ActionWorkState, **changes: object) -> ActionWorkState:
    from dataclasses import replace as _dc_replace

    return _dc_replace(state, **changes)


def project_action_state(events: tuple[Event, ...], action_id: str) -> ActionWorkState | None:
    """Classify one action from the ledger alone. Appends nothing."""
    action_events = _events_by_action(events).get(action_id)
    if not action_events:
        return None
    requested = next((e for e in action_events if e.kind == "action.requested"), None)
    if requested is None:
        return None
    started = next((e for e in action_events if e.kind == "action.execution_started"), None)
    completed = next((e for e in action_events if e.kind == "action.completed"), None)
    refused = next((e for e in action_events if e.kind == "action.replay_refused"), None)

    task_id = requested.payload.get("task_id")
    key = requested.payload.get("idempotency_key")
    marks_execution = str(requested.payload.get("recovery_version") or "") == ACTION_RECOVERY_V1
    base = ActionWorkState(
        action_id=action_id,
        task_id=str(task_id) if task_id else None,
        idempotency_key=str(key) if key else None,
        stage="requested",
        result_status=None,
        effect_state=EffectState.NONE.value,
        next_operation=ActionNextOperation.START_ACTION.value,
        duplicate_effect_risk=False,
        reason="",
        basis="record",
        reconciliations=_reconciliations(action_events),
    )

    if completed is not None:
        status = str(completed.payload.get("status"))
        reused = completed.payload.get("reused_from_action_id")
        observed = completed.payload.get("observed_state_hash")
        common = {
            "result_status": status,
            "reused_from_action_id": str(reused) if reused else None,
            "observed_state_hash": str(observed) if observed else None,
        }
        if reused:
            state = _replace(
                base,
                stage="replayed",
                effect_state=EffectState.NONE.value,
                next_operation=ActionNextOperation.NONE.value,
                reason="recorded outcome returned under the same key; no new effect by construction",
                **common,
            )
        elif status == "succeeded":
            state = _replace(
                base,
                stage="completed",
                effect_state=EffectState.OBSERVED.value,
                next_operation=ActionNextOperation.NONE.value,
                reason="completed with the runtime's own observation of the resulting state",
                **common,
            )
        elif status == "denied":
            state = _replace(
                base,
                stage="denied",
                effect_state=EffectState.NONE.value,
                next_operation=ActionNextOperation.NONE.value,
                reason="authority refused the request before the adapter was invoked",
                **common,
            )
        elif started is not None:
            state = _replace(
                base,
                stage="failed_after_execution_started",
                effect_state=EffectState.UNKNOWN.value,
                next_operation=ActionNextOperation.RECONCILE_EFFECT.value,
                duplicate_effect_risk=True,
                reason="failed after execution began; a failure status is not evidence that nothing happened",
                **common,
            )
        elif marks_execution:
            state = _replace(
                base,
                stage="failed_before_execution_started",
                effect_state=EffectState.NONE.value,
                next_operation=ActionNextOperation.FIX_INPUTS.value,
                reason="refused before the adapter was invoked, so no effect was possible",
                **common,
            )
        else:
            state = _replace(
                base,
                stage="failed",
                effect_state=EffectState.UNKNOWN.value,
                next_operation=ActionNextOperation.RECONCILE_EFFECT.value,
                duplicate_effect_risk=True,
                reason="failed, and this record predates execution marking, so the effect cannot be ruled out",
                basis="record:pre-execution-marking",
                **common,
            )
    elif refused is not None:
        state = _replace(
            base,
            stage="replay_refused",
            effect_state=EffectState.NONE.value,
            next_operation=ActionNextOperation.NONE.value,
            reason="same key, different operation: refused before any effect",
        )
    elif started is not None:
        state = _replace(
            base,
            stage="execution_started",
            effect_state=EffectState.UNKNOWN.value,
            next_operation=ActionNextOperation.RECONCILE_EFFECT.value,
            duplicate_effect_risk=True,
            reason="execution began with no completion recorded; the effect may have happened",
        )
    elif marks_execution:
        state = _replace(
            base,
            stage="requested",
            effect_state=EffectState.NONE.value,
            next_operation=ActionNextOperation.START_ACTION.value,
            reason="requested with no execution started, so no effect was possible",
        )
    else:
        state = _replace(
            base,
            stage="requested",
            effect_state=EffectState.UNKNOWN.value,
            next_operation=ActionNextOperation.RECONCILE_EFFECT.value,
            duplicate_effect_risk=True,
            reason="recorded before execution marking: it cannot be told whether execution began",
            basis="record:pre-execution-marking",
        )
    return _apply_reconciliation(state)


def project_open_effects(
    events: tuple[Event, ...], *, task_id: str | None = None
) -> tuple[ActionWorkState, ...]:
    """Every action whose record leaves the effect unknown: the orphan list.

    Chapter 11 recorded ambiguity without reporting it. This is the report.
    """
    states = []
    for action_id in _events_by_action(events):
        state = project_action_state(events, action_id)
        if state is None or state.effect_state != EffectState.UNKNOWN.value:
            continue
        if task_id is not None and state.task_id != task_id:
            continue
        states.append(state)
    return tuple(sorted(states, key=lambda s: s.action_id))


# ---------------- reconciliation ----------------


def reconcile_action(
    runtime: Runtime,
    action_id: str,
    *,
    verdict: str,
    actor_id: str,
    evidence_refs: tuple[str, ...] = (),
    basis: str | None = None,
) -> ActionWorkState:
    """Record later evidence about an unknown effect. Appends; never edits.

    Refusals are durable: an action whose effect the record already settles has
    nothing to reconcile, and saying so in the ledger is worth more than an
    exception a later reader cannot see.
    """
    verdict_value = ReconciliationVerdict(str(verdict))
    events = runtime.ledger.read_all()
    state = project_action_state(events, action_id)
    if state is None:
        _append_refusal(runtime, action_id, actor_id, "no recorded action with that id")
        raise ReconciliationRefused(action_id, "no recorded action with that id")
    if state.effect_state != EffectState.UNKNOWN.value:
        reason = f"effect is already {state.effect_state} by the record; nothing to reconcile"
        _append_refusal(runtime, action_id, actor_id, reason)
        raise ReconciliationRefused(action_id, reason)

    runtime.ledger.append(
        Event.create(
            stream_id=action_id,
            kind="action.reconciled",
            actor_id=actor_id,
            payload={
                "action_id": action_id,
                "verdict": verdict_value.value,
                "evidence_refs": list(evidence_refs),
                "basis": basis,
                "version": ACTION_RECOVERY_V1,
            },
            correlation_id=state.task_id,
        )
    )
    reconciled = project_action_state(runtime.ledger.read_all(), action_id)
    assert reconciled is not None
    return reconciled


def _append_refusal(runtime: Runtime, action_id: str, actor_id: str, reason: str) -> None:
    runtime.ledger.append(
        Event.create(
            stream_id=action_id,
            kind="action.reconcile_refused",
            actor_id=actor_id,
            payload={"action_id": action_id, "reason": reason, "version": ACTION_RECOVERY_V1},
        )
    )
