"""What is true about a task, projected from the ledger (the Chapter 28 seam).

The scheduler used to take five booleans from whoever called it. It decided
correctly and recorded nothing, so the honest reading of any decision was "a
caller asserted these flags and the policy agreed". That is not a process state;
it is a caller's opinion with a policy stamped on it.

This module separates the two halves that were conflated:

    ProcessState   what is true, derived from recorded events
    scheduler      what to do about it, a pure function of that state

Nothing here decides anything. There is deliberately no ``should_call_model``
and no ``should_ask_human``: those are policy conclusions, and keeping them out
is what lets the same state be replayed against a different policy version.

Every fact carries the events it was read from, so a decision recorded later can
be reconstructed rather than re-asserted. Where the ledger cannot establish a
fact, the projection says so instead of guessing: ``model_budget_known`` is
False when no directive budget is recorded, and an unknown budget is never
reported as exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .acceptance import TaskCompletionStatus, project_task_completion
from .actions import project_open_effects
from .authority import resolve_directive_authority
from .ledger import Event

if TYPE_CHECKING:
    from .runtime import Runtime

PROCESS_STATE_V1 = "process-state-v1"


@dataclass(frozen=True, slots=True)
class BudgetFact:
    """A limit, what has been consumed against it, and whether either is known."""

    limit: int | None
    consumed: int
    known: bool
    basis_event_ids: tuple[str, ...] = ()

    @property
    def exhausted(self) -> bool:
        return self.known and self.limit is not None and self.consumed >= self.limit


@dataclass(frozen=True, slots=True)
class ProcessState:
    task_id: str
    directive_id: str | None

    process_complete: bool

    proposal_count: int
    independent_proposal_count: int

    criteria_declared: bool
    check_required: bool
    check_satisfied: bool

    acceptance_required: bool
    acceptance_authority_available: bool
    directive_effective_capabilities: tuple[str, ...]

    model_budget: BudgetFact
    process_budget: BudgetFact

    unresolved_effects: tuple[str, ...]

    basis_event_ids: tuple[str, ...] = ()
    projection_version: str = PROCESS_STATE_V1
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def model_budget_exhausted(self) -> bool:
        return self.model_budget.exhausted

    @property
    def process_budget_exhausted(self) -> bool:
        return self.process_budget.exhausted

    def to_scheduler_input(self):
        """Hand the policy exactly these facts, and nothing it could have invented."""
        from .scheduler import SchedulerInput

        return SchedulerInput(
            has_required_verification=self.check_required and not self.check_satisfied,
            requests_independent_proposals=self.proposal_count == 0,
            requires_human_authority_for_next_effect=(
                self.acceptance_required and not self.acceptance_authority_available
            ),
            process_budget_exhausted=self.process_budget_exhausted,
            model_budget_exhausted=self.model_budget_exhausted,
            unresolved_effect=bool(self.unresolved_effects),
            process_complete=self.process_complete,
        )


def _task_event(events: tuple[Event, ...], task_id: str) -> Event | None:
    for event in events:
        if event.kind == "task.created" and str(
            event.payload.get("task_id") or event.stream_id
        ) == task_id:
            return event
    return None


def _for_task(events: tuple[Event, ...], task_id: str, kinds: tuple[str, ...]) -> list[Event]:
    out = []
    for event in events:
        if event.kind not in kinds:
            continue
        payload_task = event.payload.get("task_id")
        if str(payload_task) == task_id or event.correlation_id == task_id:
            out.append(event)
    return out


def project_process_state(runtime: Runtime, task_id: str) -> ProcessState:
    """Derive what is true about a task from its recorded events. Appends nothing."""
    events = runtime.ledger.read_all()
    basis: list[str] = []
    notes: list[str] = []

    task_event = _task_event(events, task_id)
    if task_event is not None:
        basis.append(task_event.event_id)
    directive_id = str(task_event.payload.get("directive_id")) if task_event else None
    criteria = tuple(task_event.payload.get("success_criteria") or ()) if task_event else ()
    if task_event is None:
        notes.append("no task.created event is recorded for this task")

    completion = project_task_completion(runtime, task_id)
    process_complete = completion.status == TaskCompletionStatus.COMPLETED

    calls = _for_task(events, task_id, ("call.completed",))
    succeeded = [e for e in calls if str(e.payload.get("call_status") or "") == "succeeded"]
    basis.extend(e.event_id for e in succeeded)

    # "Independent" is not a synonym for "separate": only proposals collected
    # through a recorded blind fan-out are counted as such (Chapter 23).
    fanout_calls: set[str] = set()
    for event in _for_task(events, task_id, ("fanout.completed",)):
        basis.append(event.event_id)
        for branch in event.payload.get("branches") or ():
            call_id = branch.get("call_id") if isinstance(branch, dict) else None
            if call_id:
                fanout_calls.add(str(call_id))
    independent = len(
        [e for e in succeeded if str(e.payload.get("call_id") or e.stream_id) in fanout_calls]
    )

    checks = _for_task(events, task_id, ("check.completed",))
    basis.extend(e.event_id for e in checks)
    check_satisfied = any(str(e.payload.get("verdict") or "").upper() == "PASS" for e in checks)

    # Declared is not owed. A criterion is a standing requirement; a check is
    # owed only once something exists for it to examine. Reporting an owed
    # check against nothing would make the projection assert work the record
    # cannot support, which is exactly what this seam exists to prevent.
    criteria_declared = bool(criteria)
    check_required = criteria_declared and bool(succeeded)
    if criteria_declared and not succeeded:
        notes.append("criteria are declared, but nothing has been produced to check yet")
    elif check_required and not checks:
        notes.append("the task declares criteria and no check has completed")

    # Acceptance is required once something has been proposed and the task is
    # not complete; whether the process may accept it is the directive's answer.
    acceptance_required = bool(succeeded) and not process_complete
    standing = resolve_directive_authority(events, directive_id)
    acceptance_available = standing.allows("accept") if standing.resolvable else False
    basis.extend(standing.basis_event_ids)
    if directive_id and not standing.resolvable:
        notes.append(f"authority is unresolved: {standing.reason}")

    consumed_tokens = sum(
        int(e.payload.get("total_input_tokens") or 0) + int(e.payload.get("total_output_tokens") or 0)
        for e in calls
    )
    limits = {}
    if standing.resolvable and standing.grant_chain:
        directive_event = next(
            (e for e in events if e.kind == "directive.opened" and e.stream_id == directive_id),
            None,
        )
        if directive_event is not None:
            limits = directive_event.payload.get("budget") or {}
            basis.append(directive_event.event_id)
    max_tokens = limits.get("max_tokens")
    model_budget = BudgetFact(
        limit=int(max_tokens) if max_tokens is not None else None,
        consumed=consumed_tokens,
        known=max_tokens is not None,
        basis_event_ids=tuple(e.event_id for e in calls),
    )
    if max_tokens is None:
        notes.append("no recorded token limit, so the model budget cannot be called exhausted")

    operations = _for_task(
        events, task_id, ("call.completed", "check.completed", "action.completed")
    )
    max_turns = limits.get("max_turns")
    process_budget = BudgetFact(
        limit=int(max_turns) if max_turns is not None else None,
        consumed=len(operations),
        known=max_turns is not None,
        basis_event_ids=tuple(e.event_id for e in operations),
    )
    if max_turns is None:
        notes.append("no recorded turn limit, so the process budget cannot be called exhausted")

    open_effects = project_open_effects(events, task_id=task_id)
    unresolved = tuple(state.action_id for state in open_effects)

    return ProcessState(
        task_id=task_id,
        directive_id=directive_id,
        process_complete=process_complete,
        proposal_count=len(succeeded),
        independent_proposal_count=independent,
        criteria_declared=criteria_declared,
        check_required=check_required,
        check_satisfied=check_satisfied,
        acceptance_required=acceptance_required,
        acceptance_authority_available=acceptance_available,
        directive_effective_capabilities=(
            tuple(standing.effective_capabilities) if standing.resolvable else ()
        ),
        model_budget=model_budget,
        process_budget=process_budget,
        unresolved_effects=unresolved,
        basis_event_ids=tuple(dict.fromkeys(basis)),
        notes=tuple(notes),
    )


def state_snapshot(state: ProcessState) -> dict[str, object]:
    """The facts a recorded decision rests on, in a form that can be replayed."""
    return {
        "task_id": state.task_id,
        "directive_id": state.directive_id,
        "process_complete": state.process_complete,
        "proposal_count": state.proposal_count,
        "independent_proposal_count": state.independent_proposal_count,
        "criteria_declared": state.criteria_declared,
        "check_required": state.check_required,
        "check_satisfied": state.check_satisfied,
        "acceptance_required": state.acceptance_required,
        "acceptance_authority_available": state.acceptance_authority_available,
        # Any change to the effective grant moves this digest, so a decision
        # taken before an authority transition cannot survive it.
        "directive_effective_capabilities": list(state.directive_effective_capabilities),
        "model_budget": {
            "limit": state.model_budget.limit,
            "consumed": state.model_budget.consumed,
            "known": state.model_budget.known,
        },
        "process_budget": {
            "limit": state.process_budget.limit,
            "consumed": state.process_budget.consumed,
            "known": state.process_budget.known,
        },
        "unresolved_effects": list(state.unresolved_effects),
        "projection_version": state.projection_version,
        "notes": list(state.notes),
    }
