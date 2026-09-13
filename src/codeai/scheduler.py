from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Operation(StrEnum):
    CALL = "CALL"
    CHECK = "CHECK"
    ACTION = "ACTION"
    ASK_HUMAN = "ASK_HUMAN"
    STOP = "STOP"


@dataclass(frozen=True, slots=True, init=False)
class SchedulerInput:
    has_required_verification: bool = False
    requests_independent_proposals: bool = False
    requires_human_authority_for_next_effect: bool = False
    process_budget_exhausted: bool = False
    model_budget_exhausted: bool = False

    def __init__(self, has_required_verification=False, requests_independent_proposals=False,
                 requires_destructive_capability=None, budget_exhausted=None, *,
                 requires_human_authority_for_next_effect=None,
                 process_budget_exhausted=None, model_budget_exhausted=False):
        # v1 positional/keyword compatibility is resolved once, never stored ambiguously.
        def migrate(old, new, name):
            if old is not None and new is not None and old != new:
                raise ValueError(f"conflicting legacy and explicit {name}")
            return new if new is not None else (old if old is not None else False)

        values = dict(
            has_required_verification=has_required_verification,
            requests_independent_proposals=requests_independent_proposals,
            requires_human_authority_for_next_effect=migrate(
                requires_destructive_capability, requires_human_authority_for_next_effect, "authority"),
            process_budget_exhausted=migrate(budget_exhausted, process_budget_exhausted, "budget"),
            model_budget_exhausted=model_budget_exhausted,
        )
        for name, value in values.items():
            if type(value) is not bool:
                raise TypeError(f"{name} must be a bool")
            object.__setattr__(self, name, value)

    @property
    def budget_exhausted(self):
        """Deprecated v1 spelling: process-level stop only."""
        return self.process_budget_exhausted

    @property
    def requires_destructive_capability(self):
        """Deprecated v1 spelling: gate for the next effect, not cognition."""
        return self.requires_human_authority_for_next_effect


@dataclass(frozen=True, slots=True)
class SchedulerDecision:
    operation: Operation
    reason: str
    policy_version: str = "epistemic-v2"


POLICY_VERSION = "epistemic-v2"


def decide_next_step(query: SchedulerInput) -> SchedulerDecision:
    """Process stop > check > budget-permitted cognition > human gate > stop.

    Model budget blocks CALL only. CALL/CHECK never authorize a later effect.
    Flags are caller-supplied state, not inferred facts or resource accounting.
    """
    if query.process_budget_exhausted:
        return SchedulerDecision(Operation.STOP, "process budget exhausted", POLICY_VERSION)
    if query.has_required_verification:
        return SchedulerDecision(Operation.CHECK, "required deterministic verification exists", POLICY_VERSION)
    if query.requests_independent_proposals and not query.model_budget_exhausted:
        return SchedulerDecision(Operation.CALL, "task requests independent proposals", POLICY_VERSION)
    if query.requires_human_authority_for_next_effect:
        reason = "next effect requires human authority"
        if query.model_budget_exhausted and query.requests_independent_proposals:
            reason += "; model budget blocks proposals"
        return SchedulerDecision(Operation.ASK_HUMAN, reason, POLICY_VERSION)
    if query.model_budget_exhausted and query.requests_independent_proposals:
        return SchedulerDecision(Operation.STOP, "model budget blocks proposals", POLICY_VERSION)
    return SchedulerDecision(Operation.STOP, "no epistemic operation required", POLICY_VERSION)
