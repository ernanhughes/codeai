from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Operation(StrEnum):
    CALL = "CALL"
    CHECK = "CHECK"
    ACTION = "ACTION"
    ASK_HUMAN = "ASK_HUMAN"
    STOP = "STOP"


@dataclass(frozen=True, slots=True)
class SchedulerInput:
    has_required_verification: bool = False
    requests_independent_proposals: bool = False
    requires_destructive_capability: bool = False
    budget_exhausted: bool = False


@dataclass(frozen=True, slots=True)
class SchedulerDecision:
    operation: Operation
    reason: str
    policy_version: str = "epistemic-v1"


POLICY_VERSION = "epistemic-v1"


def decide_next_step(query: SchedulerInput) -> SchedulerDecision:
    """Deterministic epistemic scheduler seam (no learning, no routing).

    Keeps *what to do next* separate from *which model to use*.
    Priority mirrors the spec example: verification > independent sampling
    > human gate for destructive work > stop on exhausted budget.
    """
    if query.budget_exhausted:
        return SchedulerDecision(Operation.STOP, "budget exhausted", POLICY_VERSION)
    if query.has_required_verification:
        return SchedulerDecision(Operation.CHECK, "required deterministic verification exists", POLICY_VERSION)
    if query.requests_independent_proposals:
        return SchedulerDecision(Operation.CALL, "task requests independent proposals", POLICY_VERSION)
    if query.requires_destructive_capability:
        return SchedulerDecision(Operation.ASK_HUMAN, "destructive capability requires human", POLICY_VERSION)
    return SchedulerDecision(Operation.STOP, "no epistemic operation required", POLICY_VERSION)
