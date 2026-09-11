from __future__ import annotations

from dataclasses import dataclass

from .domain import Authority, Budget, Capability


@dataclass(frozen=True, slots=True)
class Usage:
    tokens: int = 0
    cost_usd: float = 0.0
    seconds: int = 0
    turns: int = 0
    human_minutes: int = 0


class BudgetExceeded(RuntimeError):
    pass


class AuthorityDenied(PermissionError):
    pass


class PolicyEngine:
    def require(self, authority: Authority, capability: Capability) -> None:
        if not authority.allows(capability):
            raise AuthorityDenied(f"capability denied: {capability.value}")

    def check_budget(self, budget: Budget, usage: Usage) -> None:
        checks = (
            (budget.max_tokens, usage.tokens, "tokens"),
            (budget.max_cost_usd, usage.cost_usd, "cost_usd"),
            (budget.max_seconds, usage.seconds, "seconds"),
            (budget.max_turns, usage.turns, "turns"),
            (budget.max_human_minutes, usage.human_minutes, "human_minutes"),
        )
        for limit, actual, name in checks:
            if limit is not None and actual > limit:
                raise BudgetExceeded(f"{name} budget exceeded: {actual} > {limit}")
