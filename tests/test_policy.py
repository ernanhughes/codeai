import pytest

from codeai.domain import Authority, Budget, Capability, Directive
from codeai.policy import AuthorityDenied, BudgetExceeded, PolicyEngine, Usage


def test_child_directive_cannot_expand_authority_or_budget():
    parent = Directive(
        directive_id="d1",
        objective="solve",
        success_criteria=("verified",),
        budget=Budget(max_tokens=1000, max_cost_usd=5),
        authority=Authority(frozenset({Capability.READ, Capability.WRITE})),
    )
    child = Directive(
        directive_id="d2",
        parent_directive_id="d1",
        objective="subproblem",
        success_criteria=("done",),
        budget=Budget(max_tokens=500, max_cost_usd=2),
        authority=Authority(frozenset({Capability.READ})),
    )
    parent.validate_child(child)

    illegal = Directive(
        directive_id="d3",
        parent_directive_id="d1",
        objective="bad",
        success_criteria=("done",),
        budget=Budget(max_tokens=2000, max_cost_usd=2),
        authority=Authority(frozenset({Capability.READ, Capability.DESTRUCTIVE})),
    )
    with pytest.raises(ValueError):
        parent.validate_child(illegal)


def test_policy_enforces_authority_and_hard_budget():
    policy = PolicyEngine()
    authority = Authority(frozenset({Capability.READ}))
    policy.require(authority, Capability.READ)
    with pytest.raises(AuthorityDenied):
        policy.require(authority, Capability.WRITE)

    with pytest.raises(BudgetExceeded):
        policy.check_budget(Budget(max_turns=2), Usage(turns=3))
