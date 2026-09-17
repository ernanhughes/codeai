"""Chapter 20 seam: authority comes from the record, not from the caller.

    caller says it has authority  !=  runtime establishes authority

The load-bearing test is the forged one: a caller passing a broader grant makes
no difference, because the runtime never consults it when the request names a
directive.
"""

import pytest

from codeai.actions import ActionNextOperation, EffectState
from codeai.adapters import ActionRequest, ActionResult, ActionStatus
from codeai.authority import (
    AuthorizationStatus,
    GrantSource,
    authorize,
    resolve_directive_authority,
)
from codeai.domain import Authority, Budget, Capability, Directive
from codeai.ledger import SQLiteLedger
from codeai.runtime import Runtime


class CountingWriter:
    def __init__(self):
        self.calls = 0

    def execute(self, request: ActionRequest) -> ActionResult:
        self.calls += 1
        return ActionResult(action_id=request.action_id, status=ActionStatus.SUCCEEDED)


def directive(directive_id, capabilities, *, parent=None, tokens=100):
    return Directive(
        directive_id=directive_id,
        objective="fixture",
        success_criteria=(),
        budget=Budget(max_tokens=tokens),
        authority=Authority(frozenset(capabilities)),
        parent_directive_id=parent,
    )


def action(action_id="a1", *, directive_id="root", capability="write", key="k"):
    return ActionRequest(
        action_id=action_id,
        task_id="t",
        directive_id=directive_id,
        capability=capability,
        instruction="do the thing",
        precondition_hash=None,
        idempotency_key=key,
        requested_by="human",
        actor_id="worker",
        adapter="fixture",
    )


def runtime_with(tmp_path):
    return Runtime(SQLiteLedger(tmp_path / "ledger.sqlite"))


# ---------------- the matrix ----------------


def test_root_grant_permits_the_effect(tmp_path):
    runtime = runtime_with(tmp_path)
    runtime.open_directive(directive("root", {Capability.WRITE}))
    writer = CountingWriter()

    result = runtime.execute_action(action(), adapter=writer)

    assert result.status == ActionStatus.SUCCEEDED
    assert writer.calls == 1
    authorized = runtime.ledger.events_by_kind(("action.authorized",))
    assert authorized[0].payload["grant_source"] == GrantSource.RECORDED_DIRECTIVE
    assert authorized[0].payload["effective_capabilities"] == ["write"]


def test_child_narrowing_refuses_what_the_child_gave_up(tmp_path):
    runtime = runtime_with(tmp_path)
    runtime.open_directive(directive("root", {Capability.READ, Capability.WRITE}))
    runtime.open_directive(directive("child", {Capability.READ}, parent="root"))
    writer = CountingWriter()

    result = runtime.execute_action(action(directive_id="child"), adapter=writer)

    assert result.status == ActionStatus.DENIED
    assert writer.calls == 0
    assert "capability denied: write" in (result.error or "")


def test_unknown_directive_is_refused_durably(tmp_path):
    runtime = runtime_with(tmp_path)
    writer = CountingWriter()

    result = runtime.execute_action(action(directive_id="never-registered"), adapter=writer)

    assert result.status == ActionStatus.DENIED
    assert writer.calls == 0
    refusal = runtime.ledger.events_by_kind(("action.authorization_refused",))[0]
    assert refusal.payload["status"] == AuthorizationStatus.UNKNOWN_DIRECTIVE
    assert refusal.payload["grant_chain"] == []


def test_valid_grandchild_chain_permits_the_effect(tmp_path):
    runtime = runtime_with(tmp_path)
    runtime.open_directive(directive("root", {Capability.READ, Capability.WRITE, Capability.ACCEPT}))
    runtime.open_directive(directive("child", {Capability.READ, Capability.WRITE}, parent="root", tokens=50))
    runtime.open_directive(directive("grandchild", {Capability.WRITE}, parent="child", tokens=25))
    writer = CountingWriter()

    result = runtime.execute_action(action(directive_id="grandchild"), adapter=writer)

    assert result.status == ActionStatus.SUCCEEDED
    basis = runtime.ledger.events_by_kind(("action.authorized",))[0].payload
    assert [link["directive_id"] for link in basis["grant_chain"]] == [
        "grandchild",
        "child",
        "root",
    ]
    assert basis["effective_capabilities"] == ["write"]
    assert len(basis["basis_event_ids"]) == 3


def test_a_forged_caller_grant_changes_nothing(tmp_path):
    # The old seam: the caller handed in an Authority and the runtime believed it.
    runtime = runtime_with(tmp_path)
    runtime.open_directive(directive("root", {Capability.READ}))
    writer = CountingWriter()

    result = runtime.execute_action(
        action(capability="write"),
        adapter=writer,
        authority=Authority(frozenset({Capability.WRITE, Capability.DESTRUCTIVE})),
    )

    assert result.status == ActionStatus.DENIED, "a caller-supplied grant authorized an effect"
    assert writer.calls == 0


def test_an_action_naming_no_directive_falls_back_to_the_caller_and_says_so(tmp_path):
    runtime = runtime_with(tmp_path)
    writer = CountingWriter()

    result = runtime.execute_action(
        action(directive_id=None), adapter=writer, authority=Authority(frozenset({Capability.WRITE}))
    )

    assert result.status == ActionStatus.SUCCEEDED
    basis = runtime.ledger.events_by_kind(("action.authorized",))[0].payload
    assert basis["grant_source"] == GrantSource.CALLER_SUPPLIED
    assert basis["directive_id"] is None


def test_a_widened_child_is_refused_at_registration_and_leaves_evidence(tmp_path):
    runtime = runtime_with(tmp_path)
    runtime.open_directive(directive("root", {Capability.READ}))
    with pytest.raises(ValueError, match="authority must narrow"):
        runtime.open_directive(directive("child", {Capability.READ, Capability.WRITE}, parent="root"))

    refusal = runtime.ledger.events_by_kind(("directive.registration_refused",))[0]
    assert refusal.payload["requested_capabilities"] == ["read", "write"]
    assert refusal.payload["parent_capabilities"] == ["read"]
    assert refusal.payload["basis_event_ids"]


# ---------------- authority before disclosure ----------------


def test_an_unauthorized_caller_learns_nothing_about_the_recorded_operation(tmp_path):
    runtime = runtime_with(tmp_path)
    runtime.open_directive(directive("root", {Capability.READ, Capability.WRITE}))
    runtime.open_directive(directive("narrow", {Capability.READ}, parent="root"))
    writer = CountingWriter()
    runtime.execute_action(action("a1"), adapter=writer)

    denied = runtime.execute_action(action("a2", directive_id="narrow"), adapter=writer)

    assert denied.status == ActionStatus.DENIED
    assert denied.reused_from_action_id is None, "a denial disclosed the recorded operation"
    assert writer.calls == 1
    kinds = [e.kind for e in runtime.ledger.read_all() if e.stream_id == "a2"]
    assert kinds == ["action.requested", "action.authorization_refused", "action.completed"]


def test_a_replay_is_disclosed_under_its_own_authorization_basis(tmp_path):
    runtime = runtime_with(tmp_path)
    runtime.open_directive(directive("root", {Capability.WRITE}))
    runtime.open_directive(directive("child", {Capability.WRITE}, parent="root", tokens=50))
    writer = CountingWriter()

    runtime.execute_action(action("a1", directive_id="root"), adapter=writer)
    replayed = runtime.execute_action(action("a2", directive_id="child"), adapter=writer)

    assert replayed.reused_from_action_id == "a1"
    assert writer.calls == 1
    bases = {
        e.stream_id: e.payload["directive_id"]
        for e in runtime.ledger.events_by_kind(("action.authorized",))
    }
    # The original effect and its later disclosure rest on different grants.
    assert bases == {"a1": "root", "a2": "child"}


# ---------------- authorization meets effect recovery ----------------


def test_a_crash_after_authorization_leaves_no_effect(tmp_path):
    runtime = runtime_with(tmp_path)
    runtime.open_directive(directive("root", {Capability.WRITE}))
    request = action()
    runtime.ledger.append(
        __import__("codeai.ledger", fromlist=["Event"]).Event.create(
            stream_id=request.action_id, kind="action.requested", actor_id="human",
            payload={"action_id": "a1", "task_id": "t", "idempotency_key": "k",
                     "recovery_version": "action-recovery-v1"},
        )
    )
    decision = runtime.authorize_action(request)
    runtime.ledger.append(
        __import__("codeai.ledger", fromlist=["Event"]).Event.create(
            stream_id=request.action_id, kind="action.authorized", actor_id="human",
            payload={"action_id": "a1", **decision.basis_payload()},
        )
    )

    state = runtime.action_state("a1")
    assert state.stage == "authorized"
    assert state.effect_state == EffectState.NONE
    assert state.next_operation == ActionNextOperation.START_ACTION
    assert state.duplicate_effect_risk is False


def test_a_refused_action_is_terminal_with_no_effect(tmp_path):
    runtime = runtime_with(tmp_path)
    runtime.open_directive(directive("root", {Capability.READ}))
    runtime.execute_action(action(), adapter=CountingWriter())

    state = runtime.action_state("a1")
    assert state.result_status == "denied"
    assert state.effect_state == EffectState.NONE
    assert state.next_operation == ActionNextOperation.NONE


# ---------------- projection, independent of the runtime ----------------


def test_effective_authority_rebuilds_from_ledger_events_alone(tmp_path):
    runtime = runtime_with(tmp_path)
    runtime.open_directive(directive("root", {Capability.READ, Capability.WRITE, Capability.ACCEPT}))
    runtime.open_directive(directive("child", {Capability.READ, Capability.WRITE}, parent="root", tokens=50))
    runtime.open_directive(directive("grandchild", {Capability.WRITE}, parent="child", tokens=25))
    events = runtime.ledger.read_all()

    standing = resolve_directive_authority(events, "grandchild")

    assert standing.resolvable
    assert standing.effective_capabilities == ("write",)
    assert [link.directive_id for link in standing.grant_chain] == ["grandchild", "child", "root"]
    assert authorize(standing, "write").status == AuthorizationStatus.GRANTED
    assert authorize(standing, "accept").status == AuthorizationStatus.DENIED


def test_a_chain_that_widens_is_invalid_rather_than_quietly_intersected(tmp_path):
    # Registration refuses this now, but a ledger written before it could not.
    from codeai.ledger import Event

    runtime = runtime_with(tmp_path)
    runtime.open_directive(directive("root", {Capability.READ}))
    runtime.ledger.append(
        Event.create(
            stream_id="child", kind="directive.opened", actor_id="human",
            payload={
                "directive_id": "child",
                "objective": "widened before registration validated narrowing",
                "success_criteria": [],
                "budget": {"max_tokens": 50, "max_cost_usd": None, "max_seconds": None,
                           "max_turns": None, "max_human_minutes": None},
                "authority": {"capabilities": ["read", "write"]},
                "parent_directive_id": "root",
            },
        )
    )

    standing = runtime.directive_authority("child")
    assert not standing.resolvable
    assert "does not narrow" in standing.reason
    assert authorize(standing, "write").status == AuthorizationStatus.INVALID_CHAIN

    result = runtime.execute_action(action(directive_id="child"), adapter=CountingWriter())
    assert result.status == ActionStatus.DENIED


def test_a_cycle_is_named_rather_than_followed(tmp_path):
    from codeai.ledger import Event

    runtime = runtime_with(tmp_path)
    for name, parent in (("a", "b"), ("b", "a")):
        runtime.ledger.append(
            Event.create(
                stream_id=name, kind="directive.opened", actor_id="human",
                payload={
                    "directive_id": name, "objective": "cycle", "success_criteria": [],
                    "budget": {"max_tokens": None, "max_cost_usd": None, "max_seconds": None,
                               "max_turns": None, "max_human_minutes": None},
                    "authority": {"capabilities": ["write"]},
                    "parent_directive_id": parent,
                },
            )
        )

    standing = runtime.directive_authority("a")
    assert not standing.resolvable
    assert "cycle" in standing.reason
