"""Authority derived from the record, not from the caller (the Chapter 20 seam).

The old shape let a caller hand the runtime an ``Authority`` object and say, in
effect, "I am allowed to do this". Registration validated that a declared child
directive narrowed its recorded parent, but nothing connected that recorded
grant to the effect that followed. Chapter 29 called the result what it was: a
conventional joint.

    caller says it has authority  !=  runtime establishes authority

This module resolves the grant from the ledger. A directive's effective
capabilities are the intersection down its recorded chain, so a child can only
ever narrow; the chain, the events it was read from, and the reason are kept
beside the answer, because the decision has to be reconstructable later, not
merely correct now.

    root directive        READ + WRITE + ACCEPT
          |
    child directive       READ + WRITE
          |
    grandchild            WRITE
          |
    requested capability  WRITE  -> GRANTED, basis = the three events above

Four outcomes are distinguished, because "refused" is not one thing:

    GRANTED             the resolved chain contains the requested capability
    DENIED              it does not
    UNKNOWN_DIRECTIVE   no such directive is recorded
    INVALID_CHAIN       the recorded chain cannot be trusted: a missing parent,
                        a cycle, or a child that widens its parent

An action that names no directive falls back to the caller-supplied grant. That
is recorded as such (``caller_supplied``) rather than dressed up as derived
authority, and it stays a named limitation: name a directive and the runtime
stops trusting the caller.

Authority also changes over time, and that is a different relationship from
delegation. Both live here, and they must not be confused:

    delegation            parent -> child
                          effective(child) is a subset of effective(parent)
                          a widening child is invalid, always

    authority transition  old directive -> superseding directive
                          may add, remove or otherwise alter authority, because
                          it records a new external decision rather than a
                          delegated child claiming powers its parent lacked

Directives are immutable. A transition records a successor and leaves the
predecessor exactly as it was, so the history stays honest: what was true at T1
is still readable at T3. Resolution follows the supersession chain first, then
intersects the delegation chain of whichever directive is currently effective.

What a transition establishes is narrow: *an explicit external authority change,
attributed to this actor, entered the durable process at this point*. It does not
establish that the actor was entitled to make it. Attribution is not
authentication, here as everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .domain import Authority, Capability
from .ledger import Event

AUTHORITY_RESOLUTION_V1 = "authority-resolution-v1"
AUTHORITY_TRANSITION_V1 = "authority-transition-v1"

TRANSITIONED = "authority.transitioned"
TRANSITION_REFUSED = "authority.transition_refused"


class GrantSource(StrEnum):
    RECORDED_DIRECTIVE = "recorded_directive"
    CALLER_SUPPLIED = "caller_supplied"


class TransitionSource(StrEnum):
    """Who decided that authority should change. Attribution, not authentication."""

    HUMAN_INTERVENTION = "human_intervention"
    EXTERNAL_DECISION = "external_decision"


class AuthorityTransitionRefused(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"authority transition refused: {reason}")


class AuthorizationStatus(StrEnum):
    GRANTED = "granted"
    DENIED = "denied"
    UNKNOWN_DIRECTIVE = "unknown_directive"
    INVALID_CHAIN = "invalid_chain"


@dataclass(frozen=True, slots=True)
class GrantLink:
    """One directive in the recorded chain, with the event it was read from."""

    directive_id: str
    event_id: str
    capabilities: tuple[str, ...]
    parent_directive_id: str | None = None


@dataclass(frozen=True, slots=True)
class DirectiveAuthorityStanding:
    directive_id: str | None
    source: str
    resolvable: bool
    effective_capabilities: tuple[str, ...]
    grant_chain: tuple[GrantLink, ...]
    basis_event_ids: tuple[str, ...]
    reason: str
    version: str = AUTHORITY_RESOLUTION_V1
    # The directive actually resolved, and how the record got there from the one
    # that was named. Empty when nothing has superseded it.
    effective_directive_id: str | None = None
    supersession_chain: tuple[str, ...] = ()

    def allows(self, capability: str | Capability) -> bool:
        return self.resolvable and str(Capability(str(capability))) in self.effective_capabilities

    def as_authority(self) -> Authority:
        return Authority(frozenset(Capability(value) for value in self.effective_capabilities))


@dataclass(frozen=True, slots=True)
class AuthorizationDecision:
    capability: str
    status: str
    reason: str
    standing: DirectiveAuthorityStanding

    @property
    def granted(self) -> bool:
        return self.status == AuthorizationStatus.GRANTED

    def basis_payload(self) -> dict[str, object]:
        """The durable basis: enough to reconstruct why, not merely that."""
        return {
            "capability": self.capability,
            "status": str(self.status),
            "reason": self.reason,
            "directive_id": self.standing.directive_id,
            "grant_source": str(self.standing.source),
            "effective_capabilities": list(self.standing.effective_capabilities),
            "grant_chain": [
                {
                    "directive_id": link.directive_id,
                    "event_id": link.event_id,
                    "capabilities": list(link.capabilities),
                    "parent_directive_id": link.parent_directive_id,
                }
                for link in self.standing.grant_chain
            ],
            "basis_event_ids": list(self.standing.basis_event_ids),
            "effective_directive_id": self.standing.effective_directive_id,
            "supersession_chain": list(self.standing.supersession_chain),
            "version": AUTHORITY_RESOLUTION_V1,
        }


def _directive_events(events: tuple[Event, ...]) -> dict[str, list[Event]]:
    by_id: dict[str, list[Event]] = {}
    for event in events:
        if event.kind == "directive.opened":
            by_id.setdefault(str(event.payload.get("directive_id") or event.stream_id), []).append(event)
    return by_id


def _capabilities_of(event: Event) -> tuple[str, ...]:
    authority = event.payload.get("authority") or {}
    values = authority.get("capabilities") or ()
    return tuple(sorted(str(value) for value in values))


def resolve_supersession(
    events: tuple[Event, ...], directive_id: str
) -> tuple[str | None, tuple[str, ...], tuple[str, ...], str | None]:
    """Follow authority transitions to whichever directive is effective now.

    Returns (effective_id, chain, basis_event_ids, refusal_reason). A directive
    with two recorded successors is refused rather than silently resolved to one
    of them: an ambiguous authority epoch is not an authority.
    """
    successors: dict[str, list[Event]] = {}
    for event in events:
        if event.kind == TRANSITIONED:
            previous = str(event.payload.get("previous_directive_id") or "")
            successors.setdefault(previous, []).append(event)

    chain = [str(directive_id)]
    basis: list[str] = []
    seen = {str(directive_id)}
    current = str(directive_id)
    while current in successors:
        recorded = successors[current]
        if len(recorded) > 1:
            return (
                None,
                tuple(chain),
                tuple(basis),
                f"directive {current} has {len(recorded)} recorded successors; "
                f"the effective authority is ambiguous",
            )
        event = recorded[0]
        nxt = str(event.payload.get("new_directive_id") or "")
        if nxt in seen:
            return (
                None,
                tuple(chain),
                tuple(basis),
                f"the supersession chain revisits {nxt}: it is a cycle, not a history",
            )
        basis.append(event.event_id)
        seen.add(nxt)
        chain.append(nxt)
        current = nxt
    return current, tuple(chain), tuple(basis), None


def resolve_directive_authority(
    events: tuple[Event, ...], directive_id: str | None
) -> DirectiveAuthorityStanding:
    """Resolve the currently effective grant for a directive. Appends nothing.

    Supersession first (which directive is in force now), then delegation (what
    that directive and its recorded parents jointly allow).
    """
    if directive_id is None:
        return DirectiveAuthorityStanding(
            directive_id=None,
            source=GrantSource.CALLER_SUPPLIED.value,
            resolvable=False,
            effective_capabilities=(),
            grant_chain=(),
            basis_event_ids=(),
            reason="the request names no directive, so no recorded grant can be resolved",
        )

    named = str(directive_id)
    effective_id, supersession, supersession_basis, refusal = resolve_supersession(events, named)
    if refusal is not None:
        return DirectiveAuthorityStanding(
            directive_id=named,
            source=GrantSource.RECORDED_DIRECTIVE.value,
            resolvable=False,
            effective_capabilities=(),
            grant_chain=(),
            basis_event_ids=supersession_basis,
            reason=refusal,
            effective_directive_id=None,
            supersession_chain=supersession,
        )
    superseded = {
        "effective_directive_id": effective_id,
        "supersession_chain": supersession,
    }

    by_id = _directive_events(events)
    chain: list[GrantLink] = []
    seen: set[str] = set()
    current: str | None = effective_id
    effective: set[str] | None = None

    while current is not None:
        recorded = by_id.get(current, [])
        if not recorded:
            reason = (
                f"directive {current} is not recorded"
                if current == directive_id
                else f"parent directive {current} is not recorded"
            )
            return DirectiveAuthorityStanding(
                directive_id=str(directive_id),
                source=GrantSource.RECORDED_DIRECTIVE.value,
                resolvable=False,
                effective_capabilities=(),
                grant_chain=tuple(chain),
                basis_event_ids=tuple(link.event_id for link in chain) + supersession_basis,
                **superseded,
                reason=reason,
            )
        if len(recorded) > 1:
            return DirectiveAuthorityStanding(
                directive_id=str(directive_id),
                source=GrantSource.RECORDED_DIRECTIVE.value,
                resolvable=False,
                effective_capabilities=(),
                grant_chain=tuple(chain),
                basis_event_ids=tuple(link.event_id for link in chain) + supersession_basis,
                **superseded,
                reason=f"directive {current} has more than one recorded registration",
            )
        if current in seen:
            return DirectiveAuthorityStanding(
                directive_id=str(directive_id),
                source=GrantSource.RECORDED_DIRECTIVE.value,
                resolvable=False,
                effective_capabilities=(),
                grant_chain=tuple(chain),
                basis_event_ids=tuple(link.event_id for link in chain) + supersession_basis,
                **superseded,
                reason=f"the recorded chain revisits {current}: it is a cycle, not a hierarchy",
            )
        seen.add(current)

        event = recorded[0]
        capabilities = _capabilities_of(event)
        parent = event.payload.get("parent_directive_id")
        link = GrantLink(
            directive_id=current,
            event_id=event.event_id,
            capabilities=capabilities,
            parent_directive_id=str(parent) if parent else None,
        )
        chain.append(link)

        if effective is None:
            effective = set(capabilities)
        elif not set(chain[-2].capabilities).issubset(set(capabilities)):
            # A child that is not a subset of its parent cannot have narrowed it.
            # Registration refuses that now, but a ledger written earlier can
            # still contain one, and it must not be silently intersected away.
            return DirectiveAuthorityStanding(
                directive_id=str(directive_id),
                source=GrantSource.RECORDED_DIRECTIVE.value,
                resolvable=False,
                effective_capabilities=(),
                grant_chain=tuple(chain),
                basis_event_ids=tuple(link.event_id for link in chain) + supersession_basis,
                **superseded,
                reason=(
                    f"directive {chain[-2].directive_id} does not narrow its recorded parent "
                    f"{current}"
                ),
            )
        else:
            effective &= set(capabilities)
        current = link.parent_directive_id

    reason = f"resolved from {len(chain)} recorded directive(s)"
    if len(supersession) > 1:
        reason += f", after {len(supersession) - 1} recorded authority transition(s)"
    return DirectiveAuthorityStanding(
        directive_id=str(directive_id),
        source=GrantSource.RECORDED_DIRECTIVE.value,
        resolvable=True,
        effective_capabilities=tuple(sorted(effective or set())),
        grant_chain=tuple(chain),
        basis_event_ids=tuple(link.event_id for link in chain) + supersession_basis,
        reason=reason,
        **superseded,
    )


def standing_from_caller(authority: Authority | None) -> DirectiveAuthorityStanding:
    """A caller-supplied grant, recorded as exactly that."""
    capabilities = tuple(sorted(str(c) for c in (authority.capabilities if authority else ())))
    return DirectiveAuthorityStanding(
        directive_id=None,
        source=GrantSource.CALLER_SUPPLIED.value,
        resolvable=True,
        effective_capabilities=capabilities,
        grant_chain=(),
        basis_event_ids=(),
        reason="no directive was named; the caller supplied this grant",
    )


def authorize(
    standing: DirectiveAuthorityStanding, capability: str | Capability
) -> AuthorizationDecision:
    """Decide one capability against one standing, keeping the reason attached."""
    value = str(Capability(str(capability)))
    if not standing.resolvable:
        status = (
            AuthorizationStatus.INVALID_CHAIN
            if standing.grant_chain or "narrow" in standing.reason or "cycle" in standing.reason
            else AuthorizationStatus.UNKNOWN_DIRECTIVE
        )
        return AuthorizationDecision(
            capability=value, status=status.value, reason=standing.reason, standing=standing
        )
    if standing.allows(value):
        return AuthorizationDecision(
            capability=value,
            status=AuthorizationStatus.GRANTED.value,
            reason=(
                f"{value} is in the effective grant "
                f"{{{', '.join(standing.effective_capabilities)}}}"
            ),
            standing=standing,
        )
    return AuthorizationDecision(
        capability=value,
        status=AuthorizationStatus.DENIED.value,
        reason=f"capability denied: {value}",
        standing=standing,
    )
