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
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .domain import Authority, Capability
from .ledger import Event

AUTHORITY_RESOLUTION_V1 = "authority-resolution-v1"


class GrantSource(StrEnum):
    RECORDED_DIRECTIVE = "recorded_directive"
    CALLER_SUPPLIED = "caller_supplied"


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


def resolve_directive_authority(
    events: tuple[Event, ...], directive_id: str | None
) -> DirectiveAuthorityStanding:
    """Walk the recorded directive chain and intersect the grants. Appends nothing."""
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

    by_id = _directive_events(events)
    chain: list[GrantLink] = []
    seen: set[str] = set()
    current: str | None = str(directive_id)
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
                basis_event_ids=tuple(link.event_id for link in chain),
                reason=reason,
            )
        if len(recorded) > 1:
            return DirectiveAuthorityStanding(
                directive_id=str(directive_id),
                source=GrantSource.RECORDED_DIRECTIVE.value,
                resolvable=False,
                effective_capabilities=(),
                grant_chain=tuple(chain),
                basis_event_ids=tuple(link.event_id for link in chain),
                reason=f"directive {current} has more than one recorded registration",
            )
        if current in seen:
            return DirectiveAuthorityStanding(
                directive_id=str(directive_id),
                source=GrantSource.RECORDED_DIRECTIVE.value,
                resolvable=False,
                effective_capabilities=(),
                grant_chain=tuple(chain),
                basis_event_ids=tuple(link.event_id for link in chain),
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
                basis_event_ids=tuple(link.event_id for link in chain),
                reason=(
                    f"directive {chain[-2].directive_id} does not narrow its recorded parent "
                    f"{current}"
                ),
            )
        else:
            effective &= set(capabilities)
        current = link.parent_directive_id

    return DirectiveAuthorityStanding(
        directive_id=str(directive_id),
        source=GrantSource.RECORDED_DIRECTIVE.value,
        resolvable=True,
        effective_capabilities=tuple(sorted(effective or set())),
        grant_chain=tuple(chain),
        basis_event_ids=tuple(link.event_id for link in chain),
        reason=f"resolved from {len(chain)} recorded directive(s)",
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
