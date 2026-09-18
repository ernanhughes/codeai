# W1-E1: authority transition — extension, not repair

**Motivated by:** composition-audit probe N, raised by W1-R2 and preserved as `ABSENT` through R5.
**Check:** `experiments/W1-E1-composition-check.json`, same harness.
**Seam doc:** `docs/seams/authority-transition.md`.

## Why this is not Repair 6

The five repairs fixed things the runtime *incorrectly implied, trusted, or failed to enforce*:

```text
effect observation              a report read as an observation
acceptance authority            a caller trusted as an authority
artifact binding                a label trusted as an identity
decision/execution binding      a decision that governed nothing
claim attempt trace             an attempt that left no trace
```

Probe N is different in kind. The process reached a state from which completion is **unauthorized**,
and refused — the correct behaviour. What was missing was a *legal transition out of it*. Adding one
is extension work.

```text
CORE COMPOSITION REPAIRS
  effect observation              CLOSED
  acceptance authority            CLOSED
  artifact binding                CLOSED
  decision/execution binding      CLOSED
  claim attempt trace             CLOSED

EXTENSION
  authority transition            W1-E1
```

## The principle

> Human intervention may change durable authority. It does not bypass durable authority.

The gap-4 override shape (`source = human_override`, `governed_by_scheduler = false`) is right for
*scheduler governance*, where a person may legitimately act outside the scheduler. It is wrong for
*authority*, which is the boundary deciding whether an operation may happen at all. The two cases
stay different, and this extension is the authority one.

## What probe N does now

```text
scheduler: ASK_HUMAN (next effect requires human authority)
a human accepting without changing authority -> rejected: acceptance_not_granted:d-root
the task waits at: ASK_HUMAN

after a recorded authority transition: accept granted = True, effective directive = d-root-accept
the same acceptance, under the ordinary rule -> completed
the task now ends at: STOP
```

The acceptance itself is unchanged code: the ordinary W1-R2 rule, resolving the ordinary chain. The
only new thing is that the chain now resolves to a different directive.

## Design decisions worth recording

**Directives are immutable; supersession is a separate relation.** Nothing is mutated, so `T1 WRITE`,
`T2 WRITE+ACCEPT`, `T3 READ` all remain readable as what was true then. A completed acceptance keeps
the basis it was granted under.

**A successor takes no parent.** Delegation may only narrow; a transition may add or remove, because
it records a new external decision. Allowing a successor to sit under a parent would let one
mechanism impersonate the other, so it is refused.

**Two successors for one epoch are refused, not resolved.** `transition_authority` refuses to
supersede an already-superseded directive, and a ledger that somehow holds the ambiguity resolves to
*unresolvable* rather than picking one. An ambiguous authority is not an authority.

**Transitions expire outstanding decisions naturally.** `ProcessState` now carries
`directive_effective_capabilities`, so an authority change moves the digest W1-R4 checks decisions
against. A decision taken before the transition is `stale_basis` afterwards. The seams connect
without either knowing about the other.

**One improvement fell out of the work:** an acceptance refusal now names the *effective* directive
rather than the one the task happens to name. After a transition, the successor is what denied it,
and naming it is what tells an operator where to look.

## Composition check

```text
N  scheduler human gate -> acceptance authority     ABSENT  ->  ENFORCED
```

Nothing else moved. The matrix is now **12 ENFORCED, 2 DERIVED**.

## What it still does not establish

1. **Attribution, not authentication.** The record says an explicit transition attributed to this
   actor entered the process here. It does not say the actor was entitled to make it. That is the
   outer trust boundary, unchanged.
2. **No approval workflow** — no second signature, quorum or expiry.
3. **No scope** — a transition changes a directive's capabilities wholesale, not "for this task" or
   "until Friday".
4. **Not retroactive, not compensating** — work already accepted stays accepted.
5. **Ambiguity is refused, not repaired.**
