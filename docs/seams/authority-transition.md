# Seam: authority transition (extension W1-E1)

*Applied AI* Chapters 20 (capability is not authority) and 29 (composition) own this concept.

**This is an extension, not a repair.** The five Wave 1 repairs fixed things the runtime implied,
trusted or failed to enforce. This adds a legal state transition to a system that was already doing
the safe thing: refusing.

## The question it answers

> When the durable process reaches a state from which completion is unauthorized, is there a lawful
> way forward that does not involve bypassing the authority model?

Composition-audit probe N, at every point from the baseline through W1-R5:

```text
directive grants WRITE only
  scheduler            ASK_HUMAN   "next effect requires human authority"
  a human accepting    refused: acceptance_not_granted
  the task waits at    ASK_HUMAN, permanently
```

## The principle

> **Human intervention may change durable authority. It does not bypass durable authority.**

So the path is:

```text
current durable authority has no ACCEPT
      -> the scheduler asks for human intervention
      -> the human records an authority transition
      -> re-project
      -> acceptance is enforced by the ordinary rule
      -> completion
```

and never a special accept-anyway call.

## Delegation and transition are different relationships

```text
delegation            parent -> child
                      effective(child) is a subset of effective(parent)
                      a widening child is invalid, always

authority transition  old directive -> superseding directive
                      may add, remove or otherwise alter authority, because it
                      records a new external decision rather than a delegated
                      child claiming powers its parent lacked
```

A superseding directive therefore takes **no parent**: it declares its own authority and inherits
nothing. Declaring a parent is refused, because narrowing under a parent is delegation wearing a new
name.

## Directives are immutable

Nothing is mutated. `authority.transitioned` records the successor and leaves the predecessor exactly
as it was:

```text
previous_directive_id            new_directive_id
actor_id                         source (human_intervention | external_decision)
reason                           previous_effective_capabilities
new_effective_capabilities       previous_basis_event_ids
new_directive_event_id           version
```

so history reads as history:

```text
T1  WRITE
T2  WRITE + ACCEPT
T3  READ
```

rather than today's authority appearing to have existed yesterday. `Runtime.authority_history`
returns the chain; a completed acceptance keeps the basis it was granted under, and a later
transition does not reach back and change it.

## Resolution

`resolve_directive_authority` now resolves **supersession first, then delegation**: which directive
is in force now, then what that directive and its recorded parents jointly allow. The standing
carries `effective_directive_id` and the `supersession_chain`, and both appear in every authority
basis payload, so a decision can be reconstructed rather than re-derived.

Refusals, all durable as `authority.transition_refused`:

```text
unknown or unresolvable predecessor
a predecessor that has already been superseded   (supersede the successor instead)
a superseding directive that declares a parent
a transition with no recorded reason
```

A ledger that somehow holds **two successors for one epoch** is not resolved to either: the standing
becomes unresolvable with reason "the effective authority is ambiguous". An ambiguous authority is
not an authority.

## Connection to the other seams

`ProcessState` now carries `directive_effective_capabilities`, so an authority change moves the state
digest that scheduler decisions are checked against (W1-R4). A decision taken before a transition is
`stale_basis` afterwards, and a fresh decision must be taken — the world the old one described has
genuinely changed.

## Evidence

`tests/test_authority_transition.py` (14): the probe-N path end to end; a transition that takes
authority away; a completed acceptance that a later transition does not rewrite; a child that still
cannot widen its parent; a successor that may not declare one; unknown predecessor, missing reason
and already-superseded refusals; two competing successors refusing rather than choosing; the history
reading back as history with predecessors untouched; reopen reconstructing the same effective
authority; a transition invalidating an outstanding scheduler decision; and the actor staying
attribution.

Full suite: **629 passed**. Composition check: probe N `ABSENT` → `ENFORCED`, nothing else moved.

## What it still does not establish

1. **No authentication.** What this establishes is that *an explicit external authority change,
   attributed to this actor, entered the durable process at this point*. Not that the actor was
   entitled to make it. That remains the outer trust boundary.
2. **No approval workflow.** One recorded transition is one decision; there is no notion of a second
   signature, a quorum or an expiry.
3. **No scope on a transition.** It changes a directive's capabilities wholesale; it cannot grant
   ACCEPT "for this one task" or "until Friday".
4. **Tasks still name the directive they were created under.** Resolution follows the chain from
   there, so a task cannot be moved to an unrelated directive.
5. **A transition is not retroactive and not compensating.** Work already accepted stays accepted.
6. **Ambiguity is refused, not repaired.** A ledger with two successors for one epoch stays
   unresolvable until someone records something that resolves it.
