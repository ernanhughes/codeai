# W1-R4: decision–execution binding — repair of composition audit gap 4

**Cites:** `experiments/W1-composition-results.md` §4 gap 4 (frozen; not edited).
**Recheck:** `experiments/W1-R4-composition-recheck.json`, same harness.
**Author decision this implements:** enforce the arrow; do not redraw the lifecycle. Scoped to the
scheduler-governed path, not to every runtime call.

## The gap

```text
recorded decision: CHECK
caller executes an action anyway -> succeeded
with no decision recorded at all -> succeeded
action.requested references a decision: False
```

The scheduler decided correctly, recorded the decision with its basis, and nothing connected that
decision to what happened next in either direction.

## The rule

> A process decision must never be silently bypassed while the resulting operation still appears to
> belong to that governed process.

Two legitimate shapes, told apart in the record:

```text
scheduler-governed   names a decision; the runtime checks it and refuses a mismatch
external or manual   names none; the source is recorded, and it cannot masquerade as governed
```

A claim is checked on four questions, all of which must hold:

```text
the decision is recorded
it belongs to this task
it selected this operation class
it still refers to the state it was derived from
```

`codeai/governance.py` carries the vocabulary: `GovernanceSource` (scheduler, human_override,
external_request) and `GovernanceStatus` (governed, ungoverned, unknown_decision, wrong_task,
wrong_operation, stale_basis, not_selectable). Every governed-path operation appends
`operation.governance_recorded` or `operation.governance_refused` carrying the standing, the decision
event id, and both state digests.

Governed entry points: `invoke_recorded_call` (CALL), `run_check` (CHECK), `execute_action` (ACTION),
each taking `decision_id` and `source`. The default is `external_request`, so an operation is never
*accidentally* governed.

### Governance is not authority

```text
decision permits this operation class?   governance
authority permits this capability?       the directive chain
precondition still holds?                the operation
execute
```

Both gates run, in that order, and a refusal at either is durable. Probe C's `d-read` case shows
governance permitting an external action that authority then denies.

### The scheduler cannot select ACTION

The policy has no ACTION output, by design: effects are gated by authority, not by the scheduler. So
no action can be scheduler-governed, and an action naming any decision is refused as
`not_selectable`. This is the book's position made enforceable rather than assumed — *"a CHECK
decision cannot silently become a WRITE action"* holds because no decision can ever license a write.

## The hostile matrix, frozen

`tests/test_decision_binding.py` (13):

| Decision | Governed operation requested | Result |
|---|---|---|
| CALL | CALL | allowed, recorded `governed` |
| CHECK | CHECK | allowed, recorded `governed` |
| CHECK | ACTION | refused `not_selectable`, adapter never called |
| ASK_HUMAN | CALL | refused `wrong_operation` |
| STOP | anything governed | refused |
| unknown decision id | anything | refused `unknown_decision` |
| decision from another task | anything | refused `wrong_task` |
| stale basis | anything | refused `stale_basis`, both digests recorded |
| `source=scheduler`, no decision | anything | refused — the masquerade case |
| no claim | appropriate operation | allowed, recorded `ungoverned` / `external_request` |
| `source=human_override` | appropriate operation | allowed, recorded as an override |

## Freshness, deliberately conservative

The basis is the digest of the whole projected process state — the same digest the decision recorded.
If it has moved, the decision no longer describes the world it was about, and it permits nothing.

This is coarse on purpose: it cannot tell a relevant change from an irrelevant one, so **any recorded
operation for the task invalidates outstanding decisions**. Probe C shows exactly that: after an
external action ran, the earlier CHECK decision was stale, and a decision taken on the current state
was required. Erring toward refusal is the right direction for a first version; a relevance model is
not attempted.

It is also not a transaction. The world may move between the check and the effect. What the record
establishes is that the operation was *launched* under a decision that still described the world at
launch.

## A defect this repair found in itself

The governance event initially leaked across a sealed fan-out: `test_sibling_seal_excludes_all_new_
event_kinds` caught a branch seeing a sibling's governance record, because the payload did not carry
the call identity the seal filters on. Fixed by naming the subject in `call_id`/`lineage_ids`, so a
governance event is excludable exactly like the operation it describes (Chapter 23). The test was
written for precisely this: every new event kind must be excludable.

## Evidence

- `src/codeai/governance.py` — new module.
- `src/codeai/runtime.py` — three governed entry points, `operation_governance` projection.
- `src/codeai/verification.py` — `run_check` governs before binding; a refused check is ERROR, since
  a check that may not run produced no verification result at all.
- Sixteen existing tests updated: each asserted an exact event sequence, and every governed-path
  operation now records what it was launched under. Call governance sits *after* spec validation, so
  "a malformed request is refused before any event" still holds.
- Full suite: **604 passed**.

## Recheck against the frozen baseline

```text
C  decision -> execution     CONVENTIONAL  ->  ENFORCED
```

Nothing else moved.

## What this repair does not do

1. **It binds the operation class, not the parameters.** A governed CHECK decision permits *a* check,
   not a particular target, command or artifact.
2. **Freshness is whole-state.** Coarse, as above.
3. **The default is external.** An operation is never accidentally governed, but it can be
   accidentally external: a caller who simply forgets to pass a decision gets an honest
   `ungoverned` record rather than a refusal.
4. **`human_override` is attribution, not authentication.** The record says an override was claimed,
   never who claimed it.
5. **Nothing consumes governance downstream yet.** Acceptance and the projections do not require an
   operation to have been governed.
6. **ACTION is unreachable from the scheduler**, so the governed path for effects is empty by
   construction. If effects should ever be scheduler-selected, that is a policy change, not a
   governance one.
