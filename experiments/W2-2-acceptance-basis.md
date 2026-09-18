# W2-2: the acceptance basis

**Protocol:** `experiments/W2-2-prereg.md`, registered before any implementation choice.
**Case A evidence:** `experiments/W2-2-case-a-results.json`, produced by
`experiments/w2_2_acceptance_basis_probe.py`.
**Subject:** CodeAI `c1d4418` for the measurement; the implementation follows it.

## Case A ran first, and the gap is real

The falsifier: if a second process can already derive the action behind an acceptance from the
durable record, this relationship is DERIVED and no seam is needed. Six shapes, one derivation
procedure — from the acceptance's check ids, to each check's runtime observation, to the actions
whose completion observed that state:

```text
derivable  shape
   yes     one action, bound check
   NO      a second action that changed nothing        2 candidates
   NO      another task's action on the same state     2 candidates
   NO      the action replayed under its key           2 candidates
   NO      an acceptance-eligible check with no state binding   0 candidates
   NO      the world moved after the action            0 candidates
```

**Derivable in one shape out of six.** Worse than the chapter suggested: the failing row that matters
most is the *ordinary* one. An acceptance-eligible check binds an **artifact** (W1-R3); binding a
state as well is optional, and when it is absent no check carries a runtime observation, so there is
nothing to match an action against at all.

The registered prediction was that ambiguity would come from two actions observing the same state.
It did — three different ways, including a replay of the same operation.

## What was built

The basis is explicit and optional, because the alternative is worse than the gap:

```text
acceptance basis
  |- artifact / check basis     "I accept this verified result"
  `- effect / action basis      "I accept this verified result as the outcome of action A"
```

A task may legitimately accept an artifact that was imported, hand-written or already correct.
Forcing a fictional action into that history to satisfy a schema would manufacture a claim rather
than record one. So `AcceptanceRequest.effect_action_id` defaults to `None`, and an acceptance that
names no action is unchanged.

An acceptance that **does** name one has it enforced:

```text
action.completed
      -> the runtime's own observation H
      -> a cited check bound to H
      -> PASS
      -> acceptance naming that action
```

| Refusal | When |
|---|---|
| `effect_action_unknown:<id>` | no such action is recorded |
| `effect_action_wrong_task:<id>` | the action belongs to another task |
| `effect_action_not_observed:<id>:<state>` | the effect is `unknown`, `reported` or `none` |
| `effect_action_state_unchecked:<id>` | no cited check examined and passed that observed state |

`task.accepted` now carries `acceptance_basis`, and an acceptance with no effect basis records
`{"kind": "artifact_check", "effect": null}` — absence as a recorded fact rather than something a
reader infers from silence.

## What it establishes, said in the record itself

The payload carries this sentence, so no later reader promotes it:

> the action was followed by this runtime observation, and an accepted check examined it; not that
> the action caused it

Chapter 19's limit is unchanged: readings either side of an effect are proximity, not a transaction.
The right name for this is **action → accepted-state binding**, never causal proof.

## Evidence

`tests/test_acceptance_basis.py` (10), one per registered case:

| Case | Result |
|---|---|
| C — claims the action that produced the checked state | accepted, basis recorded with the action's completion event |
| G — claims no action | accepted, basis recorded as `artifact_check` with `effect: null` |
| B — claims an action whose state was not the checked one | `effect_action_state_unchecked`; naming the right one succeeds |
| D — claims another task's action | `effect_action_wrong_task` |
| E — claims a failed action while a later check passes | refused, **and the task still completes on the ordinary basis** |
| F — claims an `unknown` or `reported` effect | refused in both, with the state named |
| H — accepts the current state citing an older action's chain | refused; the older check still stands for what it examined |
| — unknown action | refused rather than ignored |
| — reopen | every element of the basis re-derives from the reopened ledger |
| — identity | naming an action makes it a different acceptance |

Case F needed no new machinery: W1-R1's `UNKNOWN` and `REPORTED` did the refusing, exactly as
predicted.

Full suite: **650 passed**. Wave 1 composition audit re-run: 12 ENFORCED, 2 DERIVED, nothing moved.

## What it still does not establish

1. **Not causality.** Only that the action was followed by an observation an accepted check examined.
2. **One action per acceptance.** A result produced by several actions has no way to say so.
3. **Only the runtime's observation scope.** An effect outside the resolver's reach cannot be the
   basis of anything, because it was never observed.
4. **The claim is still the acceptor's to make.** Nothing requires an acceptance to name the action
   that in fact produced the state; it requires that a named one holds up.
5. **The artifact and the state remain separately bound.** This seam links an action to the
   *observed state* a check examined, not to the artifact bytes being accepted — those are bound by
   W1-R3, and nothing forces the two bindings to describe the same thing.
