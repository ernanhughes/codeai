# Seam: action recovery

*Applied AI* Chapters 19 (observe the effect, not the report), 22 (retry ≠ replay ≠ duplicate) and
29 (composition) own this concept.

## What it establishes

Result status and knowledge of the effect are separate dimensions of one record.

```text
result status   SUCCEEDED | FAILED | DENIED      what the operation reported
effect state    NONE | UNKNOWN | OBSERVED        what the record establishes
```

The ordering that makes the second decidable:

```text
action.requested
      -> authority and precondition checks
action.execution_started      committed before adapter.execute()
      -> the adapter acts
action.completed              result plus the runtime's own observation
```

For actions recorded by a runtime that emits `execution_started`:

| Record | Effect | Next operation | Retry |
|---|---|---|---|
| requested, no execution_started | NONE | start the action | safe |
| execution_started, no completion | UNKNOWN | reconcile | never automatically |
| completed SUCCEEDED | OBSERVED | none | — |
| completed DENIED | NONE | none | — |
| completed FAILED after execution_started | UNKNOWN | reconcile | never automatically |
| completed FAILED before execution_started | NONE | fix the inputs | safe under a new key |
| completed with `reused_from_action_id` | NONE (no new effect) | none | — |

`FAILED` is never read as "nothing happened".

Reconciliation is append-only evidence about an unknown effect:
`EFFECT_CONFIRMED`, `NO_EFFECT_CONFIRMED`, `STILL_UNKNOWN`. It does not edit the action's history,
the reported status survives it, and `STILL_UNKNOWN` is a valid final answer. Reconciling an action
whose effect the record already settles is refused **durably**
(`action.reconcile_refused`), not only in memory.

`Runtime.open_effects()` lists every action whose effect is unknown — the orphan report Chapter 11
said the runtime recorded but never surfaced.

## Records written before this seam existed

An `action.requested` without `recovery_version` could have been interrupted on either side of the
adapter call. Those project as UNKNOWN with basis `record:pre-execution-marking`. The projection
does not retrofit certainty onto history it cannot see.

## Code

- `src/codeai/actions.py` — `EffectState`, `ActionWorkState`, `project_action_state`,
  `project_open_effects`, `reconcile_action`, `ReconciliationVerdict`, `ReconciliationRefused`.
- `src/codeai/runtime.py` — `Runtime.action_state`, `Runtime.open_effects`,
  `Runtime.reconcile_action`; `execute_action` commits `action.execution_started`.
- `src/codeai/ledger.py` — `SQLiteLedger.close()`.

## Tests

`tests/test_action_recovery.py` (19 tests), including a parametrized test of the governing question:
at every interruption point, can the runtime say whether another execution is safe, unsafe or
unresolved without guessing? Three existing event-sequence assertions in `tests/test_action_replay.py`
and `tests/test_runtime.py` were updated to expect `action.execution_started`; that change is the
ordering argument, not a cosmetic fix. Full suite: 476 passed.

## Example

`examples/applied_ai/ch22_action_recovery.py`, asserted by `tests/test_examples.py`.

## Repair W1-R1: the effect state stopped trusting the actor

The Wave 1 composition audit (gap 1) found the SUCCEEDED branch deriving effect
OBSERVED from the worker's report, while the runtime held two identical state readings it never
compared. Result status and effect state were separate dimensions everywhere except the one place a
lie would show.

Three dimensions now, not two:

```text
result status      what the actor said       SUCCEEDED / FAILED / DENIED
state observation  what the runtime read     CHANGED / UNCHANGED / UNAVAILABLE
effect state       what the record supports  NONE / REPORTED / UNKNOWN / OBSERVED
```

`action.execution_started` carries `state_hash_before`: the reading the runtime already took for the
precondition check, now recorded rather than discarded. `action.completed` already carried the
reading after. The projection compares them:

| Report | Observation | Effect | Next |
|---|---|---|---|
| succeeded | CHANGED | OBSERVED | none |
| succeeded | UNCHANGED | **UNKNOWN** | reconcile |
| succeeded | UNAVAILABLE | **REPORTED** | none |
| failed after start | any | UNKNOWN | reconcile |
| failed before start | any | NONE | fix inputs |
| denied | any | NONE | none |

`REPORTED` is new and deliberately weaker than `OBSERVED`: an effect was claimed and nothing
corroborates it, which is not the same as UNKNOWN, where something contradicts it.

**CHANGED does not mean the intended effect occurred.** It means the observed scope is not what it
was. A worker that writes the wrong thing yields OBSERVED. That is verification's question, and
`examples/applied_ai/ch19_effect_observation.py` shows both workers reaching the same effect state
and opposite verdicts.

## What it still does not establish

1. **No atomicity.** The adapter acts and the ledger hears afterwards. The seam makes the gap
   *visible and decidable*; it does not close it.
2. **No concurrency safety.** Nothing spans the check-to-act interval. Two processes can still
   submit the same key and both execute.
3. **Reconciliation evidence is unvalidated.** `evidence_refs` are strings the caller supplies. The
   runtime records who said what, not whether it is true.
4. **Effect scope is still the resolver's.** Both readings cover whatever the configured resolver
   reads. An effect outside it is invisible, which is exactly why UNCHANGED projects UNKNOWN rather
   than "nothing happened".
5. **No compensation.** `EFFECT_CONFIRMED` records that the world changed; undoing it is not a
   runtime operation.
6. **Pre-seam records stay unknown forever** unless someone reconciles them.
7. **Single writer.** Unchanged.
8. **The two readings are not a transaction.** Anything else may have moved the scope between them,
   so CHANGED attributes the change to this action only by proximity.
9. **`REPORTED` cannot be reconciled.** Reconciliation accepts only UNKNOWN, so a deployment with
   no resolver stays at the actor's word with no way to record later evidence about it.
