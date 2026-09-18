# W1-R1: effect observation — repair of composition audit gap 1

**Cites:** `experiments/W1-composition-results.md` §4 gap 1 (frozen; not edited).
**Baseline subject:** runtime `f7d2910`, audit output `W1-composition-results.json` (`0ab2cc2`).
**Recheck:** `experiments/W1-R1-composition-recheck.json`, same harness, same probes.
**Author decision this implements:** three dimensions — report / observation / verification — with
the caution that CHANGED does not mean the intended effect occurred.

## The gap

```text
worker changes nothing, reports SUCCEEDED
  result status   succeeded          the worker's word
  effect state    OBSERVED           derived from that word
  runtime reading pre == post        held, and never compared
```

The projection's own reason string said "completed with the runtime's own observation of the
resulting state". It had a reading; it had never compared it to anything.

## The repair

`action.execution_started` now records `state_hash_before` — the reading the runtime already took
for the precondition check, previously discarded. The completion already carried the reading after.
The projection compares them and reports a third fact of its own.

```text
result status      what the actor said       SUCCEEDED / FAILED / DENIED
state observation  what the runtime read     CHANGED / UNCHANGED / UNAVAILABLE
effect state       what the record supports  NONE / REPORTED / UNKNOWN / OBSERVED
```

| Report | Observation | Effect state | Next operation | Why |
|---|---|---|---|---|
| succeeded | CHANGED | OBSERVED | none | two readings differ across the effect |
| succeeded | UNCHANGED | **UNKNOWN** | reconcile | report and readings disagree; neither wins |
| succeeded | UNAVAILABLE | **REPORTED** | none | nothing could look; the claim is all there is |

`REPORTED` is a new `EffectState`, weaker than `OBSERVED` and different from `UNKNOWN`: nothing
corroborates the claim, and nothing contradicts it either. Without it, a deployment with no state
resolver would have had to choose between calling every report an observation (the gap) and sending
every successful action to a human (unusable).

**UNCHANGED → UNKNOWN, not "no effect".** The effect may have landed outside the resolver's scope,
or written identical bytes. The record cannot tell, so it says so and asks for reconciliation.

**CHANGED ≠ the intended effect.** Kept explicit in the projection's reason string, in the seam doc,
and executably in the example: a diligent worker and a busy worker both reach OBSERVED, and only the
declared check separates PASS from FAIL.

## Recheck against the frozen baseline

Same harness, all 13 probes, one classification changed:

```text
D  action request -> effect     RECORDED  ->  DERIVED
```

Baseline observation → repaired observation:

```text
projected effect state: observed      ->  projected effect state: unknown
effect_state trusts the report alone  ->  False
```

Everything else is identical, including the happy path (CALL → CHECK → ASK_HUMAN → STOP, reopen
identical, status `completed`). The happy path's action still projects OBSERVED, because its worker
genuinely changed the file — the repair does not make honest work look suspicious.

Probe D now computes its classification from what it sees rather than asserting one, so the same
harness reproduces the baseline at `f0c730b` and this result after the repair.

## Evidence

- `src/codeai/actions.py` — `StateObservation`, `EffectState.REPORTED`, three-way SUCCEEDED branch,
  `state_hash_before` and `state_observation` on `ActionWorkState`.
- `src/codeai/runtime.py` — `state_hash_before` on `action.execution_started`.
- `tests/test_action_recovery.py` — three new cases (CHANGED → OBSERVED, UNCHANGED → UNKNOWN,
  UNAVAILABLE → REPORTED); four existing expectations updated because the behaviour genuinely
  changed, each fixture now saying which of the three situations it is in.
- `examples/applied_ai/ch19_effect_observation.py` + `tests/test_examples.py`.
- `docs/seams/action-recovery.md` — repair section and two new limitations.
- Full suite: **565 passed**.

## What this repair does not do

1. **It does not establish that an effect happened.** It establishes what the record can support,
   which is the only thing a record can do.
2. **The two readings are not a transaction.** Anything else may move the scope between them, so
   CHANGED attributes the change to this action by proximity alone.
3. **`REPORTED` cannot be settled at all.** Reconciliation accepts only UNKNOWN, so later evidence
   about a reported-but-unobserved effect has nowhere to go. Queued as a small follow-up rather than
   widened here, because reconciliation's refusal rule is itself load-bearing.
4. **Scope is still the resolver's** — one hash for everything it watches, nothing outside it.
5. **An idle worker is not identified as dishonest.** UNCHANGED is a fact about the reading, not an
   accusation about the actor.
