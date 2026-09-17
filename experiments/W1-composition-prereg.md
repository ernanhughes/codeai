# Pre-registration: Wave 1 composition audit

**Registered:** 2026-09-18, before any probe was executed.
**Subject:** CodeAI at `11fedcd` (main, after the four Wave 1 seams are merged).
**Status when registered:** protocol frozen; harness not yet written; no probe results observed.

## Purpose

Wave 1 produced four individually tested seams:

1. action/effect recovery (`actions.py`)
2. recorded-directive authority (`authority.py`)
3. durable process state and scheduler decisions (`process_state.py`, `scheduler.py`)
4. verification extraction, binding and four-verdict semantics (`verification.py`)

Each has its own tests. That is not evidence that they compose. This audit asks one question:

> Can a second process, using only the durable record and the runtime's rules, explain and enforce
> why this task was allowed to move from intent to effect to verification to acceptance — and refuse
> every transition whose basis is missing?

## Rules of this audit

1. **No repair during discovery.** The baseline is frozen before any finding is fixed. Runtime
   behaviour is not modified while evidence is gathered.
2. **Observed, not assumed.** Every classification must come from an executed probe against the
   public runtime API. A property that cannot be expressed through that API is itself a finding.
3. **Fake cognition only.** Model quality is irrelevant here; the subject is runtime composition.
4. **A probe that cannot be written is recorded as `UNTESTABLE`**, with the reason, rather than
   silently dropped.

## The chain under audit

```text
recorded directive -> projected facts -> next-operation decision -> recorded authority
-> effect request -> effect / effect uncertainty -> observation or reconciliation
-> bound verification -> claim/task decision -> acceptance -> completion -> replay
```

For every arrow: is the relationship **enforced** by the runtime, **derived** from durable facts,
merely **recorded**, **conventional**, or **absent**?

## Classification vocabulary (fixed)

| Class | Meaning |
|---|---|
| **ENFORCED** | The runtime derives or checks the relation from durable facts and refuses a violation. |
| **DERIVED** | Deterministically reconstructable from durable facts, but not an execution gate. |
| **RECORDED** | Persisted, but trusted from the caller or producer rather than independently checked. |
| **CONVENTIONAL** | Holds only because callers currently cooperate. |
| **ABSENT** | The runtime does not represent the relationship at all. |
| **UNTESTABLE** | Cannot be expressed through the public runtime API (a finding in itself). |

## Joints to be audited

1. Directive → action authority; directive → acceptance authority
2. Durable state → next-operation decision (recomputable ≠ recorded ≠ causally binding)
3. Decision → execution
4. Action request → effect
5. Effect → recovery/reconciliation
6. Observed state → verification binding
7. Verification command → verdict semantics
8. Verification → claim evidence (including: is an errored attempt discoverable from the claim?)
9. Verification → acceptance
10. Acceptance → completion
11. Completion/replay → current authority

## Hostile cases (registered before execution)

```text
A  recorded directive denies WRITE, caller supplies WRITE            -> must not execute
B  recorded directive denies ACCEPT, caller supplies ACCEPT          -> observe; freeze asymmetry
C  decision says CHECK, caller submits an action                     -> observe enforcement
D  action reports success, observed state disagrees                  -> verification binds to observation
E  target changes between requested and observed state               -> no silent PASS on stale state
F  verification returns ERROR                                        -> claim must not be refuted
G  verification returns INCONCLUSIVE                                 -> claim unresolved, attempt durable
H  authorized action replayed after the grant narrows                -> which authority governs?
I  effect may have happened, completion absent                       -> retry must not be inferred safe
J  PASS exists for state A, acceptance targets state B               -> A's PASS must not establish B
```

Additional registered probes: decision→execution with the correct capability but a different
instruction or target; execution with no recorded decision at all; reopen/reprojection equality.

## Expected findings (registered in advance, so the audit cannot be read backwards)

These are predictions, not results. Recording them now makes a surprise visible as a surprise.

1. `directive → action authority` is **ENFORCED**; `directive → acceptance authority` is
   **RECORDED** at best (caller-supplied `Authority`). Already recorded in the book ledger as the
   authority asymmetry.
2. `decision → execution` is expected to be **CONVENTIONAL**: the scheduler records a decision and
   nothing binds a later effect to it. If confirmed, this outranks the authority asymmetry.
3. Effect-unknown is expected to survive composition (seam 1 refuses to call it retry-safe).
4. Verification binding is expected to hold against a forging verifier, and binding failure to stay
   ERROR through the claim path.
5. An **errored** verification attempt is expected to be discoverable only through `check.*` events,
   not from the claim — "no negative evidence" and "no discoverable attempt" may currently be the
   same thing. Registered as a distinct property to test.

## Deliverables

- `experiments/wave1_composition_audit.py` — the harness (one lifecycle plus the hostile cases).
- `experiments/W1-composition-results.json` — frozen machine-readable result, including the subject
  commit and the environment.
- `experiments/W1-composition-results.md` — composition matrix, happy-path trace, hostile-case table,
  book-core gaps, conventional joints, repair ordering.
- `tests/test_composition_audit.py` — the audit's own assertions, so a later repair cannot silently
  change what the baseline said without the frozen document disagreeing.

The frozen result is never edited when a later seam repairs a finding. Repairs are recorded as new
entries that cite it.
