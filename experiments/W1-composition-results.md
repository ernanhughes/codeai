# Results: Wave 1 composition audit

**Protocol:** `experiments/W1-composition-prereg.md` (registered before the harness existed).
**Subject:** CodeAI `11fedcd` — main, after all four Wave 1 seams merged.
**Harness:** `experiments/wave1_composition_audit.py` · **Frozen output:** `W1-composition-results.json`
**Pinned by:** `tests/test_composition_audit.py` (23 assertions, including the gaps).
**Repairs applied during this audit:** none.

> Can a second process, using only the durable record and the runtime's rules, explain and enforce
> why this task was allowed to move from intent to effect to verification to acceptance — and refuse
> every transition whose basis is missing?

**Answer: it can explain, but it cannot enforce.** Every transition is reconstructable from the
record. Four of them are not gated by it.

---

## 1. Composition matrix

| # | Joint | Producer → Consumer | Durable basis | Class | Caller can forge or bypass? | Failure state |
|---|---|---|---|---|---|---|
| A | directive → action authority | `directive.opened` → `execute_action` | resolved chain, intersected | **ENFORCED** | No — a forged caller grant changes nothing | `action.authorization_refused` with the chain as basis |
| B | directive → acceptance authority | `directive.opened` → `accept_task` | none: the caller's `Authority` argument | **RECORDED** | **Yes** — any caller may claim ACCEPT | none; the task completes |
| C | decision → execution | `scheduler.decision_recorded` → `execute_action` | none: no reference in either direction | **CONVENTIONAL** | **Yes** — effects run with any decision, or none | none |
| D | action request → effect | adapter report → `action.completed` | worker's status, plus the runtime's own pre/post readings | **RECORDED** | **Yes** — a worker that does nothing reports SUCCEEDED | none; effect projects OBSERVED |
| E | effect → reconciliation | `action.execution_started` → `project_action_state` | the pre-effect marker | **ENFORCED** | No | UNKNOWN + `RECONCILE_EFFECT`, scheduler escalates |
| F | observed state → verification binding | `state_resolver` → `bind_check_target` | the runtime's own single reading | **ENFORCED** | No — a verifier's self-report is discarded | ERROR, verifier never invoked |
| G | command → verdict semantics | declared policy → `CheckResult` | `verdict_policy` in `check.requested` | **ENFORCED** | No — unmapped or malformed becomes ERROR | ERROR with the reason |
| H | verification → claim evidence | `check.completed` → claim events | claim ids on the check request | **DERIVED** | No (ERROR writes nothing) | — |
| I | verification → acceptance | `check.completed` → `accept_task` | verdict + the check's `target` label | **RECORDED** | Partly — see M | `check_not_passed`, `check_wrong_artifact` |
| J | acceptance → completion | `task.accepted` → `task.completed` | causation pair, fully validated | **ENFORCED** | No | completion projects `incomplete` |
| K | replay → current authority | `idempotency_key` → authorize-then-replay | resolved chain at replay time | **ENFORCED** | No | denial, with no disclosure of the recorded operation |
| L | durable state → decision | ledger → `ProcessState` → policy | full state snapshot + sha256 + basis ids | **DERIVED** | A raw `SchedulerInput` call bypasses the projection, but records nothing | — |
| M | check → artifact identity | caller's `target` string → `accept_task` | a label, not a reading | **CONVENTIONAL** | **Yes** — a check that never opened the bytes can carry their label | none |

Sub-finding on K: **grant revocation is ABSENT.** Replay is authorized against the record *as it
stands*, which is the right rule, but a recorded grant cannot be narrowed — a second registration for
the same id is refused (durably). So "current authority" and "original authority" cannot currently
differ for a valid chain, and K's correctness is untested by the world.

---

## 2. Happy-path trace

One task, one directive chain, crossing every seam (`happy_path` in the harness):

```text
directive chain     d-child -> d-root, effective grant {write}
decision 1          CALL       task requests independent proposals        epistemic-v3
recorded call       offline synthetic transport, artifact 35f79d8b90a3...
decision 2          CHECK      required deterministic verification exists
effect              succeeded  effect OBSERVED, next NONE, adapter calls 1
verification        PASS       binding BOUND, observed 1e3f91e171d1...
decision 3          ASK_HUMAN  next effect requires human authority
acceptance          completed  on a caller-supplied Authority({ACCEPT})
decision 4          STOP       the process is already complete
```

Reopened from disk: events byte-identical, task status reprojects `completed`, next operation
reprojects `STOP`. **Reconstruction holds.**

---

## 3. Hostile-case table

| Case | Condition | Expected property | Observed | Class |
|---|---|---|---|---|
| A | record grants READ; caller passes WRITE+DESTRUCTIVE | must not execute | `denied`, 0 adapter calls, refusal carries the grant chain | ENFORCED |
| B | record grants no ACCEPT; caller passes ACCEPT | observe | **task completed**; `task.accepted` does not even name a directive | RECORDED |
| C | decision says CHECK; caller submits an action | observe | **action executed**; also executes with no decision recorded at all | CONVENTIONAL |
| D | worker reports success, changes nothing | report ≠ effect | **effect projects OBSERVED**; pre/post state hashes identical; a check of the intended post-condition returns FAIL | RECORDED |
| E | target changes between requested and observed state | no silent PASS | ERROR, binding MISMATCH, verifier never called | ENFORCED |
| F | verification returns ERROR | claim not refuted | claim stays `asserted`; no claim event written | ENFORCED |
| G | verification returns INCONCLUSIVE | attempt durable, status unmoved | `claim.check_inconclusive` recorded with reason; status `asserted` | ENFORCED |
| H | replay after attempted grant narrowing | which authority governs? | re-registration refused; replay decided against the current resolved chain; a capability the child lacks is denied on the same key | ENFORCED (revocation ABSENT) |
| I | effect may have happened, completion absent | retry not safe | effect UNKNOWN, `RECONCILE_EFFECT`, scheduler ASK_HUMAN, same-key retry replays FAILED without re-invoking the adapter | ENFORCED |
| J | PASS for artifact A, acceptance targets B | A's PASS must not establish B | `check_wrong_artifact:k-other` | ENFORCED |
| M | check labelled with an artifact it never read | — (added mid-audit) | **acceptance completed** | CONVENTIONAL |

---

## 4. Book-core gaps, ranked

Ranked by how much of the book's central argument they weaken — not by effort.

### Gap 1 — the effect projection answers Chapter 19's question with the agent's word

Chapter 19 is *"The Agent Said Done. Did Anything Change?"*. The runtime's own answer, for a worker
that did nothing and reported success:

```text
result status   succeeded          (the worker's word)
effect state    OBSERVED           (derived from that word)
projection says "completed with the runtime's own observation of the resulting state"
runtime's readings   pre be2fe827c17a   post be2fe827c17a   identical
```

Seam 1's whole premise is that result status and effect state are different dimensions. That holds
for FAILED (→ UNKNOWN) and DENIED (→ NONE) and collapses for SUCCEEDED. The runtime *holds* the
contradicting evidence — it read the state before and after — and never compares them. The reason
string in the projection overstates what happened: it observed *a* state, not *an effect*.

This was not predicted by the pre-registration. It is the audit's main discovery.

Not in scope for the repair: proving an effect occurred in general (the resolver's scope is limited,
and an unchanged hash does not prove nothing happened). In scope: refusing to call the effect
OBSERVED on a report alone, and recording the comparison the runtime already has.

### Gap 2 — acceptance authority is the caller's claim (the predicted asymmetry, confirmed)

```text
recorded directive -> action authority      ENFORCED
recorded directive -> acceptance authority  RECORDED (caller-supplied)
```

Executable reproducer: probe B. A directive granting only WRITE; a caller passing
`Authority({ACCEPT})`; the task completes. Worse than expected in one respect: `task.accepted`
carries no directive id at all, so the acceptance cannot even be *re-resolved* against a chain after
the fact. Any repair must add that reference before it can resolve anything.

### Gap 3 — acceptance trusts a label for the bytes a check examined

The verification seam established the *state* binding from the runtime's own reading. Acceptance
binds to *bytes* through `CheckRequest.target`, a string the caller writes. Probe M: a check whose
command is `sys.exit(0)` — it never opened the artifact — carries the artifact's label, passes, and
acceptance completes on it. Two different standards for "what was checked" inside one lifecycle.

### Gap 4 — a recorded decision does not gate the effect (predicted)

Confirmed as CONVENTIONAL. Two readings, and the audit does not adjudicate between them:

- *As a defect:* the runtime decides correctly and callers may do something else; nothing links an
  effect to the decision that justified it. `action.requested` has no decision field.
- *As a stated design:* the book argues effects are gated by **authority**, not by the scheduler, and
  ACTION is deliberately unreachable from the scheduler. On that reading the arrow the lifecycle
  diagram draws is one the book never claimed to enforce — in which case the honest repair is to the
  diagram, or a minimal one: record the decision a request believed it was acting under.

### Gap 5 — an errored verification leaves no trace on the claim

Registered in advance as a distinct property, and confirmed: `claim-scoped inconclusive index: 0`,
claim stream holds only `claim.recorded`. The attempt is recoverable only by scanning every
`check.requested` for the claim id. "No negative evidence" and "no claim-side trace" are currently
the same thing, which is exactly the conflation the pre-registration flagged.

---

## 5. Conventional joints — where the book draws an arrow the runtime does not enforce

1. `recorded directive → acceptance authority` (Gap 2)
2. `next-operation decision → the operation actually performed` (Gap 4)
3. `worker report → effect happened` (Gap 1)
4. `check → the artifact it claims to have examined` (Gap 3)
5. `claim → verification attempted against it` (Gap 5, one direction only)
6. `grant may be narrowed over time` — **ABSENT**: no revocation primitive exists
7. `binding → the state stays still during the run` — a reading, not a lock (already named in
   `docs/seams/verification.md`)
8. `state_resolver scope → the effect's real scope` — one hash for everything (already named in
   `docs/seams/action-recovery.md`)

---

## 6. Properties that survived composition

- Effect-unknown never became retry-safe, end to end, including the scheduler's escalation.
- `FAILED ≠ INEFFECTUAL` held; reconciliation cleared the block without rewriting the reported status.
- All four verdicts survived the claim path and the acceptance path without collapsing; INCONCLUSIVE
  and ERROR were both refused by acceptance, with different reason codes.
- A forging verifier could not describe its own subject.
- A binding failure never produced INCONCLUSIVE.
- A hand-appended `task.completed` completed nothing.
- Reopen and reprojection reconstructed identical events, status and next operation.
- Authority was decided before any disclosure of a recorded operation, on replay as on first request.

## 7. Named untested properties

- Concurrency of any kind (single-writer assumption is untested, not verified).
- Real process interruption between `execution_started` and the adapter's return; the audit simulates
  it with an exception, which is not the same as a killed process.
- Ledger tampering, identity, signing — out of scope by design.
- Multi-task and cross-directive scheduling.
- Any effect outside the state resolver's scope.
- Grant narrowing over time (no primitive exists to test against).

---

## 8. Repair ordering (recommendation)

1. **`BOOK-CORE/effect-observation`** — stop deriving effect OBSERVED from the worker's report;
   record the pre/post comparison the runtime already performs. Smallest honest version: a
   `state_change_observed` fact on the completion plus an effect state that says what the record can
   support. This is first because Chapter 19 is the chapter it contradicts, and because the evidence
   needed is already being collected and discarded.
2. **`BOOK-CORE/authority-symmetry`** — acceptance authority resolved from the recorded chain.
   Prerequisite discovered here: `task.accepted` must carry the directive id. Scope stays as
   registered: resolution from the record, not authentication.
3. **`BOOK-CORE/artifact-binding`** — make the artifact a check examined a reading rather than a
   label, or refuse acceptance on checks that cannot demonstrate it.
4. **`BOOK-CORE/decision-reference`** — decide, with the author, whether the decision→execution arrow
   should be enforced or redrawn; the minimal step is recording the decision an action believed it
   was acting under.
5. **`BOOK-CORE/claim-attempt-trace`** — a claim-side record that a verification of it errored.

Sealed fan-out (wave 2) is unaffected by all five and can proceed in parallel if desired.

---

## 9. Evidence

- Protocol frozen at `ff05dd8`, before the harness existed.
- Subject `11fedcd`; harness, JSON output and pinned tests committed together.
- The JSON records the subject commit, branch, working-tree cleanliness and Python version at run
  time.
- This document is not edited when a later seam repairs a finding. A repair writes a new entry that
  cites the gap number here.
