# W1-R5: claim attempt trace — repair of composition audit gap 5

**Cites:** `experiments/W1-composition-results.md` §4 gap 5 (frozen; not edited).
**Recheck:** `experiments/W1-R5-composition-recheck.json`, same harness.
**Scope, as registered:** discoverability, not claim semantics.

## The gap

```text
claim-scoped index for an errored check : 0
events in the claim stream              : 1   (the claim itself)
by scanning every check.requested       : ['k-error']
```

An ERROR correctly produced no evidence against the claim — and produced no claim-side trace either,
so *"no evidence was produced"* and *"no attempt is discoverable"* were the same fact.

## The repair

Two projections, deliberately separate:

```text
claim
  |- verification attempts    PASS / FAIL / INCONCLUSIVE / ERROR   all visible
  |
  `- evidentiary effect       PASS may support, FAIL may refute
                              INCONCLUSIVE and ERROR move nothing
```

`claim.verification_attempted` is a **relation, not a second opinion**. Its payload is exactly:

```text
claim_id
check_id
check_completed_event_id
version
```

with `causation_id` pointing at the check's completion. No verdict, no binding, no reason. The check
stream stays the single source of truth for what was found, so the ledger can never reach a state
where the claim says ERROR and the check says INCONCLUSIVE.

`verification_attempts_for_claim(claim_id)` walks the relations and resolves each check at read time,
reporting verdict, both binding statuses and the reason — always from the check.

**A duplicate was removed as part of this.** The previous `claim.check_inconclusive` event copied
`inconclusive_reason` onto the claim side, which is exactly the divergence risk this repair is meant
to avoid. `inconclusive_checks_for_claim` is now derived from the attempt trace, so its API is
unchanged and it cannot disagree with the checks.

## The frozen matrix

| Check outcome | Visible from claim? | Moves claim evidence? |
|---|---:|---:|
| PASS | yes | yes |
| FAIL | yes | yes |
| INCONCLUSIVE | yes | no |
| ERROR | **yes** | **no** |
| binding failure → ERROR | yes | no |
| malformed INCONCLUSIVE → ERROR | yes | no |

Also pinned in `tests/test_verification.py`:

- a check naming no claim records no attempt, and no refusal either;
- an unknown claim is **refused** (`claim.verification_attempt_refused`, reason `unknown_claim`)
  rather than linked, so no relation points at nothing;
- the same `check_id` processed twice yields one attempt;
- reopening gives the identical ordered attempt list (`k1` INCONCLUSIVE, `k2` ERROR, `k3` PASS);
- the hostile case: artifact binding `missing` → the verifier is never invoked → ERROR → **the
  attempt is discoverable from the claim** and the claim's evidence is untouched;
- an errored check is still refused as deliberate evidence (`check_errored`).

## Naming

`claim.verification_attempted` and `claim.verification_attempt_refused`. Neutral on purpose: a name
like `claim.check_failed` would collapse FAIL, INCONCLUSIVE and ERROR back together in the one place
this work has spent five repairs keeping apart.

## Evidence

- `src/codeai/verification.py` — `VerificationAttempt`, `record_verification_attempts`,
  `verification_attempts_for_claim`, `inconclusive_checks_for_claim` (now derived),
  `CLAIM_VERIFICATION_ATTEMPTED`, `CLAIM_ATTEMPT_REFUSED`.
- `src/codeai/runtime.py` — `verification_attempts_for_claim`.
- `tests/test_check_binding.py` — two fixtures name a claim that was never recorded, so they now
  show the refusal. That is the rule working.
- Full suite: **615 passed**.

## Recheck against the frozen baseline

```text
H  verification -> claim evidence     DERIVED  ->  ENFORCED
```

Nothing else moved.

## What this repair does not do

1. **It does not change claim semantics.** What moves a claim is exactly what moved it before.
2. **Attempts are recorded for claims that exist at check time.** A claim recorded *after* a check
   that named it keeps no trace of that check; the refusal is durable, but it is not retried.
3. **The trace is per claim id, not per statement.** Two claims making the same assertion are two
   traces.
4. **It reads the whole ledger.** Same O(n) shape as the other projections.
5. **Stage-18 claims and v0 claims share the trace**, and neither generation's evidence rules change.
