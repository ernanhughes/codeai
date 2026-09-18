# W1-R3: artifact binding — repair of composition audit gap 3

**Cites:** `experiments/W1-composition-results.md` §4 gap 3 (frozen; not edited).
**Recheck:** `experiments/W1-R3-composition-recheck.json`, same harness.

## The gap

The verification seam established the *state* a check examined from the runtime's own reading.
Acceptance established the *bytes* a check examined by comparing two strings:

```text
check.requested.target == "artifact:sha256:<hash>"     a label the caller wrote
acceptance.artifact_sha256                              the bytes being accepted
```

Probe M: a check whose command was `sys.exit(0)` — it never opened the artifact — carried the
artifact's label, passed, and completed an acceptance. Two standards for "what was checked" inside
one lifecycle.

## The repair

```text
target names artifact:sha256:<digest>
      -> resolve it from the artifact store
      -> verify the stored bytes hash to that digest
      -> write them where the check can read them
      -> substitute {artifact} in the command with that path
```

`ArtifactBinding` is runtime-owned and recorded on `check.completed` beside the state binding:

| Status | Meaning | Verifier runs? |
|---|---|---|
| `BOUND` | resolved, digest verified, supplied to the check | yes |
| `UNBOUND` | the check names no artifact | yes |
| `MISSING` | named, and not retrievable | no — ERROR |
| `MISMATCH` | the stored bytes do not hash to the named digest | no — ERROR |
| `UNCONSUMED` | a command check that never references `{artifact}` | no — ERROR |

`UNCONSUMED` is the rule that closes the gap: **a command that is never given the artifact cannot
have examined it**, so it is a broken measurement rather than a verdict — the same treatment a state
binding failure already got. Checks with no command (static verifiers, fakes) still receive the path
on the request, so existing verifiers keep working.

`materialized_artifact_path` is runtime-owned. A caller who sets it is overwritten, exactly as a
verifier who reports its own `observed_target_state_hash` is overwritten.

Acceptance no longer compares labels. `_validate_checks` reads the **runtime-established** binding
from `check.completed`:

```text
artifact_binding.status == "bound" and resolved_artifact_sha256 == the artifact being accepted
```

A `check.completed` written before this repair carries no binding and is refused as
`check_artifact_unestablished:<check_id>` — distinct from `check_wrong_artifact`, because "this
record cannot establish it" is not "this is the wrong artifact". The same conservative treatment
`actions.py` gives pre-seam history.

## Evidence

`tests/test_verification.py` gains 8 cases: a command given the artifact reads the verified bytes; a
command that never references it is ERROR/`unconsumed`; an unstored artifact is ERROR/`missing`; a
non-artifact target stays `unbound` and runs; a static verifier is still handed the bytes; a caller
cannot set the materialized path; corrupted stored bytes are `mismatch`, never FAIL; and the two
bindings are independent (state `mismatch` with artifact `bound`).

Existing expectations changed because the behaviour genuinely changed:

- `test_check_of_different_bytes_cannot_complete` and `test_unpreserved_artifact_cannot_complete` now
  carry an additional `:ERROR` reason — a check naming bytes that were never stored cannot run at
  all, which is true and was previously invisible;
- the audit harness now passes `{artifact}` for checks that are meant to examine the artifact.
  Probe M deliberately still does not: that is the measurement.

Full suite: **590 passed**.

## Recheck against the frozen baseline

```text
I  verification -> acceptance     RECORDED      ->  ENFORCED
M  check -> artifact identity     CONVENTIONAL  ->  ENFORCED
```

Everything else unchanged. Probe I's rows are sharper than at baseline, because each refusal now
names one reason rather than being masked by the artifact rule:

```text
PASS on a different artifact       -> check_wrong_artifact:k-other
INCONCLUSIVE on the right artifact -> check_not_passed:k-maybe:INCONCLUSIVE
ERROR on the right artifact        -> check_not_passed:k-err:ERROR
PASS with no artifact label        -> check_wrong_artifact:k-unlabelled
PASS on the accepted artifact      -> completed
```

## What this repair does not do

1. **Artifact binding is not artifact adequacy.** `BOUND` means the intended artifact was resolved,
   integrity-checked and made available through the declared verifier interface. It does not mean the
   verifier used a single byte of it. Both of these are legitimately BOUND:

   ```text
   grep expected-token {artifact}     a meaningful check
   python verifier.py {artifact}      a verifier that ignores argv[1] and returns 0
   ```

   The runtime claims only "this is the artifact that was supplied". Pushing past that generically
   means sandboxing, syscall tracing or instrumented execution — a different problem, and one this
   seam does not pretend to solve. Pinned by
   `test_artifact_binding_is_not_artifact_adequacy`.
2. **The materialized copy is transient.** It lives for the duration of the check; the artifact
   itself lives in the store. Nothing records the copy's path as durable evidence.
3. **One artifact per check.** A check that should examine several has no way to say so.
4. **`{artifact}` is a textual placeholder.** A command can smuggle it into an argument that never
   reaches a reader.
5. **Digest verification is the store's own**, so a store that lies about its bytes is trusted; the
   ledger remains the trust boundary.
6. **Pre-repair records are refused, not reinterpreted.** A ledger written before this seam cannot
   complete an acceptance without re-running its checks.
