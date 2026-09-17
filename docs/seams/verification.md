# Seam: verification — four outcomes, and what was actually checked

*Applied AI* Chapter 21 (the agent cannot grade its own homework) owns this concept; Chapters 18 and
29 depend on it.

## The question it answers

> Can CodeAI distinguish "the thing is false", "the verifier broke", and "the available check cannot
> determine the answer"?

```text
PASS          ran against the intended target, criterion established
FAIL          ran against the intended target, criterion violated
INCONCLUSIVE  ran legitimately, cannot establish PASS or FAIL
ERROR         no trustworthy verification result was obtained at all
```

Three of those demand different engineering, which is why collapsing them is expensive:

```text
FAIL         -> investigate the work
INCONCLUSIVE -> gather better evidence
ERROR        -> repair the measurement
```

## What it establishes

Binding is first-class. A check answers two separate questions — what the verifier said, and what
state it was looking at — and the second is the runtime's, never the verifier's.

```python
@dataclass(frozen=True, slots=True)
class VerificationBinding:
    requested_state_hash: str | None
    observed_state_hash: str | None
    status: str          # bound | unbound | mismatch | unavailable
    reason: str | None
```

`BOUND` and `UNBOUND` permit verification; `UNBOUND` is recorded explicitly, so "nothing pins what
this check examined" is visible rather than implied. `MISMATCH` and `UNAVAILABLE` stop the check
before the verifier runs.

Two rules keep the four outcomes apart:

1. **A binding failure is never INCONCLUSIVE.** Infrastructure uncertainty is ERROR. Anything else
   lets a process that cannot measure at all report that the world is merely unclear.
2. **An INCONCLUSIVE result must carry a reason.** An unexplained "I do not know" is not evidence, so
   it is treated as a malformed result: ERROR.

**The verdict mapping is declared before execution and recorded.** Rather than blessing a convention
like "exit 2 means inconclusive" runtime-wide, the mapping is a property of the command:

```python
LocalCommandVerifier(pass_exit_codes=(0,), fail_exit_codes=(1,), inconclusive_exit_codes=(2,))
```

or, in the form that reaches the ledger, `CheckRequest.verdict_policy`. The request wins when both
are present, because that is the one a later reader can see. An exit code the declared policy does
not map is an unmapped outcome — ERROR, not a guess. The default policy keeps the ordinary shell
reading (0 passes, anything else fails, nothing is inconclusive), so nothing changed for checks that
never declared a mapping.

**A verifier cannot say what it checked.** `finalize_verification` overwrites
`observed_target_state_hash` with the runtime's own pre-check reading, so a buggy or hostile verifier
reporting `observed_target_state_hash="whatever-I-like"` changes nothing in the record.

**verification failed ≠ claim disproved.** PASS promotes the claims the check named, FAIL refutes
them, INCONCLUSIVE appends a durable `claim.check_inconclusive` that records the attempt and moves no
status, and ERROR records nothing against the claim at all. The deliberate evidence path
(`record_claim_evidence`) now separates `check_errored` from `check_inconclusive`, which were one
reason code before.

## Recording shape

The event flow is unchanged — `check.requested` → binding → verifier → `check.completed` — and
`check.completed` is now rich enough to reconstruct the whole thing: verdict, binding (status,
requested and runtime-observed hashes, reason), declared verdict policy, the policy the verifier
applied, verifier identity and version, timestamps, artifacts, and the error or inconclusive reason.

## Code

- `src/codeai/verification.py` — `BindingStatus`, `VerificationBinding`, `ExitCodePolicy`,
  `bind_check_target`, `execute_verification`, `finalize_verification`, `record_verification`,
  `apply_verification_to_claims`, `inconclusive_checks_for_claim`, `run_check`.
- `src/codeai/verifier.py` — one concrete verifier, now with a declared exit-code policy. Protocol,
  binding, semantics and recording live in `verification.py`; running a command lives here.
- `src/codeai/runtime.py` — `run_check` is a façade; `check_binding` and
  `inconclusive_checks_for_claim` are new projections; `_store_check_artifacts` has the name its job
  actually has.
- `src/codeai/adapters.py` — `CheckRequest.verdict_policy`; `CheckResult.inconclusive_reason`,
  `verdict_policy_id`, `binding_status`.
- `src/codeai/evidence.py` — `check_errored` split from `check_inconclusive`.

## Tests

`tests/test_verification.py` (26) is the frozen acceptance matrix: bound + established → PASS; bound
+ violated → FAIL; bound + legitimately undecidable → INCONCLUSIVE from a real command and a declared
policy; mismatch → ERROR with the verifier never called; unavailable target (no resolver, a raising
resolver, a resolver returning nothing) → ERROR, verifier never called; raising verifier → ERROR;
malformed results → ERROR (not a `CheckResult`, unknown verdict, foreign check id, unexplained
INCONCLUSIVE, a policy other than the declared one); unmapped exit code → ERROR; a forging verifier
loses to the runtime observation; PASS/FAIL move only the claims the check named; INCONCLUSIVE is
durable and moves nothing; an errored check is never negative evidence; reopening the ledger
reconstructs target, policy, verifier identity and verdict.

Two existing fixtures changed, each because the behaviour genuinely changed: the fake verifiers in
`test_claim_evidence.py` and `test_task_acceptance.py` that returned a bare INCONCLUSIVE now state a
reason, because an unexplained one is no longer a verdict.

Full suite: **539 passed**.

## Example

`examples/applied_ai/ch21_verification.py`, asserted by `tests/test_examples.py`: one file, one
declared check, four situations — marker gone (PASS), marker present (FAIL), binary content the check
cannot read as prose (INCONCLUSIVE, exit 2, reason carried), and a target that moved between the
request and the run (ERROR, verifier never called).

## What it still does not establish

This seam answers one narrow question: *given this declared check and this bound target, what result
did the verification mechanism produce?* It does not answer whether the verification was any good.

1. **Verification adequacy is untouched, deliberately.** A PASS does not establish that the check was
   the right test, that the criterion was sufficient, that coverage was adequate, that the verifier
   was independent enough, or that the environment was hermetic.
2. **The binding is a reading, not a lock.** The target can move between the observation and the
   verifier's own read; nothing holds it still. The record says what was observed before the run, not
   what the verifier consumed.
3. **The state resolver is whole-repository and caller-supplied.** One hash for everything, so a
   check bound to an unrelated edit reads as a mismatch.
4. **`INCONCLUSIVE` reasons are free text.** A stable taxonomy (insufficient evidence, unsupported
   case, ambiguous result) is not enforced, and over-standardizing it now would be guessing.
5. **Verifier identity is a class name, not an identity.** No signing, no pinned version of the
   command, no checksum of the script that ran.
6. **No hermetic environment.** The command inherits the process environment plus the request's
   overrides; nothing isolates it.
7. **Claim promotion still feeds the v0 claim projection.** Stage-18 claims (`claim.extracted`) are
   moved only through `record_claim_evidence`, where a person's judgment that the check bears on the
   claim is required. The automatic path deliberately does not forge that judgment, so the two
   generations of claim records remain separate.
8. **One verifier.** `LocalCommandVerifier` is the only concrete implementation; "blind, independent,
   diverse" checking (Chapter 23) is a different seam.
