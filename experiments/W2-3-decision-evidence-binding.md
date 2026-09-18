# W2-3: decision–evidence binding

**Protocol:** `experiments/W2-3-prereg.md`, with the defeat semantics frozen before the runtime was
touched.
**Falsifier that justified it:** `experiments/W2-3-falsifier-results.json` — the original hostile
case still executed unrefused after Wave 1.

## The question

> If an effectful operation claims to rely on decision D, can the runtime establish that the
> evidentiary basis recorded for D is still admissible?

Three mechanisms now answer three different questions, and none substitutes for another:

```text
process freshness   is this still the state the scheduler decided against?   W1-R4
authority           may this effect happen at all?                           seam 2
decision evidence   do the claims this decision relied on still stand?       here
```

## Defeat, not change

`project_decision_standing` reports `basis_changed` whenever any tracked field differs — including
when *supporting* evidence is added. Gating on that would revoke a decision for becoming better
founded, so this seam uses a narrower relation.

| Since the decision | Verdict |
|---|---|
| New supporting evidence, or a higher evidence class | intact |
| Any refuting evidence recorded after the decision | **defeated** |
| Current status `refuted` or `contested` | **defeated** |
| Evidence class fell below what was recorded, or below the decision minimum | **defeated** |
| Source call status no longer `succeeded` | **defeated** |
| The claim can no longer be projected | **unknown** — refused |
| A verification attempt returned INCONCLUSIVE or ERROR | no effect |

`project_decision_evidence` returns a per-claim standing with the recorded and current values beside
each verdict, so a refusal can be read rather than merely obeyed. `Runtime.decision_evidence`
exposes it; `ActionRequest.decision_id` cites a decision; `action.decision_basis_refused` records the
refusal with the whole standing attached.

Refusal vocabulary, deliberately distinct from W1-R4's `stale_basis`:

```text
decision_unknown:<decision_id>
decision_basis_defeated:<claim_id>: <why>
decision_basis_unknown:<claim_id>: <why>
```

The check runs **after authority** — so seam 2's rule that an unauthorized caller learns only the
denial is untouched — and **before the replay lookup**, because returning a recorded result is
re-asserting an operation whose justification may be gone.

## Evidence

`tests/test_decision_evidence.py` (14), one per registered case. All twelve registered expectations
hold, two of them by finding that the case cannot arise:

| # | Registered | Result |
|---|---|---|
| 1 | supported basis permits | allowed |
| 2 | refuted basis refuses | refused, with the refuting evidence ids named |
| 3 | one of several refuted | refused, naming that claim; the others read `intact` |
| 4 | INCONCLUSIVE attempt | allowed — attempts are not standing |
| 5 | ERROR attempt | allowed — infrastructure revokes nothing |
| 6 | basis claim missing | refused as `decision_basis_unknown` |
| 7 | decision with no claim basis | **not expressible**: Chapter 18 already refuses `no_claims_relied_on` |
| 8 | action citing no decision | untouched, even while a decision it never cited is defeated |
| 9 | authority and evidence | refuse separately, for their own reasons |
| 10 | reopen after refutation | the same refusal from the reopened record |
| 11 | decision recorded on an already-refuted claim | already refused at record time |
| 12 | action that ran before the refutation | unchanged; nothing is rewritten |

Plus: added support is a `basis_changed` that is **not** a defeat, and a defeated basis refuses a
replay without disclosing the recorded operation.

Full suite: **664 passed**. Wave 1 composition audit: 12 ENFORCED, 2 DERIVED, nothing moved.

## A finding worth keeping

The registered repair for a defeated decision was "make a new decision". **That path is not
available**, and the test now records why: a claim carrying both supporting and refuting evidence
stands as `contested`, and `record_decision` already refuses to rest a new decision on a claim that
is not supported. So a refuted basis does not merely retire one decision — it blocks any decision
resting on that claim until the contest is settled, and nothing in the runtime settles a contest.

That is the honest state of affairs, not a defect introduced here: this seam made an existing
property of Chapter 18's claim semantics visible at the effect boundary. Whether contest resolution
should exist is a separate question, and it belongs to Chapter 18.

## What it does not establish

1. **Not that the decision was right.** Only that the evidence it recorded has not since been
   defeated under the runtime's own claim semantics.
2. **Not automatic.** An action that cites no decision is asked nothing, and nothing requires an
   action to cite the decision that in fact motivated it.
3. **Not retroactive.** A refutation invalidates future reliance; actions that already ran stay
   exactly as recorded.
4. **No contest resolution.** See above — a contested claim blocks new decisions, and this seam adds
   no way to settle one.
5. **Whole-ledger projection.** `project_claim_standing` per basis claim, per gated action; correct,
   not optimised.
6. **The v0 claim path is untouched.** Only Stage-18 claims (`claim.extracted`) carry the standing
   this seam consults.
