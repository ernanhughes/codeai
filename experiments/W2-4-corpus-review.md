# W2-4 Checkpoint A: corpus review

**Reviewed:** 2026-09-18, before either adjudicator begins and before any provider is contacted.
**Input:** `experiments/router_cases_v1.DRAFT.json` (68 cases, 44 marked `review_required`).
**Output:** `experiments/router_cases_v1.CANDIDATE.json` — 68/68 semantically valid, status
`REVIEWED_CANDIDATE`, hash `ee3d250ff23321d9aaadc01a00d3b87d0e91bb154089e72a2a471cb96baacd95`.

## The rule this pass obeyed

> **Repair the question, never answer it.**

No oracle label was assigned, computed or implied. The deterministic router was not consulted for any
case. Every question asked here was *"can two reasonable adjudicators apply the §1b precedence table
to this, without inventing facts?"* — never *"what is the right operation?"*

```text
corpus review    is the question well posed?          this pass
adjudication     what is the answer?                  two humans, next
```

## Result

| Verdict | Count |
|---|---|
| `VALID_AS_WRITTEN` | 43 |
| `REVISE` | 1 |
| `REJECT` | 0 |

Every reviewed case now carries its verdict and reason in `validity_reason`, so the judgement
travels with the case rather than living only here.

## The one revision

**`r2-10` — a contradiction between prose and flags.**

```text
frozen flag      has_required_verification = true
narrative        "A required check on the edited file just passed."
```

A check that just passed is not a check that remains pending. As written, the case asked the
narrative arms and the flag arms two *different* questions, so any disagreement between M-direct and
D on it would have measured the defect rather than the router. Stratum B's flags are ground truth by
design ("naturalistic state descriptions with frozen ground-truth flags"), so the prose moved:

```text
-  A required check on the edited file just passed. The following step — pushing the result to
-  the shared branch — needs human authority that has not been granted.
+  A required check on the edited file has not been run. The following step — pushing the result
+  to the shared branch — needs human authority that has not been granted.
```

| Question | Answer |
|---|---|
| Does the revision alter intended difficulty? | **No.** The case still pairs a pending check with an ungranted authority gate, which is the precedence conflict it was built for |
| Does the revision expose an answer? | **No.** It states one more fact about the world and names no operation |

## Two cases kept deliberately, with their difficulty intact

**`r2-15`** records an unknown external side effect — required coverage in §4. The v1 five-field
schema has no flag for an unresolved effect, so the state as written underdetermines the operation.
That is a fact about the schema, not a defect in the case, and the oracle has `AMBIGUOUS` for exactly
this situation. Repairing it would mean either inventing a sixth flag or choosing the answer, and
both are out of bounds here. Left as written, with the expectation recorded rather than hidden.

**`r2-17`** and **`r2-18`** tell the adjudicator that the flags may be stale or were computed
earlier. That invites distrust of the given state, which is uncomfortable and is precisely the
required "stale context / drifted flags" coverage. Kept.

## Underdetermination is not contradiction

Several narratives imply a flag more weakly than they assert it — `r2-03` calls extra proposals an
"enticing backlog item" where the flag records them as *requested*; `r2-02` has a teammate suggesting
variants while the proposals flag is false. These were **not** repaired, because recovering state
from prose is R2's entire subject matter: if a narrative underdetermines a flag, the extraction arms
will show it, and that is the decomposition result the component exists to produce. Only an outright
contradiction breaks the case, and only one case had one.

## Structural findings for the author (before freeze)

### 1. The distractor field name tells the model the distractor is irrelevant

Stratum C's model arms receive:

```python
input_value = {"narrative": case.narrative, "irrelevant_history": case.distractor}
```

The key is `irrelevant_history`. A model that simply trusts the field name will discount the
distractor by construction, so a low flip rate would partly measure the label rather than the
router's resistance to salience. That weakens the one stratum built specifically to test salience
traps.

This is a change to the harness, not to the corpus, so it is flagged rather than made: renaming the
key to something neutral (`history`, `recent_log`) is a one-line change in
`src/codeai/router_experiment.py`, and it must be settled **before** freeze because it changes what
every C decision sees. The deterministic arm is unaffected — it receives `{"state": ...}` and never
sees the distractor at all, which is correct and is what makes D invariant by construction.

### 2. R2's state vectors are concentrated

```text
R1   24 cases, 24 distinct state vectors        exhaustive by construction
R2   24 cases, 14 distinct state vectors        six share one vector
C    20 cases,  8 distinct vectors (10 pairs)   by design
```

Six R2 cases — `r2-14`, `r2-16`, `r2-18`, `r2-19`, `r2-20`, `r2-23` — share the vector
`verification-only`. Under any consistent precedence rule they therefore share one answer, and with
the neighbouring vectors included, one operation dominates R2's answer distribution.

That is a consequence of §4's required coverage, which is heavily verification-flavoured, and it is
not repairable without dropping required cases. The honest response is in the analysis, not the
corpus: **report a majority-class baseline for R2 alongside its correctness numbers**, so that no
reader mistakes a high R2 score for skill when a constant answer would score nearly as well. R1, the
load-bearing component, has 24 distinct vectors and is unaffected.

No case was rebalanced or removed for this: reshaping a corpus to flatter its own metric, after
seeing the distribution, is how a preregistration stops meaning anything.

## Sanity pass across all 68

| Check | Result |
|---|---|
| `validate_corpus(freeze=True)` | passes — R1 24, R2 24, C 20, 10 complete pairs |
| `review_required` remaining | 0 |
| Exact duplicate cases | none |
| Pair integrity (shared narrative, differing distractors) | enforced by the contract, holds |
| R1 vectors | 24 distinct, exhaustive over the five-field schema |
| Oracle labels present anywhere in the corpus | none |

## What happens next, and what must not

The candidate is ready for **Checkpoint B**: two independent adjudicators, each labelling all 68
cases with written reasons, neither seeing the other's worksheet, router outputs, model choices or
the deterministic arm's answers. Worksheets already exist at
`experiments/W2-4-oracle-worksheet-adjudicator-{a,b}.json` and must be regenerated from this
candidate so they carry the revised `r2-10` and the candidate hash.

An AI must not stand in for either adjudicator. The experiment's whole point is that the router is
judged against something outside itself, and substituting a model for a human judge would quietly
delete that. Explaining the rubric or checking a worksheet for completeness is fine; supplying the
labels is not.
