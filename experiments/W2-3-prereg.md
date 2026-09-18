# Pre-registration: W2-3, decision–evidence binding

**Registered:** 2026-09-18, before touching the runtime.
**Falsifier that justifies it:** `experiments/W2-3-falsifier-results.json` (`9c58cc1`) — the original
hostile case still executes unrefused, so Wave 1 did not consume this idea.

## The name, and why it changed

Not `decision-basis gating`: W1-R4 already owns a *basis*, the process-state digest. This seam is
about a different basis entirely, and it takes its place in the same family:

```text
state binding              which state a check examined
artifact binding           which bytes a check was given
decision/execution binding which operation a decision selected
acceptance/effect binding  which action produced an accepted state
decision/evidence binding  whether the claims a decision relied on still stand
```

**The question:**

> If an effectful operation claims to rely on decision D, can the runtime establish that the
> evidentiary basis recorded for D is still admissible?

**The invariant:**

> An effectful operation that claims decision D as its basis must not execute if a claim in D's
> recorded evidentiary basis has subsequently become inadmissible under the runtime's claim
> semantics.

And, as deliberately, what it does not say: *every action must have a decision*. External and manual
actions are unchanged. Cite a decision and the justification must hold; cite none and none is
claimed.

## Three different questions, three different mechanisms

```text
scheduler decision freshness   is this still the process state it was decided against?   W1-R4
authority                      may this effect happen at all?                            W1-R2/seam 2
decision evidence              do the claims this decision relied on still stand?        here
```

Evidence can make an action unjustified while authority permits it. Authority can forbid an action
whose evidence is impeccable. Chapter 18 keeps the first, Chapter 20 the second, and this seam is
the reason they stay apart rather than merging.

## What invalidates a basis — frozen before implementation

`project_decision_standing` already reports `basis_changed` whenever any tracked field differs. That
is **too broad to gate on**: adding *supporting* evidence changes the field set, and a decision must
not be revoked for becoming better founded. This seam uses a narrower relation, **defeat**:

| Change since the decision | Effect on the basis |
|---|---|
| New supporting evidence, or a higher evidence class | intact — a decision is not revoked for strengthening |
| Current status `refuted` or `contested` | **defeated** |
| Any refuting evidence recorded since the decision | **defeated**, even if later re-supported |
| Evidence class fell below what was recorded, or below the decision minimum | **defeated** |
| Source call status no longer `succeeded` (Chapter 17 reinterpretation) | **defeated** |
| The claim can no longer be projected | **unknown** — refuse; absence is not permission |
| A verification attempt returned INCONCLUSIVE or ERROR | **no effect** — attempts are not standing |

The last row matters most. W1-R5 separated a verification *attempt* from claim *evidence* from claim
*standing*; gating on attempts would let an infrastructure failure revoke a sound decision and undo
that distinction.

**Re-support does not resurrect a decision.** If a claim was refuted after the decision and later
supported again by better evidence, the decision stays defeated: it was a judgment made against an
evidentiary state that has since been overturned, and the honest repair is a new decision, not a
resurrection. The recorded snapshot already pins `refuting_evidence_ids` at decision time, so
"refuting evidence arrived since" is directly checkable with no new identity system.

## Refusal vocabulary, distinct from W1-R4

```text
stale_basis                          the process state moved          (W1-R4)
decision_basis_defeated:<claim_id>   the epistemic basis was overturned
decision_basis_unknown:<claim_id>    the claim can no longer be projected
decision_unknown:<decision_id>       no such decision is recorded
```

## Hostile matrix, frozen

| # | Case | Expected |
|---|---|---|
| 1 | supported claim → decision → action | allowed |
| 2 | supported claim → decision → claim refuted → action | refused, `decision_basis_defeated` |
| 3 | several basis claims, one refuted | refused, naming that claim |
| 4 | a basis claim gets an INCONCLUSIVE verification | allowed; behaviour follows standing, not attempts |
| 5 | a verification attempt ERRORs | allowed; an infrastructure failure revokes nothing |
| 6 | a basis claim missing on replay | refused, `decision_basis_unknown` |
| 7 | decision with no claim basis | allowed; no evidence relationship is claimed |
| 8 | action with no decision | unchanged external/manual semantics |
| 9 | action cites a decision from another task | refused |
| 10 | reopen after refutation | the same refusal, from the reopened record |
| 11 | decision recorded when the claim was *already* refuted | refused at record time — `record_decision` already requires supported claims; this seam adds the later half |
| 12 | an action executed *before* the refutation | remains historical; nothing is rewritten |

Case 12 is the temporal invariant: **new evidence may invalidate future reliance on a decision; it
does not rewrite the fact that an earlier action ran while the basis stood.**

## Where it sits in the order

```text
decision permits this operation class?     governance   (W1-R4)
authority permits this capability?         authority    (seam 2)
does the cited justification still stand?  here
precondition still holds?
execute
```

After authority, deliberately: seam 2's rule that an unauthorized caller learns only the denial must
not be weakened by an earlier refusal that reveals something about recorded decisions.

The check runs **before the replay lookup**, so a defeated basis refuses a replay too. Returning a
recorded result is re-asserting an operation, and an operation whose justification is gone should
not be re-asserted on request.

## Predictions

1. The defeat rules will be implementable entirely from the recorded snapshot plus
   `project_claim_standing` — no new events beyond the refusal.
2. Case 4 and 5 will pass without special handling, because attempts never touched standing.
3. The interesting design pressure will be case 12: making sure the projection of a *past* action is
   unaffected by a *present* refusal.
