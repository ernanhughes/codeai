# W2-4: the router challenger — execution addendum

**Normative design:** `docs/applied-ai/ch28-router-experiment-design.md` (book repo, revision 2) and
`experiments/router-prereg.md` (this repo). This addendum **adds nothing to the criteria**. The
falsifier (§0), thresholds (§7), arms (§3), corpus construction (§4), oracle rules (§5), metrics
(§6) and interpretation rules (§9) stand exactly as already preregistered.

What this file records: the spend authorization, the state of the artifacts, two blockers found
before any provider was contacted, and the evidence that the instrumentation works.

## Spend authorization (author, 2026-09-18, revised the same day)

```text
hard ceiling      $1.00 total model spend
authorized now    $0.25 — the R1 tranche only, 480 model calls
```

The $0.25 is **the authorized experimental scope**, not merely an instrumentation checkpoint: run the
preregistered R1 experiment and nothing else. R1 is load-bearing by the design's own words — it alone
feeds the §0 falsifier — so this executes a preregistered scope rather than trimming one.

Continuations are stated **now**, so that no choice after R1 can be performance-driven:

| R1 outcome under the frozen falsifier | Next |
|---|---|
| The challenger is rejected | Stop. The router did not earn further spend |
| Inconclusive | No promotion. Decide separately whether more evidence is scientifically worth buying |
| The challenger is supported | Then consider funding R2 and C as confirmatory and sensitivity work |

No protocol change follows R1 in any branch. Thresholds, arms, repeats and metrics are already
frozen, and the remaining ceiling exists to make a *second authorization* possible, not to permit
extension mid-experiment.

## Freeze manifest contents (author, explicit)

The manifest must record all of:

```text
corpus schema and version          scheduler policy actually executed (epistemic-v3)
oracle version                     adjudicator identities
case hashes                        prompt and template hashes
model identities                   repeat counts
promotion and falsifier thresholds
```

The v1-state / `epistemic-v3` gap is acceptable precisely because the manifest tells the truth about
which executable policy ran.

## Blocker 1: the oracle is people, and the code enforces it

`router_contract.adjudicate` refuses anything less than the design demands:

```text
two distinct adjudicator identifiers per case
a decision from {CALL, CHECK, ASK_HUMAN, STOP, AMBIGUOUS}
a written reason, the facts relied on, a version, a timestamp
disagreement -> AMBIGUOUS, with no tie-breaker
```

and the design forbids the one thing a machine could produce: *"Scoring current scheduler output as
ground truth is forbidden."* So the oracle cannot be generated, and W2-4 cannot start, until two
people label the corpus.

Prepared for them: `experiments/W2-4-oracle-worksheet-adjudicator-{a,b}.json`, one per adjudicator,
68 cases each, carrying the §1b precedence table, the state or narrative as an adjudicator sees it,
and empty decision/reason/facts fields. They contain **no arm output and no deterministic answer**,
because the oracle is established before either router runs.

A second job belongs to the author alone: **44 of the 68 cases are still
`semantic_validity: review_required`** — the corpus is `DRAFT`, and its own field says the author must
review whether each state is sufficient. Corpus review comes first, then adjudication, then freeze,
then the run. The design's order is corpus → oracle → freeze → run, and nothing here shortcuts it.

## Blocker 2: the schedule is larger than the cap

`build_schedule` over the draft corpus, with the preregistered repeat counts (5 for model arms, 3
for deterministic) and the §3 sensitivity variations:

```text
total scheduled decisions   1844
model calls                 1580      R2 600 · C 500 · R1 480
                                      M-direct 1140 · M+D 440
```

Against the $1.00 cap:

| Price per routing call | Total |
|---|---|
| $0.0005 | $0.79 — inside the cap, with no margin |
| $0.002 | $3.16 — **3× over** |
| $0.01 | $15.80 — **16× over** |

A routing prompt is short, so the cheap column is reachable with a small model; the design
nevertheless requires **two** models (primary and alternate) for its model-sensitivity dimension, and
the alternate is the one most likely to be dearer.

This is a scope question, not a threshold question, and it is the author's to settle. The options, as
they stand:

1. **Run R1 only, first.** 480 model calls, ~$0.24 at the cheap price — inside the checkpoint. R1 is
   the load-bearing experiment: the design says it *alone* feeds the §0 falsifier. R2 and C would be
   deferred to a second authorization, and their absence reported as scope, not as a result.
2. **Raise the cap** to cover the full 1,580 calls at the actual price of the chosen pair.
3. **Choose both models from the cheap tier**, accepting that "model sensitivity" then means
   sensitivity across two small models rather than across a capability gap.

What is *not* an option: reducing repeats. Repeat counts feed the variance threshold (median flip
≤10%, p95 ≤20%), and cutting them to fit a budget would quietly change a preregistered criterion.

## Instrumentation evidence, at $0.00

`experiments/w2_4_instrumentation_dryrun.py`, result frozen in
`experiments/W2-4-dryrun-results.json`. It runs the whole pipeline in the harness's synthetic mode,
which accepts only the repository's own fake adapter, and re-analyses the result with the independent
verifier:

```text
verdict                     SYNTHETIC_INSTRUMENT_TEST_ONLY
components exercised        R1, R2, C
decisions recorded          25
deterministic arms          priced at exactly 0.0
model arms                  priced, non-zero
raw output per decision     preserved and content-addressed
distractor pairs            scored per arm and variation
distinct prompts per case   3 — the sensitivity variations are real, not aliases
```

**A defect in the first version of this dry run, now a permanent regression test.** The harness
refuses it rather than reporting it: a model-backed arm whose execution succeeded must carry
accounting evidence, and its absence raises *"model decision lacks accounting evidence"* instead of
flowing through as `cost_usd: None`. A deterministic arm's `0.0` is a measured zero; a model arm's
missing usage is an instrumentation failure. Pinned by
`test_a_model_arm_without_usage_is_an_instrumentation_failure`.

The original defect, recorded because it is the point of a checkpoint: The first attempt used a fake adapter with no scripted response and no model name. The
pipeline ran, the verifier returned a verdict, and every model arm's cost came back `None` — the
accounting check passed vacuously because the accounting path had never been exercised. The fixed
version scripts the adapter exactly as the harness test does, and the prices appear. An
instrumentation check that can pass without the instrument running is worth less than no check at
all, and that is precisely what a $0.25 checkpoint is for.

## Runtime drift to record before freezing

The corpus states are the five-field v1 schema; the committed scheduler is now `epistemic-v3`, which
adds `process_complete` and `unresolved_effect`. Both default to False, so a v1 state routes through
exactly the v2 ladder the design was written against, and `router_contract.V1_STATE_FIELDS` accepts
those states deliberately rather than by accident. Arm D therefore answers the question the design
asked. The freeze manifest should record the policy version actually used (`epistemic-v3`) rather
than the version the design named, so a later reader is not misled.

## State: not frozen, not run

No case has been executed through any router under this protocol. No provider has been contacted. No
spend has occurred. `router_compare.py`'s entry point still refuses to run itself, and that has not
been weakened.
