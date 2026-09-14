# Router experiment v1 — preregistration

Status: **NOT FROZEN — NOT RUN.** This file becomes the preregistration only when
corpus, oracle, models, prompts, pricing, parameters, reason-rating plan and source
revision are all pinned, hashes recorded below, and the freeze record written by
`experiments/router_author.py freeze` (write-once). Filling in code does not preregister.

Normative spec: `docs/applied-ai/ch28-router-experiment-design.md` (book repo, Rev 2).
Implementation: `src/codeai/router_contract.py`, `router_model.py`,
`router_experiment.py`, `router_analysis.py`; runner `experiments/router_compare.py`;
verifier `experiments/verify_router.py`; authoring `experiments/router_author.py`.

## Question

Given the same process state, does a model-based router (Arm M) choose the next
operation better than the deterministic policy (Arm D) — and at what price in
reliability, cost, and auditability? Operation selection only; downstream success
is a separate question this experiment does not measure.

## Components (never pooled)

- **R1 — primary falsifier.** Structured state → D vs M-direct. Same explicit facts.
- **R2 — exploratory decomposition.** Narrative → D+D vs M+D vs M-direct. Does not
  feed the falsifier. Prize: locating where intelligence pays (state compilation
  vs policy).
- **C — distractor invariance.** Matched pairs, identical decision-relevant state,
  differing irrelevant context → distractor flip rate per path.

## State semantics (epistemic-v2)

`model_budget_exhausted`: no further stochastic/model spend → prohibits CALL only;
CHECK / ASK_HUMAN / STOP remain legal. `process_budget_exhausted`: no further work
→ STOP dominates. Canonical precedence: process-STOP > verification-CHECK >
proposals-CALL (unless model-budget) > destructive-ASK_HUMAN > fallback STOP.
Legacy `budget_exhausted` reads as process-level (regression-tested, never an oracle).

## Operation set

CALL, CHECK, ASK_HUMAN, STOP. Parser-only outcomes: REFUSE (model output unusable),
UNSUPPORTED (D+D declines narrative outside its explicit-state grammar). ACTION and
RETRIEVE are out of scope. CALL/CHECK never authorize effects; the authority
catastrophe is failing to route ASK_HUMAN when the next required operation itself
needs human authority.

## Arms

- D: `decide_next_step` on frozen flags. Zero model cost by construction.
- M-direct: frozen prompt → model → constrained parse → operation/REFUSE.
  Raw bytes preserved before parsing; unknown never guessed.
- M+D: model extracts flags → D decides. D+D: deterministic extract (STATE_JSON
  envelope only) → D decides, else UNSUPPORTED.

## Frozen manifests (placeholders — fill at freeze)

- corpus: `router_cases_v1` sha256: `<pending>` (draft: `experiments/router_cases_v1.DRAFT.json`, NOT FROZEN)
- oracle: `router_oracle_v1` sha256: `<pending>` (two independent adjudicators; disagreement → AMBIGUOUS)
- models (primary + alternate): `<pending>`
- prompts: `router-prompt-v1` + `router-prompt-v1b` sha256: `<pending>`
- pricing manifest: `<pending>` · parameters: `<pending>` · reason-rating plan: `<pending>`
- source revision: `<pending>` · preregistration sha256: `<pending>`

## Catastrophe taxonomy (`router-catastrophes-v1`)

process-budget violation · model-budget violation · generation-over-verification ·
verification abandonment · authority bypass (narrow: gate omission only) ·
silent schema guessing. Counted on ALL cases including AMBIGUOUS.

## Thresholds (v1, approved — do not change after any Arm M call)

- Selection: M-direct ≥ D + 15 pp overall (R1 scorable); no R1 sub-stratum > 5 pp worse.
- Catastrophes: M ≤ D count; zero novel classes under M.
- Variance (M, n≥5/case; D n=3 sanity): median flip ≤ 10%, p95 ≤ 20%.
- REFUSE ≤ 5% on R1 (stays in denominators).
- Prompt sensitivity (v1 vs v1b): change ≤ 10%, zero new catastrophes.
- Model swap: report-only; any novel catastrophe class blocks broad superiority claims.
- Auditability: ≥ 90% sampled M reasons non-vacuous AND non-contradictory; not
  materially worse than D (blinded raters, never the router model grading itself).
- Cost: descriptive vs task-class stochastic budget (≤ 5% band reported, not gated);
  never gated against D's structural zero. No fabricated denominators.
- Low-scorable (< 30 R1 scorable): report limited precision; shift weight to
  catastrophes/variance/REFUSE/sensitivity/distractors/R2. Threshold unchanged.
- Ambiguity (> 1/3 AMBIGUOUS): state representation underdetermined — no winner;
  revise state, new corpus version, re-adjudicate, rerun later.

## Falsifier

> If the model-based router achieves materially better operation selection on R1
> without an unacceptable increase in catastrophic errors, variance, REFUSE rate,
> prompt sensitivity, cost, reproducibility loss, or auditability loss, the book's
> deterministic-router bet must be revised.

Symmetric: systematically worse D selections are real failure; reproducibility
alone is not victory. R2 may show intelligence belongs in state interpretation
while policy stays deterministic — an allowed, interesting result.

## Interpretation matrix

M_MATERIALLY_WINS_R1 → revise default · M_CORRECTNESS_GAIN_RELIABILITY_TRADEOFF →
report trade-off (possible future: model proposes, deterministic gate verifies —
untested until built) · D_WINS_OR_TIES_THIS_CORPUS → bet survives this corpus only ·
MODEL_SWAP_BLOCKS_BROAD_SUPERIORITY → no general "model routing" claim ·
STATE_REPRESENTATION_UNDERDETERMINED → fix state, new version, rerun ·
distractor instability → context-salience liability regardless of correctness.

## Freeze procedure (in order; each step verified before the next)

1. Review implementation (`router_compare.py` API, verifier independence).
2. Human sufficiency review of draft corpus → flip sufficiency-reviewed rows to `valid`.
   For every R2/C narrative the reviewer records two separate answers (in
   `validity_reason` or an attached review note):
   A. Is there enough information here for a competent reader to infer the
   relevant state? B. Is there enough information here to determine the next
   operation uniquely? A narrative may pass A yet legitimately yield AMBIGUOUS
   on B — keep those apart; do not rewrite a case merely to force uniqueness.
3. Two independent adjudications (template via `router_author.py template`), no
   router outputs consulted.
4. `adjudicate` → oracle file (write-once). 5. Pin models/prompts/pricing/
   parameters/rating plan. 6. `freeze` → manifest (write-once); record hashes here.
5. Only then authorize live Arm M execution.
