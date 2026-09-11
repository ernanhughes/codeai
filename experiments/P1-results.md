# P1 Results — Heterogeneity Premium on `seeded-code-v1`

Status: **P1 v1 complete — null heterogeneity premium on seeded-code-v1.**
Frozen. Do not rerun or modify. Corpus `seeded-code-v1` frozen with it.

- Calibration experiment: `4b6af118-41eb-4860-87d8-79a3df8724d7` (3 tasks, 21 calls)
- Full experiment: `0ea37186-bada-4e1b-b355-ccb774aaab8e`
  (12 tasks × (C0 + C1×3 + H1×3) = 84 calls)
- Config hash: `d240118b618e8b98328008fab1cddd2bff84897dab3a09a28ff68b9825c9178c`
- Providers: local Ollama only (`qwen2.5-coder` / `mistral:7b-instruct` / `llama3.1:8b`).
  Cloud keys unavailable (OpenAI 401, Anthropic absent). Monetary cost unknown;
  token/latency budgets govern.
- Raw export: `p1-runs/p1-export.json` (84 calls, 84 candidates, 83 checks).
- Interpretation bound by `P1-interpretation.md` (commit `f9bc67e`).

## Metrics (exact)

| arm | oracle@k | solves | tokens | p50 latency |
|-----|----------|--------|--------|-------------|
| C0 (qwen×1) | 0.9167 (11/12) | 11/12 | 1827 | 2.9s |
| C1 (qwen×3) | 0.9167 (11/12) | 33/36 | 5452 | 2.8s |
| H1 (3 models) | 0.9167 (11/12) | 33/36 | 6549 | 6.8s |

- Heterogeneity premium (H1−C1): **0.0** (matched samples and matched tokens alike).
- Unique rescues: none, in any direction.
- Pairwise P(fail|fail): 1.0 for every pair (perfect failure overlap).
- H1 cost strictly more (20% more tokens, 2.4× latency) for the identical boundary.

## Per-task structure

11/12 tasks solved by all 77 passing candidates. `stale-state-average` failed
all 7 attempts across all 3 model families. All seven wrong repairs preserved
or expanded state (running accumulators, a class rewrite, a `global` SyntaxError)
instead of removing it. Observed behavior only; no claim about underlying cause.

## Preregistered interpretation applied

Rule 2 (H1 ≈ C1): **prefer homogeneous repeated sampling** until measured
evidence shows useful error decorrelation. Rule 5 (universally hard task):
investigate task framing/generator capability before collaboration machinery.
Rule 6: no conclusion about synthesis.

Working default (evidence-backed, corpus-scoped): **model diversity is not
evidence diversity** — different labels behaved as correlated sensors here.

## Limits

n=12, single-function Python, 11 tasks at ceiling. Corpus too easy to
discriminate ensemble economics. Next: P1.1 on a discriminating corpus
(`semantic-repair-v1`), same C0/C1/H1 protocol, difficulty strata.
