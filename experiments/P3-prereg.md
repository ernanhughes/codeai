# P3 Preregistration — Counterfactual as Default

Committed before any P3 model call. Frozen on first call.

## Question

Is counterfactual framing genuinely better than the normal prompt, or did P2
merely catch a favorable sample? (P2: counterfactual 16/36 candidates and 9/12
tasks vs normal 11/36 and 7/12 — within one 12-task draw.)

## Design (single variable)

- Model: `qwen-local` only. Corpus: same 12 P2 tasks (within-corpus replication).
- P3-C: normal prompt × 12 (byte-identical to P1.1/P2-C prompts).
- P3-CF: counterfactual prompt × 12 (`src/codeai/stances.py`, unchanged).
- Same starting state, verifier, sealing, sampling policy, budget accounting.
- NOT a portfolio comparison: 12 draws vs 12 draws tests the prompt distribution.

Tasks: v2-registry-pollution, v2-retry-once-accepted, v2-size-format,
v2-stable-priority, v2-splitlines-cr, v2-half-up-rounding, v2-strip-query,
v2-dt-roundtrip, v2-lsp-square, v2-unsound-cache, v2-env-timing,
v2-single-append.

## Hypotheses

- Primary: counterfactual×12 achieves higher verified oracle@12 than normal×12.
- Secondary: candidate solve rate, tokens, latency, per-task success frequency,
  arm-unique failures, conditional overlap, hard-core movement, stratum persistence.

No credit for better prose, apparent reasoning, or self-confidence.
Only verified candidate outcomes count.

## Interpretation rules (task is the unit, not pooled Bernoulli draws)

1. Counterfactual wins clearly → candidate for default framing; run out-of-sample
   replication (P3.1) before promoting.
2. Counterfactual ≈ normal → P2 difference was sampling noise. Keep normal default.
3. Counterfactual loses → drop as general default; may remain targeted diagnostic.
4. Oracle equal but candidate rate higher → record reliability separately; do not
   claim improved coverage.
5. Counterfactual uniquely solves persistent hard-core tasks → investigate those
   tasks individually before generalizing.
6. `retry-once-accepted` and `size-format` remain universally unsolved → stop
   attacking them with framing variants; treat as capability/decomposition probes.
