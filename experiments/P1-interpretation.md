# P1 Interpretation Rules — Pilot Experiment: Heterogeneity Premium

Preregistered before any experimental model call. These rules bind the
interpretation of P1 regardless of outcome. They may not be edited after
results are observed; any change requires a new dated amendment noting
what was seen first.

Experiment: P1 — Heterogeneity Premium (corpus `seeded-code-v1`, 12 tasks).
Primary comparison: `H1 oracle@3 − C1 oracle@3`, with cost shown alongside.

Two cost framings are preregistered:

- **P1A — matched number of samples**: C1 = 3× Qwen; H1 = Qwen + Claude + GPT.
- **P1B — approximately matched monetary/token budget**: C1 = enough homogeneous
  samples to spend ≈ H1 cost; H1 = heterogeneous portfolio.

## Rules

1. If H1 materially beats C1 at similar cost:
   pursue model heterogeneity and error-correlation analysis.

2. If H1 ≈ C1:
   treat repeated sampling as the simpler default;
   investigate whether heterogeneity provides task-specific unique rescues.

3. If H1 < C1:
   do not rationalize the result.
   Prefer homogeneous sampling until another experiment justifies heterogeneity.

4. If oracle@k is high but practical selection is unresolved:
   generation is not the main bottleneck; selection/verification is.

5. If both C1 and H1 remain poor:
   investigate decomposition, context, task framing, or stronger generators
   before adding collaboration machinery.

6. No conclusion about synthesis is permitted from P1.

## Notes

- `oracle@k` is best-of-k with an oracle selector, not deployable performance.
- `agreement != verification`: only hidden-verifier outcomes count as evidence.
- n=12 < 30: no claim of causal superiority from P1 alone.
- H1 is allowed to lose. A negative premium is a valid, publishable outcome.
