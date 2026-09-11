# P2 Preregistration — Diversity Without More Models

Committed before any P2 model call. Frozen on first call; amendments require a
dated note stating what was seen first.

## Question

Can manufactured error diversity inside one model (epistemic stance framing)
replace heterogeneous-model diversity?

## Hypotheses

- **H1**: stance-diverse sampling (P2-S, 12 calls) improves verified oracle over
  same-count normal sampling (P2-C, 12 calls).
- **H2**: stance diversity reduces conditional failure overlap relative to
  repeated normal sampling (stance-pair `P(fail|fail)` below normal self-overlap).
- **H3**: any gain concentrates in tasks whose reference repair is REMOVE,
  RELOCATE, or REDEFINE (premise lies in the abstraction, not its implementation).
- **Null**: no meaningful difference; extra framing merely changes wording.

Drop rule: if P2-S does not beat P2-C, drop stance framing rather than inventing
elaborate personas.

## Design (matched)

- Model: `qwen-local` only. Corpus: `semantic-repair-v1`. Verification: hidden tests.
- P2-C: normal prompt × 12 (byte-identical to P1.1 prompts — control validity).
- P2-S: normal × 3, assumption_challenge × 3, minimality × 3, counterfactual × 3.
- Stance texts: `src/codeai/stances.py` (short suffixes; only independent variable
  is reasoning stance). Recorded per call in `Variant.prompt_variant` + tags.
- Same `sealed_fanout`, isolation, verifier, metrics. No synthesis.

## Tasks (12 = 4 hard-core + 8 near-miss)

Hard-core — can stance move the competence boundary?

| task | premise label |
|------|---------------|
| v2-registry-pollution | RELOCATE |
| v2-retry-once-accepted | RELOCATE |
| v2-size-format | REDEFINE |
| v2-stable-priority | REDEFINE |

Near-miss — can stance improve reliability / decorrelate failures?

| task | premise label | P1.1 signal |
|------|---------------|-------------|
| v2-splitlines-cr | PRESERVE | C1 1/3, H1 0/3 |
| v2-half-up-rounding | REDEFINE | C1 0/3, H1 1/3 |
| v2-strip-query | PRESERVE | C1 0/3, H1 1/3 |
| v2-dt-roundtrip | REDEFINE | C1 1/3, H1 2/3 |
| v2-lsp-square | REMOVE | C0 1/1 only |
| v2-unsound-cache | REMOVE | C0 1/1 only |
| v2-env-timing | RELOCATE | C1 1/3, H1 2/3 |
| v2-single-append | REMOVE | C1 1/3, H1 1/3 |

Premise labels: PRESERVE = fix inside existing mechanism; REMOVE = delete
machinery/state; RELOCATE = move responsibility/ownership; REDEFINE = change
the governing invariant.

## Metrics

Primary: oracle(P2-S) − oracle(P2-C). Secondary: stance rescue rate vs normal
failures, stance-pair conditional failure, unique stance rescues, per-stance
candidate quality (oracle gains must not hide weaker candidates — P1.1 lesson),
tokens per additional rescue, breakdown by premise label.

## Interpretation rules

1. H1 holds materially → pursue framing diversity; test which premise types respond.
2. H2 holds without H1 → stances decorrelate but don't yet convert; selection problem.
3. H3 holds → normal prompting is preservation-biased; abstraction-level bugs need
   abstraction-questioning prompts.
4. Null → drop the idea.
5. No conclusion about synthesis permitted from P2.
