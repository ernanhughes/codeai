# Quality Trajectories (shadow mode)

Observational, deterministic measurement of structural code quality and its
change over time. No gates, no scores, no verdicts.

> Functional correctness and structural maintainability are different
> dimensions.
>
> A passing test suite does not imply that the cost of the next modification
> has remained constant.

## What is measured

All three metrics are computed with the standard library only (`ast`,
line analysis), carry explicit provenance (`tool@version`, formula,
coverage), and are reproducible from the same tree.

| metric | formula | reading |
|---|---|---|
| `loc` | non-blank, non-comment `.py` lines | contextual evidence only; growth alone is never a violation |
| `erosion` | `sum(CC(f)*sqrt(SLOC(f)) for CC(f)>10) / sum(CC(f)*sqrt(SLOC(f)))` | share of complexity mass in functions above CC 10 |
| `verbosity` | significant lines inside repeated 5-line windows / significant lines (significant = non-blank, non-comment, non-import, normalized) | codeai-specific condensability proxy reusing the duplication intuition without third-party deps |

Complexity counting (documented subset): base 1 per function plus one per
`if` / loop / `except` / boolean operand / ternary / `assert` /
comprehension generator / `match` case. Nested functions score separately.
Unparseable files are listed in `coverage.python_files_failed`; their
complexity is never invented. Empty scopes measure `0.0` with an explicit
`empty-not-clean` provenance note — missing evidence is not clean evidence.

Limitations: Python-only; CC subset undercounts `match` guards, chained
comparisons, and decorator-driven branching; the verbosity proxy sees
textual repetition, so legitimately parallel concepts can flag (that is why
output is a question, not a verdict).

## Why trajectories, not scores

Absolute repositories are never classified good/bad. Two snapshots are
compared per metric (`IMPROVING` / `STABLE` / `DEGRADING` / `STEP_CHANGE` /
`INSUFFICIENT_HISTORY`; `loc` only ever `STABLE` / `STEP_CHANGE` /
`INSUFFICIENT_HISTORY`). Bands: stable within 0.03 (erosion/verbosity) or
10% relative (loc); step at 0.15 or 50% relative.

## Quality Questions

`DEGRADING` / `STEP_CHANGE` on erosion or verbosity generates an
investigation candidate: a question with source refs, signal refs, evidence
refs, uncertainty, and a domain mapping (erosion → complexity,
maintainability, architecture; verbosity → duplication, residue,
maintainability). Generation is deterministic, stably ordered, and makes no
model calls. Candidates persist as ordinary `E0_ASSERTED` / `UNRESOLVED`
claims, so the existing claim projections (`claims_for_run`,
`disagreement_report`) accept them unchanged.

## Modification friction

`collect_episode_friction` projects existing ledger telemetry
(`call.completed`, `context.compiled`, `action.completed`,
`check.completed`, `fanout.completed`) into one analytical view
(tokens, calls, context selected, failed checks, cost, latency).
Facets without telemetry are `None`, never zero. Measurement only:
correlations with structure do not establish causality.

## Ledger representation

One `quality.snapshot` event per snapshot (single evidence record, not a
node per metric), claims for questions. No second history subsystem: repo
lineage is git; runtime lineage is the ledger.

## CLI (observational)

```bash
codeai quality measure TARGET
codeai quality compare BASE CURRENT
codeai quality trajectory TARGET --commits 20
```

Append `--json` for machine-readable output. `trajectory` replays history
read-only via `git archive` into temp dirs; the worktree is never touched.

## Recommended first experiment

Run `codeai quality trajectory . --commits 30` on this repository,
then join per-commit erosion/verbosity against per-episode friction
(context tokens, tool calls, retries, cost) from the ledger for the same
period. Look for metric-distribution shape, false positives (flagged but
frictionless changes), and whether post-consolidation episodes cost less —
before considering any gating.
