# P3 Results — Counterfactual as Default

Status: **P3 complete.** Single-variable replication of the P2 counterfactual
signal: 12 normal draws vs 12 counterfactual draws, same model, same 12 tasks.

- Experiment: `b5d147f4-d657-4204-b49e-d20af4a060bd`
- Config hash: `3800306ba9d1a93c64533edc659d2e7def37f7d71a40a434e19147476185a6ce`
- 12 tasks × (P3-C ×12 + P3-CF ×12) = 288 calls, 286 checks.
- Preregistration: `P3-prereg.md` (rules 1–6). Raw export: `p1-runs/p3-export.json`.

## Metrics (exact)

| arm | oracle@12 | candidate solves | tokens | p50 latency |
|-----|-----------|----------------|--------|-------------|
| P3-C (normal×12) | 0.75 (9/12) | 44/144 | 28325 | 3.1s |
| P3-CF (counterfactual×12) | 0.75 (9/12) | 44/144 | 41303 | 3.7s |

- Identical oracle (same 9 tasks), identical candidate totals, identical failure
  sets (pairwise conditional failure 1.00 both ways, no rescues either side).
- Counterfactual cost 46% more tokens for literally the same outcomes.

## Per-task frequencies (task is the unit)

| task | P3-C | P3-CF |
|------|------|-------|
| dt-roundtrip | 10/12 | 6/12 |
| env-timing | 10/12 | 8/12 |
| unsound-cache | 3/12 | 9/12 |
| lsp-square | 6/12 | 8/12 |
| single-append | 1/12 | 2/12 |
| half-up-rounding | 6/12 | 5/12 |
| splitlines-cr | 3/12 | 2/12 |
| stable-priority | 4/12 | 3/12 |
| registry-pollution | 1/12 | 1/12 |
| retry-once-accepted | 0/12 | 0/12 |
| size-format | 0/12 | 0/12 |
| strip-query | 0/12 | 0/12 |

Swings cut both directions (−4, −2, +6, +2) around a null mean: task×prompt
interaction noise, not a stable framing effect.

## Preregistered interpretation applied

Rule 2 fires: the P2 difference was sampling noise. **Normal stays the default.**
Rule 3 would have fired on an oracle loss; the loss is zero but the token
premium (+46%) alone rejects counterfactual as a default. Rule 6 fires again:
`retry-once-accepted` and `size-format` remain universally unsolved — stop
attacking them with framing variants. Noted for the record: `registry-pollution`
cracked 1/12 under *both* framings independently (rare but replicable).

## Limits

One local model; n=12 tasks; one counterfactual wording. P3.1 out-of-sample
replication is moot — there is nothing to replicate.
