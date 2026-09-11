# P2 Results — Diversity Without More Models

Status: **P2 complete.** Stance-diversity portfolio tested against matched-count
normal sampling, one model (`qwen-local`), 12 tasks.

- Experiment: `e16643de-2638-4097-9461-ab6d0b51f6ef`
- Config hash: `7ff4f1a401af3f161b83a591da9c7c430207514bdeb06414a7f3baf8a4f3de19`
- 12 tasks × (P2-C normal×12 + P2-S 4 stances×3) = 288 calls, 288 checks.
- Stance texts: `src/codeai/stances.py` (frozen pre-call at `a348010`).
- Preregistration: `P2-prereg.md` (H1/H2/H3/Null + drop rule).
- Raw export: `p1-runs/p2-export.json`.

## Metrics (exact)

| arm | oracle | candidate solves | tokens |
|-----|--------|----------------|--------|
| P2-C (normal×12) | **0.8333 (10/12)** | 51/144 | 28273 |
| P2-S (portfolio) | 0.7500 (9/12) | 53/144 | 34594 |

- Primary (H1): **fails**. Portfolio underperforms same-count normal by one task
  at ~22% more tokens. P2-C uniquely solved `v2-registry-pollution` (3/12 draws);
  no stance solved anything P2-C missed.
- Per-stance candidate quality (36 calls each): normal 0.306, assumption 0.306,
  minimality 0.417, **counterfactual 0.444**. Task solve sets: 7/7/7/9.
- H2 (decorrelation): **partial**. Normal/assumption/minimality solve sets are
  identical (pairwise P(fail|fail) = 1.00). Counterfactual differs:
  P(counterfactual solves | another stance fails) = 0.40, and every
  counterfactual failure is failed by all others — its 9-task set is a strict
  superset of each other stance's 7. It alone equals the full portfolio oracle.
- H3 (premise concentration): **no support**. P2-S oracle matches P2-C on
  PRESERVE (2/2), REMOVE (3/3), REDEFINE (3/4); worse on RELOCATE (1/3 vs 2/3).

## Preregistered interpretation applied

Null holds at portfolio level: per the drop rule, **stance diversification as a
portfolio strategy is dropped** — it cost more and won less. The surviving
observation is narrower and is recorded as hypothesis only, not conclusion:
counterfactual framing alone shows the best per-candidate rate and the largest
solve set, consistent with attacking premise preservation directly. Testing
counterfactual×N as a *default* prompt requires its own preregistered experiment.

## Standing lesson (spans P1–P2)

Oracle diversity and candidate competence are different quantities — in both
directions now: H1 had the better oracle with worse candidates (P1.1);
P2-S has better candidates in two stances with a worse oracle (P2).

## Limits

One local model; n=12; stance texts are one arbitrary wording each.
Cloud-model generality untested. Cost unknown (tokens/latency only).
