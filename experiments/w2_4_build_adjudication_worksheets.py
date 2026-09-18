"""Build the blinded adjudication worksheets the router oracle requires.

`router_contract.adjudicate` enforces the design: two distinct adjudicators per
case, each with a decision, a written reason, the facts they relied on, a version
and a timestamp. No tie-breaker; disagreement stays AMBIGUOUS. Nothing in the
runtime can supply that, and scoring the scheduler's own output as ground truth
is forbidden — so the oracle is people, and this script prepares their work.

Each worksheet contains, per case: the case id, its component and sub-stratum,
the state or narrative as an adjudicator would see it, and empty fields to fill.
It deliberately does **not** contain the deterministic router's answer, or any
model's, because the oracle is established before either runs.

Run it:

    PYTHONPATH=src python experiments/w2_4_build_adjudication_worksheets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codeai.router_contract import ORACLE_VERSION, digest, validate_corpus  # noqa: E402

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "router_cases_v1.CANDIDATE.json"   # reviewed at Checkpoint A

PRECEDENCE = """Canonical precedence (Ch28 design section 1b), which every reason must cite:

    process-STOP  >  verification-CHECK  >  proposals-CALL  >  destructive-ASK_HUMAN  >  fallback STOP

    model_budget_exhausted    no further model spend authorized; prohibits CALL only
    process_budget_exhausted  no further work of any kind; STOP dominates everything
    has_required_verification a deterministic check remains pending; CHECK unless a
                              process-level stop dominates

Label each case with one of CALL, CHECK, ASK_HUMAN, STOP, or AMBIGUOUS.

AMBIGUOUS is a real answer, not a failure: use it whenever the state as written
does not determine the operation. Those cases still run, and they inform
divergence and catastrophe counts; they simply leave the correctness numerators.

Do not consult the scheduler, any model, or any arm's output. Scoring the current
implementation as ground truth is forbidden by the design: the oracle exists to
judge the implementation, not to echo it."""


def render(case) -> dict[str, object]:
    row: dict[str, object] = {
        "case_id": case.case_id,
        "component": case.component,
        "sub_stratum": case.sub_stratum,
        "semantic_validity_as_authored": case.semantic_validity,
    }
    if case.narrative:
        row["narrative"] = case.narrative
    else:
        row["state"] = case.state
    if case.distractor:
        row["distractor"] = case.distractor
    if case.base_case_id:
        row["pair_of"] = case.base_case_id
    row["decision"] = ""          # CALL | CHECK | ASK_HUMAN | STOP | AMBIGUOUS
    row["reason"] = ""            # why, citing the precedence table
    row["facts"] = ""             # the state facts relied on
    return row


def main() -> dict[str, object]:
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    cases = validate_corpus(corpus)
    rows = [render(case) for case in cases]

    written = []
    for adjudicator in ("adjudicator-a", "adjudicator-b"):
        payload = {
            "version": ORACLE_VERSION,
            "corpus_hash": digest(corpus),
            "corpus_status": corpus.get("status"),
            "adjudicator_id": f"REPLACE-WITH-YOUR-IDENTIFIER ({adjudicator})",
            "instructions": PRECEDENCE,
            "timestamp": "REPLACE-WITH-ISO-8601-WHEN-COMPLETE",
            "adjudications": rows,
        }
        destination = HERE / f"W2-4-oracle-worksheet-{adjudicator}.json"
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written.append(destination.name)

    summary = {
        "cases": len(rows),
        "by_component": {
            component: sum(1 for c in cases if c.component == component)
            for component in sorted({c.component for c in cases})
        },
        "needing_author_review": sum(
            1 for c in cases if c.semantic_validity == "review_required"
        ),
        "corpus_hash": digest(corpus),
        "corpus_status": corpus.get("status"),
        "worksheets": written,
    }
    print("=" * 78)
    print("W2-4 oracle worksheets")
    print("=" * 78)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()
    print("Two independent people must each complete one worksheet. The contract")
    print("refuses a single adjudicator, a missing reason, and any tie-breaker.")
    return summary


if __name__ == "__main__":
    main()
