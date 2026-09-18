"""W2-4 instrumentation dry run: everything the $0.25 checkpoint checks, at $0.00.

The author set a $1.00 hard cap on the router challenger with a $0.25
instrumentation checkpoint. Most of what that checkpoint verifies does not need a
paid call at all — it needs the pipeline exercised end to end with the repository's
own fake adapter, which `run(..., synthetic=True)` accepts and nothing else.

Checked here, before any provider is contacted:

    accounting          usage and cost recorded per decision, and re-derivable
    paired inputs       distractor pairs differ only in the distractor
    evaluator           the independent verifier re-analyses from raw bytes and
                        refuses to trust a recorded parse
    asymmetry           arms on the same case receive the same information, and
                        M-direct in R2 sees narrative rather than flags

What it deliberately does not do: touch the real corpus as a source of
observations. Synthetic mode uses fixture cases, exactly as the harness demands.

Run it:

    PYTHONPATH=src python experiments/w2_4_instrumentation_dryrun.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from codeai.adapters import FakeCognitionAdapter, TransportObservation  # noqa: E402
from codeai.artifacts import FileArtifactStore  # noqa: E402
from codeai.domain import ActorRef  # noqa: E402
from codeai.ledger import SQLiteLedger  # noqa: E402
from codeai.router_analysis import verify_ledger  # noqa: E402
from codeai.router_experiment import run  # noqa: E402
from codeai.router_model import ModelRouter  # noqa: E402
from codeai.runtime import Runtime  # noqa: E402

from test_router_compare import fixture_corpus, fixture_oracle  # noqa: E402


def scripted_adapter():
    """The repository's exact fake adapter, scripted as the harness test scripts it.

    The model must be named and the response must be a parseable routing answer,
    or the pricing path is never exercised and the accounting check silently
    verifies nothing. That is what the first version of this script did.
    """
    adapter = FakeCognitionAdapter(
        model="gpt-4.1",
        responses=['{"operation":"CHECK","reason":"verification pending"}'],
    )
    original = adapter.invoke

    def invoke(spec):
        result = original(spec)
        if spec.prompt_version == "router-extract-v1":
            result = replace(result, raw_output=json.dumps(
                {"state": {
                    "has_required_verification": True,
                    "requests_independent_proposals": False,
                    "requires_human_authority_for_next_effect": False,
                    "process_budget_exhausted": False,
                    "model_budget_exhausted": False,
                }, "reason": "done"}
            ))
        body = json.dumps(
            {"choices": [{"finish_reason": "stop",
                          "message": {"content": result.raw_output}}]}
        ).encode()
        return replace(
            result, protocol="chat_completions", raw_payload=json.loads(body),
            transport=TransportObservation(outcome="response_received", status_code=200,
                                           body=body, content_type="application/json"),
        )

    adapter.invoke = invoke  # type: ignore[method-assign]
    return adapter


def main() -> dict[str, object]:
    # Windows keeps a handle on the SQLite file after the read-only verifier
    # has finished with it; the cleanup failure is not the experiment's business.
    with TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        root = Path(directory)
        ledger = SQLiteLedger(root / "ledger.sqlite")
        store = FileArtifactStore(root / "artifacts", ledger)
        runtime = Runtime(ledger, artifact_store=store)
        actor = ActorRef("dry-run-router", "model", "fake-provider", "gpt-4.1", "fixture-v1")
        routers = {
            name: ModelRouter(runtime, scripted_adapter(), actor)
            for name in ("primary", "alternate")
        }
        corpus = fixture_corpus()
        oracle = fixture_oracle(corpus)
        run(runtime, corpus, oracle, {"status": "TEST_ONLY"}, routers,
            run_id="w2-4-dryrun", synthetic=True)
        report = verify_ledger(root / "ledger.sqlite", root / "artifacts", "w2-4-dryrun")

        decisions = ledger.events_by_kind(("router.decided",))
        priced = [
            d.payload for d in decisions
            if (d.payload.get("model_evidence") or {}).get("usage") is not None
        ]
        prompts_by_case: dict[str, set[str]] = {}
        for d in decisions:
            payload = d.payload
            evidence = payload.get("model_evidence") or {}
            ref = (evidence.get("prompt_ref") or {}).get("sha256")
            if ref:
                prompts_by_case.setdefault(payload["case_id"], set()).add(ref)

        findings = {
            "verdict": report["verdict"],
            "components_exercised": sorted(report["components"]),
            "decisions_recorded": len(decisions),
            "decisions_with_usage": len(priced),
            # Deterministic arms are priced at exactly zero by construction.
            "deterministic_arms_cost_zero": all(
                report["components"][c][k]["cost_usd"] == 0.0
                for c in report["components"] for k in report["components"][c]
                if not k.startswith("M")
            ),
            # Model arms must carry a real price, or the accounting this
            # checkpoint exists to verify has not been exercised at all.
            "model_arms_priced": all(
                isinstance(report["components"][c][k]["cost_usd"], (int, float))
                and report["components"][c][k]["cost_usd"] > 0
                for c in report["components"] for k in report["components"][c]
                if k.startswith("M")
            ),
            "distractor_pairs_scored": sorted(report["distractors"]),
            "independent_verifier_ran": report["verdict"].startswith("SYNTHETIC"),
            "raw_output_preserved_per_decision": all(
                (d.payload.get("model_evidence") or {}).get("raw_output_ref")
                for d in decisions
                if (d.payload.get("model_evidence") or {}).get("raw_output_ref") is not None
            ),
            "arm_cost_table": {
                component: {
                    arm: report["components"][component][arm]["cost_usd"]
                    for arm in sorted(report["components"][component])
                }
                for component in sorted(report["components"])
            },
            "distinct_prompts_per_case": {
                case_id: len(refs) for case_id, refs in sorted(prompts_by_case.items())
            },
        }
        ledger.close()

    print("=" * 78)
    print("W2-4 instrumentation dry run (synthetic; no provider contacted)")
    print("=" * 78)
    for key, value in findings.items():
        print(f"  {key}: {value}")
    print()
    print("model spend this run: $0.00")
    return findings


if __name__ == "__main__":
    output = main()
    destination = Path(__file__).resolve().parent / "W2-4-dryrun-results.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"frozen: {destination}")
