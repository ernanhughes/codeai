"""W2-3 falsifier: is decision-basis gating superseded by Wave 1?

Chapter 20 proposes a seam: an action that names a Chapter 18 decision should
check that decision's current standing before executing, and refuse or seek
renewed approval when the basis has moved.

Wave 1 built two things that sound adjacent, and the question is whether they
consumed this idea:

    W1-R4 decision/execution binding   which operation a decision selected
    W1-R4 freshness                    whole projected-state digest

Before building anything, run the original hostile case and see whether it still
does what the chapter says it does.

    a decision rests on a supported claim
    an action is performed, naming that decision
    the claim is later refuted, so the decision's standing becomes basis_changed
    -> does anything refuse, or even notice?

Run it:

    PYTHONPATH=src python experiments/w2_3_decision_basis_falsifier.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codeai.adapters import ActionRequest, ActionResult, ActionStatus  # noqa: E402
from codeai.artifacts import FileArtifactStore  # noqa: E402
from codeai.domain import (  # noqa: E402
    Authority,
    Budget,
    Capability,
    Claim,
    Directive,
    Task,
)
from codeai.governance import GovernanceSource, resolve_governance  # noqa: E402
from codeai.ledger import Event, SQLiteLedger  # noqa: E402
from codeai.runtime import Runtime  # noqa: E402


class Worker:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ActionRequest) -> ActionResult:
        self.calls += 1
        return ActionResult(action_id=request.action_id, status=ActionStatus.SUCCEEDED)


def main() -> dict[str, object]:
    findings: dict[str, object] = {}
    with TemporaryDirectory() as directory:
        root = Path(directory)
        ledger = SQLiteLedger(root / "ledger.sqlite")
        runtime = Runtime(ledger, artifact_store=FileArtifactStore(root / "artifacts", ledger))
        runtime.open_directive(
            Directive(directive_id="d", objective="fixture", success_criteria=(),
                      budget=Budget(),
                      authority=Authority(frozenset({Capability.WRITE, Capability.ACCEPT})))
        )
        runtime.create_task(Task("t1", "d", "fixture", (), Budget(), Authority()))

        # A decision resting on a claim. The v0 claim path is enough to ask the
        # question; what matters is whether anything downstream consults it.
        runtime.record_claim(
            Claim(claim_id="c1", task_id="t1", statement="the cache is safe to widen",
                  source_call_id="m")
        )
        ledger.append(
            Event.create(stream_id="dec-1", kind="decision.recorded", actor_id="reviewer",
                         payload={"decision_id": "dec-1", "task_id": "t1",
                                  "statement": "widen the cache",
                                  "relied_on_claim_ids": ["c1"]},
                         correlation_id="t1")
        )

        # An action performed under that decision, naming it the only way the
        # contract allows: in application payload data.
        worker = Worker()
        result = runtime.execute_action(
            ActionRequest(action_id="a1", task_id="t1", directive_id="d", capability="write",
                          instruction="widen the cache", precondition_hash=None,
                          idempotency_key="k-a1", requested_by="human", actor_id="worker",
                          adapter="fixture", payload={"decision_id": "dec-1"}),
            adapter=worker,
        )
        findings["action_under_decision"] = {
            "status": str(result.status), "adapter_calls": worker.calls,
        }

        # Does the action record connect to the decision in any way the runtime
        # would act on?
        requested = next(
            e for e in ledger.events_by_kind(("action.requested",))
            if e.payload.get("action_id") == "a1"
        )
        findings["action_request_fields"] = sorted(requested.payload)
        findings["decision_only_in_application_payload"] = (
            "decision_id" in (requested.payload.get("payload") or {})
            and "decision_id" not in requested.payload
        )

        # Now the basis moves: the claim the decision relied on is refuted.
        ledger.append(
            Event.create(stream_id="c1", kind="claim.status", actor_id="verifier",
                         payload={"claim_id": "c1", "status": "refuted",
                                  "details": "a later check refuted it"},
                         correlation_id="t1")
        )

        # 1. Does a second, identical action refuse now?
        again = Worker()
        second = runtime.execute_action(
            ActionRequest(action_id="a2", task_id="t1", directive_id="d", capability="write",
                          instruction="widen the cache", precondition_hash=None,
                          idempotency_key="k-a2", requested_by="human", actor_id="worker",
                          adapter="fixture", payload={"decision_id": "dec-1"}),
            adapter=again,
        )
        findings["action_after_basis_moved"] = {
            "status": str(second.status), "adapter_calls": again.calls,
        }

        # 2. Does W1-R4 governance cover this at all?
        standing = resolve_governance(
            runtime, task_id="t1", operation="ACTION", decision_id="dec-1",
            source=GovernanceSource.SCHEDULER,
        )
        findings["governance_on_an_action_naming_a_decision"] = {
            "status": str(standing.status), "reason": standing.reason,
        }

        # 3. Does the freshness digest notice a refuted claim?
        from codeai.governance import current_basis_sha256

        before = current_basis_sha256(runtime, "t1")
        ledger.append(
            Event.create(stream_id="c1", kind="claim.status", actor_id="verifier",
                         payload={"claim_id": "c1", "status": "refuted",
                                  "details": "and again"},
                         correlation_id="t1")
        )
        after = current_basis_sha256(runtime, "t1")
        findings["claim_movement_moves_the_freshness_digest"] = before != after

        ledger.close()

    print("=" * 78)
    print("W2-3 falsifier: does decision-basis gating survive Wave 1?")
    print("=" * 78)
    for key, value in findings.items():
        print(f"  {key}: {value}")
    survives = (
        findings["action_after_basis_moved"]["status"] == "succeeded"
        and findings["action_after_basis_moved"]["adapter_calls"] == 1
        and not findings["claim_movement_moves_the_freshness_digest"]
    )
    print()
    print(f"the original hostile case still executes unrefused: {survives}")
    findings["falsifier_survives"] = survives
    return findings


if __name__ == "__main__":
    output = main()
    destination = Path(__file__).resolve().parent / "W2-3-falsifier-results.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"frozen: {destination}")
