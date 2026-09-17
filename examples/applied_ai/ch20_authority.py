"""Chapter 20: the grant that authorized an effect is one the record establishes.

Run it:

    python examples/applied_ai/ch20_authority.py

Three requests against the same recorded chain:

    root directive     READ + WRITE
          |
    child directive    WRITE
          |
    write   -> granted, with the chain as its basis
    destroy -> denied: the child never held that capability
    write, with a caller-supplied grant claiming DESTRUCTIVE
            -> still decided from the record; what the caller passed is ignored

The reduction a chapter can print:

    grant = runtime.directive_authority(request.directive_id)
    if not grant.allows(request.capability):
        refuse(request, grant)
    else:
        record_authorized(request, grant)
        execute(request)
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from codeai.adapters import ActionRequest, ActionResult, ActionStatus
from codeai.domain import Authority, Budget, Capability, Directive
from codeai.ledger import SQLiteLedger
from codeai.runtime import Runtime


class CountingWorker:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ActionRequest) -> ActionResult:
        self.calls += 1
        return ActionResult(action_id=request.action_id, status=ActionStatus.SUCCEEDED)


def request(action_id: str, capability: str, key: str) -> ActionRequest:
    return ActionRequest(
        action_id=action_id,
        task_id="demo",
        directive_id="review-child",
        capability=capability,
        instruction=f"{capability} the target",
        precondition_hash=None,
        idempotency_key=key,
        requested_by="human-approver",
        actor_id="file-worker",
        adapter="fixture",
    )


def main() -> dict[str, object]:
    with TemporaryDirectory() as directory:
        ledger = SQLiteLedger(Path(directory) / "ledger.sqlite")
        runtime = Runtime(ledger)
        worker = CountingWorker()

        runtime.open_directive(
            Directive(
                directive_id="review-root",
                objective="review and repair the cache configuration",
                success_criteria=(),
                budget=Budget(max_tokens=1000),
                authority=Authority(frozenset({Capability.READ, Capability.WRITE})),
            )
        )
        runtime.open_directive(
            Directive(
                directive_id="review-child",
                parent_directive_id="review-root",
                objective="apply the approved edit",
                success_criteria=(),
                budget=Budget(max_tokens=500),
                authority=Authority(frozenset({Capability.WRITE})),
            )
        )

        standing = runtime.directive_authority("review-child")
        chain = " -> ".join(link.directive_id for link in standing.grant_chain)
        print(f"resolved chain  : {chain}")
        print(f"effective grant : {set(standing.effective_capabilities)}")

        allowed = runtime.execute_action(request("edit", "write", "demo:edit"), adapter=worker)
        print(f"write           : {allowed.status}")

        refused = runtime.execute_action(
            request("destroy", "destructive", "demo:destroy"), adapter=worker
        )
        print(f"destructive     : {refused.status}  ({refused.error})")

        forged = runtime.execute_action(
            request("forged", "destructive", "demo:forged"),
            adapter=worker,
            authority=Authority(frozenset({Capability.DESTRUCTIVE})),
        )
        print(f"forged grant    : {forged.status}  <- the caller's claim is not consulted")
        print(f"worker calls    : {worker.calls}")

        basis = ledger.events_by_kind(("action.authorized",))[0].payload
        print(f"authorized basis: {basis['grant_source']}, {len(basis['basis_event_ids'])} events")

        summary = {
            "chain": [link.directive_id for link in standing.grant_chain],
            "effective": list(standing.effective_capabilities),
            "write": str(allowed.status),
            "destructive": str(refused.status),
            "forged": str(forged.status),
            "worker_calls": worker.calls,
            "grant_source": basis["grant_source"],
            "basis_events": len(basis["basis_event_ids"]),
        }
        ledger.close()
        return summary


if __name__ == "__main__":
    main()
