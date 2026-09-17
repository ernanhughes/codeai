"""Chapters 19, 22 and 29: what a record can say about an effect it did not see.

Run it:

    python examples/applied_ai/ch22_action_recovery.py

The fixture is a worker that appends a line to a file and then loses its
connection. The effect happened. The completion says FAILED. Those are two
different facts, and the runtime keeps them apart:

    result status   FAILED        what the operation reported
    effect state    UNKNOWN       what the record establishes about the world

The ordering is what makes that decidable: action.execution_started is
committed before the adapter runs, so an interruption on either side of the
effect leaves a different record. Reconciliation then adds later evidence
without editing what was written at the time, and "still unknown" stays a
legitimate answer.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from codeai.actions import ReconciliationVerdict
from codeai.adapters import ActionRequest
from codeai.domain import Authority, Capability
from codeai.ledger import SQLiteLedger
from codeai.runtime import Runtime


class LosesTheConnectionAfterWriting:
    """Writes, then fails. The failure is real; so is the line in the file."""

    def __init__(self, target: Path) -> None:
        self.target = target
        self.calls = 0

    def execute(self, request: ActionRequest):
        self.calls += 1
        with self.target.open("a", encoding="utf-8") as handle:
            handle.write("one line\n")
        raise ConnectionError("connection reset after the write")


def main() -> dict[str, object]:
    with TemporaryDirectory() as directory:
        workspace = Path(directory)
        target = workspace / "target.txt"
        target.write_text("", encoding="utf-8")

        ledger = SQLiteLedger(workspace / "ledger.sqlite")
        runtime = Runtime(ledger)
        worker = LosesTheConnectionAfterWriting(target)
        request = ActionRequest(
            action_id="append-one-line",
            task_id="demo",
            capability="write",
            instruction="append one line to the target",
            precondition_hash=None,
            idempotency_key="demo:append:1",
            requested_by="human-approver",
            actor_id="file-worker",
            adapter="fixture",
        )

        result = runtime.execute_action(
            request, authority=Authority(frozenset({Capability.WRITE})), adapter=worker
        )
        after_crash = runtime.action_state("append-one-line")

        print(f"reported status : {result.status}")
        print(f"effect state    : {after_crash.effect_state}   <- not 'nothing happened'")
        print(f"next operation  : {after_crash.next_operation}")
        print(f"retry is unsafe : {after_crash.duplicate_effect_risk}")
        print(f"open effects    : {[s.action_id for s in runtime.open_effects()]}")

        # A person reads the target and records what they found. The action's
        # own history is not touched; the verdict is appended beside it.
        lines = target.read_text(encoding="utf-8").splitlines()
        verdict = (
            ReconciliationVerdict.EFFECT_CONFIRMED
            if lines
            else ReconciliationVerdict.NO_EFFECT_CONFIRMED
        )
        reconciled = runtime.reconcile_action(
            "append-one-line",
            verdict=verdict,
            actor_id="operator",
            evidence_refs=(f"file:{target.name}#lines={len(lines)}",),
            basis="read the target after the crash",
        )

        print(f"reconciled as   : {reconciled.effect_state} ({reconciled.basis})")
        print(f"reported status : {reconciled.result_status}   <- still what it was")
        print(f"adapter calls   : {worker.calls}")

        ledger.close()
        return {
            "reported_status": str(result.status),
            "effect_before_reconciliation": str(after_crash.effect_state),
            "retry_unsafe": after_crash.duplicate_effect_risk,
            "effect_after_reconciliation": str(reconciled.effect_state),
            "result_status_after_reconciliation": str(reconciled.result_status),
            "adapter_calls": worker.calls,
            "lines_written": len(lines),
        }


if __name__ == "__main__":
    main()
