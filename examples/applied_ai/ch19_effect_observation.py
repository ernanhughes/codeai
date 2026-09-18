"""Chapter 19: the agent said done. Did anything change?

Run it:

    python examples/applied_ai/ch19_effect_observation.py

Four workers report exactly the same thing -- SUCCEEDED -- and the record says
four different things, none of them on the worker's say-so:

    diligent   the scope changed, and the change was the right one
    busy       the scope changed, and the change was the wrong one
    idle       the scope did not change at all
    unwatched  nothing could observe the scope

Three dimensions, kept apart:

    result status      what the actor said       SUCCEEDED / FAILED / DENIED
    state observation  what the runtime read     CHANGED / UNCHANGED / UNAVAILABLE
    effect state       what the record supports  NONE / REPORTED / UNKNOWN / OBSERVED

The diligent and busy workers produce the *same* effect state. OBSERVED means
the scope the runtime can see is not what it was; it does not mean the intended
change was made. Only a declared check against a bound target separates them:

    reported   the actor's claim
    observed   the runtime's own before/after readings
    verified   a declared check against a bound target

The reduction a chapter can print:

    before = observe()
    record(execution_started, state_hash_before=before)
    result = adapter.execute(request)
    after  = observe()

    if   before is None or after is None:  effect = REPORTED   # nobody looked
    elif after != before:                  effect = OBSERVED   # something moved
    else:                                  effect = UNKNOWN    # report and reading disagree
"""

from __future__ import annotations

import sys
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from codeai.adapters import ActionRequest, ActionResult, ActionStatus, CheckRequest
from codeai.domain import Authority, Budget, Capability, Directive
from codeai.ledger import SQLiteLedger
from codeai.runtime import Runtime
from codeai.verifier import LocalCommandVerifier

# The criterion: the paragraph explains the cache instead of quoting a figure.
CRITERION = (
    "import sys, pathlib; "
    "text = pathlib.Path(sys.argv[1]).read_text(); "
    "sys.exit(0 if 'cache' in text and '%' not in text else 1)"
)

ORIGINAL = "Pages load 73% faster.\n"
REPAIRED = "The cache is intended to make page loads faster [S1].\n"
WRONG = "Page loads are faster than before.\n"


class Worker:
    """Reports success. Whether it does anything is the point of the example."""

    def __init__(self, target: Path | None = None, text: str = "") -> None:
        self.target, self.text, self.calls = target, text, 0

    def execute(self, request: ActionRequest) -> ActionResult:
        self.calls += 1
        if self.target is not None:
            self.target.write_text(self.text, encoding="utf-8")
        return ActionResult(action_id=request.action_id, status=ActionStatus.SUCCEEDED)


def request(action_id: str) -> ActionRequest:
    return ActionRequest(
        action_id=action_id,
        task_id="t1",
        directive_id="d1",
        capability="write",
        instruction="rewrite the paragraph so it explains the cache",
        precondition_hash=None,
        idempotency_key=f"key:{action_id}",
        requested_by="human",
        actor_id="worker",
        adapter="fixture",
    )


def build(directory: Path, *, observable: bool) -> tuple[Runtime, Path, SQLiteLedger]:
    """One runtime over its own directory, watched or unwatched."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "paragraph.txt"
    target.write_text(ORIGINAL, encoding="utf-8")
    ledger = SQLiteLedger(directory / "ledger.sqlite")
    resolver = (lambda: sha256(target.read_bytes()).hexdigest()) if observable else None
    runtime = Runtime(ledger, state_resolver=resolver)
    runtime.open_directive(
        Directive(
            directive_id="d1", objective="repair the paragraph", success_criteria=(),
            budget=Budget(max_tokens=1000),
            authority=Authority(frozenset({Capability.WRITE})),
        )
    )
    return runtime, target, ledger


def show(label: str, state) -> None:
    print(
        f"{label}: report={state.result_status:9} observation={state.state_observation:11} "
        f"effect={state.effect_state:8} next={state.next_operation}"
    )


def verify(runtime: Runtime, target: Path, check_id: str) -> str:
    result = runtime.run_check(
        CheckRequest(
            check_id=check_id, task_id="t1",
            command=(sys.executable, "-c", CRITERION, str(target)),
            cwd=str(target.parent),
            target_state_hash=sha256(target.read_bytes()).hexdigest(),
        ),
        verifier=LocalCommandVerifier(),
    )
    return str(result.verdict)


def main() -> dict[str, object]:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        runtime, target, ledger = build(root / "watched", observable=True)

        runtime.execute_action(request("a-diligent"), adapter=Worker(target, REPAIRED))
        diligent = runtime.action_state("a-diligent")
        show("diligent worker ", diligent)
        diligent_verdict = verify(runtime, target, "k-diligent")
        print(f"                  verification: {diligent_verdict}")

        runtime.execute_action(request("a-busy"), adapter=Worker(target, WRONG))
        busy = runtime.action_state("a-busy")
        show("busy worker     ", busy)
        busy_verdict = verify(runtime, target, "k-busy")
        print(f"                  verification: {busy_verdict}  <- same effect state, "
              f"different answer")

        runtime.execute_action(request("a-idle"), adapter=Worker())
        idle = runtime.action_state("a-idle")
        show("idle worker     ", idle)
        open_effects = [s.action_id for s in runtime.open_effects(task_id="t1")]
        print(f"                  open effects awaiting a person: {open_effects}")

        blind_runtime, _, blind_ledger = build(root / "unwatched", observable=False)
        blind_runtime.execute_action(request("a-unwatched"), adapter=Worker())
        unwatched = blind_runtime.action_state("a-unwatched")
        show("unwatched worker", unwatched)

        print()
        print("every worker reported SUCCEEDED; the record did not take any of their words for it")

        summary = {
            "reports": sorted({
                diligent.result_status, busy.result_status, idle.result_status,
                unwatched.result_status,
            }),
            "diligent": {"observation": str(diligent.state_observation),
                         "effect": str(diligent.effect_state), "verdict": diligent_verdict},
            "busy": {"observation": str(busy.state_observation),
                     "effect": str(busy.effect_state), "verdict": busy_verdict},
            "idle": {"observation": str(idle.state_observation),
                     "effect": str(idle.effect_state), "next": str(idle.next_operation),
                     "duplicate_effect_risk": idle.duplicate_effect_risk},
            "unwatched": {"observation": str(unwatched.state_observation),
                          "effect": str(unwatched.effect_state),
                          "next": str(unwatched.next_operation)},
            "open_effects": open_effects,
        }
        ledger.close()
        blind_ledger.close()
        return summary


if __name__ == "__main__":
    main()
