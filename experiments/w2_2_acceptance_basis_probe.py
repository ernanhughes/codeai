"""W2-2 case A: can a second process derive the action behind an acceptance?

Protocol: experiments/W2-2-prereg.md, registered before this file existed.

The Chapter 29 gap says an acceptance names its call and its checks but never an
action, so the effect joins the acceptance only through a hash equality that
happened to hold. Before building anything, measure whether that equality is
enough: given only the durable record, can a reader name the action unambiguously?

The derivation a second process would have to perform, using nothing but records:

    acceptance -> check ids
              -> each check.completed's runtime observation H
              -> actions whose completion observed H
              -> exactly one? then the action is derivable

Run it:

    PYTHONPATH=src python experiments/w2_2_acceptance_basis_probe.py
"""

from __future__ import annotations

import json
import sys
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codeai.acceptance import artifact_target  # noqa: E402
from codeai.adapters import (  # noqa: E402
    ActionRequest,
    ActionResult,
    ActionStatus,
    CheckRequest,
    CheckResult,
    CheckVerdict,
)
from codeai.artifacts import FileArtifactStore  # noqa: E402
from codeai.domain import Authority, Budget, Capability, Directive, Task  # noqa: E402
from codeai.ledger import SQLiteLedger  # noqa: E402
from codeai.runtime import Runtime  # noqa: E402


class PassingVerifier:
    def run(self, request: CheckRequest) -> CheckResult:
        return CheckResult(check_id=request.check_id, verdict=CheckVerdict.PASS)


class Writer:
    """Writes the text it was given, so the observed state actually moves."""

    def __init__(self, target: Path, text: str) -> None:
        self.target, self.text, self.calls = target, text, 0

    def execute(self, request: ActionRequest) -> ActionResult:
        self.calls += 1
        self.target.write_text(self.text, encoding="utf-8")
        return ActionResult(action_id=request.action_id, status=ActionStatus.SUCCEEDED)


class Idle:
    """Reports success and changes nothing."""

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ActionRequest) -> ActionResult:
        self.calls += 1
        return ActionResult(action_id=request.action_id, status=ActionStatus.SUCCEEDED)


class Bench:
    def __init__(self, directory: Path) -> None:
        self.root = directory
        self.target = directory / "target.txt"
        self.target.write_text("before\n", encoding="utf-8")
        self.ledger = SQLiteLedger(directory / "ledger.sqlite")
        self.runtime = Runtime(
            self.ledger,
            artifact_store=FileArtifactStore(directory / "artifacts", self.ledger),
            state_resolver=self.state_hash,
        )
        self.runtime.open_directive(
            Directive(directive_id="d", objective="fixture", success_criteria=(),
                      budget=Budget(), authority=Authority(frozenset({Capability.WRITE,
                                                                     Capability.ACCEPT})))
        )

    def state_hash(self) -> str:
        return sha256(self.target.read_bytes()).hexdigest()

    def task(self, task_id="t1"):
        self.runtime.create_task(Task(task_id, "d", "fixture", (), Budget(), Authority()))

    def action(self, action_id, adapter, task_id="t1"):
        return self.runtime.execute_action(
            ActionRequest(action_id=action_id, task_id=task_id, directive_id="d",
                          capability="write", instruction="write", precondition_hash=None,
                          idempotency_key=f"k-{action_id}", requested_by="human",
                          actor_id="worker", adapter="fixture"),
            adapter=adapter,
        )

    def check(self, check_id, *, bind_state=True, task_id="t1"):
        return self.runtime.run_check(
            CheckRequest(check_id=check_id, task_id=task_id,
                         target_state_hash=self.state_hash() if bind_state else None),
            verifier=PassingVerifier(),
        )

    def close(self):
        self.ledger.close()


# ---------------------------------------------------------------- the derivation


def derive_actions(ledger: SQLiteLedger, check_ids: tuple[str, ...]) -> dict[str, object]:
    """What a second process can work out, from records alone.

    It knows only the acceptance's check ids. It may read any event. It may not
    ask the producer anything.
    """
    completed = {
        event.stream_id: event.payload
        for event in ledger.events_by_kind(("check.completed",))
    }
    observations = []
    for check_id in check_ids:
        payload = completed.get(check_id) or {}
        observations.append(payload.get("observed_target_state_hash"))

    actions = [
        event.payload
        for event in ledger.events_by_kind(("action.completed",))
    ]
    candidates: list[str] = []
    for observation in observations:
        if observation is None:
            continue
        for action in actions:
            if action.get("observed_state_hash") == observation:
                candidates.append(str(action.get("action_id")))
    unique = sorted(set(candidates))
    return {
        "check_observations": observations,
        "candidate_actions": unique,
        "derivable": len(unique) == 1,
        "reason": (
            "no check carried a runtime observation" if all(o is None for o in observations)
            else "exactly one action observed the checked state" if len(unique) == 1
            else "no action observed the checked state" if not unique
            else f"{len(unique)} actions observed the checked state"
        ),
    }


# ---------------------------------------------------------------- the shapes


def shape_simple(b: Bench) -> tuple[str, ...]:
    """One action, one bound check: the shape the capstone ran."""
    b.task()
    b.action("a1", Writer(b.target, "after\n"))
    b.check("k1")
    return ("k1",)


def shape_idle_collision(b: Bench) -> tuple[str, ...]:
    """A real action, then one that changed nothing: two actions, one state."""
    b.task()
    b.action("a1", Writer(b.target, "after\n"))
    b.action("a2", Idle())
    b.check("k1")
    return ("k1",)


def shape_other_task(b: Bench) -> tuple[str, ...]:
    """Another task's action lands on the same state."""
    b.task("t1")
    b.task("t2")
    b.action("a1", Writer(b.target, "after\n"))
    b.action("a-other", Idle(), task_id="t2")
    b.check("k1")
    return ("k1",)


def shape_replay(b: Bench) -> tuple[str, ...]:
    """The same operation replayed under its key."""
    b.task()
    writer = Writer(b.target, "after\n")
    b.action("a1", writer)
    b.runtime.execute_action(
        ActionRequest(action_id="a1-again", task_id="t1", directive_id="d", capability="write",
                      instruction="write", precondition_hash=None, idempotency_key="k-a1",
                      requested_by="human", actor_id="worker", adapter="fixture"),
        adapter=writer,
    )
    b.check("k1")
    return ("k1",)


def shape_unbound_check(b: Bench) -> tuple[str, ...]:
    """The acceptance-eligible check binds an artifact, not a state."""
    b.task()
    b.action("a1", Writer(b.target, "after\n"))
    b.check("k1", bind_state=False)
    return ("k1",)


def shape_state_moved(b: Bench) -> tuple[str, ...]:
    """The action ran, the world moved again, then the check bound the new state."""
    b.task()
    b.action("a1", Writer(b.target, "after\n"))
    b.target.write_text("after, edited by something else\n", encoding="utf-8")
    b.check("k1")
    return ("k1",)


SHAPES = (
    ("one action, bound check", shape_simple),
    ("a second action that changed nothing", shape_idle_collision),
    ("another task's action on the same state", shape_other_task),
    ("the action replayed under its key", shape_replay),
    ("an acceptance-eligible check with no state binding", shape_unbound_check),
    ("the world moved after the action", shape_state_moved),
)


def main() -> dict[str, object]:
    findings = []
    for name, build in SHAPES:
        with TemporaryDirectory() as directory:
            b = Bench(Path(directory))
            try:
                check_ids = build(b)
                result = derive_actions(b.ledger, check_ids)
            finally:
                b.close()
        findings.append({"shape": name, **result})

    print("=" * 78)
    print("W2-2 case A: can a second process derive the action behind an acceptance?")
    print("=" * 78)
    for f in findings:
        mark = "yes" if f["derivable"] else "NO "
        print(f"  derivable: {mark}  {f['shape']}")
        print(f"             candidates={f['candidate_actions']}  ({f['reason']})")
    derivable = sum(1 for f in findings if f["derivable"])
    print()
    print(f"derivable in {derivable} of {len(findings)} shapes")
    return {"findings": findings, "derivable": derivable, "shapes": len(findings)}


if __name__ == "__main__":
    output = main()
    destination = Path(__file__).resolve().parent / "W2-2-case-a-results.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"frozen: {destination}")
