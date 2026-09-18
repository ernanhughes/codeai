"""W2-2: accepting a result is not accepting an action as its producer.

Chapter 29 found the effect joining the acceptance through a hash equality that
happened to hold. The case-A probe measured how far that goes: the action was
derivable from the record in one of six shapes, and *not* in the ordinary one,
where the acceptance-eligible check binds an artifact rather than a state.

So the basis is now explicit, and optional:

    artifact / check basis   I accept this verified result
    effect / action basis    I accept this verified result as the outcome of A

An acceptance that names no action is untouched — a task may legitimately accept
an artifact without caring how it was produced. An acceptance that names one has
that relationship enforced:

    same task, the action exists, its effect state is OBSERVED, and a cited
    check examined that same observed state and passed

which establishes a binding, never a cause.
"""

from __future__ import annotations

import json
import uuid
from hashlib import sha256
from pathlib import Path

import pytest

from codeai.acceptance import (
    TASK_ACCEPTED,
    AcceptanceRejected,
    AcceptanceRequest,
    artifact_target,
    criteria_sha256,
    text_sha256,
)
from codeai.adapters import (
    ActionRequest,
    ActionResult,
    ActionStatus,
    CallSpec,
    CheckRequest,
    CheckResult,
    CheckVerdict,
)
from codeai.artifacts import FileArtifactStore
from codeai.context import ContextCompiler
from codeai.domain import ActorRef, Authority, Budget, Capability, Directive, Task
from codeai.ledger import SQLiteLedger
from codeai.providers import HttpResponse, OpenCodeCognitionAdapter
from codeai.runtime import Runtime

CRITERIA = ("no percentage figure",)
REPAIRED = "The cache is intended to make page loads faster."


class PassingVerifier:
    def run(self, request: CheckRequest) -> CheckResult:
        return CheckResult(check_id=request.check_id, verdict=CheckVerdict.PASS)


class Writer:
    def __init__(self, target: Path, text: str) -> None:
        self.target, self.text, self.calls = target, text, 0

    def execute(self, request: ActionRequest) -> ActionResult:
        self.calls += 1
        self.target.write_text(self.text, encoding="utf-8")
        return ActionResult(action_id=request.action_id, status=ActionStatus.SUCCEEDED)


class Idle:
    def execute(self, request: ActionRequest) -> ActionResult:
        return ActionResult(action_id=request.action_id, status=ActionStatus.SUCCEEDED)


class Crashes:
    def __init__(self, target: Path) -> None:
        self.target = target

    def execute(self, request: ActionRequest) -> ActionResult:
        self.target.write_text("half written\n", encoding="utf-8")
        raise ConnectionError("connection reset after the write")


class Bench:
    """A task with a real recorded call, so acceptance has something to accept."""

    def __init__(self, path: Path, *, observable: bool = True) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.root = path
        self.target = path / "target.txt"
        self.target.write_text("before\n", encoding="utf-8")
        ledger = SQLiteLedger(path / "ledger.sqlite")
        self.ledger = ledger
        self.runtime = Runtime(
            ledger,
            artifact_store=FileArtifactStore(path / "artifacts", ledger),
            state_resolver=(self.state_hash if observable else None),
        )
        self.runtime.open_directive(
            Directive(directive_id="d", objective="fixture", success_criteria=(),
                      budget=Budget(),
                      authority=Authority(frozenset({Capability.WRITE, Capability.ACCEPT})))
        )

    def state_hash(self) -> str:
        return sha256(self.target.read_bytes()).hexdigest()

    def task(self, task_id="t1"):
        self.runtime.create_task(Task(task_id, "d", "fixture", CRITERIA, Budget(), Authority()))

    def produce(self, task_id="t1"):
        body = json.dumps(
            {"id": "s", "choices": [{"finish_reason": "stop", "message": {"content": REPAIRED}}]}
        ).encode()

        def post(url, payload, headers, timeout):
            return HttpResponse(200, {"request-id": "s"}, body, "application/json")

        act = ActorRef("repairer", "model", provider="opencode", model="mimo-v2.5")
        context = ContextCompiler().compile(
            task_id=task_id, actor=act, prompt="Repair.", prompt_version="v1"
        )
        spec = CallSpec(str(uuid.uuid4()), task_id, act, context, str(uuid.uuid4()),
                        chamber="deep-review", parameters={"max_tokens": 64})
        adapter = OpenCodeCognitionAdapter(
            model="mimo-v2.5", protocol="chat_completions", gateway_plan="go",
            api_key="offline-decoy", http_post=post, timeout=60,
        )
        call = self.runtime.invoke_recorded_call(spec, adapter=adapter, max_attempts=1)
        attempt = call.attempts[-1]
        interpretation = self.runtime.interpretations_for_attempt(attempt.attempt_id)[-1]
        output = json.loads(
            self.runtime.artifact_store.read_text(attempt.raw_artifact.artifact_id)
        )["output_text"]
        self.runtime.artifact_store.store_text(output, artifact_type="candidate_output")
        self.produced = {
            "call_id": call.call_id, "attempt_id": attempt.attempt_id,
            "interpretation_id": interpretation.interpretation_id,
            "artifact_sha256": text_sha256(output), "task_id": task_id,
        }
        return self.produced

    def action(self, action_id, adapter, task_id="t1"):
        return self.runtime.execute_action(
            ActionRequest(action_id=action_id, task_id=task_id, directive_id="d",
                          capability="write", instruction="write", precondition_hash=None,
                          idempotency_key=f"k-{action_id}", requested_by="human",
                          actor_id="worker", adapter="fixture"),
            adapter=adapter,
        )

    def check(self, check_id, *, bind_state=True, task_id="t1"):
        """An acceptance-eligible check: artifact-bound, and optionally state-bound."""
        return self.runtime.run_check(
            CheckRequest(
                check_id=check_id, task_id=task_id,
                target=artifact_target(self.produced["artifact_sha256"]),
                target_state_hash=self.state_hash() if bind_state else None,
            ),
            verifier=PassingVerifier(),
        )

    def request(self, check_ids, *, action=None, task_id="t1"):
        return AcceptanceRequest(
            acceptance_id=str(uuid.uuid4()), task_id=task_id, actor_id="reviewer",
            criteria_sha256=criteria_sha256(CRITERIA),
            artifact_sha256=self.produced["artifact_sha256"],
            source_call_id=self.produced["call_id"],
            source_attempt_id=self.produced["attempt_id"],
            source_interpretation_id=self.produced["interpretation_id"],
            check_ids=tuple(check_ids), effect_action_id=action,
        )

    def accept(self, check_ids, *, action=None, task_id="t1"):
        return self.runtime.accept_task(self.request(check_ids, action=action, task_id=task_id))

    def refusal(self, check_ids, *, action=None, task_id="t1"):
        with pytest.raises(AcceptanceRejected) as refused:
            self.accept(check_ids, action=action, task_id=task_id)
        return refused.value.reasons

    def close(self):
        self.ledger.close()


def bench(tmp_path, **kwargs):
    return Bench(tmp_path / "run", **kwargs)


# ---------------- C: the claim the record can stand behind ----------------


def test_c_an_acceptance_may_claim_the_action_that_produced_the_checked_state(tmp_path):
    b = bench(tmp_path)
    b.task()
    b.produce()
    b.action("a1", Writer(b.target, "after\n"))
    b.check("k1")

    assert b.accept(["k1"], action="a1").status == "completed"
    [accepted] = b.runtime.ledger.events_by_kind((TASK_ACCEPTED,))
    basis = accepted.payload["acceptance_basis"]
    assert basis["kind"] == "artifact_check_and_effect"
    assert basis["effect"]["action_id"] == "a1"
    assert basis["effect"]["effect_state"] == "observed"
    assert basis["effect"]["checks_examining_that_state"] == ["k1"]
    assert basis["effect"]["action_completed_event_id"]
    # The record says what it establishes, so no later reader promotes it.
    assert "not that the action caused it" in basis["effect"]["establishes"]


# ---------------- G: a task that never had an action ----------------


def test_g_an_acceptance_that_claims_no_action_is_untouched(tmp_path):
    b = bench(tmp_path)
    b.task()
    b.produce()
    b.check("k1", bind_state=False)   # the ordinary artifact-bound check

    assert b.accept(["k1"]).status == "completed"
    [accepted] = b.runtime.ledger.events_by_kind((TASK_ACCEPTED,))
    basis = accepted.payload["acceptance_basis"]
    # Absence is recorded, not left for a reader to infer from silence.
    assert basis["kind"] == "artifact_check"
    assert basis["effect"] is None


# ---------------- B, D, E, F, H: the claims it must refuse ----------------


def test_b_an_acceptance_cannot_claim_an_action_that_produced_a_different_state(tmp_path):
    b = bench(tmp_path)
    b.task()
    b.produce()
    b.action("a1", Writer(b.target, "state one\n"))
    b.action("a2", Writer(b.target, "state two\n"))
    b.check("k1")   # bound to state two

    assert b.refusal(["k1"], action="a1") == ("effect_action_state_unchecked:a1",)
    # The same acceptance, naming the action that did produce it, is fine.
    assert b.accept(["k1"], action="a2").status == "completed"


def test_d_an_acceptance_cannot_claim_another_task_s_action(tmp_path):
    b = bench(tmp_path)
    b.task("t1")
    b.task("t2")
    b.produce("t1")
    b.action("a-other", Writer(b.target, "after\n"), task_id="t2")
    b.check("k1")

    assert b.refusal(["k1"], action="a-other") == ("effect_action_wrong_task:a-other",)


def test_e_a_failed_action_is_not_a_producer_even_when_a_later_check_passes(tmp_path):
    b = bench(tmp_path)
    b.task()
    b.produce()
    b.action("a-crash", Crashes(b.target))
    b.check("k1")

    reasons = b.refusal(["k1"], action="a-crash")
    assert reasons[0].startswith("effect_action_not_observed:a-crash:")
    # ...and the task is still acceptable on the ordinary basis. The work may be
    # fine; what is unestablished is that this action produced it.
    assert b.accept(["k1"]).status == "completed"


def test_f_an_unobserved_effect_cannot_be_claimed_as_the_producer(tmp_path):
    """UNKNOWN and REPORTED both refuse, for the same reason."""
    unknown = bench(tmp_path / "unknown")
    unknown.task()
    unknown.produce()
    unknown.action("a-idle", Idle())      # reported success, scope unchanged
    unknown.check("k1")
    assert unknown.refusal(["k1"], action="a-idle") == (
        "effect_action_not_observed:a-idle:unknown",
    )
    unknown.close()

    reported = bench(tmp_path / "reported", observable=False)  # nothing can observe
    reported.task()
    reported.produce()
    reported.action("a-unwatched", Idle())
    reported.check("k1", bind_state=False)
    assert reported.refusal(["k1"], action="a-unwatched") == (
        "effect_action_not_observed:a-unwatched:reported",
    )
    reported.close()


def test_h_an_acceptance_cannot_borrow_a_chain_the_world_has_moved_past(tmp_path):
    b = bench(tmp_path)
    b.task()
    b.produce()
    b.action("a1", Writer(b.target, "after\n"))
    b.check("k-then")                                   # bound to the action's state
    b.target.write_text("something else again\n", encoding="utf-8")
    b.check("k-now")                                    # bound to the state now

    # Accepting against the current state cannot cite the old action's chain.
    assert b.refusal(["k-now"], action="a1") == ("effect_action_state_unchecked:a1",)
    # The old check still stands for what it examined.
    assert b.accept(["k-then"], action="a1").status == "completed"


def test_an_unknown_action_is_refused_rather_than_ignored(tmp_path):
    b = bench(tmp_path)
    b.task()
    b.produce()
    b.check("k1", bind_state=False)
    assert b.refusal(["k1"], action="a-never-happened") == (
        "effect_action_unknown:a-never-happened",
    )


# ---------------- what the record supports afterwards ----------------


def test_the_basis_is_reconstructable_from_the_reopened_record(tmp_path):
    b = bench(tmp_path)
    b.task()
    b.produce()
    b.action("a1", Writer(b.target, "after\n"))
    b.check("k1")
    b.accept(["k1"], action="a1")
    events = b.ledger.read_all()
    b.close()

    reopened = SQLiteLedger(tmp_path / "run" / "ledger.sqlite")
    assert reopened.read_all() == events
    [accepted] = reopened.events_by_kind((TASK_ACCEPTED,))
    effect = accepted.payload["acceptance_basis"]["effect"]
    # Everything a second process needs to re-derive the link, without guessing.
    completion = next(
        e for e in reopened.events_by_kind(("action.completed",))
        if e.event_id == effect["action_completed_event_id"]
    )
    assert completion.payload["observed_state_hash"] == effect["observed_state_hash"]
    checks = {e.stream_id: e.payload for e in reopened.events_by_kind(("check.completed",))}
    for check_id in effect["checks_examining_that_state"]:
        assert checks[check_id]["observed_target_state_hash"] == effect["observed_state_hash"]
        assert checks[check_id]["verdict"] == "PASS"
    reopened.close()


def test_the_effect_basis_changes_what_the_acceptance_is(tmp_path):
    """Two acceptances of the same artifact are not the same decision."""
    b = bench(tmp_path)
    b.task()
    b.produce()
    b.action("a1", Writer(b.target, "after\n"))
    b.check("k1")
    plain = b.request(["k1"])
    with_effect = b.request(["k1"], action="a1")
    assert plain.identity() != with_effect.identity()
