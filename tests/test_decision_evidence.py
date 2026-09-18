"""W2-3: if you cite a decision as justification, the justification must still stand.

Three different questions, answered by three different mechanisms:

    process freshness   is this still the state the scheduler decided against?   W1-R4
    authority           may this effect happen at all?                           seam 2
    decision evidence   do the claims this decision relied on still stand?       here

Evidence can make an action unjustified while authority permits it; authority can
forbid one whose evidence is impeccable. Chapter 18 owns the first, Chapter 20
the second, and this seam is why they stay apart.

Defeat is narrower than change (experiments/W2-3-prereg.md). A decision that has
*gained* supporting evidence has changed and has not been defeated, and attempts
are not standing: an INCONCLUSIVE or ERROR verification revokes nothing.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

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
from codeai.evidence import (
    CHECK,
    REFUTES,
    SOURCE_PASSAGE,
    SUPPORTS,
    ClaimExtraction,
    DecisionRequest,
    EvidenceRecord,
)
from codeai.ledger import SQLiteLedger
from codeai.providers import HttpResponse, OpenCodeCognitionAdapter
from codeai.runtime import Runtime

SOURCE = "The cache ttl is 60 seconds and the retry budget is three attempts."
OUTPUT = "The ttl is 60 seconds. Retries are capped at three."
TTL = "The ttl is 60 seconds."
RETRY = "Retries are capped at three."


class Worker:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ActionRequest) -> ActionResult:
        self.calls += 1
        return ActionResult(action_id=request.action_id, status=ActionStatus.SUCCEEDED)


class Verdict:
    def __init__(self, verdict) -> None:
        self.verdict = verdict

    def run(self, request: CheckRequest) -> CheckResult:
        return CheckResult(
            check_id=request.check_id, verdict=self.verdict,
            inconclusive_reason=(
                "the fixture cannot settle it" if self.verdict == CheckVerdict.INCONCLUSIVE
                else None
            ),
        )


class Bench:
    """A claim with real evidence, a decision resting on it, and an action citing it."""

    def __init__(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        ledger = SQLiteLedger(path / "ledger.sqlite")
        self.ledger = ledger
        self.runtime = Runtime(ledger, artifact_store=FileArtifactStore(path / "artifacts", ledger))
        self.runtime.open_directive(
            Directive(directive_id="d", objective="fixture", success_criteria=(),
                      budget=Budget(), authority=Authority(frozenset({Capability.WRITE})))
        )
        self.runtime.create_task(Task("t1", "d", "fixture", (), Budget(), Authority()))
        self.recorded = self._call()

    def _call(self):
        body = json.dumps(
            {"id": "s", "choices": [{"finish_reason": "stop", "message": {"content": OUTPUT}}]}
        ).encode()

        def post(url, payload, headers, timeout):
            return HttpResponse(200, {"request-id": "s"}, body, "application/json")

        actor = ActorRef("summariser", "model", provider="opencode", model="mimo-v2.5")
        context = ContextCompiler().compile(
            task_id="t1", actor=actor, prompt="Summarise.", prompt_version="v1"
        )
        spec = CallSpec("call-1", "t1", actor, context, str(uuid.uuid4()),
                        chamber="deep-review", parameters={"max_tokens": 64})
        adapter = OpenCodeCognitionAdapter(
            model="mimo-v2.5", protocol="chat_completions", gateway_plan="go",
            api_key="offline-decoy", http_post=post, timeout=60,
        )
        return self.runtime.invoke_recorded_call(spec, adapter=adapter, max_attempts=1)

    def claim(self, claim_id: str, sentence: str):
        attempt = self.recorded.attempts[-1]
        start = OUTPUT.index(sentence)
        self.runtime.extract_claim(
            ClaimExtraction(claim_id, "t1", "call-1", attempt.attempt_id,
                            start, start + len(sentence), sentence, sentence, "extractor")
        )
        return claim_id

    def support(self, claim_id: str, sentence: str, *, evidence_id=None, actor="reviewer"):
        """Support a claim with an exact passage of a preserved source document."""
        source = self.runtime.artifact_store.store_text(SOURCE, artifact_type="source_document")
        start = SOURCE.index("60 seconds") if "ttl" in sentence else SOURCE.index("three attempts")
        text = SOURCE[start:start + 10] if "ttl" in sentence else SOURCE[start:start + 14]
        return self.runtime.record_claim_evidence(
            EvidenceRecord(evidence_id or f"ev-{uuid.uuid4()}", claim_id, SOURCE_PASSAGE,
                           SUPPORTS, actor, source_artifact_id=source.artifact_id,
                           passage_start=start, passage_end=start + len(text), passage=text)
        )

    def refute(self, claim_id: str, *, actor="second-reviewer"):
        source = self.runtime.artifact_store.store_text(
            "Actually the ttl was changed to 300 seconds last week.",
            artifact_type="source_document",
        )
        text = "changed to 300 seconds"
        start = "Actually the ttl was changed to 300 seconds last week.".index(text)
        return self.runtime.record_claim_evidence(
            EvidenceRecord(f"ev-ref-{uuid.uuid4()}", claim_id, SOURCE_PASSAGE, REFUTES, actor,
                           source_artifact_id=source.artifact_id, passage_start=start,
                           passage_end=start + len(text), passage=text)
        )

    def decide(self, decision_id: str, claim_ids, *, actor="decider"):
        return self.runtime.record_decision(
            DecisionRequest(decision_id, "t1", actor, "widen the cache", tuple(claim_ids))
        )

    def act(self, action_id: str, *, decision=None, task_id="t1"):
        worker = Worker()
        result = self.runtime.execute_action(
            ActionRequest(action_id=action_id, task_id=task_id, directive_id="d",
                          capability="write", instruction="widen", precondition_hash=None,
                          idempotency_key=f"k-{action_id}", requested_by="human",
                          actor_id="worker", adapter="fixture", decision_id=decision),
            adapter=worker,
        )
        return result, worker

    def close(self):
        self.ledger.close()


def bench(tmp_path):
    return Bench(tmp_path / "run")


def supported_decision(b: Bench, decision_id="dec-1", claims=("c-ttl",)):
    for claim_id in claims:
        sentence = TTL if claim_id == "c-ttl" else RETRY
        b.claim(claim_id, sentence)
        b.support(claim_id, sentence)
    return b.decide(decision_id, claims)


# ---------------- 1, 2, 3: the core rule ----------------


def test_1_a_supported_basis_permits_the_action(tmp_path):
    b = bench(tmp_path)
    supported_decision(b)
    result, worker = b.act("a1", decision="dec-1")
    assert (result.status, worker.calls) == (ActionStatus.SUCCEEDED, 1)
    assert b.runtime.decision_evidence("dec-1").admissible is True


def test_2_a_refuted_basis_refuses_the_action(tmp_path):
    b = bench(tmp_path)
    supported_decision(b)
    assert b.act("a1", decision="dec-1")[0].status == ActionStatus.SUCCEEDED

    b.refute("c-ttl")
    result, worker = b.act("a2", decision="dec-1")
    assert result.status == ActionStatus.FAILED
    assert worker.calls == 0
    assert "decision_basis_defeated:c-ttl" in (result.error or "")

    standing = b.runtime.decision_evidence("dec-1")
    assert standing.admissible is False
    assert [c.claim_id for c in standing.defeated] == ["c-ttl"]
    assert standing.defeated[0].new_refuting_evidence_ids

    [refusal] = b.ledger.events_by_kind(("action.decision_basis_refused",))
    assert refusal.payload["action_id"] == "a2"
    assert refusal.payload["claims"][0]["verdict"] == "defeated"


def test_3_one_refuted_claim_among_several_is_enough(tmp_path):
    b = bench(tmp_path)
    supported_decision(b, claims=("c-ttl", "c-retry"))
    b.refute("c-retry")
    result, worker = b.act("a1", decision="dec-1")
    assert (result.status, worker.calls) == (ActionStatus.FAILED, 0)
    assert "decision_basis_defeated:c-retry" in (result.error or "")
    standing = b.runtime.decision_evidence("dec-1")
    verdicts = {c.claim_id: c.verdict for c in standing.claims}
    assert verdicts == {"c-ttl": "intact", "c-retry": "defeated"}


# ---------------- 4, 5: attempts are not standing ----------------


def test_4_an_inconclusive_verification_does_not_defeat_a_decision(tmp_path):
    b = bench(tmp_path)
    supported_decision(b)
    b.runtime.run_check(
        CheckRequest("k-maybe", "t1", claim_ids=("c-ttl",)),
        verifier=Verdict(CheckVerdict.INCONCLUSIVE),
    )
    assert b.runtime.inconclusive_checks_for_claim("c-ttl")   # the attempt is on the record
    result, worker = b.act("a1", decision="dec-1")
    assert (result.status, worker.calls) == (ActionStatus.SUCCEEDED, 1)
    assert b.runtime.decision_evidence("dec-1").admissible is True


def test_5_an_errored_verification_revokes_nothing(tmp_path):
    """An infrastructure failure must not be able to withdraw a justification."""
    b = bench(tmp_path)
    supported_decision(b)
    errored = b.runtime.run_check(
        CheckRequest("k-error", "t1", claim_ids=("c-ttl",),
                     target=f"artifact:sha256:{'0' * 64}"),
        verifier=Verdict(CheckVerdict.PASS),
    )
    assert errored.verdict == CheckVerdict.ERROR          # the artifact is not retrievable
    assert b.runtime.verification_attempts_for_claim("c-ttl")[0].verdict == "ERROR"

    result, worker = b.act("a1", decision="dec-1")
    assert (result.status, worker.calls) == (ActionStatus.SUCCEEDED, 1)


def test_added_support_is_not_a_change_that_defeats(tmp_path):
    """The projection that reports *change* would fire here. Defeat does not."""
    b = bench(tmp_path)
    supported_decision(b)
    b.support("c-ttl", TTL, actor="a-third-reviewer")
    assert b.runtime.decision_standing("dec-1").standing == "basis_changed"
    assert b.runtime.decision_evidence("dec-1").admissible is True
    assert b.act("a1", decision="dec-1")[0].status == ActionStatus.SUCCEEDED


# ---------------- 6, 7, 8, 9: the edges ----------------


def test_6_an_unrecorded_decision_is_refused_rather_than_ignored(tmp_path):
    b = bench(tmp_path)
    result, worker = b.act("a1", decision="dec-never-recorded")
    assert (result.status, worker.calls) == (ActionStatus.FAILED, 0)
    assert "decision_unknown:dec-never-recorded" in (result.error or "")


def test_7_a_decision_with_no_claim_basis_cannot_be_recorded_at_all(tmp_path):
    """Registered as "allowed; no evidence relationship claimed" — and it turns
    out the case cannot arise. Chapter 18 already refuses a decision that relies
    on nothing, so there is no such decision for this seam to be lenient about.
    """
    from codeai.evidence import DecisionRefused

    b = bench(tmp_path)
    with pytest.raises(DecisionRefused) as refused:
        b.runtime.record_decision(
            DecisionRequest("dec-empty", "t1", "decider", "proceed on judgment", ())
        )
    assert "no_claims_relied_on" in refused.value.reasons
    assert b.runtime.decision_evidence("dec-empty").found is False


def test_8_an_action_citing_no_decision_is_asked_nothing(tmp_path):
    b = bench(tmp_path)
    supported_decision(b)
    b.refute("c-ttl")          # the decision is defeated...
    result, worker = b.act("a1")   # ...and this action never invoked it
    assert (result.status, worker.calls) == (ActionStatus.SUCCEEDED, 1)
    assert not b.ledger.events_by_kind(("action.decision_basis_refused",))


def test_9_authority_and_evidence_refuse_for_different_reasons(tmp_path):
    b = bench(tmp_path)
    supported_decision(b)
    b.runtime.open_directive(
        Directive(directive_id="d-read", objective="read only", success_criteria=(),
                  budget=Budget(), authority=Authority(frozenset({Capability.READ})))
    )
    b.runtime.create_task(Task("t-read", "d-read", "fixture", (), Budget(), Authority()))
    worker = Worker()
    denied = b.runtime.execute_action(
        ActionRequest(action_id="a-denied", task_id="t-read", directive_id="d-read",
                      capability="write", instruction="widen", precondition_hash=None,
                      idempotency_key="k-denied", requested_by="human", actor_id="worker",
                      adapter="fixture", decision_id="dec-1"),
        adapter=worker,
    )
    # Authority answers first, and its answer is its own.
    assert denied.status == ActionStatus.DENIED
    assert worker.calls == 0
    assert "capability denied" in (denied.error or "")
    assert not b.ledger.events_by_kind(("action.decision_basis_refused",))


# ---------------- 10, 12: the record over time ----------------


def test_10_reopening_gives_the_same_refusal(tmp_path):
    b = bench(tmp_path)
    supported_decision(b)
    b.refute("c-ttl")
    before = b.runtime.decision_evidence("dec-1")
    events = b.ledger.read_all()
    b.close()

    reopened = SQLiteLedger(tmp_path / "run" / "ledger.sqlite")
    second = Runtime(reopened, artifact_store=FileArtifactStore(tmp_path / "run" / "artifacts",
                                                                reopened))
    after = second.decision_evidence("dec-1")
    assert reopened.read_all() == events
    assert (after.admissible, [c.claim_id for c in after.defeated]) == (
        before.admissible, [c.claim_id for c in before.defeated]
    )
    reopened.close()


def test_12_a_refutation_does_not_rewrite_an_action_that_already_ran(tmp_path):
    """New evidence invalidates future reliance. It does not edit history."""
    b = bench(tmp_path)
    supported_decision(b)
    ran, worker = b.act("a-before", decision="dec-1")
    assert (ran.status, worker.calls) == (ActionStatus.SUCCEEDED, 1)
    completed_before = [
        e.payload for e in b.ledger.events_by_kind(("action.completed",))
        if e.payload["action_id"] == "a-before"
    ]

    b.refute("c-ttl")
    assert b.act("a-after", decision="dec-1")[0].status == ActionStatus.FAILED

    completed_after = [
        e.payload for e in b.ledger.events_by_kind(("action.completed",))
        if e.payload["action_id"] == "a-before"
    ]
    assert completed_after == completed_before
    assert b.runtime.action_state("a-before").result_status == "succeeded"


def test_a_defeated_basis_refuses_a_replay_too(tmp_path):
    """Returning a recorded result is re-asserting an operation."""
    b = bench(tmp_path)
    supported_decision(b)
    first, worker = b.act("a1", decision="dec-1")
    assert first.status == ActionStatus.SUCCEEDED

    b.refute("c-ttl")
    replay = b.runtime.execute_action(
        ActionRequest(action_id="a1-again", task_id="t1", directive_id="d", capability="write",
                      instruction="widen", precondition_hash=None, idempotency_key="k-a1",
                      requested_by="human", actor_id="worker", adapter="fixture",
                      decision_id="dec-1"),
        adapter=worker,
    )
    assert replay.status == ActionStatus.FAILED
    assert "decision_basis_defeated" in (replay.error or "")
    assert replay.reused_from_action_id is None, "a refusal disclosed the recorded operation"


def test_re_support_does_not_resurrect_a_defeated_decision(tmp_path):
    """A judgment made against an overturned state is replaced, not revived."""
    b = bench(tmp_path)
    supported_decision(b)
    b.refute("c-ttl")
    b.support("c-ttl", TTL, actor="a-fourth-reviewer")   # better evidence arrives

    standing = b.runtime.decision_evidence("dec-1")
    assert standing.admissible is False
    assert standing.defeated[0].reason.startswith("refuting evidence was recorded after")
    assert b.act("a1", decision="dec-1")[0].status == ActionStatus.FAILED

    # And the obvious repair is not available either, which is worth knowing.
    # A claim carrying both supporting and refuting evidence stands as
    # CONTESTED, and Chapter 18 already refuses to rest a new decision on one.
    # So the path forward is to settle the contest, not to re-decide on top of
    # it -- and nothing in the runtime settles a contest today.
    from codeai.evidence import DecisionRefused

    with pytest.raises(DecisionRefused) as refused:
        b.decide("dec-2", ("c-ttl",), actor="decider")
    assert any("claim_not_supported:c-ttl" in r for r in refused.value.reasons)
    assert b.runtime.claim_standing("c-ttl").status == "contested"
