"""Stage 18: claims, evidence and decisions.

Offline. A synthetic provider returns a review whose sentences become claims.
The review, the configuration files and the check verdicts are fixtures, not
model evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeai.adapters import CallSpec, CheckRequest, CheckResult, CheckVerdict
from codeai.artifacts import FileArtifactStore
from codeai.context import ContextCompiler
from codeai.domain import ActorRef, Authority, Budget, Task
from codeai.evidence import (
    CHECK,
    REFUTES,
    SOURCE_PASSAGE,
    SUPPORTS,
    ClaimExtraction,
    ClaimRefused,
    DecisionRefused,
    DecisionRequest,
    EvidenceRecord,
    EvidenceRefused,
    text_sha256,
)
from codeai.interpretation import (
    ATTEMPT_POLICY_V1,
    ATTEMPT_POLICY_V2,
    INTERPRETER_V1,
    INTERPRETER_V2,
)
from codeai.ledger import Event, SQLiteLedger
from codeai.providers import HttpResponse, OpenCodeCognitionAdapter
from codeai.runtime import Runtime

TASK = "task-18"
REVIEWER = ActorRef("reviewer", "model", provider="opencode", model="mimo-v2.5")
SECOND = ActorRef("second-reviewer", "model", provider="opencode", model="minimax-m2.7")
TTL = "The cache TTL is set to 60 seconds in config/cache.toml."
RETRY = "The retry test passes."
LATENCY = "p99 latency stays under 200 ms."
REVIEW = f"{TTL} {RETRY} {LATENCY}"
CONFIG = "[cache]\nttl_seconds = 60\nmax_entries = 10000\n"
DEPLOYED = "[cache]\nttl_seconds = 600\nmax_entries = 10000\n"


class Provider:
    def __init__(self, text: str = REVIEW, finish: str = "stop") -> None:
        self.requests = 0
        self.text = text
        self.finish = finish

    def __call__(self, url, body, headers, timeout):
        self.requests += 1
        payload = {"id": "synthetic-18", "model": "mimo-v2.5",
                   "choices": [{"finish_reason": self.finish,
                                "message": {"role": "assistant", "content": self.text}}]}
        return HttpResponse(200, {}, json.dumps(payload).encode(), "application/json")


class Verdict:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict

    def run(self, request):
        return CheckResult(
            check_id=request.check_id,
            verdict=self.verdict,
            inconclusive_reason=(
                "the fixture cannot disambiguate this criterion"
                if self.verdict == CheckVerdict.INCONCLUSIVE
                else None
            ),
        )


def make_runtime(path: Path) -> Runtime:
    path.mkdir(parents=True, exist_ok=True)
    ledger = SQLiteLedger(path / "ledger.sqlite")
    runtime = Runtime(ledger, artifact_store=FileArtifactStore(path / "artifacts", ledger))
    if not ledger.events_by_kind(("task.created",)):
        runtime.create_task(Task(TASK, "run-18", "Review the cache change", ("names its evidence",),
                                 Budget(), Authority()))
    return runtime


def review(runtime, provider, *, call_id="call-18", actor=REVIEWER,
           interpreter=INTERPRETER_V2, policy=ATTEMPT_POLICY_V2):
    package = ContextCompiler().compile(task_id=TASK, actor=actor, prompt="Review the change.")
    spec = CallSpec(call_id=call_id, task_id=TASK, actor=actor, context=package,
                    idempotency_key=f"key-{call_id}", chamber="deep-review",
                    parameters={"max_tokens": 128})
    adapter = OpenCodeCognitionAdapter(model=actor.model, protocol="chat_completions",
                                       gateway_plan="go", api_key="offline-decoy",
                                       http_post=provider, timeout=5)
    return runtime.invoke_recorded_call(spec, adapter=adapter, interpreter_version=interpreter,
                                        policy_version=policy)


def extraction(recorded, claim_id, sentence, *, quote=None, start=None, text=REVIEW):
    start = text.index(sentence) if start is None else start
    return ClaimExtraction(claim_id, TASK, recorded.call_id, recorded.attempts[-1].attempt_id,
                           start, start + len(sentence), sentence if quote is None else quote,
                           sentence, "claim-extractor")


def passage(claim_id, evidence_id, source, text, *, verdict=SUPPORTS, actor="human-reviewer",
            artifact_id=None, source_text=None):
    start = (source_text or text).index(text) if source_text else None
    return EvidenceRecord(evidence_id, claim_id, SOURCE_PASSAGE, verdict, actor,
                          source_artifact_id=artifact_id or source.artifact_id,
                          passage_start=start, passage_end=start + len(text), passage=text)


def source_evidence(runtime, claim_id, evidence_id, document, text, **kwargs):
    source = runtime.artifact_store.store_text(document, artifact_type="source_document")
    return runtime.record_claim_evidence(
        passage(claim_id, evidence_id, source, text, source_text=document, **kwargs))


def check_evidence(runtime, claim_id, evidence_id, check_id, verdict, *, targets=None,
                   claimed=SUPPORTS):
    runtime.run_check(CheckRequest(check_id=check_id, task_id=TASK,
                                   claim_ids=tuple(targets or (claim_id,))),
                      verifier=Verdict(verdict))
    return runtime.record_claim_evidence(
        EvidenceRecord(evidence_id, claim_id, CHECK, claimed, "verification-reviewer",
                       check_id=check_id))


def supported_review(tmp_path):
    runtime, provider = make_runtime(tmp_path), Provider()
    recorded = review(runtime, provider)
    for claim_id, sentence in (("c-ttl", TTL), ("c-retry", RETRY), ("c-latency", LATENCY)):
        runtime.extract_claim(extraction(recorded, claim_id, sentence))
    source_evidence(runtime, "c-ttl", "ev-config", CONFIG, "ttl_seconds = 60")
    check_evidence(runtime, "c-retry", "ev-retry", "check-retry", CheckVerdict.PASS)
    return runtime, provider, recorded


MERGE = DecisionRequest("dec-merge", TASK, "release-manager", "Merge the cache change",
                        ("c-ttl", "c-retry"), ("c-latency",))


def kinds(runtime):
    return [e.kind for e in runtime.ledger.read_all()]


# ---------------- attribution ----------------


def test_an_extracted_claim_is_attributed_not_supported(tmp_path):
    runtime, provider = make_runtime(tmp_path), Provider()
    recorded = review(runtime, provider)
    standing = runtime.extract_claim(extraction(recorded, "c-ttl", TTL))
    body = runtime.get_attempt_observation(recorded.attempts[-1].attempt_id)["response_body_artifact"]
    assert (standing.status, standing.evidence_class) == ("unresolved", "E1_ATTRIBUTED")
    assert standing.quote == TTL and standing.quote_sha256 == text_sha256(TTL)
    assert standing.observation_sha256 == body["sha256"]
    assert standing.source_call_status == "succeeded" and provider.requests == 1
    again = runtime.extract_claim(extraction(recorded, "c-ttl", TTL))
    assert again == standing and kinds(runtime).count("claim.extracted") == 1


def test_a_quote_the_source_never_said_is_refused_and_the_refusal_recorded(tmp_path):
    runtime, provider = make_runtime(tmp_path), Provider()
    recorded = review(runtime, provider)
    invented = TTL.replace("60", "600").replace(".", "")  # same length as the real span minus one
    with pytest.raises(ClaimRefused) as refused:
        runtime.extract_claim(extraction(recorded, "c-ttl", TTL, quote=invented))
    assert refused.value.reasons == ("quote_not_in_source",)
    with pytest.raises(ClaimRefused) as beyond:
        runtime.extract_claim(extraction(recorded, "c-far", TTL, start=len(REVIEW)))
    assert beyond.value.reasons == ("span_out_of_range",)
    assert "claim.extracted" not in kinds(runtime)
    assert kinds(runtime)[-2:] == ["claim.refused", "claim.refused"]


@pytest.mark.parametrize("damage", ["delete", "corrupt"])
def test_a_claim_cannot_cite_bytes_that_cannot_be_read(tmp_path, damage):
    runtime, provider = make_runtime(tmp_path), Provider()
    recorded = review(runtime, provider)
    ref = runtime.get_attempt_observation(recorded.attempts[-1].attempt_id)["response_body_artifact"]
    path = Path(runtime.ledger.read_artifact(ref["artifact_id"]).uri)
    if damage == "delete":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes().replace(b"TTL", b"ttl"))
    with pytest.raises(ClaimRefused) as refused:
        runtime.extract_claim(extraction(recorded, "c-ttl", TTL))
    assert refused.value.reasons == ("observation_unavailable",)


def test_asserted_promotions_are_ignored(tmp_path):
    runtime, provider = make_runtime(tmp_path), Provider()
    recorded = review(runtime, provider)
    runtime.extract_claim(extraction(recorded, "c-ttl", TTL))
    for kind, payload in (("claim.evidence", {"evidence_class": "E4_ROBUST", "status": "supported"}),
                          ("claim.status", {"status": "supported"})):
        runtime.ledger.append(Event.create(stream_id="c-ttl", kind=kind, actor_id="reviewer",
                                           payload={"claim_id": "c-ttl", **payload}))
    standing = runtime.claim_standing("c-ttl")
    assert (standing.status, standing.evidence_class) == ("unresolved", "E1_ATTRIBUTED")
    assert len(standing.ignored_legacy_event_ids) == 2


# ---------------- evidence ----------------


def test_a_source_passage_supports_at_source_checked(tmp_path):
    runtime, provider = make_runtime(tmp_path), Provider()
    recorded = review(runtime, provider)
    runtime.extract_claim(extraction(recorded, "c-ttl", TTL))
    standing = source_evidence(runtime, "c-ttl", "ev-config", CONFIG, "ttl_seconds = 60")
    assert (standing.status, standing.evidence_class) == ("supported", "E2_SOURCE_CHECKED")
    assert standing.supporting_evidence_ids == ("ev-config",)


def test_passage_evidence_is_refused_when_absent_self_authored_or_circular(tmp_path):
    runtime, provider = make_runtime(tmp_path), Provider()
    recorded = review(runtime, provider)
    runtime.extract_claim(extraction(recorded, "c-ttl", TTL))
    source = runtime.artifact_store.store_text(CONFIG, artifact_type="source_document")
    absent = EvidenceRecord("ev-absent", "c-ttl", SOURCE_PASSAGE, SUPPORTS, "human-reviewer",
                            source_artifact_id=source.artifact_id, passage_start=8,
                            passage_end=25, passage="ttl_seconds = 600")
    with pytest.raises(EvidenceRefused) as refused:
        runtime.record_claim_evidence(absent)
    assert refused.value.reasons == ("passage_not_in_source",)
    with pytest.raises(EvidenceRefused) as self_made:
        source_evidence(runtime, "c-ttl", "ev-self", CONFIG, "ttl_seconds = 60", actor="reviewer")
    assert self_made.value.reasons == ("self_evidence",)
    body = runtime.get_attempt_observation(recorded.attempts[-1].attempt_id)["response_body_artifact"]
    body_text = runtime.artifact_store.read_text(body["artifact_id"])
    start = body_text.index(TTL)
    with pytest.raises(EvidenceRefused) as circular:
        runtime.record_claim_evidence(EvidenceRecord(
            "ev-circular", "c-ttl", SOURCE_PASSAGE, SUPPORTS, "human-reviewer",
            source_artifact_id=body["artifact_id"], passage_start=start,
            passage_end=start + len(TTL), passage=TTL))
    assert circular.value.reasons == ("source_is_claim_origin",)
    assert runtime.claim_standing("c-ttl").status == "unresolved"


def test_only_a_completed_check_that_targeted_the_claim_reproduces_it(tmp_path):
    runtime, provider = make_runtime(tmp_path), Provider()
    recorded = review(runtime, provider)
    runtime.extract_claim(extraction(recorded, "c-retry", RETRY))
    runtime.extract_claim(extraction(recorded, "c-ttl", TTL))
    with pytest.raises(EvidenceRefused) as untargeted:
        check_evidence(runtime, "c-retry", "ev-other", "check-other", CheckVerdict.PASS,
                       targets=("c-ttl",))
    assert untargeted.value.reasons == ("check_not_targeting_claim",)
    with pytest.raises(EvidenceRefused) as inconclusive:
        check_evidence(runtime, "c-retry", "ev-maybe", "check-maybe", CheckVerdict.INCONCLUSIVE)
    assert inconclusive.value.reasons == ("check_inconclusive",)
    standing = check_evidence(runtime, "c-retry", "ev-retry", "check-retry", CheckVerdict.PASS)
    assert (standing.status, standing.evidence_class) == ("supported", "E3_REPRODUCED")
    with pytest.raises(EvidenceRefused) as contradicted:
        runtime.record_claim_evidence(EvidenceRecord("ev-spin", "c-retry", CHECK, REFUTES,
                                                     "verification-reviewer",
                                                     check_id="check-retry"))
    assert contradicted.value.reasons == ("verdict_contradicts_check",)


def test_support_and_refutation_together_are_contested(tmp_path):
    runtime, _provider, _recorded = supported_review(tmp_path)
    standing = source_evidence(runtime, "c-ttl", "ev-deployed", DEPLOYED, "ttl_seconds = 600",
                               verdict=REFUTES, actor="on-call-engineer")
    assert (standing.status, standing.evidence_class) == ("contested", "E2_SOURCE_CHECKED")
    assert standing.refuting_evidence_ids == ("ev-deployed",)


def test_agreement_between_models_is_not_evidence(tmp_path):
    runtime, provider = make_runtime(tmp_path), Provider()
    first = review(runtime, provider)
    second = review(runtime, provider, call_id="call-18b", actor=SECOND)
    a = runtime.extract_claim(extraction(first, "c-ttl-a", TTL))
    b = runtime.extract_claim(extraction(second, "c-ttl-b", TTL))
    assert {(a.status, a.evidence_class), (b.status, b.evidence_class)} == {
        ("unresolved", "E1_ATTRIBUTED")}
    with pytest.raises(DecisionRefused) as refused:
        runtime.record_decision(DecisionRequest("dec-agree", TASK, "release-manager",
                                                "Merge because two models agree",
                                                ("c-ttl-a", "c-ttl-b")))
    assert set(refused.value.reasons) == {
        "claim_not_supported:c-ttl-a", "evidence_below_policy:c-ttl-a",
        "claim_not_supported:c-ttl-b", "evidence_below_policy:c-ttl-b"}


# ---------------- decisions ----------------


def test_a_decision_cannot_rely_on_an_open_claim(tmp_path):
    runtime, _provider, _recorded = supported_review(tmp_path)
    request = DecisionRequest("dec-fast", TASK, "release-manager", "Ship on the latency claim",
                              ("c-ttl", "c-latency"))
    with pytest.raises(DecisionRefused) as refused:
        runtime.record_decision(request)
    assert set(refused.value.reasons) == {"claim_not_supported:c-latency",
                                          "evidence_below_policy:c-latency"}
    assert kinds(runtime)[-1] == "decision.refused" and "decision.recorded" not in kinds(runtime)


def test_a_decision_records_its_basis_and_repeats_idempotently(tmp_path):
    runtime, _provider, _recorded = supported_review(tmp_path)
    standing = runtime.record_decision(MERGE)
    assert standing.standing == "basis_intact" and standing.changes == ()
    [event] = runtime.ledger.events_by_kind(("decision.recorded",))
    basis = {item["claim_id"]: item for item in event.payload["basis"]}
    assert (basis["c-ttl"]["status"], basis["c-ttl"]["evidence_class"]) == (
        "supported", "E2_SOURCE_CHECKED")
    assert basis["c-retry"]["evidence_class"] == "E3_REPRODUCED"
    assert event.payload["acknowledged_unresolved"] == [
        {"claim_id": "c-latency", "status": "unresolved", "evidence_class": "E1_ATTRIBUTED",
         "source_call_status": "succeeded"}]
    count = len(runtime.ledger.read_all())
    assert runtime.record_decision(MERGE) == standing and len(runtime.ledger.read_all()) == count
    other = DecisionRequest("dec-merge", TASK, "release-manager", "Merge something else",
                            ("c-ttl",))
    with pytest.raises(DecisionRefused) as conflicting:
        runtime.record_decision(other)
    assert conflicting.value.reasons == ("conflicting_decision",)


def test_new_refuting_evidence_changes_the_standing_not_the_record(tmp_path):
    runtime, _provider, _recorded = supported_review(tmp_path)
    runtime.record_decision(MERGE)
    [before] = runtime.ledger.events_by_kind(("decision.recorded",))

    source_evidence(runtime, "c-ttl", "ev-deployed", DEPLOYED, "ttl_seconds = 600",
                    verdict=REFUTES, actor="on-call-engineer")

    standing = runtime.decision_standing("dec-merge")
    assert standing.standing == "basis_changed"
    assert {(c["claim_id"], c["field"]) for c in standing.changes} == {
        ("c-ttl", "status"), ("c-ttl", "refuting_evidence_ids")}
    status = next(c for c in standing.changes if c["field"] == "status")
    assert (status["recorded"], status["current"]) == ("supported", "contested")
    [after] = runtime.ledger.events_by_kind(("decision.recorded",))
    assert after == before
    assert [d.decision_id for d in runtime.decisions_resting_on("c-ttl")] == ["dec-merge"]
    assert runtime.decisions_resting_on("c-latency") == ()


def test_reinterpreting_the_source_call_changes_the_decisions_standing(tmp_path):
    runtime, provider = make_runtime(tmp_path), Provider(finish="length")
    recorded = review(runtime, provider, interpreter=INTERPRETER_V1, policy=ATTEMPT_POLICY_V1)
    runtime.extract_claim(extraction(recorded, "c-ttl", TTL))
    runtime.extract_claim(extraction(recorded, "c-retry", RETRY))
    source_evidence(runtime, "c-ttl", "ev-config", CONFIG, "ttl_seconds = 60")
    check_evidence(runtime, "c-retry", "ev-retry", "check-retry", CheckVerdict.PASS)
    request = DecisionRequest("dec-merge", TASK, "release-manager", "Merge the cache change",
                              ("c-ttl", "c-retry"))
    assert runtime.record_decision(request).standing == "basis_intact"

    runtime.reinterpret_call("call-18", interpreter_version=INTERPRETER_V2,
                             policy_version=ATTEMPT_POLICY_V2)

    standing = runtime.decision_standing("dec-merge")
    assert standing.standing == "basis_changed"
    source = [c for c in standing.changes if c["field"] == "source_call_status"]
    assert {(c["claim_id"], c["recorded"], c["current"]) for c in source} == {
        ("c-ttl", "succeeded", "unresolved"), ("c-retry", "succeeded", "unresolved")}
    again = DecisionRequest("dec-merge-2", TASK, "release-manager", "Merge the cache change",
                            ("c-ttl", "c-retry"))
    with pytest.raises(DecisionRefused) as refused:
        runtime.record_decision(again)
    assert set(refused.value.reasons) == {"source_call_not_succeeded:c-ttl",
                                          "source_call_not_succeeded:c-retry"}
    assert provider.requests == 1
