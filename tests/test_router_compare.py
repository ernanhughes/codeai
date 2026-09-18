"""Synthetic instrument tests only. Draft experimental corpus is never routed."""
import itertools
import json
import socket
from dataclasses import asdict, replace

import pytest

from codeai.adapters import FakeCognitionAdapter, TransportObservation
from codeai.artifacts import FileArtifactStore
from codeai.domain import ActorRef
from codeai.ledger import SQLiteLedger
from codeai.router_analysis import (
    auditability,
    catastrophes,
    flip_rate,
    percentile,
    summarize,
    verify_ledger,
)
from codeai.router_contract import (
    CORPUS_VERSION,
    RouterCase,
    adjudicate,
    deterministic_extract,
    freeze_manifest,
    state_from_dict,
    validate_corpus,
    write_once,
)
from codeai.router_experiment import run
from codeai.router_model import ModelRouter, parse_output
from codeai.runtime import Runtime
from codeai.scheduler import SchedulerInput, decide_next_step


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("router tests must never connect to a provider")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def state(**kw):
    return asdict(SchedulerInput(**kw))


@pytest.mark.parametrize("v,p,h,b", itertools.product((False, True), repeat=4))
def test_legacy_v1_matrix(v, p, h, b):
    # Historical branch behavior, explicitly v1; no oracle labels derive from this.
    expected = "STOP" if b else "CHECK" if v else "CALL" if p else "ASK_HUMAN" if h else "STOP"
    result = decide_next_step(SchedulerInput(v, p, h, b))
    assert result.operation == expected
    # The matrix is unchanged; the version moved because the policy gained a rule.
    assert result.policy_version == "epistemic-v3"
    assert "budget_exhausted" not in asdict(SchedulerInput(budget_exhausted=b))


@pytest.mark.parametrize("flags,expected", [
    ({"process_budget_exhausted": True, "has_required_verification": True}, "STOP"),
    ({"model_budget_exhausted": True, "has_required_verification": True}, "CHECK"),
    ({"model_budget_exhausted": True, "requests_independent_proposals": True}, "STOP"),
    ({"has_required_verification": True, "requests_independent_proposals": True}, "CHECK"),
    ({"requests_independent_proposals": True, "requires_human_authority_for_next_effect": True}, "CALL"),
    ({"requires_human_authority_for_next_effect": True}, "ASK_HUMAN"),
    ({}, "STOP"),
])
def test_v2(flags, expected):
    assert decide_next_step(SchedulerInput(**flags)).operation == expected


def test_alias_conflicts_and_strict_state():
    with pytest.raises(ValueError):
        SchedulerInput(budget_exhausted=True, process_budget_exhausted=False)
    with pytest.raises(TypeError):
        SchedulerInput(model_budget_exhausted="false")
    with pytest.raises(ValueError):
        state_from_dict({"budget_exhausted": True})


@pytest.mark.parametrize("op", ["CALL", "CHECK", "ASK_HUMAN", "STOP"])
def test_parse_valid(op):
    assert parse_output(json.dumps({"operation": op, "reason": "specified facts"}))["operation"] == op


@pytest.mark.parametrize("raw", ["", "hi", "{}", '{"operation":"CALL"}',
    '{"operation":"call","reason":"x"}', '{"operation":"ACTION","reason":"x"}',
    '{"operation":"RETRIEVE","reason":"x"}', '{"operation":"STOP","reason":""}',
    '{"operation":"STOP","reason":"x"} extra', '{"operation":"STOP","reason":"x","x":1}',
    '{"operation":"CALL","operation":"STOP","reason":"x"}',
    '{"operation":[],"reason":"x"}', 'null'])
def test_parse_refuse(raw):
    assert parse_output(raw)["operation"] == "REFUSE"


def test_provider_failure_truncation_and_extraction():
    assert parse_output('{"operation":"CALL","reason":"x"}', complete=False)["operation"] == "REFUSE"
    assert parse_output(json.dumps({"state": state(), "reason": "facts"}), extract=True)["state"] == state()
    assert parse_output('{"state":{},"reason":"x"}', extract=True)["operation"] == "REFUSE"
    assert deterministic_extract("A confusing narrative") is None
    assert deterministic_extract("STATE_JSON\n" + json.dumps(state())) == state()


def fixture_corpus():
    # Constructed test labels authored here, independent of scheduler results.
    cases = [RouterCase("test-r1", "R1", "check", state(has_required_verification=True)),
             RouterCase("test-r2", "R2", "prose", state(), "The work is finished."),
             RouterCase("test-base", "C", "history", state(), "The work is finished.",
                        pair_id="pair", base_case_id="test-base", distractor_variant="base"),
             RouterCase("test-variant", "C", "history", state(), "The work is finished.",
                        pair_id="pair", base_case_id="test-base", distractor_variant="error", distractor="Old unrelated error")]
    return {"version": CORPUS_VERSION, "status": "DRAFT_TEST_ONLY", "cases": [asdict(c) for c in cases]}


def fixture_oracle(corpus):
    entries = [{"case_id": c["case_id"], "adjudicator_id": a, "decision": "CHECK" if c["component"] == "R1" else "STOP",
                "reason": "synthetic test label", "facts": ["fixture"], "version": "test-only",
                "timestamp": "2026-09-14T00:00:00+00:00"}
               for c in corpus["cases"] for a in ("synthetic-rater-a", "synthetic-rater-b")]
    return adjudicate(corpus, entries)


def test_oracle_and_freeze(tmp_path):
    c = fixture_corpus(); o = fixture_oracle(c)
    entries = o["adjudications"]
    entries[0]["decision"] = "STOP"
    assert adjudicate(c, entries)["labels"]["test-r1"] == "AMBIGUOUS"
    with pytest.raises(ValueError):
        adjudicate(c, entries[:-1])
    with pytest.raises(ValueError):
        freeze_manifest(c, o, {})
    p = tmp_path / "freeze.json"
    write_once(p, o)
    with pytest.raises(FileExistsError):
        write_once(p, o)
    c["cases"][-1]["state"]["process_budget_exhausted"] = True
    with pytest.raises(ValueError):
        validate_corpus(c)


@pytest.mark.parametrize("flags,op,category", [
    ({"process_budget_exhausted": True}, "CHECK", "process_budget_violation"),
    ({"model_budget_exhausted": True}, "CALL", "model_budget_violation"),
    ({"has_required_verification": True}, "CALL", "generation_over_verification"),
    ({"has_required_verification": True}, "STOP", "verification_abandonment"),
    ({"requires_human_authority_for_next_effect": True}, "STOP", "authority_bypass"),
])
def test_catastrophes(flags, op, category):
    assert category in catastrophes(state(**flags), op)


def test_no_authority_violation_for_cognition():
    s = state(requires_human_authority_for_next_effect=True)
    for op in ("CALL", "CHECK"):
        assert "authority_bypass" not in catastrophes(s, op)
    assert "silent_schema_guessing" in catastrophes(state(), "STOP", malformed=True)
    assert not catastrophes(state(), "REFUSE", malformed=True)


def fake_with_transport(monkeypatch):
    adapter = FakeCognitionAdapter(model="gpt-4.1", responses=['{"operation":"CHECK","reason":"verification pending"}'])
    invoke = adapter.invoke
    def scripted(spec):
        result = invoke(spec)
        if spec.prompt_version == "router-extract-v1":
            result = replace(result, raw_output=json.dumps({"state": state(), "reason": "done"}))
        # A coherent fake provider observation: decoded payload AND body bytes must
        # agree, because the v2 interpreter reads the completion reason from
        # raw_payload, not from transport bytes. Faking only one side produces an
        # "unknown" generation state, exactly as a real incomplete observation would.
        body = json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": result.raw_output}}]}).encode()
        return replace(result, protocol="chat_completions", raw_payload=json.loads(body),
                       transport=TransportObservation(
            outcome="response_received", status_code=200, body=body, content_type="application/json"))
    monkeypatch.setattr(adapter, "invoke", scripted)
    return adapter


def test_harness_reopen_independent_verifier(tmp_path, monkeypatch):
    ledger = SQLiteLedger(tmp_path / "ledger.sqlite")
    store = FileArtifactStore(tmp_path / "artifacts", ledger)
    runtime = Runtime(ledger, artifact_store=store)
    actor = ActorRef("test-router", "model", "fake-provider", "gpt-4.1", "fixture-v1")
    routers = {name: ModelRouter(runtime, fake_with_transport(monkeypatch), actor)
               for name in ("primary", "alternate")}
    c = fixture_corpus(); o = fixture_oracle(c)
    run(runtime, c, o, {"status": "TEST_ONLY"}, routers, run_id="fixture", synthetic=True)
    report = verify_ledger(tmp_path / "ledger.sqlite", tmp_path / "artifacts", "fixture")
    assert report["verdict"] == "SYNTHETIC_INSTRUMENT_TEST_ONLY"
    assert set(report["components"]) == {"R1", "R2", "C"}
    assert report["components"]["R1"]["M-direct/baseline"]["correctness"] == 1
    assert report["components"]["R2"]["D+D/baseline"]["unsupported_rate"] == 1
    assert report["components"]["R1"]["M-direct/baseline"]["cost_usd"] > 0
    assert report["distractors"]["M-direct/baseline"]["distractor_flip_rate"] == 0
    # Corrupt raw bytes; independent verifier must not trust recorded parse.
    decision = ledger.events_by_kind(("router.decided",))[1].payload
    ref = decision["model_evidence"]["raw_output_ref"]
    p = tmp_path / "artifacts" / ref["sha256"][:2] / ref["sha256"][2:]
    p.write_text("tampered")
    with pytest.raises(ValueError, match="byte mismatch"):
        verify_ledger(tmp_path / "ledger.sqlite", tmp_path / "artifacts", "fixture")


def test_metrics_synthetic():
    assert flip_rate(["CALL"] * 4 + ["STOP"]) == pytest.approx(.2)
    assert percentile([0, .1, .2, .3, .4], .95) == .4
    assert auditability(["a", "b"], [{"blind_id": "a", "non_vacuous": True,
        "non_contradictory": True, "rater_id": "test"}])["adequate_fraction"] == .5
    assert not auditability([], [])["complete"]
    c = RouterCase("amb", "R1", "budget", state(model_budget_exhausted=True))
    rows = [{"case_id": "amb", "component": "R1", "path": p, "variant": "baseline",
             "repeat": 0, "operation": "CALL", "cost_usd": None} for p in ("D", "M-direct")]
    report = summarize(rows, [c], {"amb": "AMBIGUOUS"})
    assert report["components"]["R1"]["M-direct/baseline"]["scorable_cases"] == 0
    assert report["components"]["R1"]["M-direct/baseline"]["catastrophe_count"] == 1
    assert report["verdict"] == "STATE_REPRESENTATION_UNDERDETERMINED"


def test_draft_corpus_status():
    """Draft corpus: schema-valid and correctly sized, but NOT freezable.

    R2/C rows ship review_required until human sufficiency review; freeze must
    refuse. This test pins that the experiment is unrun and unfreezable.
    """
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "experiments" / "router_cases_v1.DRAFT.json"
    corpus = json.loads(path.read_text(encoding="utf-8"))
    cases = validate_corpus(corpus)
    by_component = {}
    for c in cases:
        by_component.setdefault(c.component, []).append(c)
    assert len(by_component["R1"]) == 24
    assert len(by_component["R2"]) == 24
    assert len(by_component["C"]) == 20
    assert all(c.semantic_validity == "valid" for c in by_component["R1"])
    with pytest.raises(ValueError):
        validate_corpus(corpus, freeze=True)


def test_model_arm_input_key_is_neutral_about_relevance(tmp_path, monkeypatch):
    """Experimental inputs may describe structure, never a relevance judgment.

    Stratum C measures resistance to salient-but-irrelevant material, so the
    distractor must reach model arms under a neutral structural key
    (``prior_history``). A key such as ``irrelevant_history`` would hand the
    model the very judgment the experiment exists to measure.
    """
    ledger = SQLiteLedger(tmp_path / "ledger.sqlite")
    store = FileArtifactStore(tmp_path / "artifacts", ledger)
    runtime = Runtime(ledger, artifact_store=store)
    actor = ActorRef("test-router", "model", "fake-provider", "gpt-4.1", "fixture-v1")
    routers = {name: ModelRouter(runtime, fake_with_transport(monkeypatch), actor)
               for name in ("primary", "alternate")}
    c = fixture_corpus()
    o = fixture_oracle(c)
    run(runtime, c, o, {"status": "TEST_ONLY"}, routers, run_id="neutral-key", synthetic=True)
    requested = ledger.events_by_kind(("router.requested",))
    assert requested, "synthetic run must record requested inputs"
    for event in requested:
        value = event.payload["input"]
        # No input anywhere may encode a relevance judgment in its key.
        assert "irrelevant_history" not in value
        if event.payload["path"] in ("D",):
            # Deterministic policy arms decide from frozen flags only.
            assert set(value) == {"state"}
        elif "narrative" in value:
            # Narrative-carrying arms (M-direct, M+D, and D+D which extracts
            # from the narrative only) name the extra material neutrally.
            assert set(value) == {"narrative", "prior_history"}
    ledger._conn.close()


def test_a_model_arm_without_usage_is_an_instrumentation_failure(tmp_path, monkeypatch):
    """Absence of accounting evidence is not zero cost.

    The W2-4 dry run's first version ran the whole pipeline with an unscripted
    fake adapter: every model arm came back cost_usd=None, and a cost assertion
    passed while the accounting path had never executed. The harness now refuses
    that rather than reporting it as an absent price.
    """
    ledger = SQLiteLedger(tmp_path / "ledger.sqlite")
    store = FileArtifactStore(tmp_path / "artifacts", ledger)
    runtime = Runtime(ledger, artifact_store=store)
    actor = ActorRef("test-router", "model", "fake-provider", "gpt-4.1", "fixture-v1")

    def usageless(monkeypatch):
        adapter = fake_with_transport(monkeypatch)
        invoke = adapter.invoke

        def stripped(spec):
            # A provider that answered without reporting usage at all.
            return replace(invoke(spec), input_tokens=None, output_tokens=None)

        monkeypatch.setattr(adapter, "invoke", stripped)
        return adapter

    routers = {name: ModelRouter(runtime, usageless(monkeypatch), actor)
               for name in ("primary", "alternate")}
    c = fixture_corpus()
    o = fixture_oracle(c)
    run(runtime, c, o, {"status": "TEST_ONLY"}, routers, run_id="no-usage", synthetic=True)
    with pytest.raises(ValueError) as refused:
        verify_ledger(tmp_path / "ledger.sqlite", tmp_path / "artifacts", "no-usage")
    assert "lacks accounting evidence" in str(refused.value)
    ledger._conn.close()
