"""Stage 13: versioned usage semantics.

Expected values below are authored from the dialect documentation, not from
function output. Cases marked synthetic are specifications, not provider
observations. The three live-shaped cases copy the usage members of the
Stage 12 captures exactly.
"""

from __future__ import annotations

import json

import pytest

from codeai.artifacts import FileArtifactStore
from codeai.context import ContextCompiler
from codeai.domain import ActorRef, CallSpec
from codeai.ledger import SQLiteLedger
from codeai.providers import HttpResponse, OpenCodeCognitionAdapter
from codeai.runtime import Runtime
from codeai.usage_semantics import (
    USAGE_SEMANTICS_V1,
    USAGE_SEMANTICS_V2,
    ComponentStatus,
    interpret_usage,
)

REPORTED = ComponentStatus.REPORTED.value
NOT_REPORTED = ComponentStatus.NOT_REPORTED.value
INVALID = ComponentStatus.INVALID.value


def v2(protocol, parsed, **kw):
    return interpret_usage(protocol, parsed, version=USAGE_SEMANTICS_V2, **kw)


def comp(result, name):
    c = result.component(name)
    return (c.value, c.status)


def der(result, name):
    d = result.derived_value(name)
    return (d.value, d.lower_bound)


# --- U01-U13 synthetic specification corpus -------------------------------------


def test_u01_chat_absent_usage_is_not_zero():
    r = v2("chat_completions", {"choices": []})
    assert r.usage_present is False
    assert comp(r, "input") == (None, NOT_REPORTED)
    assert comp(r, "output") == (None, NOT_REPORTED)
    assert der(r, "processed_total") == (None, None)
    assert "usage_not_reported" in r.diagnostics


def test_u02_chat_measured():
    r = v2("chat_completions", {"usage": {"prompt_tokens": 100, "completion_tokens": 20}})
    assert comp(r, "input") == (100, REPORTED) and comp(r, "output") == (20, REPORTED)
    assert der(r, "processed_total") == (120, 120)
    assert der(r, "total_input") == (100, 100)
    assert der(r, "fresh_input") == (None, None)  # cache components not reported
    assert comp(r, "total_reported") == (None, NOT_REPORTED)
    assert r.conflicts == ()


def test_u03_responses_reported_total_kept_separately():
    r = v2("responses", {"usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}})
    assert comp(r, "total_reported") == (120, REPORTED)
    assert der(r, "processed_total") == (120, 120)
    assert r.conflicts == ()


def test_u04_responses_conflicting_total_survives():
    r = v2("responses", {"usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 150}})
    assert comp(r, "total_reported") == (150, REPORTED)
    assert der(r, "processed_total") == (120, 120)
    assert r.conflicts == ("reported_total_differs_from_derived",)


def test_u05_chat_cache_read_is_subset_not_addend():
    r = v2(
        "chat_completions",
        {"usage": {"prompt_tokens": 100, "completion_tokens": 20,
                   "prompt_tokens_details": {"cached_tokens": 80}}},
    )
    assert comp(r, "cache_read") == (80, REPORTED)
    assert r.component("cache_read").relation == "subset_of_input"
    assert der(r, "processed_total") == (120, 120)  # not 200
    assert der(r, "fresh_input") == (None, None)  # cache_write not reported


def test_u06_responses_reasoning_is_subset_of_output():
    r = v2(
        "responses",
        {"usage": {"input_tokens": 100, "output_tokens": 20,
                   "output_tokens_details": {"reasoning_tokens": 5}}},
    )
    assert comp(r, "reasoning") == (5, REPORTED)
    assert der(r, "processed_total") == (120, 120)  # not 125


def test_u07_messages_additive_cache():
    r = v2(
        "messages",
        {"usage": {"input_tokens": 20, "output_tokens": 6,
                   "cache_read_input_tokens": 15, "cache_creation_input_tokens": 2}},
    )
    assert r.component("cache_read").relation == "additive_to_input"
    assert der(r, "total_input") == (37, 37)
    assert der(r, "fresh_input") == (20, 20)
    assert der(r, "processed_total") == (43, 43)


def test_u08_messages_missing_cache_read_is_not_zero():
    r = v2(
        "messages",
        {"usage": {"input_tokens": 20, "output_tokens": 6, "cache_creation_input_tokens": 2}},
    )
    assert comp(r, "cache_read") == (None, NOT_REPORTED)
    assert der(r, "total_input") == (None, 22)
    assert der(r, "processed_total") == (None, 28)
    assert der(r, "fresh_input") == (20, 20)


def test_u09_chat_partial_usage_gives_lower_bound_only():
    r = v2("chat_completions", {"usage": {"prompt_tokens": 100}})
    assert comp(r, "output") == (None, NOT_REPORTED)
    assert der(r, "processed_total") == (None, 100)


def test_u10_measured_zero_differs_from_absent():
    r = v2("responses", {"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}})
    assert comp(r, "input") == (0, REPORTED)
    assert der(r, "processed_total") == (0, 0)
    assert r.usage_present is True and "usage_not_reported" not in r.diagnostics


def test_u11_estimate_never_fills_measured_slots():
    estimate = {"input_tokens": 100, "output_tokens": 20, "method": "fixture-estimator-v1"}
    r = v2("responses", {"output": []}, estimate=estimate)
    assert comp(r, "input") == (None, NOT_REPORTED)
    assert der(r, "processed_total") == (None, None)
    assert r.estimate == estimate


def test_u12_gateway_cost_is_an_observation_not_a_bill():
    r = v2(
        "chat_completions",
        {"usage": {"prompt_tokens": 100, "completion_tokens": 20}, "cost": "0"},
    )
    assert r.provider_cost == {"path": "cost", "raw": "0", "accounting": "unknown"}


def test_u13_unrecognized_detail_is_preserved_not_assumed():
    r = v2("responses", {"usage": {"input_tokens": 20, "output_tokens": 6, "cached_tokens": 15}})
    assert "usage.cached_tokens" in r.unrecognized_paths
    assert comp(r, "cache_read") == (None, NOT_REPORTED)
    assert der(r, "processed_total") == (26, 26)
    assert der(r, "fresh_input") == (None, None)


def test_combined_overlap_matches_draft_example():
    r = v2(
        "chat_completions",
        {"usage": {"prompt_tokens": 100, "completion_tokens": 20,
                   "prompt_tokens_details": {"cached_tokens": 80},
                   "completion_tokens_details": {"reasoning_tokens": 5}}},
    )
    assert der(r, "processed_total") == (120, 120)  # not 205


# --- adversarial ----------------------------------------------------------------


@pytest.mark.parametrize("bad", [-5, "100", True, 3.5])
def test_invalid_counts_are_diagnosed_never_coerced(bad):
    r = v2("chat_completions", {"usage": {"prompt_tokens": bad, "completion_tokens": 20}})
    assert comp(r, "input") == (None, INVALID)
    assert der(r, "processed_total") == (None, 20)
    assert "invalid_value:usage.prompt_tokens" in r.diagnostics


def test_null_count_is_not_reported_with_diagnostic():
    r = v2("chat_completions", {"usage": {"prompt_tokens": None, "completion_tokens": 20}})
    assert comp(r, "input") == (None, NOT_REPORTED)
    assert "null_value:usage.prompt_tokens" in r.diagnostics


def test_subset_larger_than_parent_is_a_conflict():
    r = v2(
        "chat_completions",
        {"usage": {"prompt_tokens": 100, "completion_tokens": 20,
                   "prompt_tokens_details": {"cached_tokens": 150, "cache_write_tokens": 0}}},
    )
    assert "cache_components_exceed_input" in r.conflicts
    assert der(r, "fresh_input") == (None, None)


def test_reasoning_larger_than_output_is_a_conflict():
    r = v2(
        "responses",
        {"usage": {"input_tokens": 100, "output_tokens": 20,
                   "output_tokens_details": {"reasoning_tokens": 30}}},
    )
    assert "reasoning_exceeds_output" in r.conflicts


def test_unknown_protocol_derives_nothing():
    r = v2("gemini", {"usage": {"promptTokenCount": 10, "candidatesTokenCount": 3}})
    assert r.rule_id is None
    assert all(d.value is None for d in r.derived)
    assert "usage.promptTokenCount" in r.unrecognized_paths


def test_unknown_version_fails_loudly():
    with pytest.raises(ValueError):
        interpret_usage("chat_completions", {}, version="usage-semantics-v9")


# --- Stage 12 live usage members, copied exactly ----------------------------------

LUNA = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "x"}]}],
        "usage": {"input_tokens": 38, "output_tokens": 26, "total_tokens": 64,
                  "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 0}},
        "cost": "0"}
MIMO = {"choices": [{"message": {"content": "x", "reasoning": "hidden reasoning text"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 279, "completion_tokens": 166, "total_tokens": 445,
                  "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 192,
                                            "cache_write_tokens": 0},
                  "completion_tokens_details": {"audio_tokens": 0, "reasoning_tokens": 0}},
        "cost": "0"}
MINIMAX = {"content": [{"type": "thinking", "thinking": "hidden"}, {"type": "text", "text": "x"}],
           "stop_reason": "end_turn",
           "usage": {"input_tokens": 73, "output_tokens": 220},
           "cost": "0"}


def test_live_luna_responses():
    r = v2("responses", LUNA)
    assert der(r, "total_input") == (38, 38)
    assert der(r, "fresh_input") == (38, 38)
    assert der(r, "processed_total") == (64, 64)
    assert r.conflicts == () and r.route_conformance == "unverified"


def test_live_mimo_chat():
    r = v2("chat_completions", MIMO)
    assert der(r, "total_input") == (279, 279)
    assert der(r, "fresh_input") == (87, 87)
    assert der(r, "processed_total") == (445, 445)
    assert r.conflicts == ()
    assert "reasoning_tokens_zero_with_reasoning_content" in r.diagnostics
    assert "usage.prompt_tokens_details.audio_tokens" in r.unrecognized_paths


def test_live_minimax_messages():
    r = v2("messages", MINIMAX)
    assert der(r, "fresh_input") == (73, 73)
    assert der(r, "total_input") == (None, 73)
    assert der(r, "processed_total") == (None, 293)
    assert "reasoning_content_present_reasoning_tokens_not_reported" in r.diagnostics


@pytest.mark.parametrize("protocol,parsed,expected", [
    ("responses", LUNA, (38, 26)),
    ("chat_completions", MIMO, (279, 166)),
    ("messages", MINIMAX, (73, 220)),
])
def test_v1_reproduces_historical_adapter_view(protocol, parsed, expected):
    r = interpret_usage(protocol, parsed, version=USAGE_SEMANTICS_V1)
    assert (r.component("input").value, r.component("output").value) == expected
    assert r.derived == () and r.conflicts == ()


def test_projection_is_deterministic():
    assert v2("chat_completions", MIMO) == v2("chat_completions", MIMO)


# --- runtime projection over preserved bytes -----------------------------------------


def test_runtime_usage_projection_reads_bytes_and_never_writes(tmp_path):
    body = json.dumps(MIMO).encode()

    def post(url, request, headers, timeout):
        return HttpResponse(200, {}, body, "application/json")

    ledger = SQLiteLedger(tmp_path / "ledger.sqlite")
    store = FileArtifactStore(tmp_path / "artifacts", ledger)
    runtime = Runtime(ledger, artifact_store=store)
    actor = ActorRef("reviewer", "model", provider="opencode", model="mimo-v2.5")
    spec = CallSpec("call", "task", actor,
                    ContextCompiler().compile(task_id="task", actor=actor, prompt="Review P"),
                    "unique", parameters={})
    adapter = OpenCodeCognitionAdapter(model="mimo-v2.5", protocol="chat_completions",
                                       api_key="decoy", http_post=post)
    call = runtime.invoke_recorded_call(spec, adapter=adapter)
    attempt = call.attempts[0]
    assert (attempt.usage.input_tokens, attempt.usage.output_tokens) == (279, 166)
    events_before = len(ledger.read_all())

    projected = runtime.interpret_usage_as(attempt.attempt_id, version=USAGE_SEMANTICS_V2)

    assert len(ledger.read_all()) == events_before
    observation = runtime.get_attempt_observation(attempt.attempt_id)
    ref = observation["response_body_artifact"]
    assert projected["evidence_class"] == "transport_body"
    assert projected["source_sha256"] == ref["sha256"]
    assert store.read_bytes(ref["artifact_id"]) == body
    interp = projected["interpretation"]
    assert der(interp, "fresh_input") == (87, 87)
    assert "reasoning_tokens_zero_with_reasoning_content" in interp.diagnostics
    # The recorded attempt keeps its historical usage view.
    assert (call.attempts[0].usage.input_tokens, call.attempts[0].usage.output_tokens) == (279, 166)


def test_runtime_usage_projection_unknown_attempt(tmp_path):
    ledger = SQLiteLedger(tmp_path / "ledger.sqlite")
    runtime = Runtime(ledger, artifact_store=FileArtifactStore(tmp_path / "a", ledger))
    assert runtime.interpret_usage_as("missing", version=USAGE_SEMANTICS_V2) is None
