"""Stage 29B: an execution ladder with recorded escalation.

Offline. Scripted OpenCode-shaped providers return fixed bodies and count the
requests they receive; the check is a toy. Live rungs are exercised only by the
evidence producer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeai.adapters import ActionResult, ActionStatus, CheckVerdict
from codeai.artifacts import FileArtifactStore
from codeai.domain import Authority, Budget, Capability, Task
from codeai.ladder import (
    HUMAN,
    MODEL,
    RULE,
    CheckOutcome,
    LadderRequest,
    LadderSpec,
    LadderValidationError,
    RuleResult,
    Rung,
    project_ladder_run,
    render_receipt,
    run_ladder,
)
from codeai.ledger import SQLiteLedger
from codeai.providers import HttpResponse, OpenCodeCognitionAdapter
from codeai.runtime import Runtime

TASK = "task-29b"
SPEC = LadderSpec("toy", "v1", (
    Rung("rule", RULE),
    Rung("cheap", MODEL, model="cheap", route="opencode-go", pricing_schedule_id="test",
         input_usd_per_million=0.10, output_usd_per_million=0.20),
    Rung("strong", MODEL, model="strong", route="opencode-go", pricing_schedule_id="test",
         input_usd_per_million=1.00, output_usd_per_million=4.00),
    Rung("person", HUMAN),
))


class Provider:
    def __init__(self, text: str = "OK", status: int = 200) -> None:
        self.requests = 0
        self.text = text
        self.status = status

    def __call__(self, url, body, headers, timeout):
        self.requests += 1
        if self.status != 200:
            error = {"type": "error", "error": {"type": "FreeUsageLimitError",
                                                "message": "Rate limit exceeded"}}
            return HttpResponse(self.status, {}, json.dumps(error).encode(), "application/json")
        payload = {"id": "synthetic-29b", "model": "m",
                   "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
                   "choices": [{"finish_reason": "stop",
                                "message": {"role": "assistant", "content": self.text}}]}
        return HttpResponse(200, {}, json.dumps(payload).encode(), "application/json")


class Apply:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request):
        self.calls += 1
        return ActionResult(action_id=request.action_id, status=ActionStatus.SUCCEEDED)


def adapter(provider: Provider, model: str) -> OpenCodeCognitionAdapter:
    return OpenCodeCognitionAdapter(model=model, protocol="chat_completions", gateway_plan="go",
                                    api_key="offline-decoy", http_post=provider, timeout=5)


def make_runtime(path: Path) -> Runtime:
    path.mkdir(parents=True, exist_ok=True)
    ledger = SQLiteLedger(path / "ledger.sqlite")
    runtime = Runtime(ledger, artifact_store=FileArtifactStore(path / "artifacts", ledger))
    if not ledger.events_by_kind(("task.created",)):
        runtime.create_task(Task(TASK, "run-29b", "Answer", ("checked",), Budget(), Authority()))
    return runtime


def rule(text: str) -> RuleResult:
    if text.startswith("ttl="):
        return RuleResult(output="OK rule")
    return RuleResult(decline_reason="no structured ttl")


def check(input_text: str, output: str) -> CheckOutcome:
    if output.startswith("OK"):
        return CheckOutcome(True, "starts with OK")
    return CheckOutcome(False, "does not start with OK")


def request(run_id: str, text: str, **kwargs) -> LadderRequest:
    return LadderRequest(run_id, TASK, text, "Answer with OK.", **kwargs)


def climb(runtime, req, cheap, strong, **kwargs):
    return run_ladder(runtime, SPEC, req, rule=rule, check=check,
                      adapters={"cheap": adapter(cheap, "cheap"),
                                "strong": adapter(strong, "strong")}, **kwargs)


def declines(outcome):
    return [a.decline_reason for a in outcome.attempts if a.outcome == "declined"]


def test_a_rule_answers_without_calling_a_model(tmp_path):
    runtime, cheap, strong = make_runtime(tmp_path), Provider(), Provider()
    outcome = climb(runtime, request("r1", "ttl=90"), cheap, strong)
    assert (outcome.status, outcome.resolved_rung_id, outcome.model_calls) == ("resolved", "rule", 0)
    assert cheap.requests == strong.requests == 0
    assert (outcome.known_spend_usd, outcome.unknown_spend_attempts) == (0.0, 0)
    assert not runtime.ledger.events_by_kind(("call.requested",))


def test_a_declined_rule_escalates_to_the_cheap_model(tmp_path):
    runtime, cheap, strong = make_runtime(tmp_path), Provider("OK cheap"), Provider()
    outcome = climb(runtime, request("r2", "prose"), cheap, strong)
    assert (outcome.status, outcome.resolved_rung_id) == ("resolved", "cheap")
    assert (cheap.requests, strong.requests, outcome.escalations) == (1, 0, 1)
    assert declines(outcome) == ["rule_declined:no structured ttl"]
    assert outcome.attempts[1].entered_reason == "escalated from rule: rule_declined:no structured ttl"
    assert outcome.known_spend_usd == pytest.approx(0.0002)


def test_a_failed_check_escalates_to_the_stronger_model(tmp_path):
    runtime, cheap, strong = make_runtime(tmp_path), Provider("wrong"), Provider("OK strong")
    outcome = climb(runtime, request("r3", "prose"), cheap, strong)
    assert (outcome.status, outcome.resolved_rung_id, outcome.model_calls) == ("resolved", "strong", 2)
    assert declines(outcome) == ["rule_declined:no structured ttl",
                                 "check_failed:does not start with OK"]
    assert outcome.attempts[1].check_verdict == CheckVerdict.FAIL
    assert outcome.answer_text == "OK strong"
    assert outcome.known_spend_usd == pytest.approx(0.0002 + 0.003)


def test_an_unavailable_rung_escalates_and_its_spend_stays_unknown(tmp_path):
    runtime, cheap, strong = make_runtime(tmp_path), Provider(status=429), Provider("OK strong")
    outcome = climb(runtime, request("r4", "prose"), cheap, strong)
    assert outcome.resolved_rung_id == "strong"
    assert declines(outcome)[1].startswith("call_not_succeeded:")
    assert "rate_limited" in declines(outcome)[1]
    assert outcome.unknown_spend_attempts == 1


def test_when_no_rung_meets_the_check_a_person_is_asked(tmp_path):
    runtime, cheap, strong = make_runtime(tmp_path), Provider("bad"), Provider("worse")
    outcome = climb(runtime, request("r5", "prose", input_needed="the TTL, read by a person"),
                    cheap, strong)
    assert (outcome.status, outcome.resolved_rung_id, outcome.model_calls) == ("asked_human", None, 2)
    assert outcome.human_reason == "escalated from strong: check_failed:does not start with OK"
    receipt = render_receipt(outcome)
    assert receipt.startswith("Needs a person: the TTL, read by a person.")
    assert "Tried cheap: check_failed:does not start with OK." in receipt


def test_missing_authority_asks_a_person_before_any_effect(tmp_path):
    runtime, cheap, strong, applier = make_runtime(tmp_path), Provider(), Provider(), Apply()
    denied = climb(runtime, request("r6", "ttl=90", effect_capability="write", requested_by="op"),
                   cheap, strong, authority=Authority(frozenset({Capability.READ})), apply=applier)
    assert (denied.status, denied.resolved_rung_id, denied.effect_status) == (
        "asked_human", "rule", "denied")
    assert denied.human_reason == "authority_missing:write" and applier.calls == 0
    granted = climb(runtime, request("r7", "ttl=90", effect_capability="write"), cheap, strong,
                    authority=Authority(frozenset({Capability.WRITE})), apply=applier)
    assert (granted.status, granted.effect_status, applier.calls) == ("resolved", "succeeded", 1)


def test_invalid_ladders_are_refused_before_anything_is_appended(tmp_path):
    runtime = make_runtime(tmp_path)
    count = len(runtime.ledger.read_all())
    with pytest.raises(LadderValidationError):
        run_ladder(runtime, SPEC, request("r8", "x"), rule=rule, check=check,
                   adapters={"cheap": adapter(Provider(), "cheap")})
    person_first = LadderSpec("toy", "v1", (Rung("person", HUMAN), Rung("rule", RULE)))
    with pytest.raises(LadderValidationError):
        run_ladder(runtime, person_first, request("r9", "x"), rule=rule, check=check, adapters={})
    with pytest.raises(LadderValidationError):
        climb(runtime, request("r10", "ttl=1", effect_capability="write"), Provider(), Provider())
    assert len(runtime.ledger.read_all()) == count


def test_an_unpriced_rung_counts_as_unknown_spend_not_zero(tmp_path):
    runtime, provider = make_runtime(tmp_path), Provider("OK")
    spec = LadderSpec("toy", "v1", (Rung("cheap", MODEL, model="cheap"),))
    outcome = run_ladder(runtime, spec, request("r11", "prose"), check=check,
                         adapters={"cheap": adapter(provider, "cheap")})
    assert (outcome.status, outcome.known_spend_usd, outcome.unknown_spend_attempts) == (
        "resolved", 0.0, 1)


def test_a_crashing_check_is_an_error_not_a_pass(tmp_path):
    runtime = make_runtime(tmp_path)

    def broken(input_text, output):
        raise ValueError("bad check")

    spec = LadderSpec("toy", "v1", (Rung("rule", RULE), Rung("person", HUMAN)))
    outcome = run_ladder(runtime, spec, request("r12", "ttl=1"), rule=rule, check=broken,
                         adapters={})
    assert outcome.status == "asked_human"
    assert declines(outcome) == ["check_error:ValueError: bad check"]


def test_the_projection_survives_reopening_and_a_repeat_runs_nothing(tmp_path):
    runtime, cheap, strong = make_runtime(tmp_path), Provider("wrong"), Provider("OK strong")
    outcome = climb(runtime, request("r13", "prose"), cheap, strong)
    reopened = make_runtime(tmp_path)
    assert project_ladder_run(reopened, "r13") == outcome
    count = len(reopened.ledger.read_all())
    again = climb(reopened, request("r13", "prose"), cheap, strong)
    assert again == outcome and len(reopened.ledger.read_all()) == count
    assert (cheap.requests, strong.requests) == (1, 1)


def test_the_receipt_is_built_from_the_record(tmp_path):
    runtime, cheap, strong = make_runtime(tmp_path), Provider("wrong"), Provider("OK strong")
    outcome = climb(runtime, request("r14", "prose"), cheap, strong)
    receipt = render_receipt(outcome, answer_line=lambda text: f"The answer on record: {text}")
    assert receipt.splitlines() == [
        "Answered at rung strong (model); check r14:strong:check passed.",
        "The answer on record: OK strong",
        "Tried rule: rule_declined:no structured ttl.",
        "Tried cheap: check_failed:does not start with OK.",
        "Model calls: 2; known spend: $0.003200.",
    ]
