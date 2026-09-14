"""Execution ladder with recorded escalation (Stage 29B).

Useful applied AI is ordinary software with selectively intelligent
boundaries. A ladder answers a request at the lowest rung that meets a
deterministic acceptance check: a rule first, then model rungs in declared
order, then a person. Every rung entered and why, every decline and its
reason, every check, and either the resolution or the question put to a
person are ledger events. The router is the loop in ``run_ladder``; its
decision path contains no model call.

The check sees only the request input and one candidate output. It can
establish that a candidate meets the stated criterion, not that the answer is
correct; experiments score correctness separately, against labels the runtime
never sees.

Spend is derived, never asserted: a model rung declares its pricing schedule
and ``project_ladder_run`` multiplies recorded usage by it. A rung that is
unpriced, or a call that reported no usage, is counted as unknown spend,
never as zero.

A run id is used once. Calling ``run_ladder`` again for a started run returns
its projection and appends nothing; an interrupted run is not resumed here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .acceptance import artifact_target
from .adapters import ActionRequest, ActionStatus, CheckRequest, CheckResult, CheckVerdict
from .artifacts import ArtifactCorruptionError
from .context import ContextCompiler
from .domain import ActorRef, Authority, CallSpec, Capability
from .evidence import preserved_output_text
from .interpretation import ObservationUnavailable
from .ledger import Event

if TYPE_CHECKING:
    from .adapters import CognitionAdapter, ExecutionAdapter
    from .runtime import Runtime

LADDER_ROUTER_V1 = "ladder-router-v1"
RULE, MODEL, HUMAN = "rule", "model", "human"

LADDER_STARTED = "ladder.started"
RUNG_ENTERED = "ladder.rung_entered"
RUNG_DECLINED = "ladder.rung_declined"
LADDER_RESOLVED = "ladder.resolved"
ASKED_HUMAN = "ladder.asked_human"
LADDER_EXHAUSTED = "ladder.exhausted"


class LadderValidationError(ValueError):
    """The ladder or request cannot run; raised before anything is appended."""


@dataclass(frozen=True, slots=True)
class Rung:
    rung_id: str
    kind: str  # "rule" | "model" | "human"
    model: str | None = None
    route: str | None = None  # e.g. "opencode-zen", "opencode-go"
    pricing_schedule_id: str | None = None
    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LadderSpec:
    ladder_id: str
    version: str
    rungs: tuple[Rung, ...]
    router_version: str = LADDER_ROUTER_V1


@dataclass(frozen=True, slots=True)
class RuleResult:
    """A rule either produces a candidate output or declines with a reason."""

    output: str | None = None
    decline_reason: str | None = None


@dataclass(frozen=True, slots=True)
class CheckOutcome:
    passed: bool
    reason: str
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LadderRequest:
    run_id: str
    task_id: str
    input_text: str
    instruction: str
    effect_capability: str | None = None  # capability an effect of the answer requires
    effect_instruction: str | None = None
    requested_by: str | None = None
    input_needed: str = "a person to answer the request"


@dataclass(frozen=True, slots=True)
class RungAttempt:
    rung_id: str
    kind: str
    entered_reason: str
    outcome: str  # "resolved" | "declined" | "asked_human" | "in_progress"
    decline_reason: str | None = None
    call_id: str | None = None
    model: str | None = None
    route: str | None = None
    provider_attempts: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
    spend_usd: float | None = 0.0
    check_id: str | None = None
    check_verdict: str | None = None
    check_ms: int | None = None
    candidate_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class LadderOutcome:
    run_id: str
    task_id: str
    ladder_id: str
    ladder_version: str
    status: str  # "resolved" | "asked_human" | "exhausted" | "in_progress" | "not_found"
    attempts: tuple[RungAttempt, ...] = ()
    resolved_rung_id: str | None = None
    candidate_sha256: str | None = None
    answer_text: str | None = None
    human_reason: str | None = None
    input_needed: str | None = None
    effect_status: str | None = None
    model_calls: int = 0
    escalations: int = 0
    known_spend_usd: float = 0.0
    unknown_spend_attempts: int = 0
    model_latency_ms: int = 0
    check_ms: int = 0


# ---------------- running ----------------


def run_ladder(
    runtime: Runtime,
    spec: LadderSpec,
    request: LadderRequest,
    *,
    check: Callable[[str, str], CheckOutcome],
    adapters: Mapping[str, CognitionAdapter],
    rule: Callable[[str], RuleResult] | None = None,
    authority: Authority | None = None,
    apply: ExecutionAdapter | None = None,
) -> LadderOutcome:
    """Climb the ladder once for one request, recording every transition."""
    _validate(runtime, spec, request, rule=rule, adapters=adapters, apply=apply)
    if _ladder_events(runtime, request.run_id):
        return project_ladder_run(runtime, request.run_id)

    _append(runtime, request, LADDER_STARTED, {
        "ladder_id": spec.ladder_id,
        "ladder_version": spec.version,
        "router_version": spec.router_version,
        "rungs": [_rung_payload(rung) for rung in spec.rungs],
        "input_sha256": _sha(request.input_text),
        "instruction_sha256": _sha(request.instruction),
        "effect_capability": request.effect_capability,
    })
    reason = "first rung"
    declined: list[dict[str, str]] = []
    for rung in spec.rungs:
        _append(runtime, request, RUNG_ENTERED,
                {"rung_id": rung.rung_id, "kind": rung.kind, "reason": reason})
        if rung.kind == HUMAN:
            _append(runtime, request, ASKED_HUMAN, {
                "rung_id": rung.rung_id, "reason": reason,
                "input_needed": request.input_needed, "declined": declined})
            return project_ladder_run(runtime, request.run_id)

        output, call_id, decline = _candidate(runtime, rung, request, rule, adapters)
        checked: dict[str, Any] = {}
        if decline is None and output is not None:
            assert runtime.artifact_store is not None
            candidate = runtime.artifact_store.store_text(output, artifact_type="ladder_candidate")
            check_id = f"{request.run_id}:{rung.rung_id}:check"
            result = runtime.run_check(
                CheckRequest(check_id=check_id, task_id=request.task_id,
                             target=artifact_target(candidate.sha256)),
                verifier=_InProcessCheck(check, request.input_text, output),
            )
            checked = {"check_id": check_id, "candidate_sha256": candidate.sha256}
            if result.verdict == CheckVerdict.PASS:
                _append(runtime, request, LADDER_RESOLVED,
                        {"rung_id": rung.rung_id, "call_id": call_id, **checked})
                if request.effect_capability is not None:
                    _effect(runtime, request, candidate.sha256, authority, apply, declined)
                return project_ladder_run(runtime, request.run_id)
            decline = _check_decline(result)
        _append(runtime, request, RUNG_DECLINED,
                {"rung_id": rung.rung_id, "reason": decline, "call_id": call_id, **checked})
        declined.append({"rung_id": rung.rung_id, "reason": str(decline)})
        reason = f"escalated from {rung.rung_id}: {decline}"
    _append(runtime, request, LADDER_EXHAUSTED, {"declined": declined})
    return project_ladder_run(runtime, request.run_id)


def _candidate(
    runtime: Runtime,
    rung: Rung,
    request: LadderRequest,
    rule: Callable[[str], RuleResult] | None,
    adapters: Mapping[str, CognitionAdapter],
) -> tuple[str | None, str | None, str | None]:
    """Return (output, call_id, decline_reason) for a rule or model rung."""
    if rung.kind == RULE:
        assert rule is not None
        result = rule(request.input_text)
        if result.output is None:
            return None, None, f"rule_declined:{result.decline_reason or 'no reason given'}"
        return result.output, None, None
    call_id = f"{request.run_id}:{rung.rung_id}"
    actor = ActorRef(rung.rung_id, "model", provider=rung.route, model=rung.model)
    package = ContextCompiler().compile(
        task_id=request.task_id, actor=actor, prompt=request.input_text
    )
    spec = CallSpec(call_id=call_id, task_id=request.task_id, actor=actor, context=package,
                    idempotency_key=call_id, chamber=rung.rung_id,
                    instruction=request.instruction, parameters=dict(rung.parameters))
    recorded = runtime.invoke_recorded_call(spec, adapter=adapters[rung.rung_id], max_attempts=1)
    if recorded.status != "succeeded":
        kind = recorded.attempts[-1].error_kind if recorded.attempts else None
        return None, call_id, f"call_not_succeeded:{recorded.status}:{kind or 'unknown'}"
    try:
        return preserved_output_text(runtime, recorded.attempts[-1].attempt_id), call_id, None
    except ObservationUnavailable as exc:
        return None, call_id, f"output_unavailable:{exc.reason}"


def _effect(
    runtime: Runtime,
    request: LadderRequest,
    candidate_sha256: str,
    authority: Authority | None,
    apply: ExecutionAdapter | None,
    declined: list[dict[str, str]],
) -> None:
    assert apply is not None and request.effect_capability is not None
    action = ActionRequest(
        action_id=f"{request.run_id}:effect",
        task_id=request.task_id,
        capability=request.effect_capability,
        instruction=request.effect_instruction or "apply the resolved answer",
        precondition_hash=None,
        idempotency_key=f"{request.run_id}:effect",
        actor_id="ladder",
        adapter="ladder",
        payload={"candidate_sha256": candidate_sha256},
        requested_by=request.requested_by,
        adapter_id="ladder-effect",
    )
    result = runtime.execute_action(action, authority=authority or Authority(), adapter=apply)
    if result.status == ActionStatus.DENIED:
        _append(runtime, request, ASKED_HUMAN, {
            "rung_id": None,
            "reason": f"authority_missing:{request.effect_capability}",
            "input_needed": f"authority to {action.instruction}",
            "action_id": action.action_id,
            "declined": declined,
        })


class _InProcessCheck:
    """Runs the caller's deterministic check as a recorded verification."""

    def __init__(self, check: Callable[[str, str], CheckOutcome], input_text: str, output: str):
        self._check = check
        self._input = input_text
        self._output = output

    def run(self, request: CheckRequest) -> CheckResult:
        started = _now()
        try:
            outcome = self._check(self._input, self._output)
        except Exception as exc:  # noqa: BLE001 - a check that crashes is an error, never a pass
            return CheckResult(check_id=request.check_id, verdict=CheckVerdict.ERROR,
                               started_at=started, completed_at=_now(),
                               error=f"{type(exc).__name__}: {exc}")
        details = json.dumps({"reason": outcome.reason, "details": dict(outcome.details)},
                             sort_keys=True, default=str)
        return CheckResult(check_id=request.check_id,
                           verdict=CheckVerdict.PASS if outcome.passed else CheckVerdict.FAIL,
                           started_at=started, completed_at=_now(), details=details)


def _check_decline(result: CheckResult) -> str:
    if result.verdict == CheckVerdict.FAIL:
        try:
            reason = json.loads(result.details or "{}").get("reason", "")
        except ValueError:
            reason = result.details or ""
        return f"check_failed:{reason}"
    return f"check_error:{result.error or result.verdict}"


# ---------------- projection and presentation ----------------


def project_ladder_run(runtime: Runtime, run_id: str) -> LadderOutcome:
    """Derive a ladder run's outcome, spend and escalations from the ledger. Appends nothing."""
    events = _ladder_events(runtime, run_id)
    if not events or events[0].kind != LADDER_STARTED:
        return LadderOutcome(run_id, "", "", "", "not_found")
    start = events[0].payload
    rungs = {str(r["rung_id"]): r for r in start["rungs"]}
    task_id = str(events[0].correlation_id or "")
    partial: list[dict[str, Any]] = []
    resolved: dict[str, Any] | None = None
    human: dict[str, Any] | None = None
    exhausted = False
    for event in events[1:]:
        p = event.payload
        if event.kind == RUNG_ENTERED:
            partial.append({"rung_id": p["rung_id"], "kind": p["kind"],
                            "entered_reason": p["reason"], "outcome": "in_progress"})
        elif event.kind == RUNG_DECLINED:
            partial[-1].update(outcome="declined", decline_reason=p["reason"],
                               call_id=p.get("call_id"), check_id=p.get("check_id"),
                               candidate_sha256=p.get("candidate_sha256"))
        elif event.kind == LADDER_RESOLVED:
            resolved = p
            partial[-1].update(outcome="resolved", call_id=p.get("call_id"),
                               check_id=p.get("check_id"),
                               candidate_sha256=p.get("candidate_sha256"))
        elif event.kind == ASKED_HUMAN:
            human = p
            if p.get("rung_id") is not None:
                partial[-1].update(outcome="asked_human")
        elif event.kind == LADDER_EXHAUSTED:
            exhausted = True

    attempts = tuple(_attempt(runtime, rungs[str(item["rung_id"])], item) for item in partial)
    status = ("asked_human" if human is not None else "resolved" if resolved is not None
              else "exhausted" if exhausted else "in_progress")
    answer = None
    if resolved is not None and runtime.artifact_store is not None:
        try:
            answer = runtime.artifact_store.read_text(str(resolved["candidate_sha256"]))
        except (FileNotFoundError, ArtifactCorruptionError):
            answer = None
    effect = [e for e in runtime.ledger.events_by_kind(("action.completed",))
              if str(e.payload.get("action_id")) == f"{run_id}:effect"]
    return LadderOutcome(
        run_id=run_id,
        task_id=task_id,
        ladder_id=str(start["ladder_id"]),
        ladder_version=str(start["ladder_version"]),
        status=status,
        attempts=attempts,
        resolved_rung_id=str(resolved["rung_id"]) if resolved is not None else None,
        candidate_sha256=str(resolved["candidate_sha256"]) if resolved is not None else None,
        answer_text=answer,
        human_reason=str(human["reason"]) if human is not None else None,
        input_needed=str(human["input_needed"]) if human is not None else None,
        effect_status=str(effect[-1].payload.get("status")) if effect else None,
        model_calls=sum(a.provider_attempts for a in attempts),
        escalations=sum(1 for a in attempts if a.outcome == "declined"),
        known_spend_usd=sum(a.spend_usd for a in attempts if a.spend_usd is not None),
        unknown_spend_attempts=sum(1 for a in attempts if a.spend_usd is None),
        model_latency_ms=sum(a.latency_ms or 0 for a in attempts),
        check_ms=sum(a.check_ms or 0 for a in attempts),
    )


def _attempt(runtime: Runtime, rung: Mapping[str, Any], item: Mapping[str, Any]) -> RungAttempt:
    provider_attempts, input_tokens, output_tokens, latency = 0, None, None, None
    spend: float | None = 0.0
    call_id = item.get("call_id")
    if rung["kind"] == MODEL and call_id:
        recorded = runtime.get_recorded_call(str(call_id))
        records = recorded.attempts if recorded is not None else ()
        provider_attempts = len(records)
        ins = [a.usage.input_tokens for a in records]
        outs = [a.usage.output_tokens for a in records]
        input_tokens = None if not ins or None in ins else sum(i for i in ins if i is not None)
        output_tokens = None if not outs or None in outs else sum(o for o in outs if o is not None)
        latency = sum(a.latency_ms or 0 for a in records)
        prices = (rung.get("input_usd_per_million"), rung.get("output_usd_per_million"))
        if None in prices or input_tokens is None or output_tokens is None:
            spend = None
        else:
            spend = (input_tokens * float(prices[0]) + output_tokens * float(prices[1])) / 1_000_000
    check_id = item.get("check_id")
    verdict, check_ms = None, None
    if check_id:
        completed = [e for e in runtime.ledger.events_by_kind(("check.completed",))
                     if e.stream_id == check_id]
        if completed:
            verdict = str(completed[-1].payload.get("verdict"))
            check_ms = _elapsed_ms(completed[-1].payload.get("started_at"),
                                   completed[-1].payload.get("completed_at"))
    return RungAttempt(
        rung_id=str(item["rung_id"]), kind=str(item["kind"]),
        entered_reason=str(item["entered_reason"]), outcome=str(item["outcome"]),
        decline_reason=item.get("decline_reason"), call_id=call_id,
        model=rung.get("model"), route=rung.get("route"),
        provider_attempts=provider_attempts, input_tokens=input_tokens,
        output_tokens=output_tokens, latency_ms=latency, spend_usd=spend,
        check_id=check_id, check_verdict=verdict, check_ms=check_ms,
        candidate_sha256=item.get("candidate_sha256"),
    )


def render_receipt(
    outcome: LadderOutcome, *, answer_line: Callable[[str], str] | None = None
) -> str:
    """User-facing account built only from the projection, never from model prose."""
    lines: list[str] = []
    resolved = next((a for a in outcome.attempts if a.outcome == "resolved"), None)
    if outcome.status == "resolved" and resolved is not None:
        lines.append(f"Answered at rung {resolved.rung_id} ({resolved.kind}); "
                     f"check {resolved.check_id} passed.")
        if outcome.answer_text is not None:
            lines.append(answer_line(outcome.answer_text) if answer_line
                         else f"Answer: {outcome.answer_text}")
    elif outcome.status == "asked_human":
        lines.append(f"Needs a person: {outcome.input_needed}.")
        lines.append(f"Why: {outcome.human_reason}.")
        if resolved is not None and outcome.answer_text is not None:
            lines.append(answer_line(outcome.answer_text) if answer_line
                         else f"Proposed answer: {outcome.answer_text}")
    elif outcome.status == "exhausted":
        lines.append("No rung met the check, and no person rung was declared.")
    else:
        lines.append(f"Run {outcome.run_id}: {outcome.status}.")
    lines.extend(f"Tried {a.rung_id}: {a.decline_reason}."
                 for a in outcome.attempts if a.outcome == "declined")
    unknown = (f" plus {outcome.unknown_spend_attempts} attempt(s) of unknown spend"
               if outcome.unknown_spend_attempts else "")
    lines.append(f"Model calls: {outcome.model_calls}; known spend: "
                 f"${outcome.known_spend_usd:.6f}{unknown}.")
    return "\n".join(lines)


# ---------------- helpers ----------------


def _validate(
    runtime: Runtime,
    spec: LadderSpec,
    request: LadderRequest,
    *,
    rule: Callable[[str], RuleResult] | None,
    adapters: Mapping[str, CognitionAdapter],
    apply: ExecutionAdapter | None,
) -> None:
    problems: list[str] = []
    ids = [rung.rung_id for rung in spec.rungs]
    if not spec.rungs:
        problems.append("the ladder has no rungs")
    if len(set(ids)) != len(ids):
        problems.append("rung ids are not unique")
    for index, rung in enumerate(spec.rungs):
        if rung.kind not in (RULE, MODEL, HUMAN):
            problems.append(f"{rung.rung_id}: unknown kind {rung.kind!r}")
        if rung.kind == RULE and rule is None:
            problems.append(f"{rung.rung_id}: rule rung without a rule")
        if rung.kind == MODEL and (not rung.model or rung.rung_id not in adapters):
            problems.append(f"{rung.rung_id}: model rung needs a model and an adapter")
        if rung.kind == HUMAN and index != len(spec.rungs) - 1:
            problems.append(f"{rung.rung_id}: the person rung must be last")
    if runtime.artifact_store is None:
        problems.append("an artifact store is required")
    if request.effect_capability is not None:
        try:
            Capability(request.effect_capability)
        except ValueError:
            problems.append(f"unknown effect capability {request.effect_capability!r}")
        if apply is None:
            problems.append("an effect needs an apply adapter")
    if problems:
        raise LadderValidationError("; ".join(problems))


def _ladder_events(runtime: Runtime, run_id: str) -> list[Event]:
    return [e for e in runtime.ledger.read_all()
            if e.stream_id == run_id and e.kind.startswith("ladder.")]


def _append(runtime: Runtime, request: LadderRequest, kind: str, payload: dict[str, Any]) -> Event:
    event = Event.create(stream_id=request.run_id, kind=kind, actor_id="ladder",
                         payload={"run_id": request.run_id, **payload},
                         correlation_id=request.task_id)
    runtime.ledger.append(event)
    return event


def _rung_payload(rung: Rung) -> dict[str, Any]:
    return {
        "rung_id": rung.rung_id, "kind": rung.kind, "model": rung.model, "route": rung.route,
        "pricing_schedule_id": rung.pricing_schedule_id,
        "input_usd_per_million": rung.input_usd_per_million,
        "output_usd_per_million": rung.output_usd_per_million,
        "parameters": dict(rung.parameters),
    }


def _elapsed_ms(started: Any, completed: Any) -> int | None:
    try:
        delta = datetime.fromisoformat(str(completed)) - datetime.fromisoformat(str(started))
    except ValueError:
        return None
    return int(delta.total_seconds() * 1000)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()
