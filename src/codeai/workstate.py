"""Working state that survives the process (Stage 16).

Restart is reopening the ledger. Resume is being able to say, from recorded
facts alone, what was being attempted, from which inputs, what failed, what
succeeded, what is unresolved, and which next operation is safe.

``project_work_state`` is a pure projection over the ledger; it appends
nothing. ``resume_call`` acts on it in exactly one case: a call whose record
proves no provider effect can have happened (no ``attempt.started``). It
re-issues the recorded request, under the same idempotency key, as a new call
linked by ``call.resumed``, and refuses when the recorded intent no longer
reproduces. Every other interrupted state gets a named next operation and no
automatic action.

A call is classified by the last evidence recorded for it:

- ``call.replayed``: replayed, nothing to do.
- ``call.completed``: completed.
- ``call.preparation_failed``: failed before effect; fix the inputs.
- ``call.requested`` only, or ``call.manifest`` without ``attempt.started``:
  no provider effect is possible; safe to start.
- ``attempt.started`` with no observation, interpretation, decision or
  completion: the provider may or may not have served the request; reconcile,
  never repeat automatically.
- observation without interpretation: reinterpret the preserved evidence.
- interpretation without decision: re-derive the decision.
- decision to retry with no next attempt: start the next attempt.
- decision without call status or completion: finalize the call.

A no-effect or failed-before-effect call is superseded when another call with
the same idempotency key completed.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from .acceptance import TaskCompletionStatus, project_task_completion
from .domain import ActorRef, Budget, CallSpec, ContextPackage, Seal, Variant
from .ledger import Event
from .rendering import ContextResolutionError, render_context

if TYPE_CHECKING:
    from .adapters import CognitionAdapter
    from .domain import RecordedCall
    from .runtime import Runtime

WORK_STATE_V1 = "work-state-v1"


class NextOperation(StrEnum):
    NONE = "none"
    START_CALL = "start_call"
    RECONCILE_EFFECT = "reconcile_effect"
    REINTERPRET = "reinterpret"
    REDECIDE = "redecide"
    START_NEXT_ATTEMPT = "start_next_attempt"
    FINALIZE_CALL = "finalize_call"
    FIX_INPUTS = "fix_inputs"
    RECOMPILE = "recompile"
    CHECK_AND_ACCEPT = "check_and_accept"
    REPEAT_ACCEPTANCE = "repeat_acceptance"


class ResumeRefused(RuntimeError):
    def __init__(self, call_id: str, next_operation: str, reason: str) -> None:
        self.call_id = call_id
        self.next_operation = next_operation
        self.reason = reason
        super().__init__(f"resume refused for {call_id}: {next_operation}: {reason}")


@dataclass(frozen=True, slots=True)
class CompilationState:
    compilation_id: str
    stage: str  # "interrupted" | "failed" | "compiled"
    next_operation: str
    reason: str
    offered_ids: tuple[str, ...] = ()
    required_ids: tuple[str, ...] = ()
    missing_required_ids: tuple[str, ...] = ()
    failure: str | None = None
    package_id: str | None = None
    included_ids: tuple[str, ...] = ()
    excluded: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class CallState:
    call_id: str
    idempotency_key: str | None
    stage: str
    provider_effect: str  # "none" | "unknown" | "observed"
    next_operation: str
    duplicate_effect_risk: bool
    reason: str
    attempt_count: int = 0
    call_status: str | None = None
    superseded_by: str | None = None
    resumed_as: str | None = None
    status_basis: str | None = None  # "execution:<policy>" or "reinterpretation:<interpreter>/<policy>"


@dataclass(frozen=True, slots=True)
class WorkState:
    task_id: str
    version: str
    ledger_event_count: int
    compilations: tuple[CompilationState, ...]
    calls: tuple[CallState, ...]
    task_completion: str
    task_next_operation: str
    task_reason: str


# ---------------- projection ----------------


def project_work_state(runtime: Runtime, task_id: str) -> WorkState:
    """Project the task's working state from the ledger alone. Appends nothing."""
    events = runtime.ledger.read_all()
    compilations = _compilation_states(events, task_id)
    calls = _call_states(events, task_id)
    completion = project_task_completion(runtime, task_id)
    task_next, task_reason = _task_next(completion, calls)
    return WorkState(
        task_id=task_id,
        version=WORK_STATE_V1,
        ledger_event_count=len(events),
        compilations=compilations,
        calls=calls,
        task_completion=completion.status,
        task_next_operation=task_next,
        task_reason=task_reason,
    )


def _compilation_states(events: tuple[Event, ...], task_id: str) -> tuple[CompilationState, ...]:
    outcomes: dict[str, Event] = {}
    for event in events:
        if event.kind in ("context.compiled", "context.compilation_failed"):
            compilation_id = event.payload.get("compilation_id")
            if compilation_id:
                outcomes[str(compilation_id)] = event
    states = []
    for requested in events:
        if requested.kind != "context.compilation_requested":
            continue
        if str(requested.payload.get("task_id")) != task_id:
            continue
        compilation_id = str(requested.payload["compilation_id"])
        candidates = requested.payload.get("candidates") or []
        offered = tuple(str(c["candidate_id"]) for c in candidates)
        explicit = {str(i) for i in requested.payload.get("required_ids") or ()}
        required = tuple(sorted(explicit | {str(c["candidate_id"]) for c in candidates if c.get("required")}))
        missing = tuple(sorted(explicit - set(offered)))
        outcome = outcomes.get(compilation_id)
        common = {"offered_ids": offered, "required_ids": required, "missing_required_ids": missing}
        if outcome is None:
            states.append(CompilationState(
                compilation_id, "interrupted", NextOperation.RECOMPILE.value,
                "compilation requested with no recorded outcome; compiling has no external effect",
                **common))
        elif outcome.kind == "context.compilation_failed":
            states.append(CompilationState(
                compilation_id, "failed", NextOperation.FIX_INPUTS.value,
                str(outcome.payload.get("message", "")), failure=str(outcome.payload.get("exception")),
                **common))
        else:
            trace = outcome.payload.get("trace") or []
            states.append(CompilationState(
                compilation_id, "compiled", NextOperation.NONE.value, "compiled",
                package_id=outcome.payload.get("package_id"),
                included_ids=tuple(str(i) for i in outcome.payload.get("included_ids") or ()),
                excluded=tuple((str(t["candidate_id"]), str(t["reason"])) for t in trace
                               if t.get("decision") == "excluded"),
                **common))
    return tuple(states)


def _call_id(event: Event) -> str:
    return str(event.payload.get("call_id") or event.stream_id)


def _call_states(events: tuple[Event, ...], task_id: str) -> tuple[CallState, ...]:
    by_call: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        if event.kind.startswith(("call.", "attempt.")):
            by_call[_call_id(event)].append(event)
    requested = [e for e in events if e.kind == "call.requested" and str(e.payload.get("task_id")) == task_id]

    completed_by_key: dict[str, list[str]] = defaultdict(list)
    for request in requested:
        call_id = _call_id(request)
        kinds = {e.kind for e in by_call[call_id]}
        if "call.completed" in kinds and "call.manifest" in kinds:
            completed_by_key[str(request.payload.get("idempotency_key"))].append(call_id)

    states = []
    for request in requested:
        call_id = _call_id(request)
        key = request.payload.get("idempotency_key")
        call_events = by_call[call_id]
        kinds = {e.kind for e in call_events}
        attempts = sorted((e for e in call_events if e.kind == "attempt.started"),
                          key=lambda e: int(e.payload.get("attempt_index") or 0))
        status_events = [e for e in call_events if e.kind in ("call.status_decided", "call.reinterpreted")]
        call_status = str(status_events[-1].payload.get("status")) if status_events else None
        status_basis = None
        if status_events:
            latest = status_events[-1].payload
            status_basis = (
                f"reinterpretation:{latest.get('interpreter_version')}/{latest.get('policy_version')}"
                if status_events[-1].kind == "call.reinterpreted"
                else f"execution:{latest.get('policy_version')}"
            )
        resumed_as = next((str(e.payload["to_call_id"]) for e in call_events if e.kind == "call.resumed"), None)
        effect = "observed" if attempts else "none"
        risk = False

        if "call.replayed" in kinds:
            stage, operation = "replayed", NextOperation.NONE
            reason = "idempotency replay of a completed call; no provider effect"
            effect = "none"
        elif "call.completed" in kinds:
            stage, operation = "completed", NextOperation.NONE
            reason = "call completed; the same idempotency key replays without a provider effect"
        elif "call.preparation_failed" in kinds:
            stage, operation = "preparation_failed", NextOperation.FIX_INPUTS
            failed = next(e for e in call_events if e.kind == "call.preparation_failed")
            reason = f"failed before any provider effect at {failed.payload.get('stage')}"
        elif "call.manifest" not in kinds:
            stage, operation = "requested", NextOperation.START_CALL
            reason = "no manifest recorded, so no attempt can have started"
        elif not attempts:
            stage, operation = "manifest_recorded", NextOperation.START_CALL
            reason = "manifest recorded and no attempt started: no provider effect"
        else:
            last = attempts[-1]
            attempt_id = last.payload.get("attempt_id")
            attempt_kinds = {e.kind for e in call_events if e.payload.get("attempt_id") == attempt_id}
            if not attempt_kinds & {"attempt.observed", "attempt.interpreted", "attempt.retry_decided",
                                    "attempt.completed"}:
                stage, operation, effect, risk = "effect_unknown", NextOperation.RECONCILE_EFFECT, "unknown", True
                reason = ("attempt started with no observation: the provider may or may not have served the "
                          "request; repeating could duplicate the effect")
            elif "attempt.interpreted" not in attempt_kinds:
                stage, operation = "observed", NextOperation.REINTERPRET
                reason = "transport observation preserved; interpretation missing"
            elif "attempt.retry_decided" not in attempt_kinds:
                stage, operation = "interpreted", NextOperation.REDECIDE
                reason = "interpretation recorded; attempt decision missing"
            else:
                decision = next(e for e in call_events
                                if e.kind == "attempt.retry_decided" and e.payload.get("attempt_id") == attempt_id)
                if decision.payload.get("executed"):
                    stage, operation = "retry_decided", NextOperation.START_NEXT_ATTEMPT
                    reason = "policy decided to retry; the next attempt was not started"
                else:
                    stage, operation = "decided", NextOperation.FINALIZE_CALL
                    reason = "attempt decided; call status or completion not recorded"

        superseded_by = None
        if stage in ("requested", "manifest_recorded", "preparation_failed") and key is not None:
            later = [c for c in completed_by_key.get(str(key), []) if c != call_id]
            if later:
                superseded_by = later[0]
                operation = NextOperation.NONE
                reason = f"superseded by completed call {later[0]} with the same idempotency key"

        states.append(CallState(
            call_id=call_id, idempotency_key=None if key is None else str(key), stage=stage,
            provider_effect=effect, next_operation=operation.value, duplicate_effect_risk=risk, reason=reason,
            attempt_count=len(attempts), call_status=call_status, superseded_by=superseded_by,
            resumed_as=resumed_as, status_basis=status_basis,
        ))
    return tuple(states)


def _task_next(completion: Any, calls: tuple[CallState, ...]) -> tuple[str, str]:
    if completion.status == TaskCompletionStatus.COMPLETED.value:
        return NextOperation.NONE.value, "task.completed caused by a validated acceptance"
    if completion.acceptance_pending_completion:
        return NextOperation.REPEAT_ACCEPTANCE.value, "acceptance recorded; completion not appended"
    unresolved = [c for c in calls if c.next_operation != NextOperation.NONE.value]
    if unresolved:
        first = unresolved[0]
        return first.next_operation, f"call {first.call_id}: {first.reason}"
    if any(c.stage == "completed" and c.call_status == "succeeded" for c in calls):
        return NextOperation.CHECK_AND_ACCEPT.value, "a call succeeded; the task needs a check and an acceptance"
    return NextOperation.START_CALL.value, "no succeeded call recorded for the task"


# ---------------- resume ----------------


def spec_from_payload(payload: dict[str, Any]) -> CallSpec:
    """Rebuild a CallSpec from its call.requested payload."""
    context = payload["context"]
    seal = context.get("seal") or {}
    package = ContextPackage(
        package_id=str(context["package_id"]),
        task_id=str(context["task_id"]),
        actor=ActorRef(**context["actor"]),
        prompt=str(context.get("prompt") or ""),
        event_ids=tuple(context.get("event_ids") or ()),
        artifact_ids=tuple(context.get("artifact_ids") or ()),
        seal=Seal(**{name: frozenset(seal.get(name) or ()) for name in (
            "forbidden_event_ids", "forbidden_call_ids", "forbidden_artifact_ids", "forbidden_lineage_ids")}),
        metadata=dict(context.get("metadata") or {}),
        objective=context.get("objective"),
        claim_ids=tuple(context.get("claim_ids") or ()),
        budget_tokens=context.get("budget_tokens"),
        prompt_version=context.get("prompt_version"),
        trace_hash=context.get("trace_hash"),
        provenance=dict(context.get("provenance") or {}),
    )
    budget = payload.get("budget")
    variant = dict(payload.get("variant") or {})
    variant["tags"] = dict(variant.get("tags") or {})
    return CallSpec(
        call_id=str(payload["call_id"]),
        task_id=str(payload["task_id"]),
        actor=ActorRef(**payload["actor"]),
        context=package,
        idempotency_key=str(payload["idempotency_key"]),
        pattern=str(payload.get("pattern") or "single"),
        parameters=dict(payload.get("parameters") or {}),
        directive_id=payload.get("directive_id"),
        run_id=payload.get("run_id"),
        adapter_id=payload.get("adapter_id"),
        instruction=str(payload.get("instruction") or ""),
        prompt_version=payload.get("prompt_version"),
        budget=Budget(**budget) if isinstance(budget, dict) else None,
        variant=Variant(**variant),
        metadata=dict(payload.get("metadata") or {}),
        experiment_id=payload.get("experiment_id"),
        arm=payload.get("arm"),
        chamber=payload.get("chamber"),
        logical_model=payload.get("logical_model"),
        context_render=payload.get("context_render"),
    )


def resume_call(
    runtime: Runtime,
    call_id: str,
    *,
    adapter: CognitionAdapter,
    model_config: Any | None = None,
) -> RecordedCall:
    """Continue an interrupted call only where no provider effect is recorded."""
    events = runtime.ledger.read_all()
    request = next((e for e in events if e.kind == "call.requested" and _call_id(e) == call_id), None)
    if request is None:
        raise ResumeRefused(call_id, "unknown", "no call.requested event")
    task_id = str(request.payload.get("task_id"))
    state = next((c for c in project_work_state(runtime, task_id).calls if c.call_id == call_id), None)
    if state is None or state.next_operation != NextOperation.START_CALL.value:
        raise ResumeRefused(call_id, state.next_operation if state else "unknown",
                            state.reason if state else "no projected state")
    spec = spec_from_payload(request.payload)
    new_spec = replace(spec, call_id=str(uuid.uuid4()))
    manifest = next((e.payload for e in events if e.kind == "call.manifest" and _call_id(e) == call_id), None)
    if manifest is not None:
        _check_intent_reproduces(runtime, call_id, new_spec, manifest, adapter)
    runtime.ledger.append(Event.create(
        stream_id=call_id,
        kind="call.resumed",
        actor_id="runtime",
        payload={
            "from_call_id": call_id,
            "to_call_id": new_spec.call_id,
            "task_id": task_id,
            "idempotency_key": spec.idempotency_key,
            "prior_stage": state.stage,
            "basis": state.reason,
            "work_state_version": WORK_STATE_V1,
        },
        causation_id=request.event_id,
        correlation_id=task_id,
    ))
    return runtime.invoke_recorded_call(new_spec, adapter=adapter, model_config=model_config)


def _check_intent_reproduces(
    runtime: Runtime, call_id: str, spec: CallSpec, manifest: dict[str, Any], adapter: CognitionAdapter
) -> None:
    prepare = getattr(adapter, "prepare", None)
    if not callable(prepare):
        raise ResumeRefused(call_id, NextOperation.START_CALL.value,
                            "a manifest was recorded but this adapter cannot prepare a request to compare with it")
    rendered = None
    probe = replace(spec, rendered_context=None)
    if spec.context_render is not None:
        try:
            rendered = render_context(spec.context, ledger=runtime.ledger, artifact_store=runtime.artifact_store,
                                      version=spec.context_render)
        except ContextResolutionError as exc:
            raise ResumeRefused(call_id, NextOperation.START_CALL.value,
                                f"recorded context no longer resolves: {exc}") from exc
        probe = replace(probe, rendered_context=rendered)
    prepared = prepare(probe)
    drift = []
    if manifest.get("request_body_sha256") != prepared.body_sha256:
        drift.append("request_body_sha256")
    if manifest.get("rendered_context_sha256") != (rendered.sha256 if rendered is not None else None):
        drift.append("rendered_context_sha256")
    if manifest.get("context_package_id") != spec.context.package_id:
        drift.append("context_package_id")
    if drift:
        raise ResumeRefused(call_id, NextOperation.START_CALL.value,
                            f"recorded intent does not reproduce: {', '.join(drift)}")
