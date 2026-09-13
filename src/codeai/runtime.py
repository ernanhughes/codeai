from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, is_dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .acceptance import (
    AcceptanceRequest,
    TaskCompletion,
    project_task_completion,
)
from .acceptance import (
    accept_task as _accept_task,
)
from .adapters import (
    NORMALIZER_VERSION,
    RAW_OBSERVATION_KIND,
    ActionRequest,
    ActionResult,
    ActionStatus,
    CallResult,
    CallSpec,
    CheckRequest,
    CheckResult,
    CheckVerdict,
    CognitionAdapter,
    ExecutionAdapter,
    PreparedCognitionRequest,
    VerificationAdapter,
    sanitize_effective_params,
)
from .artifacts import ArtifactCorruptionError, FileArtifactStore
from .context import (
    CompilationTrace,
    ContextBudgetUnsatisfiable,
    ContextCompiler,
    ContextSealViolation,
)
from .domain import (
    ActorRef,
    ArtifactRef,
    AttemptDecision,
    AttemptInterpretation,
    AttemptRecord,
    AttemptStatus,
    Authority,
    CallManifest,
    Capability,
    Claim,
    ClaimRelationship,
    ClaimStatus,
    ContextPackage,
    CostSource,
    Directive,
    EvidenceClass,
    LogicalCallStatus,
    RecordedCall,
    Seal,
    Task,
    Usage,
    UsageSource,
    Variant,
)
from .interpretation import (
    ATTEMPT_POLICY_V2,
    INTERPRETER_V2,
    InterpretationInput,
    ObservationUnavailable,
    decide_attempt,
    interpret_attempt,
)
from .ledger import Event, SQLiteLedger
from .policy import AuthorityDenied, PolicyEngine
from .providers import PRICING_VERSION, estimate_cost_usd, output_text_for
from .rendering import (
    KNOWN_RENDERERS,
    ContextResolutionError,
    RenderBindingError,
    compose_model_input,
    prepared_input_text,
    render_context,
)
from .workstate import WorkState, project_work_state
from .workstate import resume_call as _resume_call


class PreconditionMismatch(RuntimeError):
    pass


class IdempotencyConflictError(ValueError):
    """Same idempotency key, materially different logical request.

    Raised before any provider effect when a replay lookup finds a completed
    recorded call whose request fingerprint does not match the incoming spec.
    Names the mismatching dimension, never secret values.
    """


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def _git_output(args: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(cwd), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return completed.stdout


def default_repository_state_hash(path: str | Path) -> str:
    """Deterministic repository identity for optimistic concurrency.

    Policy: HEAD commit SHA + hash of staged diff + hash of unstaged diff +
    hash of untracked *path list* (presence, not full contents).

    An action planned against state A must fail loudly when executed against
    materially different state B. Untracked contents are intentionally not
    hashed (cheap presence policy); use an explicit artifact/check when
    untracked contents matter.
    """
    root = Path(path)
    git_dir = root / ".git"
    if git_dir.exists():
        head = _git_output(["rev-parse", "HEAD"], root)
        if head is not None:
            head = head.strip()
            staged = _git_output(["diff", "--cached"], root) or ""
            unstaged = _git_output(["diff"], root) or ""
            status = _git_output(["status", "--short", "--untracked-files=all"], root) or ""
            untracked_paths = sorted(
                line[3:] for line in status.splitlines() if line.startswith("?? ")
            )
            canonical = "\0".join(
                [head, staged, unstaged, "\n".join(untracked_paths)]
            )
            return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        # .git exists but HEAD unresolvable (e.g. fresh repo): fall through.

    digest = hashlib.sha256()
    for entry in sorted(root.rglob("*")):
        if entry.is_dir() or ".codeai" in entry.parts:
            continue
        relative = entry.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(entry.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class Runtime:
    """Small orchestration kernel. Scheduling is deliberately not learned in v0."""

    def __init__(
        self,
        ledger: SQLiteLedger,
        *,
        artifact_store: FileArtifactStore | None = None,
        state_resolver: Callable[[], str] | None = None,
        policy: PolicyEngine | None = None,
    ) -> None:
        self.ledger = ledger
        self.artifact_store = artifact_store
        self.state_resolver = state_resolver
        self.policy = policy or PolicyEngine()

    def open_directive(self, directive: Directive, *, actor_id: str = "human") -> Event:
        event = Event.create(
            stream_id=directive.directive_id,
            kind="directive.opened",
            actor_id=actor_id,
            payload=asdict(directive),
            correlation_id=directive.directive_id,
        )
        self.ledger.append(event)
        return event

    def create_task(self, task: Task, *, actor_id: str = "runtime") -> Event:
        event = Event.create(
            stream_id=task.directive_id,
            kind="task.created",
            actor_id=actor_id,
            payload=asdict(task),
            correlation_id=task.directive_id,
        )
        self.ledger.append(event)
        return event

    def execute_action(
        self,
        request: ActionRequest,
        *,
        authority: Authority,
        adapter: ExecutionAdapter,
    ) -> ActionResult:
        request_event = Event.create(
            stream_id=request.action_id,
            kind="action.requested",
            actor_id=request.effective_requester(),
            payload=asdict(request),
            correlation_id=request.task_id,
        )
        self.ledger.append(request_event)

        existing = self._find_action_result(request.idempotency_key)
        if existing is not None:
            replay = replace(existing, action_id=request.action_id, reused_from_action_id=existing.action_id)
            self._append_action_result_event(replay, actor_id=request.actor_id)
            return replay

        started_at = now_utc()
        try:
            capability = Capability(str(request.capability))
            self.policy.require(authority, capability)
            current_state = self._current_state_hash()
            if (
                request.precondition_hash is not None
                and current_state is not None
                and request.precondition_hash != current_state
            ):
                raise PreconditionMismatch(
                    f"precondition mismatch: expected {request.precondition_hash}, observed {current_state}"
                )
            result = adapter.execute(request)
            observed_state = self._current_state_hash()
            finalized = self._finalize_action_result(
                result,
                started_at=started_at,
                completed_at=now_utc(),
                resulting_state_hash=observed_state,
            )
        except (AuthorityDenied, PreconditionMismatch) as exc:
            finalized = ActionResult(
                action_id=request.action_id,
                status=ActionStatus.DENIED
                if isinstance(exc, AuthorityDenied)
                else ActionStatus.FAILED,
                started_at=started_at,
                completed_at=now_utc(),
                resulting_state_hash=self._current_state_hash(),
                error=str(exc),
            )
        except RuntimeError as exc:
            finalized = ActionResult(
                action_id=request.action_id,
                status=ActionStatus.FAILED,
                started_at=started_at,
                completed_at=now_utc(),
                resulting_state_hash=self._current_state_hash(),
                error=str(exc),
            )

        self._append_action_result_event(finalized, actor_id=request.actor_id)
        return finalized

    def run_check(
        self,
        request: CheckRequest,
        *,
        verifier: VerificationAdapter,
    ) -> CheckResult:
        request_event = Event.create(
            stream_id=request.check_id,
            kind="check.requested",
            actor_id="verifier",
            payload=asdict(request),
            correlation_id=request.task_id,
        )
        self.ledger.append(request_event)

        if (
            request.target_state_hash is not None
            and self._current_state_hash() is not None
            and request.target_state_hash != self._current_state_hash()
        ):
            result = CheckResult(
                check_id=request.check_id,
                verdict=CheckVerdict.ERROR,
                started_at=now_utc(),
                completed_at=now_utc(),
                error=(
                    f"target state mismatch: expected {request.target_state_hash}, "
                    f"observed {self._current_state_hash()}"
                ),
            )
        else:
            result = verifier.run(request)
            result = self._finalize_check_result(result)

        event = Event.create(
            stream_id=request.check_id,
            kind="check.completed",
            actor_id="verifier",
            payload=asdict(result),
            causation_id=request_event.event_id,
            correlation_id=request.task_id,
        )
        self.ledger.append(event)
        self._apply_check_to_claims(request, result)
        return result

    # ------------------------------------------------------------------
    # Task acceptance: the only path to task.completed
    # ------------------------------------------------------------------

    def accept_task(self, request: AcceptanceRequest, *, authority: Authority) -> TaskCompletion:
        """Accept an exact artifact for a task; see codeai.acceptance.

        Raises AcceptanceRejected (after recording task.acceptance_rejected)
        when any reference fails validation.
        """
        return _accept_task(self, request, authority=authority)

    def task_completion(self, task_id: str) -> TaskCompletion:
        """Project task completion from the ledger without appending."""
        return project_task_completion(self, task_id)

    # ------------------------------------------------------------------
    # Working state: restart is reopening; resume is knowing what is safe next
    # ------------------------------------------------------------------

    def work_state(self, task_id: str) -> WorkState:
        """Project attempted, failed, unresolved and safe-next work; appends nothing."""
        return project_work_state(self, task_id)

    def resume_call(
        self, call_id: str, *, adapter: CognitionAdapter, model_config: Any | None = None
    ) -> RecordedCall:
        """Continue an interrupted call only where no provider effect is recorded.

        See codeai.workstate. Raises ResumeRefused for every other state.
        """
        return _resume_call(self, call_id, adapter=adapter, model_config=model_config)

    # ------------------------------------------------------------------
    # Cognition boundary: CallRequest -> CallResult -> artifact -> claims
    # ------------------------------------------------------------------

    def invoke_call(
        self,
        spec: CallSpec,
        *,
        adapter: CognitionAdapter,
    ) -> CallResult:
        """Record a cognitive observation. A model response is never truth."""
        request_event = Event.create(
            stream_id=spec.call_id,
            kind="call.requested",
            actor_id=spec.actor.actor_id,
            payload=self._call_spec_payload(spec),
            correlation_id=spec.task_id,
        )
        self.ledger.append(request_event)

        existing = self._find_call_result(spec.idempotency_key)
        if existing is not None:
            replay = replace(existing, call_id=spec.call_id)
            self._append_call_completed(replay, spec=spec, causation_id=request_event.event_id)
            return replay

        started_at = now_utc()
        try:
            result = adapter.invoke(spec)
            completed_at = now_utc()
            finalized = self._finalize_call_result(
                result,
                spec=spec,
                started_at=started_at,
                completed_at=completed_at,
            )
        except RuntimeError as exc:
            finalized = CallResult(
                call_id=spec.call_id,
                raw_output="",
                status="failed",
                error=str(exc),
                started_at=started_at,
                completed_at=now_utc(),
                provider=spec.actor.provider,
                model=spec.actor.model,
                model_version=spec.actor.version,
            )
        self._append_call_completed(finalized, spec=spec, causation_id=request_event.event_id)
        return finalized

    # ------------------------------------------------------------------
    # Recorded cognition: Task -> Logical Call -> Attempt(s) -> Observation
    # ------------------------------------------------------------------

    def invoke_recorded_call(
        self,
        spec: CallSpec,
        *,
        adapter: CognitionAdapter,
        max_attempts: int = 1,
        interpreter_version: str = INTERPRETER_V2,
        policy_version: str = ATTEMPT_POLICY_V2,
        model_config: Any | None = None,
    ) -> RecordedCall:
        """Execute one logical cognition call with explicit attempt accounting.

        Crash-window contract (clean-restart durability only, no atomicity
        claims): the sequence per attempt is ``attempt.started`` -> provider
        effect -> response-body blob stored -> ``attempt.observed`` appended
        -> ``attempt.interpreted`` appended -> ``attempt.retry_decided``
        appended -> legacy mixed envelope stored -> ``attempt.completed``
        appended, then ``call.status_decided`` and ``call.completed``.
        Ambiguity windows, from newest evidence backwards:
        - crash after ``attempt.interpreted`` but before ``attempt.retry_decided``:
          interpretation exists, decision absent (do not infer the decision;
          the deterministic policy can re-derive it, but re-derivation is not
          silently recorded as history);
        - crash after ``attempt.observed`` but before ``attempt.interpreted``:
          transport observation exists without interpretation; recoverable by
          re-running the recorded interpreter version, transparently;
        - decision says retry, crash before the next ``attempt.started``:
          intent to retry exists with no second provider effect;
        - decision says no retry, crash before ``attempt.completed``:
          policy intent exists though the compatibility projection is absent;
        - crash after body-blob storage but before ``attempt.observed``:
          an unreferenced content-addressed response body may exist;
        - crash between the provider effect and body storage: only
          ``attempt.started`` exists and the observation is unknown
          (the provider may or may not have served the request).
        ``call.manifest`` is appended before any attempt, so an interrupted
        call is always inspectable as UNRESOLVED. Exactly-once execution is
        NOT promised; retries create new attempts, never new tasks.

        Replay: when the idempotency key matches a completed recorded call
        with the same request fingerprint, no provider effect occurs. Only
        ``call.requested`` and ``call.replayed`` are appended and the
        original RecordedCall is returned marked replayed. A key whose
        fingerprint differs raises IdempotencyConflictError before any
        effect. Keys matching only legacy (non-recorded) completions, or no
        terminal completion at all, execute fresh.

        Cognition completion (!= task completion): a "succeeded" recorded
        call only means the cognition operation produced an output. It says
        nothing about correctness, support, authorization, or verification.

        No automatic retry policy lives here: ``max_attempts`` defaults to 1.
        Passing ``max_attempts > 1`` explicitly bounds re-attempts; whether
        any single attempt is retried is decided by the versioned attempt
        policy from that attempt's interpretation, never by heuristics here.
        """
        recorded, _, _ = self._invoke_recorded_call_detailed(
            spec,
            adapter=adapter,
            max_attempts=max_attempts,
            interpreter_version=interpreter_version,
            policy_version=policy_version,
            model_config=model_config,
        )
        return recorded

    def _invoke_recorded_call_detailed(
        self,
        spec: CallSpec,
        *,
        adapter: CognitionAdapter,
        max_attempts: int = 1,
        interpreter_version: str = INTERPRETER_V2,
        policy_version: str = ATTEMPT_POLICY_V2,
        model_config: Any | None = None,
    ) -> tuple[RecordedCall, CallResult, bool]:
        """Recorded invocation returning (recorded call, final result, replayed)."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if spec.context_render is not None:
            # Rendering is bound to a request only through prepare(); refuse
            # before any event rather than record an unbindable call.
            if spec.context_render not in KNOWN_RENDERERS:
                raise ValueError(f"unknown context renderer version: {spec.context_render!r}")
            if not callable(getattr(adapter, "prepare", None)):
                raise ValueError("context rendering requires an adapter with prepare()/send()")
        # The runtime renders; a caller-supplied rendering is never trusted.
        spec = replace(spec, rendered_context=None)
        request_event = Event.create(
            stream_id=spec.call_id,
            kind="call.requested",
            actor_id=spec.actor.actor_id,
            payload=self._call_spec_payload(spec),
            correlation_id=spec.task_id,
        )
        self.ledger.append(request_event)

        rendered = None
        if spec.context_render is not None:
            try:
                rendered = render_context(
                    spec.context,
                    ledger=self.ledger,
                    artifact_store=self.artifact_store,
                    version=spec.context_render,
                )
            except ContextResolutionError as exc:
                self._append_preparation_failed(
                    spec, request_event, stage="context_render",
                    details={"unresolved": list(exc.unresolved)},
                )
                raise
            spec = replace(spec, rendered_context=rendered)

        replayed_completion = self._find_recorded_completion(spec.idempotency_key)
        if replayed_completion is not None:
            original_call_id, completed_payload = replayed_completion
            original = self.get_recorded_call(original_call_id)
            assert original is not None
            self._check_replay_fingerprint(spec, original)
            self.ledger.append(
                Event.create(
                    stream_id=spec.call_id,
                    kind="call.replayed",
                    actor_id=spec.actor.actor_id,
                    payload={
                        "requested_call_id": spec.call_id,
                        "task_id": spec.task_id,
                        "idempotency_key": spec.idempotency_key,
                        "original_call_id": original_call_id,
                        "original_attempt_ids": [
                            attempt.attempt_id for attempt in original.attempts
                        ],
                        "replayed_at": now_utc(),
                        "replay_reason": "idempotency-key hit on completed recorded call",
                        "original_call_status": original.status,
                    },
                    causation_id=request_event.event_id,
                    correlation_id=spec.task_id,
                )
            )
            result = replace(
                self._call_result_from_payload(completed_payload),
                call_id=original_call_id,
                replayed=True,
            )
            return replace(original, replayed=True), result, True

        # Prepare once: adapters exposing prepare()/send() produce a single
        # prepared request that is both recorded (manifest) and sent (every
        # attempt). Unknown/invalid controls raise here, before any event or
        # provider effect. Adapters without this capability use the legacy
        # invoke path unchanged.
        prepare = getattr(adapter, "prepare", None)
        prepared = prepare(spec) if callable(prepare) else None
        input_layout: tuple[dict[str, Any], ...] = ()
        if rendered is not None and prepared is not None:
            composed, input_layout = compose_model_input(
                spec.instruction, rendered, spec.context.prompt
            )
            if prepared_input_text(prepared.body) != composed:
                self._append_preparation_failed(
                    spec, request_event, stage="render_binding",
                    details={"rendered_context_sha256": rendered.sha256},
                )
                raise RenderBindingError("prepared request does not carry the rendered model input")
            if self.artifact_store is not None:
                self.artifact_store.store_text(
                    rendered.text, media_type="text/plain", artifact_type="rendered_context"
                )
        if prepared is not None:
            manifest = self._build_manifest_from_prepared(spec, prepared, input_layout=input_layout)
        else:
            manifest = self._build_manifest(spec, adapter=adapter, model_config=model_config)
        self.ledger.append(
            Event.create(
                stream_id=spec.call_id,
                kind="call.manifest",
                actor_id=spec.actor.actor_id,
                payload=_manifest_payload(manifest),
                causation_id=request_event.event_id,
                correlation_id=spec.task_id,
            )
        )

        attempts: list[AttemptRecord] = []
        interpretations: list[AttemptInterpretation] = []
        decisions: list[AttemptDecision] = []
        last_result: CallResult | None = None
        for attempt_index in range(1, max_attempts + 1):
            attempt_id = str(uuid.uuid4())
            started_at = now_utc()
            self.ledger.append(
                Event.create(
                    stream_id=attempt_id,
                    kind="attempt.started",
                    actor_id=spec.actor.actor_id,
                    payload={
                        "attempt_id": attempt_id,
                        "call_id": spec.call_id,
                        "task_id": spec.task_id,
                        "attempt_index": attempt_index,
                        "started_at": started_at,
                        "provider": manifest.provider,
                        "resolved_model_id": manifest.resolved_model_id,
                    },
                    causation_id=request_event.event_id,
                    correlation_id=spec.task_id,
                )
            )
            try:
                if prepared is not None:
                    observed = adapter.send(prepared)
                else:
                    observed = adapter.invoke(spec)
            except RuntimeError as exc:
                observed = CallResult(
                    call_id=spec.call_id,
                    raw_output="",
                    input_tokens=None,
                    output_tokens=None,
                    usage_source=UsageSource.UNAVAILABLE.value,
                    error_kind="adapter_exception",
                    pricing_version=PRICING_VERSION,
                    cost_source=CostSource.UNKNOWN.value,
                    normalizer_version=NORMALIZER_VERSION,
                    status="failed",
                    error=str(exc),
                    provider=manifest.provider,
                    model=manifest.resolved_model_id,
                    started_at=started_at,
                    completed_at=now_utc(),
                )
            finished_at = now_utc()
            attempt, interpretation, decision = self._record_attempt(
                spec,
                manifest=manifest,
                result=observed,
                attempt_id=attempt_id,
                attempt_index=attempt_index,
                started_at=started_at,
                finished_at=finished_at,
                causation_id=request_event.event_id,
                interpreter_version=interpreter_version,
                policy_version=policy_version,
                max_attempts=max_attempts,
            )
            attempts.append(attempt)
            interpretations.append(interpretation)
            decisions.append(decision)
            last_result = self._interpret_attempt(spec, manifest=manifest, attempt=attempt, result=observed)
            if decision.executed:
                continue
            break

        assert last_result is not None
        call_status, status_reason = _decide_call_status(attempts, interpretations, decisions)
        self.ledger.append(
            Event.create(
                stream_id=spec.call_id,
                kind="call.status_decided",
                actor_id=spec.actor.actor_id,
                payload={
                    "call_id": spec.call_id,
                    "task_id": spec.task_id,
                    "status": call_status,
                    "reason": status_reason,
                    "interpretation_ids": [i.interpretation_id for i in interpretations],
                    "policy_version": policy_version,
                },
                causation_id=request_event.event_id,
                correlation_id=spec.task_id,
            )
        )
        totals = _aggregate_totals(attempts)
        self._append_recorded_call_completed(
            last_result,
            spec=spec,
            manifest=manifest,
            attempts=attempts,
            call_status=call_status,
            totals=totals,
            causation_id=request_event.event_id,
            decision_basis_interpretation_id=(
                interpretations[-1].interpretation_id if interpretations else None
            ),
            decision_policy_version=policy_version,
        )
        recorded = RecordedCall(
            call_id=spec.call_id,
            task_id=spec.task_id,
            chamber=manifest.chamber,
            manifest=manifest,
            attempts=tuple(attempts),
            status=call_status,
            total_input_tokens=totals["input_tokens"],
            total_output_tokens=totals["output_tokens"],
            total_cost_usd=totals["cost_usd"],
        )
        return recorded, last_result, False

    def get_recorded_call(self, call_id: str) -> RecordedCall | None:
        """Reconstruct a logical call (manifest + attempts + status) from the ledger.

        Falls back through call.replayed: a requested call id that only ever
        replayed resolves to its original recorded call.
        """
        manifest_event = next(
            (
                event
                for event in self.ledger.events_by_kind(("call.manifest",))
                if event.stream_id == call_id
                or str(event.payload.get("call_id", "")) == call_id
            ),
            None,
        )
        attempt_events = [
            event
            for event in self.ledger.events_by_kind(("attempt.completed",))
            if str(event.payload.get("call_id", "")) == call_id
        ]
        if manifest_event is None:
            # No manifest: either unknown, or a requested id that only replayed.
            return self._resolve_replay(call_id)
        manifest = _manifest_from_payload(manifest_event.payload)
        attempts = tuple(
            sorted(
                (_attempt_from_payload(event.payload) for event in attempt_events),
                key=lambda attempt: attempt.attempt_index,
            )
        )
        completed = next(
            (
                event
                for event in self.ledger.events_by_kind(("call.completed",))
                if str(event.payload.get("call_id", "")) == call_id
            ),
            None,
        )
        if completed is not None and completed.payload.get("call_status"):
            status = str(completed.payload["call_status"])
        elif attempts:
            status = (
                LogicalCallStatus.SUCCEEDED.value
                if attempts[-1].status == AttemptStatus.SUCCEEDED.value
                else LogicalCallStatus.FAILED.value
            )
        else:
            status = LogicalCallStatus.UNRESOLVED.value
        totals = _aggregate_totals(list(attempts))
        return RecordedCall(
            call_id=call_id,
            task_id=manifest.task_id,
            chamber=manifest.chamber,
            manifest=manifest,
            attempts=attempts,
            status=status,
            total_input_tokens=totals["input_tokens"],
            total_output_tokens=totals["output_tokens"],
            total_cost_usd=totals["cost_usd"],
        )

    def list_call_attempts(self, call_id: str) -> tuple[AttemptRecord, ...]:
        recorded = self.get_recorded_call(call_id)
        return () if recorded is None else recorded.attempts

    def _resolve_replay(self, requested_call_id: str) -> RecordedCall | None:
        """Resolve a requested call id that only ever replayed to its original."""
        for event in self.ledger.events_by_kind(("call.replayed",)):
            if str(event.payload.get("requested_call_id", "")) == requested_call_id:
                original_call_id = str(event.payload.get("original_call_id", ""))
                if original_call_id and original_call_id != requested_call_id:
                    return self.get_recorded_call(original_call_id)
                return None
        return None

    def _find_recorded_completion(
        self, idempotency_key: str
    ) -> tuple[str, dict[str, Any]] | None:
        """Locate a completed RECORDED call by idempotency key, ledger order.

        Only call.completed events with a matching call.manifest qualify:
        legacy (non-recorded) completions and keys without terminal evidence
        are misses, so replay never crosses execution contracts and never
        fabricates completion for crashed/incomplete calls.
        """
        manifests = {
            str(event.payload.get("call_id", event.stream_id))
            for event in self.ledger.events_by_kind(("call.manifest",))
        }
        for event in self.ledger.events_by_kind(("call.completed",)):
            if event.payload.get("idempotency_key") != idempotency_key:
                continue
            call_id = str(event.payload.get("call_id", event.stream_id))
            if call_id in manifests:
                return call_id, dict(event.payload)
        return None

    def _check_replay_fingerprint(self, spec: CallSpec, original: RecordedCall) -> None:
        """Reject same-key, materially-different requests before any effect."""
        rendered_sha = spec.rendered_context.sha256 if spec.rendered_context is not None else None
        expected = (
            original.manifest.prompt_hash,
            original.manifest.chamber,
            original.manifest.requested_model,
            original.manifest.rendered_context_sha256,
        )
        actual = (*_request_fingerprint(spec), rendered_sha)
        if actual != expected:
            dimensions = ("prompt_hash", "chamber", "requested_model", "rendered_context_sha256")
            mismatched = sorted(
                dimension
                for dimension, want, got in zip(dimensions, expected, actual)
                if want != got
            )
            raise IdempotencyConflictError(
                f"idempotency key {spec.idempotency_key!r} matches completed call "
                f"{original.call_id!r} but the request differs in: {', '.join(mismatched)}"
            )

    def _append_preparation_failed(
        self,
        spec: CallSpec,
        request_event: Event,
        *,
        stage: str,
        details: Mapping[str, object],
    ) -> None:
        """Record a failure after call.requested and before any provider effect.

        No manifest or attempt follows, so the call can never read as succeeded.
        """
        self.ledger.append(
            Event.create(
                stream_id=spec.call_id,
                kind="call.preparation_failed",
                actor_id=spec.actor.actor_id,
                payload={
                    "call_id": spec.call_id,
                    "task_id": spec.task_id,
                    "stage": stage,
                    "context_package_id": spec.context.package_id,
                    "context_render_version": spec.context_render,
                    "provider_effect": False,
                    **dict(details),
                },
                causation_id=request_event.event_id,
                correlation_id=spec.task_id,
            )
        )

    # ---------------- recorded-call internals ----------------

    def _build_manifest(
        self,
        spec: CallSpec,
        *,
        adapter: CognitionAdapter,
        model_config: Any | None = None,
    ) -> CallManifest:
        chamber = spec.chamber
        logical = spec.logical_model or chamber
        requested_model = logical or spec.actor.model
        provider = spec.actor.provider
        resolved_model_id = spec.actor.model
        if model_config is not None and logical is not None:
            try:
                mapping = model_config.resolve_chamber(logical)  # type: ignore[attr-defined]
            except AttributeError:
                mapping = model_config.resolve(logical)
            provider = mapping.adapter if mapping.adapter else provider
            resolved_model_id = mapping.model or resolved_model_id
        # Pre-execution: revision genuinely unknown until a provider reports
        # one. Never synthesize from timestamps.
        requested_parameters = sanitize_effective_params(dict(spec.parameters))
        effective = self._adapter_effective_params(spec, adapter)
        prompt_hash = hashlib.sha256(
            f"{spec.instruction}\0{spec.context.prompt}".encode()
        ).hexdigest()
        return CallManifest(
            call_id=spec.call_id,
            task_id=spec.task_id,
            chamber=chamber,
            requested_model=requested_model,
            provider=provider,
            resolved_model_id=resolved_model_id,
            provider_revision=None,
            revision_source=None,
            pricing_version=PRICING_VERSION,
            context_package_id=spec.context.package_id,
            prompt_hash=prompt_hash,
            requested_parameters=requested_parameters,
            effective_parameters=effective,
            created_at=now_utc(),
        )

    def _build_manifest_from_prepared(
        self,
        spec: CallSpec,
        prepared: PreparedCognitionRequest,
        *,
        input_layout: tuple[dict[str, Any], ...] = (),
    ) -> CallManifest:
        """Build the call manifest from the single prepared request.

        The manifest records the exact effective controls, routing, omission
        record, and body identity of the object that will actually be sent —
        not a reconstruction. Requested parameters keep the raw caller map
        (sanitized); requested_controls is the declared logical view.
        """
        chamber = spec.chamber
        logical = spec.logical_model or chamber
        prompt_hash = hashlib.sha256(
            f"{spec.instruction}\0{spec.context.prompt}".encode()
        ).hexdigest()
        rendered = spec.rendered_context
        return CallManifest(
            call_id=spec.call_id,
            task_id=spec.task_id,
            chamber=chamber,
            requested_model=logical or spec.actor.model,
            provider=prepared.gateway,
            resolved_model_id=prepared.model,
            provider_revision=None,
            revision_source=None,
            pricing_version=PRICING_VERSION,
            context_package_id=spec.context.package_id,
            prompt_hash=prompt_hash,
            requested_parameters=sanitize_effective_params(dict(spec.parameters)),
            effective_parameters=_jsonable_mapping(prepared.recorded_effective()),
            requested_controls=dict(prepared.requested_controls),
            omitted_unsupported=tuple(prepared.omitted_unsupported),
            defaulted_parameters=dict(prepared.defaulted_controls),
            request_plan_version=prepared.plan_version,
            request_body_sha256=prepared.body_sha256,
            created_at=now_utc(),
            context_render_version=rendered.version if rendered is not None else None,
            rendered_context_sha256=rendered.sha256 if rendered is not None else None,
            rendered_items=(
                tuple(asdict(item) for item in rendered.items) if rendered is not None else ()
            ),
            input_layout=tuple(dict(part) for part in input_layout),
        )

    @staticmethod
    def _adapter_effective_params(
        spec: CallSpec, adapter: CognitionAdapter
    ) -> dict[str, object]:
        describe = getattr(adapter, "effective_request", None)
        if callable(describe):
            try:
                effective = describe(spec)
            except (ValueError, TypeError, AttributeError):
                effective = dict(spec.parameters)
            if isinstance(effective, Mapping):
                return sanitize_effective_params(dict(effective))
        return sanitize_effective_params(dict(spec.parameters))

    def _record_attempt(
        self,
        spec: CallSpec,
        *,
        manifest: CallManifest,
        result: CallResult,
        attempt_id: str,
        attempt_index: int,
        started_at: str,
        finished_at: str,
        causation_id: str | None,
        interpreter_version: str = INTERPRETER_V2,
        policy_version: str = ATTEMPT_POLICY_V2,
        max_attempts: int = 1,
    ) -> tuple[AttemptRecord, AttemptInterpretation, AttemptDecision]:
        provider = result.provider or manifest.provider
        resolved_model_id = result.model or manifest.resolved_model_id
        # Revision epistemics: only a provider-reported version counts as
        # observed. A version string identical to the model id carries no
        # extra revision information -> unknown (None).
        reported = result.model_version
        if reported and reported != resolved_model_id:
            provider_revision: str | None = reported
            revision_source: str | None = "reported"
        else:
            provider_revision = None
            revision_source = None
        usage_source = (result.usage_source or "").lower()
        if usage_source not in ("measured", "estimated", "unavailable"):
            usage_source = (
                "measured"
                if (result.input_tokens is not None or result.output_tokens is not None)
                else "unavailable"
            )
        if usage_source == UsageSource.UNAVAILABLE.value:
            in_tokens: int | None = None
            out_tokens: int | None = None
        else:
            in_tokens = result.input_tokens
            out_tokens = result.output_tokens
        usage = Usage(
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            source=UsageSource(usage_source),
        )
        cost = estimate_cost_usd(resolved_model_id, in_tokens, out_tokens)
        # Authoritative derivation: the pricing table (at pricing_version)
        # owns rate interpretation. An adapter-supplied cost figure is NOT
        # trusted here: for a model absent from the pricing table the cost
        # stays unknown (None), never an adapter-claimed number and never 0.
        # Adapter cost figures remain visible on the legacy CallResult path.
        pricing_version = result.pricing_version or PRICING_VERSION
        cost_source = (
            CostSource.ESTIMATED.value if cost is not None else CostSource.UNKNOWN.value
        )
        latency_ms = result.latency_ms
        if latency_ms is None:
            latency_ms = None
        effective = (
            sanitize_effective_params(dict(result.effective_parameters))
            if result.effective_parameters
            else dict(manifest.effective_parameters)
        )
        # Transport evidence first: body bytes (if any) -> blob, then the
        # attempt.observed event. Interpretation follows observation;
        # decisions follow interpretation.
        observed_event_id, body_ref = self._store_transport_observation(
            spec=spec,
            result=result,
            attempt_id=attempt_id,
            attempt_index=attempt_index,
            provider=provider,
            causation_id=causation_id,
        )
        interpretation = interpret_attempt(
            _interpretation_input(spec, manifest, result, attempt_id),
            version=interpreter_version,
            observation_event_id=observed_event_id,
            response_body_artifact=body_ref,
        )
        self._append_attempt_interpreted(
            spec=spec, interpretation=interpretation, causation_id=causation_id
        )
        decision_name, decision_reason = decide_attempt(
            interpretation,
            policy_version=policy_version,
            has_output_text=bool(result.raw_output),
        )
        executed = decision_name == "retry" and attempt_index < max_attempts
        decision = AttemptDecision(decision=decision_name, reason=decision_reason, executed=executed)
        self.ledger.append(
            Event.create(
                stream_id=attempt_id,
                kind="attempt.retry_decided",
                actor_id=spec.actor.actor_id,
                payload={
                    "attempt_id": attempt_id,
                    "call_id": spec.call_id,
                    "task_id": spec.task_id,
                    "decision": decision.decision,
                    "reason": decision.reason,
                    "executed": decision.executed,
                    "interpretation_id": interpretation.interpretation_id,
                    "policy_version": policy_version,
                },
                causation_id=causation_id,
                correlation_id=spec.task_id,
            )
        )
        # Compatibility projections: status/error_kind mirror the
        # execution-time interpretation + policy, exactly as the old
        # classifier produced (accept->succeeded, retry->transient_failure,
        # terminal->failed). Authoritative provenance lives in the
        # interpreted/decided events, not here.
        if decision.decision == "accept":
            status = AttemptStatus.SUCCEEDED.value
        elif decision.decision == "retry":
            status = AttemptStatus.TRANSIENT_FAILURE.value
        else:
            status = AttemptStatus.FAILED.value
        error_kind = interpretation.error_kind
        raw_artifact = self._store_raw_observation(
            spec=spec,
            manifest=manifest,
            result=result,
            attempt_id=attempt_id,
            attempt_index=attempt_index,
            provider=provider,
            resolved_model_id=resolved_model_id,
            provider_revision=provider_revision,
            revision_source=revision_source,
            usage=usage,
            pricing_version=pricing_version,
            cost_usd=cost,
            cost_source=cost_source,
            status=status,
            error_kind=error_kind,
            effective=effective,
        )
        attempt = AttemptRecord(
            attempt_id=attempt_id,
            call_id=spec.call_id,
            task_id=spec.task_id,
            attempt_index=attempt_index,
            started_at=result.started_at or started_at,
            finished_at=result.completed_at or finished_at,
            latency_ms=latency_ms,
            provider=provider,
            resolved_model_id=resolved_model_id,
            provider_revision=provider_revision,
            revision_source=revision_source,
            protocol=result.protocol,
            provider_request_id=result.provider_call_id or result.request_id,
            status=status,
            error_kind=error_kind,
            error=result.error,
            usage=usage,
            pricing_version=pricing_version,
            cost_usd=cost,
            cost_source=cost_source,
            currency="USD",
            raw_artifact=raw_artifact,
            raw_observation_kind=result.raw_observation_kind or RAW_OBSERVATION_KIND,
            normalizer_version=result.normalizer_version or NORMALIZER_VERSION,
            effective_parameters=effective,
            interpretation_id=interpretation.interpretation_id,
            policy_version=policy_version,
        )
        self.ledger.append(
            Event.create(
                stream_id=attempt_id,
                kind="attempt.completed",
                actor_id=spec.actor.actor_id,
                payload=_attempt_payload(attempt, fingerprint=result.fingerprint),
                causation_id=causation_id,
                correlation_id=spec.task_id,
            )
        )
        return attempt, interpretation, decision

    def _store_raw_observation(
        self,
        *,
        spec: CallSpec,
        manifest: CallManifest,
        result: CallResult,
        attempt_id: str,
        attempt_index: int,
        provider: str | None,
        resolved_model_id: str | None,
        provider_revision: str | None,
        revision_source: str | None,
        usage: Usage,
        pricing_version: str | None,
        cost_usd: float | None,
        cost_source: str,
        status: str,
        error_kind: str | None,
        effective: dict[str, object],
    ) -> ArtifactRef | None:
        if self.artifact_store is None:
            return None
        observation = {
            "kind": result.raw_observation_kind or RAW_OBSERVATION_KIND,
            "call_id": spec.call_id,
            "attempt_id": attempt_id,
            "attempt_index": attempt_index,
            "task_id": spec.task_id,
            "provider": provider,
            "protocol": result.protocol,
            "model": resolved_model_id,
            "provider_revision": provider_revision,
            "revision_source": revision_source,
            "provider_call_id": result.provider_call_id,
            "output_text": result.raw_output,
            "provider_response": dict(result.raw_payload) if result.raw_payload else None,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "source": usage.source.value,
            },
            "status": status,
            "error_kind": error_kind,
            "error": result.error,
            "pricing_version": pricing_version,
            "cost_usd": cost_usd,
            "cost_source": cost_source,
            "currency": "USD",
            "normalizer_version": result.normalizer_version or NORMALIZER_VERSION,
            "effective_parameters": effective,
        }
        content = json.dumps(observation, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return self.artifact_store.store_bytes(
            content,
            media_type="application/json",
            artifact_type="raw_provider_observation",
        )

    def _store_transport_observation(
        self,
        *,
        spec: CallSpec,
        result: CallResult,
        attempt_id: str,
        attempt_index: int,
        provider: str | None,
        causation_id: str | None,
    ) -> tuple[str | None, ArtifactRef | None]:
        """Persist pure transport evidence for one attempt, if the adapter saw any.

        Stores the exact response body bytes (when they exist) as a
        content-addressed blob, then appends attempt.observed with transport
        metadata and the blob reference. Returns (observed_event_id, body_ref).
        Adapters without transport evidence (fakes, legacy dict fixtures)
        produce no event: absent observation means unavailable under this
        schema, never an empty observation.
        """
        transport = result.transport
        if transport is None or self.artifact_store is None:
            if transport is None:
                return None, None
            # No artifact store: observation cannot be preserved; record the
            # metadata anyway so the transport outcome is not silently lost.
            event_id = self._append_attempt_observed(
                spec=spec,
                result=result,
                attempt_id=attempt_id,
                attempt_index=attempt_index,
                provider=provider,
                body_ref=None,
                byte_length=None,
                causation_id=causation_id,
            )
            return event_id, None
        body_ref = None
        byte_length = None
        if transport.body is not None:
            body_ref = self.artifact_store.store_bytes(
                transport.body,
                media_type=transport.content_type or "application/octet-stream",
                artifact_type="provider_response_body",
            )
            record = self.ledger.read_artifact(body_ref.artifact_id)
            byte_length = record.byte_length if record is not None else len(transport.body)
        event_id = self._append_attempt_observed(
            spec=spec,
            result=result,
            attempt_id=attempt_id,
            attempt_index=attempt_index,
            provider=provider,
            body_ref=body_ref,
            byte_length=byte_length,
            causation_id=causation_id,
        )
        return event_id, body_ref

    def _append_attempt_observed(
        self,
        *,
        spec: CallSpec,
        result: CallResult,
        attempt_id: str,
        attempt_index: int,
        provider: str | None,
        body_ref: ArtifactRef | None,
        byte_length: int | None,
        causation_id: str | None,
    ) -> str:
        transport = result.transport
        assert transport is not None
        event = Event.create(
            stream_id=attempt_id,
            kind="attempt.observed",
            actor_id=spec.actor.actor_id,
            payload={
                "attempt_id": attempt_id,
                "call_id": spec.call_id,
                "task_id": spec.task_id,
                "attempt_index": attempt_index,
                "provider": provider,
                "protocol": result.protocol,
                "endpoint": transport.endpoint,
                "transport_outcome": transport.outcome,
                "http_status": transport.status_code,
                "observed_at": transport.observed_at or now_utc(),
                "response_body_artifact": None
                if body_ref is None
                else {
                    "artifact_id": body_ref.artifact_id,
                    "sha256": body_ref.sha256,
                    "media_type": body_ref.media_type,
                    "byte_length": byte_length,
                },
                "content_type": transport.content_type,
                "headers": _allowlist_response_headers(transport.headers),
                "exception_type": transport.exception_type,
            },
            causation_id=causation_id,
            correlation_id=spec.task_id,
        )
        self.ledger.append(event)
        return event.event_id

    def get_attempt_observation(self, attempt_id: str) -> dict[str, Any] | None:
        """Return the persisted transport metadata for one attempt, if any.

        Returns the attempt.observed payload (transport metadata plus the
        response-body artifact reference). Body bytes themselves are loaded
        through the artifact store by artifact_id; they are not inlined here.
        """
        for event in self.ledger.events_by_kind(("attempt.observed",)):
            if event.stream_id == attempt_id or str(event.payload.get("attempt_id")) == attempt_id:
                return dict(event.payload)
        return None

    def _append_attempt_interpreted(
        self,
        *,
        spec: CallSpec,
        interpretation: AttemptInterpretation,
        causation_id: str | None,
    ) -> str:
        event = Event.create(
            stream_id=interpretation.attempt_id,
            kind="attempt.interpreted",
            actor_id=spec.actor.actor_id,
            payload=_interpretation_payload(interpretation),
            causation_id=causation_id,
            correlation_id=spec.task_id,
        )
        self.ledger.append(event)
        return event.event_id

    def interpretations_for_attempt(self, attempt_id: str) -> list[AttemptInterpretation]:
        """All persisted interpretations for one attempt, in ledger order."""
        return [
            _interpretation_from_payload(event.payload)
            for event in self.ledger.events_by_kind(("attempt.interpreted",))
            if event.stream_id == attempt_id
            or str(event.payload.get("attempt_id", "")) == attempt_id
        ]

    def interpret_attempt_as(
        self, attempt_id: str, *, version: str
    ) -> AttemptInterpretation | None:
        """Recompute an interpretation of preserved evidence without appending.

        Pure projection: loads the attempt.observed event (when present),
        the response-body bytes (when referenced), and the compatibility
        fields of attempt.completed, then runs the named interpreter version.
        Returns None when the attempt has no completed record to project from.
        Raises ObservationUnavailable when the observation references response
        bytes that are missing or corrupt: the derived envelope is never used
        in their place.
        History is never modified; the result carries a fresh
        interpretation_id that belongs to no ledger event.
        """
        completed = next(
            (
                event
                for event in self.ledger.events_by_kind(("attempt.completed",))
                if str(event.payload.get("attempt_id", "")) == attempt_id
            ),
            None,
        )
        if completed is None:
            return None
        finished = completed.payload
        observed = self.get_attempt_observation(attempt_id)
        body: bytes | None = None
        if observed is not None:
            ref = observed.get("response_body_artifact")
            if isinstance(ref, dict) and self.artifact_store is not None:
                # The observation names these bytes; never fall back to the
                # CodeAI-derived envelope when they cannot be read.
                try:
                    body = self.artifact_store.read_bytes(str(ref["artifact_id"]))
                except FileNotFoundError as exc:
                    raise ObservationUnavailable(
                        attempt_id, "response body bytes are missing"
                    ) from exc
                except ArtifactCorruptionError as exc:
                    raise ObservationUnavailable(
                        attempt_id, "response body bytes do not match their recorded sha256"
                    ) from exc
                except KeyError as exc:
                    raise ObservationUnavailable(
                        attempt_id, "response body reference is malformed"
                    ) from exc
        output_text = ""
        parsed: dict[str, Any] = {}
        raw_ref = finished.get("raw_artifact")
        if isinstance(raw_ref, dict) and self.artifact_store is not None:
            try:
                envelope = json.loads(
                    self.artifact_store.read_text(str(raw_ref["artifact_id"]))
                )
                if isinstance(envelope, dict):
                    output_text = str(envelope.get("output_text", "") or "")
                    provider_response = envelope.get("provider_response")
                    if isinstance(provider_response, dict):
                        parsed = provider_response
            except (FileNotFoundError, KeyError, RuntimeError, ValueError):
                pass
        if observed is not None:
            transport_outcome: str | None = str(observed.get("transport_outcome"))
            http_status = observed.get("http_status")
            http_status = None if http_status is None else int(http_status)
            exception_type = observed.get("exception_type")
            observation_event_id = next(
                (
                    event.event_id
                    for event in self.ledger.events_by_kind(("attempt.observed",))
                    if event.stream_id == attempt_id
                    or str(event.payload.get("attempt_id", "")) == attempt_id
                ),
                None,
            )
            ref = observed.get("response_body_artifact")
            body_artifact = None
            if isinstance(ref, dict):
                try:
                    body_artifact = ArtifactRef(
                        artifact_id=str(ref["artifact_id"]),
                        sha256=str(ref["sha256"]),
                        media_type=str(ref.get("media_type", "application/octet-stream")),
                        uri=ref.get("uri"),
                    )
                except KeyError:
                    body_artifact = None
        else:
            transport_outcome = None
            http_status = None
            exception_type = None
            observation_event_id = None
            body_artifact = None
        error_text = finished.get("error")
        entry = InterpretationInput(
            attempt_id=attempt_id,
            call_id=str(finished.get("call_id", "")),
            task_id=str(finished.get("task_id", "")),
            protocol=finished.get("protocol"),
            transport_outcome=transport_outcome,
            http_status=http_status,
            exception_type=exception_type,
            failure_message=str(error_text) if error_text is not None else None,
            body_bytes=body,
            parsed=parsed,
            output_text=output_text,
            adapter_status="failed" if error_text else "succeeded",
            adapter_error_kind=finished.get("error_kind"),
        )
        return interpret_attempt(
            entry,
            version=version,
            observation_event_id=observation_event_id,
            response_body_artifact=body_artifact,
        )

    def interpret_usage_as(self, attempt_id: str, *, version: str) -> dict[str, Any] | None:
        """Project usage semantics over preserved evidence without appending.

        Prefers the attempt's preserved transport body; falls back to the
        legacy decoded envelope. Returns the evidence class and source hash
        with the interpretation, or None when the attempt has no completed
        record. Recorded usage and decisions are never modified.
        """
        from .usage_semantics import interpret_usage

        completed = next(
            (
                event
                for event in self.ledger.events_by_kind(("attempt.completed",))
                if str(event.payload.get("attempt_id", "")) == attempt_id
            ),
            None,
        )
        if completed is None:
            return None
        parsed: dict[str, Any] = {}
        evidence_class = "none"
        source_sha256: str | None = None
        observed = self.get_attempt_observation(attempt_id)
        ref = observed.get("response_body_artifact") if observed else None
        if isinstance(ref, dict) and self.artifact_store is not None:
            try:
                loaded = json.loads(self.artifact_store.read_bytes(str(ref["artifact_id"])))
                source_sha256 = str(ref.get("sha256"))
                if isinstance(loaded, dict):
                    parsed, evidence_class = loaded, "transport_body"
                else:
                    evidence_class = "transport_body_not_object"
            except (FileNotFoundError, KeyError, RuntimeError, ValueError):
                evidence_class = "transport_body_unreadable"
        if evidence_class == "none":
            raw_ref = completed.payload.get("raw_artifact")
            if isinstance(raw_ref, dict) and self.artifact_store is not None:
                try:
                    envelope = json.loads(self.artifact_store.read_text(str(raw_ref["artifact_id"])))
                    provider_response = (
                        envelope.get("provider_response") if isinstance(envelope, dict) else None
                    )
                    if isinstance(provider_response, dict):
                        parsed, evidence_class = provider_response, "legacy_decoded_envelope"
                        source_sha256 = str(raw_ref.get("sha256"))
                except (FileNotFoundError, KeyError, RuntimeError, ValueError):
                    pass
        return {
            "attempt_id": attempt_id,
            "evidence_class": evidence_class,
            "source_sha256": source_sha256,
            "interpretation": interpret_usage(
                completed.payload.get("protocol"), parsed, version=version
            ),
        }

    # ------------------------------------------------------------------
    # Reinterpretation: preserve once, interpret many times (Stage 17)
    # ------------------------------------------------------------------

    def reinterpret_attempt(
        self, attempt_id: str, *, interpreter_version: str, actor_id: str = "runtime"
    ) -> AttemptInterpretation:
        """Interpret an attempt again from its preserved observation, and record it.

        The only evidence is the ``attempt.observed`` event and the response bytes
        it names, read with integrity checks. The CodeAI-derived envelope and
        adapter messages are not observations and are never used. No completion
        record is needed, so an attempt interrupted after observation can be
        interpreted. One interpretation is recorded per version: a repeat
        returns it and appends nothing. Raises ObservationUnavailable, appending
        nothing, when the observation or its bytes cannot be read. No provider
        is contacted.
        """
        interpretation, _text, observed_event = self._interpret_from_observation(
            attempt_id, interpreter_version
        )
        existing = [
            item for item in self.interpretations_for_attempt(attempt_id)
            if item.interpreter_version == interpreter_version
        ]
        if existing:
            return existing[-1]
        self._append_reinterpretation(interpretation, observed_event, actor_id=actor_id)
        return interpretation

    def reinterpret_call(
        self,
        call_id: str,
        *,
        interpreter_version: str,
        policy_version: str,
        actor_id: str = "runtime",
    ) -> dict[str, Any]:
        """Re-decide a recorded call from its preserved observations under named versions.

        Every attempt is interpreted from its observation, and every decision is
        derived, before anything is appended. Then missing interpretations and one
        ``call.reinterpreted`` record are appended: status, reason, per-attempt
        decisions, the interpretations and observation hashes used, and the status
        record it supersedes as the adopted view. ``call.status_decided`` and
        every earlier event are unchanged. A repeat with the same versions returns
        the existing record. Reinterpretation never starts a new attempt.
        """
        recorded = self.get_recorded_call(call_id)
        if recorded is None or not recorded.attempts:
            raise ObservationUnavailable(call_id, "no recorded attempts for this call")
        computed = []
        for attempt in recorded.attempts:
            interpretation, text, observed_event = self._interpret_from_observation(
                attempt.attempt_id, interpreter_version
            )
            existing = [
                item for item in self.interpretations_for_attempt(attempt.attempt_id)
                if item.interpreter_version == interpreter_version
            ]
            decision, reason = decide_attempt(
                existing[-1] if existing else interpretation,
                policy_version=policy_version,
                has_output_text=bool(text),
            )
            computed.append((attempt, interpretation, bool(existing), existing, decision, reason,
                             observed_event))
        prior = [
            event for event in self.ledger.events_by_kind(("call.reinterpreted",))
            if event.stream_id == call_id
            and event.payload.get("interpreter_version") == interpreter_version
            and event.payload.get("policy_version") == policy_version
        ]
        if prior:
            return dict(prior[-1].payload)
        latest = self._latest_status_record(call_id)
        interpretations: list[AttemptInterpretation] = []
        decisions: list[AttemptDecision] = []
        for attempt, interpretation, had, existing, decision, reason, observed_event in computed:
            if had:
                interpretation = existing[-1]
            else:
                self._append_reinterpretation(interpretation, observed_event, actor_id=actor_id)
            interpretations.append(interpretation)
            decisions.append(AttemptDecision(decision=decision, reason=reason, executed=False))
        status, status_reason = _decide_call_status(list(recorded.attempts), interpretations, decisions)
        if decisions[-1].decision == "retry":
            status_reason = "reinterpretation decides retry; reinterpretation never starts an attempt"
        payload = {
            "call_id": call_id,
            "task_id": recorded.task_id,
            "interpreter_version": interpreter_version,
            "policy_version": policy_version,
            "status": status,
            "reason": status_reason,
            "interpretation_ids": [item.interpretation_id for item in interpretations],
            "decisions": [
                {"attempt_id": attempt.attempt_id, "interpretation_id": item.interpretation_id,
                 "decision": decided.decision, "reason": decided.reason}
                for (attempt, *_rest), item, decided in zip(computed, interpretations, decisions)
            ],
            "observations": [
                {"attempt_id": item.attempt_id,
                 "sha256": item.response_body_artifact.sha256 if item.response_body_artifact else None}
                for item in interpretations
            ],
            "prior_status": latest.payload.get("status") if latest is not None else None,
            "prior_record_kind": latest.kind if latest is not None else None,
            "prior_record_event_id": latest.event_id if latest is not None else None,
            "provider_effect": False,
        }
        self.ledger.append(
            Event.create(
                stream_id=call_id,
                kind="call.reinterpreted",
                actor_id=actor_id,
                payload=payload,
                causation_id=latest.event_id if latest is not None else None,
                correlation_id=recorded.task_id,
            )
        )
        return payload

    def _latest_status_record(self, call_id: str) -> Event | None:
        records = [
            event
            for event in self.ledger.events_by_kind(("call.status_decided", "call.reinterpreted"))
            if event.stream_id == call_id
        ]
        return records[-1] if records else None

    def _interpret_from_observation(
        self, attempt_id: str, interpreter_version: str
    ) -> tuple[AttemptInterpretation, str, Event]:
        observed_event = next(
            (
                event
                for event in self.ledger.events_by_kind(("attempt.observed",))
                if event.stream_id == attempt_id
                or str(event.payload.get("attempt_id", "")) == attempt_id
            ),
            None,
        )
        if observed_event is None:
            raise ObservationUnavailable(
                attempt_id, "no attempt.observed event: nothing was preserved to interpret"
            )
        observed = observed_event.payload
        ref = observed.get("response_body_artifact")
        body: bytes | None = None
        body_artifact: ArtifactRef | None = None
        if isinstance(ref, dict):
            if self.artifact_store is None:
                raise ObservationUnavailable(attempt_id, "no artifact store to read the response body")
            try:
                body = self.artifact_store.read_bytes(str(ref["artifact_id"]))
                body_artifact = ArtifactRef(
                    artifact_id=str(ref["artifact_id"]),
                    sha256=str(ref["sha256"]),
                    media_type=str(ref.get("media_type", "application/octet-stream")),
                    uri=ref.get("uri"),
                )
            except FileNotFoundError as exc:
                raise ObservationUnavailable(attempt_id, "response body bytes are missing") from exc
            except ArtifactCorruptionError as exc:
                raise ObservationUnavailable(
                    attempt_id, "response body bytes do not match their recorded sha256"
                ) from exc
            except KeyError as exc:
                raise ObservationUnavailable(attempt_id, "response body reference is malformed") from exc
        protocol = observed.get("protocol")
        parsed: dict[str, Any] = {}
        text = ""
        if body is not None:
            try:
                loaded = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                loaded = None
            if isinstance(loaded, dict):
                parsed = loaded
                text = output_text_for(protocol, loaded)
        http_status = observed.get("http_status")
        entry = InterpretationInput(
            attempt_id=attempt_id,
            call_id=str(observed.get("call_id", "")),
            task_id=str(observed.get("task_id", "")),
            protocol=protocol,
            transport_outcome=observed.get("transport_outcome"),
            http_status=None if http_status is None else int(http_status),
            exception_type=observed.get("exception_type"),
            failure_message=None,
            body_bytes=body,
            parsed=parsed,
            output_text=text,
            adapter_status="succeeded" if text else "failed",
            adapter_error_kind=None,
        )
        interpretation = interpret_attempt(
            entry,
            version=interpreter_version,
            observation_event_id=observed_event.event_id,
            response_body_artifact=body_artifact,
        )
        return interpretation, text, observed_event

    def _append_reinterpretation(
        self, interpretation: AttemptInterpretation, observed_event: Event, *, actor_id: str
    ) -> None:
        payload = _interpretation_payload(interpretation)
        payload["evidence_source"] = "attempt.observed"
        payload["recorded_by"] = "reinterpretation"
        self.ledger.append(
            Event.create(
                stream_id=interpretation.attempt_id,
                kind="attempt.interpreted",
                actor_id=actor_id,
                payload=payload,
                causation_id=observed_event.event_id,
                correlation_id=interpretation.task_id,
            )
        )

    def project_call_as(
        self, call_id: str, *, interpreter_version: str, policy_version: str
    ) -> dict[str, Any] | None:
        """Counterfactual call projection under named versions, without writes.

        Reinterprets every attempt of the call and re-decides, returning the
        status history *would* have produced. Never claims to be what the
        runtime decided at the time; see call.status_decided for that.
        """
        recorded = self.get_recorded_call(call_id)
        if recorded is None:
            return None
        per_attempt = []
        for attempt in recorded.attempts:
            reinterpreted = self.interpret_attempt_as(
                attempt.attempt_id, version=interpreter_version
            )
            if reinterpreted is None:
                continue
            decision, reason = decide_attempt(
                reinterpreted,
                policy_version=policy_version,
                has_output_text=bool(self._envelope_output_text(attempt)),
            )
            per_attempt.append(
                {
                    "attempt_id": attempt.attempt_id,
                    "attempt_index": attempt.attempt_index,
                    "interpretation_id": reinterpreted.interpretation_id,
                    "error_kind": reinterpreted.error_kind,
                    "generation_state": reinterpreted.generation_state,
                    "decision": decision,
                    "reason": reason,
                }
            )
        status = _projected_call_status(per_attempt)
        return {
            "call_id": call_id,
            "interpreter_version": interpreter_version,
            "policy_version": policy_version,
            "status": status,
            "attempts": per_attempt,
        }

    def _envelope_output_text(self, attempt: AttemptRecord) -> str:
        """Best-effort output text for policy projection from the preserved envelope."""
        raw = attempt.raw_artifact
        if raw is None or self.artifact_store is None:
            return ""
        try:
            envelope = json.loads(self.artifact_store.read_text(raw.artifact_id))
        except (FileNotFoundError, KeyError, RuntimeError, ValueError):
            return ""
        if isinstance(envelope, dict):
            return str(envelope.get("output_text", "") or "")
        return ""

    def _interpret_attempt(
        self,
        spec: CallSpec,
        *,
        manifest: CallManifest,
        attempt: AttemptRecord,
        result: CallResult,
    ) -> CallResult:
        return replace(
            result,
            # The logical call identity comes from the spec: gateway adapters'
            # send() returns an empty call_id, which call.completed must not inherit.
            call_id=spec.call_id,
            input_tokens=attempt.usage.input_tokens,
            output_tokens=attempt.usage.output_tokens,
            cost_usd=attempt.cost_usd,
            provider=attempt.provider,
            model=attempt.resolved_model_id,
            model_version=result.model_version,
            provider_call_id=result.provider_call_id,
            request_id=result.request_id or spec.idempotency_key,
            started_at=attempt.started_at,
            completed_at=attempt.finished_at,
            status="succeeded" if attempt.status == AttemptStatus.SUCCEEDED.value else "failed",
            error_kind=attempt.error_kind,
            usage_source=attempt.usage.source.value,
            pricing_version=attempt.pricing_version,
            cost_source=attempt.cost_source,
            normalizer_version=attempt.normalizer_version,
            attempt_id=attempt.attempt_id,
            attempt_index=attempt.attempt_index,
            effective_parameters=dict(attempt.effective_parameters),
            raw_artifact=attempt.raw_artifact,
        )

    def _append_recorded_call_completed(
        self,
        result: CallResult,
        *,
        spec: CallSpec,
        manifest: CallManifest,
        attempts: list[AttemptRecord],
        call_status: str,
        totals: dict[str, float | int | None],
        causation_id: str | None,
        decision_basis_interpretation_id: str | None = None,
        decision_policy_version: str | None = None,
    ) -> None:
        payload = _call_result_event_payload(result)
        payload["idempotency_key"] = spec.idempotency_key
        payload["task_id"] = spec.task_id
        payload["directive_id"] = spec.directive_id
        payload["run_id"] = spec.run_id
        payload["adapter_id"] = spec.adapter_id
        payload["experiment_id"] = spec.experiment_id
        payload["arm"] = spec.arm
        payload["prompt_version"] = spec.prompt_version
        payload["context_package_id"] = spec.context.package_id
        # Recorded-cognition provenance (additive; legacy readers ignore).
        payload["chamber"] = manifest.chamber
        payload["requested_model"] = manifest.requested_model
        payload["resolved_model_id"] = manifest.resolved_model_id
        payload["provider_revision"] = attempts[-1].provider_revision if attempts else None
        payload["revision_source"] = attempts[-1].revision_source if attempts else None
        payload["attempt_ids"] = [attempt.attempt_id for attempt in attempts]
        payload["attempt_count"] = len(attempts)
        payload["call_status"] = call_status
        payload["decision_basis_interpretation_id"] = decision_basis_interpretation_id
        payload["decision_policy_version"] = decision_policy_version
        payload["total_input_tokens"] = totals["input_tokens"]
        payload["total_output_tokens"] = totals["output_tokens"]
        payload["total_cost_usd"] = totals["cost_usd"]
        payload["pricing_version"] = manifest.pricing_version
        self.ledger.append(
            Event.create(
                stream_id=result.call_id,
                kind="call.completed",
                actor_id=spec.actor.actor_id,
                payload=payload,
                causation_id=causation_id,
                correlation_id=spec.task_id,
            )
        )

    def sealed_fanout(
        self,
        *,
        task_id: str,
        base_prompt: str,
        branches: list[dict[str, object]],
        directive_id: str | None = None,
        run_id: str | None = None,
        base_events: tuple[Event, ...] = (),
        base_artifact_ids: tuple[str, ...] = (),
        base_claim_ids: tuple[str, ...] = (),
        base_seal: Seal | None = None,
        budget_tokens: int | None = None,
        objective: str | None = None,
        prompt_version: str | None = None,
        instruction: str = "",
        experiment_id: str | None = None,
        arm: str | None = None,
    ) -> tuple[CallResult, ...]:
        """First genuine collaboration primitive: N sealed independent calls.

        Each branch receives the same base problem; sibling outputs are
        invisible via lineage-aware seals; raw results persist independently;
        one branch failing never erases successful siblings; no synthesis.
        Works for same-model x N and heterogeneous-models x N alike.

        Stochastic branches execute through invoke_recorded_call, so each
        gains manifest, prepared-request provenance, attempt, transport
        observation, interpretation, and decision evidence. The return
        contract stays tuple[CallResult, ...]: replayed branches carry
        replayed=True and the original outcome, never a new status.
        """
        import uuid as _uuid

        call_ids = [str(branch.get("call_id", str(_uuid.uuid4()))) for branch in branches]
        sibling_set = set(call_ids)
        results: list[CallResult] = []

        fanout_id = str(_uuid.uuid4())
        self.ledger.append(
            Event.create(
                stream_id=fanout_id,
                kind="fanout.requested",
                actor_id="runtime",
                payload={
                    "fanout_id": fanout_id,
                    "task_id": task_id,
                    "directive_id": directive_id,
                    "run_id": run_id,
                    "call_ids": call_ids,
                    "base_prompt": base_prompt,
                    "experiment_id": experiment_id,
                    "arm": arm,
                },
                correlation_id=task_id,
            )
        )

        compiler = ContextCompiler()
        for branch, call_id in zip(branches, call_ids):
            actor = branch["actor"]
            assert isinstance(actor, ActorRef)
            adapter = branch["adapter"]
            assert hasattr(adapter, "invoke")
            variant = branch.get("variant") if isinstance(branch.get("variant"), Variant) else Variant()
            branch_prompt = str(branch.get("prompt", base_prompt))
            branch_instruction = str(branch.get("instruction", instruction))
            parameters = dict(branch.get("parameters", {})) if isinstance(branch.get("parameters"), dict) else {}
            # Seal this branch from all siblings (present and future outputs).
            siblings = frozenset(sibling_set - {call_id})
            seal = Seal(
                forbidden_event_ids=(base_seal.forbidden_event_ids if base_seal else frozenset()),
                forbidden_call_ids=frozenset(
                    set(base_seal.forbidden_call_ids if base_seal else frozenset()) | siblings
                ),
                forbidden_artifact_ids=(base_seal.forbidden_artifact_ids if base_seal else frozenset()),
                forbidden_lineage_ids=frozenset(
                    set(base_seal.forbidden_lineage_ids if base_seal else frozenset()) | siblings
                ),
            )
            package, trace = compiler.compile_with_trace(
                task_id=task_id,
                actor=actor,
                prompt=branch_prompt,
                events=base_events,
                artifact_ids=base_artifact_ids,
                seal=seal,
                objective=objective,
                claim_ids=base_claim_ids,
                budget_tokens=budget_tokens,
                prompt_version=prompt_version or branch.get("prompt_version"),  # type: ignore[arg-type]
            )
            self._append_context_compiled(package, trace, actor_id=actor.actor_id, task_id=task_id)
            spec = CallSpec(
                call_id=call_id,
                task_id=task_id,
                actor=actor,
                context=package,
                idempotency_key=str(branch.get("idempotency_key", f"fanout:{fanout_id}:{call_id}")),
                pattern="fanout",
                parameters=parameters,
                directive_id=str(directive_id) if directive_id else None,
                run_id=str(run_id) if run_id else None,
                adapter_id=str(branch.get("adapter_id", getattr(adapter, "model", None) or actor.provider or "unknown")),
                instruction=branch_instruction,
                prompt_version=str(prompt_version) if prompt_version else None,
                variant=variant,  # type: ignore[arg-type]
                metadata={"fanout_id": fanout_id},
                experiment_id=str(branch.get("experiment_id", experiment_id))
                if branch.get("experiment_id", experiment_id) is not None
                else None,
                arm=str(branch.get("arm", arm)) if branch.get("arm", arm) is not None else None,
            )
            try:
                raw_attempts = branch.get("max_attempts", 1)
                if (
                    isinstance(raw_attempts, bool)
                    or not isinstance(raw_attempts, int)
                    or raw_attempts < 1
                ):
                    raise ValueError(f"invalid max_attempts: {raw_attempts!r}")
                _, result, _ = self._invoke_recorded_call_detailed(
                    spec,
                    adapter=adapter,
                    max_attempts=raw_attempts,
                )
            except (ValueError, RuntimeError) as exc:  # never let one branch kill siblings
                result = CallResult(call_id=call_id, raw_output="", status="failed", error=str(exc))
                self.ledger.append(
                    Event.create(
                        stream_id=call_id,
                        kind="call.completed",
                        actor_id=actor.actor_id,
                        payload=_call_result_event_payload(result),
                        correlation_id=task_id,
                    )
                )
            results.append(result)

        self.ledger.append(
            Event.create(
                stream_id=fanout_id,
                kind="fanout.completed",
                actor_id="runtime",
                payload={
                    "fanout_id": fanout_id,
                    "task_id": task_id,
                    "call_ids": call_ids,
                    "statuses": [r.status for r in results],
                },
                correlation_id=task_id,
            )
        )
        return tuple(results)

    # ---------------- Claims: atomic assertions over raw outputs ----------------

    def record_claim(self, claim: Claim, *, actor_id: str | None = None) -> Event:
        event = Event.create(
            stream_id=claim.claim_id,
            kind="claim.recorded",
            actor_id=actor_id or claim.source_call_id,
            payload=asdict(claim),
            correlation_id=claim.task_id,
        )
        self.ledger.append(event)
        return event

    def link_claims(self, relationship: ClaimRelationship, *, actor_id: str = "runtime") -> Event:
        event = Event.create(
            stream_id=relationship.relationship_id,
            kind="claim.linked",
            actor_id=actor_id,
            payload=asdict(relationship),
            correlation_id=relationship.run_id,
        )
        self.ledger.append(event)
        return event

    def claims_for_run(self, run_id: str) -> dict[str, Claim]:
        from .claims import project_claims

        relevant = tuple(
            e
            for e in self.ledger.read_all()
            if e.kind in ("claim.recorded", "claim.evidence", "claim.status")
            and (e.correlation_id == run_id or str(e.payload.get("run_id", "")) == run_id or str(e.payload.get("task_id", "")) == run_id)
        )
        # Also include claims whose task belongs to this run.
        task_ids = {
            str(e.payload["task_id"])
            for e in self.ledger.events_by_kind(("task.created",))
            if str(e.payload.get("directive_id", "")) == run_id
        }
        if task_ids:
            relevant = tuple(
                e
                for e in self.ledger.read_all()
                if e.kind in ("claim.recorded", "claim.evidence", "claim.status")
                and (e.correlation_id in task_ids or str(e.payload.get("run_id", "")) == run_id)
            )
        return project_claims(relevant)

    def claims_for_task(self, task_id: str) -> dict[str, Claim]:
        from .claims import project_claims

        relevant = tuple(
            e
            for e in self.ledger.read_all()
            if e.kind in ("claim.recorded", "claim.evidence", "claim.status")
            and (e.correlation_id == task_id or str(e.payload.get("task_id", "")) == task_id)
        )
        return project_claims(relevant)

    def disagreement_for_run(self, run_id: str) -> dict[str, object]:
        from .claims import disagreement_report, project_relationships

        claims = self.claims_for_run(run_id)
        relationships = project_relationships(self.ledger.read_all())
        run_claim_ids = set(claims)
        relationships = tuple(
            r for r in relationships if r.from_claim_id in run_claim_ids and r.to_claim_id in run_claim_ids
        )
        return disagreement_report(claims, relationships)

    def compile_and_record_context(
        self, *, actor_id: str = "runtime", task_id: str, **kwargs: Any
    ) -> tuple[ContextPackage, CompilationTrace]:
        """Compile context and record the attempt, its inventory and its outcome.

        ``context.compilation_requested`` (the offered candidate inventory with
        requirements, sizes and lineage, plus seal, budget and prompt) is
        appended before compiling; then ``context.compiled`` or
        ``context.compilation_failed``, causally linked by compilation_id.
        """
        compiler = ContextCompiler()
        compilation_id = str(uuid.uuid4())
        for key in ("events", "artifact_ids", "claim_ids", "required_event_ids",
                    "required_artifact_ids", "required_claim_ids"):
            if key in kwargs:
                kwargs[key] = tuple(kwargs[key])  # inventory and compiler see the same inputs
        offer_keys = ("events", "artifact_ids", "claim_ids", "required_event_ids",
                      "required_artifact_ids", "required_claim_ids", "event_token_sizes",
                      "claim_lineage", "artifact_lineage")
        offered = {key: kwargs[key] for key in offer_keys if key in kwargs}
        candidates = compiler.offered_candidates(**offered)
        seal = kwargs.get("seal")
        actor = kwargs.get("actor")
        requested = Event.create(
            stream_id=compilation_id,
            kind="context.compilation_requested",
            actor_id=actor_id,
            payload={
                "compilation_id": compilation_id,
                "task_id": task_id,
                "actor": asdict(actor) if actor is not None else None,
                "prompt": kwargs.get("prompt"),
                "prompt_version": kwargs.get("prompt_version"),
                "objective": kwargs.get("objective"),
                "budget_tokens": kwargs.get("budget_tokens"),
                "seal": _seal_payload(seal) if seal is not None else None,
                "candidates": [asdict(candidate) for candidate in candidates],
                "required_ids": sorted({
                    str(item)
                    for key in ("required_event_ids", "required_artifact_ids", "required_claim_ids")
                    for item in offered.get(key, ())
                }),
                "size_basis": ("declared event_token_sizes where given; otherwise estimated from "
                               "serialized event payload length; artifact and claim candidates count zero"),
            },
            correlation_id=task_id,
        )
        self.ledger.append(requested)
        try:
            package, trace = compiler.compile_with_trace(task_id=task_id, **kwargs)
        except (ContextSealViolation, ContextBudgetUnsatisfiable) as exc:
            self.ledger.append(
                Event.create(
                    stream_id=compilation_id,
                    kind="context.compilation_failed",
                    actor_id=actor_id,
                    payload={
                        "compilation_id": compilation_id,
                        "task_id": task_id,
                        "exception": type(exc).__name__,
                        "message": str(exc),
                    },
                    causation_id=requested.event_id,
                    correlation_id=task_id,
                )
            )
            raise
        self._append_context_compiled(
            package, trace, actor_id=actor_id, task_id=task_id,
            compilation_id=compilation_id, causation_id=requested.event_id,
        )
        return package, trace

    def decide_next(self, query: object) -> object:
        from .scheduler import decide_next_step

        return decide_next_step(query)  # type: ignore[arg-type]

    def list_runs(self) -> tuple[dict[str, object], ...]:
        tasks_by_run: dict[str, int] = {}
        for event in self.ledger.events_by_kind(("task.created",)):
            directive_id = str(event.payload["directive_id"])
            tasks_by_run[directive_id] = tasks_by_run.get(directive_id, 0) + 1

        runs = []
        for event in self.ledger.events_by_kind(("directive.opened",)):
            runs.append(
                {
                    "run_id": str(event.payload["directive_id"]),
                    "objective": str(event.payload["objective"]),
                    "created_at": event.created_at,
                    "task_count": tasks_by_run.get(str(event.payload["directive_id"]), 0),
                    "status": "active",
                }
            )
        return tuple(runs)

    def show_run(self, run_id: str) -> dict[str, object] | None:
        directive_event = next(
            (
                event
                for event in self.ledger.events_by_kind(("directive.opened",))
                if str(event.payload["directive_id"]) == run_id
            ),
            None,
        )
        if directive_event is None:
            return None

        tasks = [
            event.payload
            for event in self.ledger.events_by_kind(("task.created",))
            if str(event.payload["directive_id"]) == run_id
        ]
        task_ids = {str(task["task_id"]) for task in tasks}
        actions = [
            event.payload
            for event in self.ledger.events_by_kind(("action.completed",))
            if event.correlation_id in task_ids
        ]
        checks = [
            event.payload
            for event in self.ledger.events_by_kind(("check.completed",))
            if event.correlation_id in task_ids
        ]
        calls = [
            event.payload
            for event in self.ledger.events_by_kind(("call.completed",))
            if event.correlation_id in task_ids
        ]
        claims = [
            event.payload
            for event in self.ledger.events_by_kind(("claim.recorded",))
            if event.correlation_id in task_ids or str(event.payload.get("run_id", "")) == run_id
        ]
        return {
            "run_id": run_id,
            "created_at": directive_event.created_at,
            "objective": directive_event.payload["objective"],
            "success_criteria": directive_event.payload["success_criteria"],
            "tasks": tasks,
            "actions": actions,
            "checks": checks,
            "calls": calls,
            "claims": claims,
        }

    # ---------------- call persistence helpers ----------------

    @staticmethod
    @staticmethod
    def _call_spec_payload(spec: CallSpec) -> dict[str, object]:
        payload = asdict(spec)
        payload["idempotency_key"] = spec.idempotency_key
        # Never persist credential-like caller parameters: the requested map
        # is provenance of intent, not a transport channel. Declared controls
        # survive; unknown values may still be rejected pre-effect by prepare.
        payload["parameters"] = sanitize_effective_params(dict(spec.parameters))
        return payload

    def _find_call_result(self, idempotency_key: str) -> CallResult | None:
        for event in self.ledger.events_by_kind(("call.completed",)):
            payload = event.payload
            if payload.get("idempotency_key") != idempotency_key:
                continue
            return self._call_result_from_payload(payload)
        return None

    def _finalize_call_result(
        self,
        result: CallResult,
        *,
        spec: CallSpec,
        started_at: str,
        completed_at: str,
    ) -> CallResult:
        raw_artifact = result.raw_artifact
        if self.artifact_store is not None and result.raw_output:
            raw_artifact = self.artifact_store.store_text(
                result.raw_output,
                media_type="text/plain",
                artifact_type="raw_model_output",
            )
        return replace(
            result,
            started_at=result.started_at or started_at,
            completed_at=result.completed_at or completed_at,
            provider=result.provider or spec.actor.provider,
            model=result.model or spec.actor.model,
            model_version=result.model_version or spec.actor.version,
            request_id=result.request_id or spec.idempotency_key,
            raw_artifact=raw_artifact or result.raw_artifact,
            pricing_version=result.pricing_version or PRICING_VERSION,
            normalizer_version=result.normalizer_version or NORMALIZER_VERSION,
        )

    def _append_call_completed(
        self, result: CallResult, *, spec: CallSpec, causation_id: str | None
    ) -> None:
        payload = _call_result_event_payload(result)
        payload["idempotency_key"] = spec.idempotency_key
        payload["task_id"] = spec.task_id
        payload["directive_id"] = spec.directive_id
        payload["run_id"] = spec.run_id
        payload["adapter_id"] = spec.adapter_id
        payload["experiment_id"] = spec.experiment_id
        payload["arm"] = spec.arm
        payload["prompt_version"] = spec.prompt_version
        payload["context_package_id"] = spec.context.package_id
        self.ledger.append(
            Event.create(
                stream_id=result.call_id,
                kind="call.completed",
                actor_id=spec.actor.actor_id,
                payload=payload,
                causation_id=causation_id,
                correlation_id=spec.task_id,
            )
        )

    def _call_result_from_payload(self, payload: dict[str, object]) -> CallResult:
        raw_artifact = None
        raw = payload.get("raw_artifact")
        if isinstance(raw, dict):
            try:
                raw_artifact = ArtifactRef(**{k: raw[k] for k in ("artifact_id", "sha256", "media_type") if k in raw} | ({"uri": raw.get("uri")} if "uri" in raw else {}))
            except TypeError:
                raw_artifact = None
        effective = payload.get("effective_parameters")
        raw_payload = payload.get("raw_payload")
        return CallResult(
            call_id=str(payload["call_id"]),
            raw_output=str(payload.get("raw_output", "")),
            input_tokens=_optional_int(payload.get("input_tokens")),
            output_tokens=_optional_int(payload.get("output_tokens")),
            cost_usd=self._payload_float(payload, "cost_usd"),
            latency_ms=self._payload_int(payload, "latency_ms"),
            provider=self._payload_value(payload, "provider"),
            model=self._payload_value(payload, "model"),
            model_version=self._payload_value(payload, "model_version"),
            fingerprint=self._payload_value(payload, "fingerprint"),
            provider_call_id=self._payload_value(payload, "provider_call_id"),
            request_id=self._payload_value(payload, "request_id"),
            started_at=self._payload_value(payload, "started_at"),
            completed_at=self._payload_value(payload, "completed_at"),
            status=str(payload.get("status", "succeeded")),
            error=self._payload_value(payload, "error"),
            raw_artifact=raw_artifact,
            usage_source=str(payload.get("usage_source", "measured") or "measured"),
            error_kind=self._payload_value(payload, "error_kind"),
            pricing_version=self._payload_value(payload, "pricing_version"),
            cost_source=self._payload_value(payload, "cost_source"),
            normalizer_version=self._payload_value(payload, "normalizer_version"),
            attempt_id=self._payload_value(payload, "attempt_id"),
            attempt_index=self._payload_int(payload, "attempt_index"),
            effective_parameters=dict(effective) if isinstance(effective, dict) else {},
            protocol=self._payload_value(payload, "protocol"),
            raw_observation_kind=self._payload_value(payload, "raw_observation_kind"),
            raw_payload=dict(raw_payload) if isinstance(raw_payload, dict) else {},
            replayed=bool(payload.get("replayed", False)),
        )

    def _append_context_compiled(
        self,
        package: ContextPackage,
        trace: CompilationTrace,
        *,
        actor_id: str,
        task_id: str,
        compilation_id: str | None = None,
        causation_id: str | None = None,
    ) -> None:
        self.ledger.append(
            Event.create(
                stream_id=package.package_id,
                kind="context.compiled",
                actor_id=actor_id,
                payload={
                    "package_id": package.package_id,
                    "task_id": package.task_id,
                    "actor": asdict(package.actor),
                    "prompt": package.prompt,
                    "prompt_version": package.prompt_version,
                    "objective": package.objective,
                    "event_ids": list(package.event_ids),
                    "artifact_ids": list(package.artifact_ids),
                    "claim_ids": list(package.claim_ids),
                    "budget_tokens": package.budget_tokens,
                    "seal": {
                        "forbidden_event_ids": sorted(package.seal.forbidden_event_ids),
                        "forbidden_call_ids": sorted(package.seal.forbidden_call_ids),
                        "forbidden_artifact_ids": sorted(package.seal.forbidden_artifact_ids),
                        "forbidden_lineage_ids": sorted(package.seal.forbidden_lineage_ids),
                    },
                    "trace_id": trace.trace_id,
                    "trace": [
                        {"candidate_id": e.candidate_id, "decision": e.decision, "reason": e.reason}
                        for e in trace.entries
                    ],
                    "included_ids": list(trace.included_ids),
                    "excluded_ids": list(trace.excluded_ids),
                    "compilation_id": compilation_id,
                },
                causation_id=causation_id,
                correlation_id=task_id,
            )
        )

    def _apply_check_to_claims(self, request: CheckRequest, result: CheckResult) -> None:
        """Scoped evidence promotion: only claims the check actually targeted."""
        if not request.claim_ids:
            return
        if result.verdict == CheckVerdict.PASS:
            for claim_id in request.claim_ids:
                self.ledger.append(
                    Event.create(
                        stream_id=str(claim_id),
                        kind="claim.evidence",
                        actor_id="verifier",
                        payload={
                            "claim_id": str(claim_id),
                            "evidence_class": EvidenceClass.REPRODUCED.value,
                            "check_id": result.check_id,
                            "details": "deterministic check passed for targeted claim",
                        },
                        correlation_id=request.task_id,
                    )
                )
        elif result.verdict == CheckVerdict.FAIL:
            for claim_id in request.claim_ids:
                self.ledger.append(
                    Event.create(
                        stream_id=str(claim_id),
                        kind="claim.status",
                        actor_id="verifier",
                        payload={
                            "claim_id": str(claim_id),
                            "status": ClaimStatus.REFUTED.value,
                            "check_id": result.check_id,
                            "details": "deterministic check failed for targeted claim",
                        },
                        correlation_id=request.task_id,
                    )
                )

    def _find_action_result(self, idempotency_key: str) -> ActionResult | None:
        for event in self.ledger.events_by_kind(("action.completed",)):
            payload = event.payload
            if payload.get("idempotency_key") != idempotency_key:
                continue
            return self._action_result_from_payload(payload)
        return None

    def _finalize_action_result(
        self,
        result: ActionResult,
        *,
        started_at: str,
        completed_at: str,
        resulting_state_hash: str | None,
    ) -> ActionResult:
        artifacts = list(result.artifacts)
        if self.artifact_store is not None:
            if result.transcript:
                artifacts.append(
                    self.artifact_store.store_text(
                        result.transcript,
                        media_type="text/plain",
                        artifact_type="raw_model_output",
                    )
                )
            if result.stdout:
                artifacts.append(
                    self.artifact_store.store_text(
                        result.stdout,
                        media_type="text/plain",
                        artifact_type="command_stdout",
                    )
                )
            if result.stderr:
                artifacts.append(
                    self.artifact_store.store_text(
                        result.stderr,
                        media_type="text/plain",
                        artifact_type="command_stderr",
                    )
                )

        return replace(
            result,
            started_at=result.started_at or started_at,
            completed_at=result.completed_at or completed_at,
            resulting_state_hash=result.resulting_state_hash or resulting_state_hash,
            artifacts=tuple(artifacts),
        )

    def _append_action_result_event(self, result: ActionResult, *, actor_id: str) -> None:
        payload = asdict(result)
        payload["idempotency_key"] = self._idempotency_key_for_action(result.action_id)
        event = Event.create(
            stream_id=result.action_id,
            kind="action.completed",
            actor_id=actor_id,
            payload=payload,
            correlation_id=self._task_id_for_action(result.action_id),
        )
        self.ledger.append(event)

    def _task_id_for_action(self, action_id: str) -> str | None:
        request_event = next(
            (
                event
                for event in self.ledger.events_by_kind(("action.requested",))
                if event.stream_id == action_id
            ),
            None,
        )
        return None if request_event is None else str(request_event.payload["task_id"])

    def _idempotency_key_for_action(self, action_id: str) -> str | None:
        request_event = next(
            (
                event
                for event in self.ledger.events_by_kind(("action.requested",))
                if event.stream_id == action_id
            ),
            None,
        )
        return None if request_event is None else str(request_event.payload["idempotency_key"])

    def _action_result_from_payload(self, payload: dict[str, object]) -> ActionResult:
        return ActionResult(
            action_id=str(payload["action_id"]),
            status=str(payload["status"]),
            started_at=self._payload_value(payload, "started_at"),
            completed_at=self._payload_value(payload, "completed_at"),
            artifacts=self._payload_artifacts(payload.get("artifacts")),
            state_hash=self._payload_value(payload, "state_hash"),
            resulting_state_hash=self._payload_value(payload, "resulting_state_hash"),
            transcript=self._payload_value(payload, "transcript"),
            stdout=self._payload_value(payload, "stdout"),
            stderr=self._payload_value(payload, "stderr"),
            exit_code=self._payload_int(payload, "exit_code"),
            cost_usd=self._payload_float(payload, "cost_usd"),
            input_tokens=int(payload.get("input_tokens", 0)),
            output_tokens=int(payload.get("output_tokens", 0)),
            error=self._payload_value(payload, "error"),
            reused_from_action_id=self._payload_value(payload, "reused_from_action_id"),
        )

    def _finalize_check_result(self, result: CheckResult) -> CheckResult:
        artifacts = list(result.artifacts)
        if self.artifact_store is not None:
            if result.stdout:
                artifacts.append(
                    self.artifact_store.store_text(
                        result.stdout,
                        media_type="text/plain",
                        artifact_type="command_stdout",
                    )
                )
            if result.stderr:
                artifacts.append(
                    self.artifact_store.store_text(
                        result.stderr,
                        media_type="text/plain",
                        artifact_type="command_stderr",
                    )
                )
        return replace(result, artifacts=tuple(artifacts))

    @staticmethod
    def _payload_artifacts(payload: object) -> tuple[ArtifactRef, ...]:
        if not isinstance(payload, list):
            return ()
        return tuple(ArtifactRef(**item) for item in payload if isinstance(item, dict))

    @staticmethod
    def _payload_value(payload: dict[str, object], key: str) -> str | None:
        value = payload.get(key)
        return None if value is None else str(value)

    @staticmethod
    def _payload_int(payload: dict[str, object], key: str) -> int | None:
        value = payload.get(key)
        return None if value is None else int(value)

    @staticmethod
    def _payload_float(payload: dict[str, object], key: str) -> float | None:
        value = payload.get(key)
        return None if value is None else float(value)

    def _current_state_hash(self) -> str | None:
        if self.state_resolver is None:
            return None
        return self.state_resolver()


# ---------------- recorded-cognition payload helpers ----------------


def _prompt_hash_for_spec(spec: CallSpec) -> str:
    """Stable prompt identity shared by manifest construction and replay checks."""
    return hashlib.sha256(f"{spec.instruction}\0{spec.context.prompt}".encode()).hexdigest()


def _request_fingerprint(spec: CallSpec) -> tuple[Any, ...]:
    """Logical request identity for idempotency matching: prompt bytes,
    chamber, and requested logical model.

    Task identity is deliberately excluded: reruns legitimately create new
    tasks for the same logical operation (run_arm mints a task per run), and
    the idempotency key remains authoritative for operation identity.
    Resolved occupants are excluded: re-resolution under the same key
    replays the original effect.
    """
    chamber = spec.chamber
    logical = spec.logical_model or chamber
    return (
        _prompt_hash_for_spec(spec),
        chamber,
        logical or spec.actor.model,
    )


def _interpretation_input(
    spec: CallSpec, manifest: CallManifest, result: CallResult, attempt_id: str
) -> InterpretationInput:
    """Assemble preserved evidence for the interpreter. Conclusions live in
    the interpretation, never here."""
    transport = result.transport
    parsed = result.raw_payload
    return InterpretationInput(
        attempt_id=attempt_id,
        call_id=spec.call_id,
        task_id=spec.task_id,
        protocol=result.protocol,
        transport_outcome=transport.outcome if transport is not None else None,
        http_status=transport.status_code if transport is not None else None,
        exception_type=transport.exception_type if transport is not None else None,
        failure_message=result.error,
        body_bytes=transport.body if transport is not None else None,
        parsed=dict(parsed) if isinstance(parsed, Mapping) else {},
        output_text=result.raw_output or "",
        adapter_status=result.status,
        adapter_error_kind=result.error_kind,
    )


def _interpretation_payload(interpretation: AttemptInterpretation) -> dict[str, Any]:
    ref = interpretation.response_body_artifact
    return {
        "interpretation_id": interpretation.interpretation_id,
        "attempt_id": interpretation.attempt_id,
        "call_id": interpretation.call_id,
        "task_id": interpretation.task_id,
        "observation_event_id": interpretation.observation_event_id,
        "response_body_artifact": None
        if ref is None
        else {
            "artifact_id": ref.artifact_id,
            "sha256": ref.sha256,
            "media_type": ref.media_type,
            "uri": ref.uri,
        },
        "interpreter_version": interpretation.interpreter_version,
        "completion_map_version": interpretation.completion_map_version,
        "classifier_version": interpretation.classifier_version,
        "created_at": interpretation.created_at,
        "transport_state": interpretation.transport_state,
        "generation_state": interpretation.generation_state,
        "provider_reason": interpretation.provider_reason,
        "provider_reason_source": interpretation.provider_reason_source,
        "error_kind": interpretation.error_kind,
        "classification_basis": interpretation.classification_basis,
        "basis_detail": _jsonable_mapping(interpretation.basis_detail),
    }


def _interpretation_from_payload(payload: dict[str, Any]) -> AttemptInterpretation:
    ref = payload.get("response_body_artifact")
    body_artifact = None
    if isinstance(ref, dict):
        try:
            body_artifact = ArtifactRef(
                artifact_id=str(ref["artifact_id"]),
                sha256=str(ref["sha256"]),
                media_type=str(ref.get("media_type", "application/octet-stream")),
                uri=ref.get("uri"),
            )
        except KeyError:
            body_artifact = None
    detail = payload.get("basis_detail")
    return AttemptInterpretation(
        interpretation_id=str(payload["interpretation_id"]),
        attempt_id=str(payload["attempt_id"]),
        call_id=str(payload.get("call_id", "")),
        task_id=str(payload.get("task_id", "")),
        observation_event_id=payload.get("observation_event_id"),
        response_body_artifact=body_artifact,
        interpreter_version=str(payload.get("interpreter_version", "")),
        completion_map_version=payload.get("completion_map_version"),
        classifier_version=payload.get("classifier_version"),
        created_at=payload.get("created_at"),
        transport_state=str(payload.get("transport_state", "")),
        generation_state=str(payload.get("generation_state", "unknown")),
        provider_reason=payload.get("provider_reason"),
        provider_reason_source=payload.get("provider_reason_source"),
        error_kind=payload.get("error_kind"),
        classification_basis=payload.get("classification_basis"),
        basis_detail=dict(detail) if isinstance(detail, dict) else {},
    )


def _decide_call_status(
    attempts: list[AttemptRecord],
    interpretations: list[AttemptInterpretation],
    decisions: list[AttemptDecision],
) -> tuple[str, str]:
    """Project the logical-call status from the final execution-time decision.

    Accepted final attempt -> succeeded. A terminal truncated/filtered
    generation is not completed cognition -> unresolved (least overclaiming
    among the existing compatible statuses). Anything else -> failed.
    """
    if not attempts or not decisions:
        return LogicalCallStatus.UNRESOLVED.value, "no attempts recorded"
    last_decision = decisions[-1]
    last_interpretation = interpretations[-1] if interpretations else None
    generation = last_interpretation.generation_state if last_interpretation else "unknown"
    if last_decision.decision == "accept":
        return LogicalCallStatus.SUCCEEDED.value, "final attempt accepted"
    if last_decision.decision == "retry":
        return (
            LogicalCallStatus.FAILED.value,
            "retry decided but attempt budget exhausted",
        )
    if generation in ("truncated", "filtered"):
        return (
            LogicalCallStatus.UNRESOLVED.value,
            f"generation {generation}, not treated as completed cognition",
        )
    return LogicalCallStatus.FAILED.value, f"terminal: {last_decision.reason}"


def _projected_call_status(per_attempt: list[dict[str, Any]]) -> str:
    """Counterfactual status for project_call_as: same rule, no persistence."""
    if not per_attempt:
        return LogicalCallStatus.UNRESOLVED.value
    last = per_attempt[-1]
    if last["decision"] == "accept":
        return LogicalCallStatus.SUCCEEDED.value
    if last["generation_state"] in ("truncated", "filtered"):
        return LogicalCallStatus.UNRESOLVED.value
    return LogicalCallStatus.FAILED.value


def _call_result_event_payload(result: CallResult) -> dict[str, Any]:
    """Serialize a CallResult for ledger/export payloads.

    The ephemeral transport observation is excluded: response bytes travel
    via the artifact store (plus an attempt.observed reference), never
    inside JSON payloads. (Event.create would otherwise coerce bytes with
    default=str into a silent "b'...'" corruption.)
    """
    payload = asdict(result)
    payload.pop("transport", None)
    return payload


# Response headers approved for persistence (case-insensitive match).
# Request IDs, retry guidance, rate-limit accounting, and provider routing
# tokens cannot be reconstructed later; everything else is default-deny.
# Never persisted: authorization, proxy-authorization, cookie, set-cookie,
# x-api-key, or any header outside this list.
_ALLOWED_RESPONSE_HEADERS = frozenset(
    {
        "request-id",
        "x-request-id",
        "retry-after",
        "ratelimit-limit",
        "ratelimit-remaining",
        "ratelimit-reset",
        "x-opencode-session",
    }
)
_ALLOWED_RESPONSE_HEADER_PREFIXES = ("ratelimit-", "x-ratelimit-")


def _allowlist_response_headers(headers: Mapping[str, Any]) -> dict[str, str]:
    """Keep only explicitly approved response headers, keyed lower-cased."""
    kept: dict[str, str] = {}
    for key, value in headers.items():
        lowered = str(key).lower()
        if lowered in _ALLOWED_RESPONSE_HEADERS or lowered.startswith(
            _ALLOWED_RESPONSE_HEADER_PREFIXES
        ):
            kept[lowered] = str(value)
    return kept


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)  # type: ignore[arg-type]


def _jsonable_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, item in value.items():
        if is_dataclass(item):
            out[str(key)] = asdict(item)
        elif isinstance(item, Enum):
            out[str(key)] = item.value
        elif isinstance(item, Mapping):
            out[str(key)] = _jsonable_mapping(item)
        elif isinstance(item, (list, tuple)):
            out[str(key)] = list(item)
        else:
            out[str(key)] = item
    return out


def _seal_payload(seal: Seal) -> dict[str, list[str]]:
    return {
        "forbidden_event_ids": sorted(seal.forbidden_event_ids),
        "forbidden_call_ids": sorted(seal.forbidden_call_ids),
        "forbidden_artifact_ids": sorted(seal.forbidden_artifact_ids),
        "forbidden_lineage_ids": sorted(seal.forbidden_lineage_ids),
    }


def _manifest_payload(manifest: CallManifest) -> dict[str, Any]:
    return {
        "call_id": manifest.call_id,
        "task_id": manifest.task_id,
        "chamber": manifest.chamber,
        "requested_model": manifest.requested_model,
        "provider": manifest.provider,
        "resolved_model_id": manifest.resolved_model_id,
        "provider_revision": manifest.provider_revision,
        "revision_source": manifest.revision_source,
        "pricing_version": manifest.pricing_version,
        "context_package_id": manifest.context_package_id,
        "prompt_hash": manifest.prompt_hash,
        "requested_parameters": _jsonable_mapping(manifest.requested_parameters),
        "effective_parameters": _jsonable_mapping(manifest.effective_parameters),
        "requested_controls": _jsonable_mapping(manifest.requested_controls),
        "omitted_unsupported": list(manifest.omitted_unsupported),
        "defaulted_parameters": _jsonable_mapping(manifest.defaulted_parameters),
        "request_plan_version": manifest.request_plan_version,
        "request_body_sha256": manifest.request_body_sha256,
        "created_at": manifest.created_at,
        "context_render_version": manifest.context_render_version,
        "rendered_context_sha256": manifest.rendered_context_sha256,
        "rendered_items": [dict(item) for item in manifest.rendered_items],
        "input_layout": [dict(part) for part in manifest.input_layout],
    }


def _manifest_from_payload(payload: dict[str, Any]) -> CallManifest:
    requested = payload.get("requested_parameters")
    effective = payload.get("effective_parameters")
    requested_controls = payload.get("requested_controls")
    defaulted = payload.get("defaulted_parameters")
    omitted = payload.get("omitted_unsupported")
    return CallManifest(
        call_id=str(payload["call_id"]),
        task_id=str(payload["task_id"]),
        chamber=payload.get("chamber"),
        requested_model=payload.get("requested_model"),
        provider=payload.get("provider"),
        resolved_model_id=payload.get("resolved_model_id"),
        provider_revision=payload.get("provider_revision"),
        revision_source=payload.get("revision_source"),
        pricing_version=payload.get("pricing_version"),
        context_package_id=payload.get("context_package_id"),
        prompt_hash=payload.get("prompt_hash"),
        requested_parameters=dict(requested) if isinstance(requested, dict) else {},
        effective_parameters=dict(effective) if isinstance(effective, dict) else {},
        requested_controls=dict(requested_controls) if isinstance(requested_controls, dict) else {},
        omitted_unsupported=tuple(omitted) if isinstance(omitted, list) else (),
        defaulted_parameters=dict(defaulted) if isinstance(defaulted, dict) else {},
        request_plan_version=payload.get("request_plan_version"),
        request_body_sha256=payload.get("request_body_sha256"),
        created_at=payload.get("created_at"),
        context_render_version=payload.get("context_render_version"),
        rendered_context_sha256=payload.get("rendered_context_sha256"),
        rendered_items=tuple(dict(item) for item in payload.get("rendered_items") or ()),
        input_layout=tuple(dict(part) for part in payload.get("input_layout") or ()),
    )


def _attempt_payload(attempt: AttemptRecord, *, fingerprint: str | None = None) -> dict[str, Any]:
    return {
        "attempt_id": attempt.attempt_id,
        "call_id": attempt.call_id,
        "task_id": attempt.task_id,
        "attempt_index": attempt.attempt_index,
        "started_at": attempt.started_at,
        "finished_at": attempt.finished_at,
        "latency_ms": attempt.latency_ms,
        "provider": attempt.provider,
        "resolved_model_id": attempt.resolved_model_id,
        "provider_revision": attempt.provider_revision,
        "revision_source": attempt.revision_source,
        "protocol": attempt.protocol,
        "provider_request_id": attempt.provider_request_id,
        "status": attempt.status,
        "error_kind": attempt.error_kind,
        "error": attempt.error,
        "usage": {
            "input_tokens": attempt.usage.input_tokens,
            "output_tokens": attempt.usage.output_tokens,
            "source": attempt.usage.source.value,
        },
        "pricing_version": attempt.pricing_version,
        "cost_usd": attempt.cost_usd,
        "cost_source": attempt.cost_source,
        "currency": attempt.currency,
        "raw_artifact": None
        if attempt.raw_artifact is None
        else {
            "artifact_id": attempt.raw_artifact.artifact_id,
            "sha256": attempt.raw_artifact.sha256,
            "media_type": attempt.raw_artifact.media_type,
            "uri": attempt.raw_artifact.uri,
        },
        "raw_observation_kind": attempt.raw_observation_kind,
        "normalizer_version": attempt.normalizer_version,
        "effective_parameters": _jsonable_mapping(attempt.effective_parameters),
        "fingerprint": fingerprint,
        "interpretation_id": attempt.interpretation_id,
        "policy_version": attempt.policy_version,
    }


def _attempt_from_payload(payload: dict[str, Any]) -> AttemptRecord:
    usage_payload = payload.get("usage")
    if isinstance(usage_payload, dict):
        source_raw = str(usage_payload.get("source", "unavailable") or "unavailable").lower()
        try:
            source = UsageSource(source_raw)
        except ValueError:
            source = UsageSource.UNAVAILABLE
        usage = Usage(
            input_tokens=_optional_int(usage_payload.get("input_tokens")),
            output_tokens=_optional_int(usage_payload.get("output_tokens")),
            source=source,
        )
    else:
        usage = Usage(
            input_tokens=_optional_int(payload.get("input_tokens")),
            output_tokens=_optional_int(payload.get("output_tokens")),
            source=UsageSource(str(payload.get("usage_source", "unavailable") or "unavailable").lower())
            if str(payload.get("usage_source", "")).lower() in ("measured", "estimated", "unavailable")
            else UsageSource.UNAVAILABLE,
        )
    raw = payload.get("raw_artifact")
    raw_artifact = None
    if isinstance(raw, dict):
        try:
            raw_artifact = ArtifactRef(
                artifact_id=str(raw["artifact_id"]),
                sha256=str(raw["sha256"]),
                media_type=str(raw.get("media_type", "application/json")),
                uri=raw.get("uri"),
            )
        except KeyError:
            raw_artifact = None
    effective = payload.get("effective_parameters")
    return AttemptRecord(
        attempt_id=str(payload["attempt_id"]),
        call_id=str(payload["call_id"]),
        task_id=str(payload.get("task_id", "")),
        attempt_index=int(payload.get("attempt_index", 1)),
        started_at=payload.get("started_at"),
        finished_at=payload.get("finished_at"),
        latency_ms=_optional_int(payload.get("latency_ms")),
        provider=payload.get("provider"),
        resolved_model_id=payload.get("resolved_model_id") or payload.get("model"),
        provider_revision=payload.get("provider_revision"),
        revision_source=payload.get("revision_source"),
        protocol=payload.get("protocol"),
        provider_request_id=payload.get("provider_request_id"),
        status=str(payload.get("status", "failed")),
        error_kind=payload.get("error_kind"),
        error=payload.get("error"),
        usage=usage,
        pricing_version=payload.get("pricing_version"),
        cost_usd=None if payload.get("cost_usd") is None else float(payload["cost_usd"]),  # type: ignore[arg-type]
        cost_source=str(payload.get("cost_source", "unknown") or "unknown"),
        currency=str(payload.get("currency", "USD") or "USD"),
        raw_artifact=raw_artifact,
        raw_observation_kind=payload.get("raw_observation_kind"),
        normalizer_version=payload.get("normalizer_version"),
        effective_parameters=dict(effective) if isinstance(effective, dict) else {},
        interpretation_id=payload.get("interpretation_id"),
        policy_version=payload.get("policy_version"),
    )


def _aggregate_totals(attempts: list[AttemptRecord]) -> dict[str, float | int | None]:
    """Derive call-level totals from persisted attempts.

    Unknowns propagate: if any attempt's usage is unknown, the call total is
    unknown (None), never a partial sum presented as authoritative. Cost sums
    only fully-known estimated costs.
    """
    total_in: int | None = 0
    total_out: int | None = 0
    total_cost: float | None = 0.0
    for attempt in attempts:
        if attempt.usage.input_tokens is None or attempt.usage.output_tokens is None:
            total_in = None
            total_out = None
        else:
            if total_in is not None:
                total_in += attempt.usage.input_tokens
            if total_out is not None:
                total_out += attempt.usage.output_tokens
        if attempt.cost_usd is None:
            total_cost = None
        elif total_cost is not None:
            total_cost += attempt.cost_usd
    if not attempts:
        return {"input_tokens": None, "output_tokens": None, "cost_usd": None}
    return {"input_tokens": total_in, "output_tokens": total_out, "cost_usd": total_cost}
