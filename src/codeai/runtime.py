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
    VerificationAdapter,
    sanitize_effective_params,
)
from .artifacts import FileArtifactStore
from .context import CompilationTrace, ContextCompiler
from .domain import (
    ActorRef,
    ArtifactRef,
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
from .ledger import Event, SQLiteLedger
from .policy import AuthorityDenied, PolicyEngine
from .providers import PRICING_VERSION, estimate_cost_usd


class PreconditionMismatch(RuntimeError):
    pass


# Attempt classifications that may be retried inside one logical call when the
# caller explicitly opts into more than one attempt (max_attempts > 1).
# This is a representability seam, not an autonomous retry policy: the default
# is max_attempts=1 (no retry), and no backoff, swapping, or scoring happens
# here.
RETRYABLE_ERROR_KINDS = ("transient_failure", "empty_output", "rate_limited")


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
        retry_on: tuple[str, ...] = RETRYABLE_ERROR_KINDS,
        model_config: Any | None = None,
    ) -> RecordedCall:
        """Execute one logical cognition call with explicit attempt accounting.

        Crash-window contract (clean-restart durability only, no atomicity
        claims): the sequence per attempt is ``attempt.started`` -> provider
        effect -> response-body blob stored -> ``attempt.observed`` appended
        -> legacy mixed envelope stored -> ``attempt.completed`` appended,
        then ``call.completed``. Ambiguity windows, from newest evidence
        backwards:
        - crash after ``attempt.observed`` but before ``attempt.completed``:
          transport observation exists, interpretation completion absent;
        - crash after body-blob storage but before ``attempt.observed``:
          an unreferenced content-addressed response body may exist;
        - crash between the provider effect and body storage: only
          ``attempt.started`` exists and the observation is unknown
          (the provider may or may not have served the request).
        ``call.manifest`` is appended before any attempt, so an interrupted
        call is always inspectable as UNRESOLVED. Exactly-once execution is
        NOT promised; retries create new attempts, never new tasks.

        Cognition completion (!= task completion): a "succeeded" recorded
        call only means the cognition operation produced an output. It says
        nothing about correctness, support, authorization, or verification.

        No automatic retry policy lives here: ``max_attempts`` defaults to 1.
        Passing ``max_attempts > 1`` explicitly opts into bounded re-attempts
        for classifications in ``retry_on`` within this single logical call.
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        request_event = Event.create(
            stream_id=spec.call_id,
            kind="call.requested",
            actor_id=spec.actor.actor_id,
            payload=self._call_spec_payload(spec),
            correlation_id=spec.task_id,
        )
        self.ledger.append(request_event)

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
            attempt = self._record_attempt(
                spec,
                manifest=manifest,
                result=observed,
                attempt_id=attempt_id,
                attempt_index=attempt_index,
                started_at=started_at,
                finished_at=finished_at,
                causation_id=request_event.event_id,
            )
            attempts.append(attempt)
            last_result = self._interpret_attempt(spec, manifest=manifest, attempt=attempt, result=observed)
            if attempt.status == AttemptStatus.SUCCEEDED.value:
                break
            if attempt_index < max_attempts and (attempt.error_kind in retry_on):
                continue
            break

        assert last_result is not None
        call_status = (
            LogicalCallStatus.SUCCEEDED.value
            if attempts and attempts[-1].status == AttemptStatus.SUCCEEDED.value
            else LogicalCallStatus.FAILED.value
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
        )
        return RecordedCall(
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

    def get_recorded_call(self, call_id: str) -> RecordedCall | None:
        """Reconstruct a logical call (manifest + attempts + status) from the ledger."""
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
        if manifest_event is None and not attempt_events:
            return None
        if manifest_event is None:
            return None
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
    ) -> AttemptRecord:
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
        status, error_kind = _classify_attempt(result)
        latency_ms = result.latency_ms
        if latency_ms is None:
            latency_ms = None
        effective = (
            sanitize_effective_params(dict(result.effective_parameters))
            if result.effective_parameters
            else dict(manifest.effective_parameters)
        )
        # Pure transport evidence first: body bytes (if any) -> blob, then the
        # attempt.observed event. The legacy mixed envelope below is kept
        # unchanged for compatibility; interpretation still flows from result.
        self._store_transport_observation(
            spec=spec,
            result=result,
            attempt_id=attempt_id,
            attempt_index=attempt_index,
            provider=provider,
            causation_id=causation_id,
        )
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
        return attempt

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
    ) -> None:
        """Persist pure transport evidence for one attempt, if the adapter saw any.

        Stores the exact response body bytes (when they exist) as a
        content-addressed blob, then appends attempt.observed with transport
        metadata and the blob reference. Adapters without transport evidence
        (fakes, legacy dict fixtures) produce no event: absent observation
        means unavailable under this schema, never an empty observation.
        """
        transport = result.transport
        if transport is None or self.artifact_store is None:
            if transport is None:
                return
            # No artifact store: observation cannot be preserved; record the
            # metadata anyway so the transport outcome is not silently lost.
            self._append_attempt_observed(
                spec=spec,
                result=result,
                attempt_id=attempt_id,
                attempt_index=attempt_index,
                provider=provider,
                body_ref=None,
                byte_length=None,
                causation_id=causation_id,
            )
            return
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
        self._append_attempt_observed(
            spec=spec,
            result=result,
            attempt_id=attempt_id,
            attempt_index=attempt_index,
            provider=provider,
            body_ref=body_ref,
            byte_length=byte_length,
            causation_id=causation_id,
        )

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
    ) -> None:
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
                result = self.invoke_call(spec, adapter=adapter)
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

    def compile_and_record_context(self, *, actor_id: str = "runtime", task_id: str, **kwargs: object) -> tuple[ContextPackage, CompilationTrace]:
        compiler = ContextCompiler()
        package, trace = compiler.compile_with_trace(task_id=task_id, **kwargs)  # type: ignore[arg-type]
        self._append_context_compiled(package, trace, actor_id=actor_id, task_id=task_id)
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
    def _call_spec_payload(spec: CallSpec) -> dict[str, object]:
        payload = asdict(spec)
        payload["idempotency_key"] = spec.idempotency_key
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
        )

    def _append_context_compiled(
        self, package: ContextPackage, trace: CompilationTrace, *, actor_id: str, task_id: str
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
                },
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
        "created_at": manifest.created_at,
    }


def _manifest_from_payload(payload: dict[str, Any]) -> CallManifest:
    requested = payload.get("requested_parameters")
    effective = payload.get("effective_parameters")
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
        created_at=payload.get("created_at"),
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
    )


def _classify_attempt(result: CallResult) -> tuple[str, str | None]:
    """Derive attempt status + error kind from an adapter interpretation."""
    if result.status == "succeeded" and result.raw_output:
        return AttemptStatus.SUCCEEDED.value, None
    if result.status == "succeeded" and not result.raw_output:
        # Adapter claimed success but produced nothing observable.
        return AttemptStatus.TRANSIENT_FAILURE.value, "empty_output"
    error_kind = result.error_kind
    if error_kind is None:
        error_kind = "provider_error"
    if error_kind in ("transient_failure", "empty_output", "timeout", "rate_limited"):
        return AttemptStatus.TRANSIENT_FAILURE.value, error_kind
    return AttemptStatus.FAILED.value, error_kind


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
