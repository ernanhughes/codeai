from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from .adapters import (
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
)
from .artifacts import FileArtifactStore
from .context import CompilationTrace, ContextCompiler
from .domain import (
    ActorRef,
    ArtifactRef,
    Authority,
    Capability,
    Claim,
    ClaimRelationship,
    ClaimStatus,
    ContextPackage,
    Directive,
    EvidenceClass,
    Seal,
    Task,
    Variant,
)
from .ledger import Event, SQLiteLedger
from .policy import AuthorityDenied, PolicyEngine


class PreconditionMismatch(RuntimeError):
    pass


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
                        payload=asdict(result),
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
        )

    def _append_call_completed(
        self, result: CallResult, *, spec: CallSpec, causation_id: str | None
    ) -> None:
        payload = asdict(result)
        payload["idempotency_key"] = spec.idempotency_key
        payload["task_id"] = spec.task_id
        payload["directive_id"] = spec.directive_id
        payload["run_id"] = spec.run_id
        payload["adapter_id"] = spec.adapter_id
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
                raw_artifact = ArtifactRef(**raw)
            except TypeError:
                raw_artifact = None
        return CallResult(
            call_id=str(payload["call_id"]),
            raw_output=str(payload.get("raw_output", "")),
            input_tokens=int(payload.get("input_tokens", 0) or 0),
            output_tokens=int(payload.get("output_tokens", 0) or 0),
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
