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
    CheckRequest,
    CheckResult,
    CheckVerdict,
    ExecutionAdapter,
    VerificationAdapter,
)
from .artifacts import FileArtifactStore
from .domain import ArtifactRef, Authority, Capability, Directive, Task
from .ledger import Event, SQLiteLedger
from .policy import AuthorityDenied, PolicyEngine


class PreconditionMismatch(RuntimeError):
    pass


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def default_repository_state_hash(path: str | Path) -> str:
    root = Path(path)
    git_dir = root / ".git"
    if git_dir.exists():
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--short", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return hashlib.sha256(f"{head}\0{status}".encode()).hexdigest()

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
            actor_id=request.actor_id,
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
        return result

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
        return {
            "run_id": run_id,
            "created_at": directive_event.created_at,
            "objective": directive_event.payload["objective"],
            "success_criteria": directive_event.payload["success_criteria"],
            "tasks": tasks,
            "actions": actions,
            "checks": checks,
        }

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
