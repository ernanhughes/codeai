from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime

from .adapters import CheckRequest, CheckResult, CheckVerdict, VerificationAdapter


class LocalCommandVerifier(VerificationAdapter):
    """Deterministic local command verifier with explicit argv/cwd/timeout inputs."""

    def run(self, request: CheckRequest) -> CheckResult:
        started_at = datetime.now(UTC).isoformat()
        if not request.command:
            completed_at = datetime.now(UTC).isoformat()
            return CheckResult(
                check_id=request.check_id,
                verdict=CheckVerdict.ERROR,
                started_at=started_at,
                completed_at=completed_at,
                error="verification command is required",
            )

        environment = os.environ.copy()
        environment.update(request.environment)
        try:
            completed = subprocess.run(
                request.command,
                cwd=request.cwd,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=request.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            completed_at = datetime.now(UTC).isoformat()
            return CheckResult(
                check_id=request.check_id,
                verdict=CheckVerdict.ERROR,
                started_at=started_at,
                completed_at=completed_at,
                stdout=_ensure_text(exc.stdout),
                stderr=_ensure_text(exc.stderr),
                error=f"verification timed out after {request.timeout_seconds} seconds",
            )
        except OSError as exc:
            completed_at = datetime.now(UTC).isoformat()
            return CheckResult(
                check_id=request.check_id,
                verdict=CheckVerdict.ERROR,
                started_at=started_at,
                completed_at=completed_at,
                error=str(exc),
            )

        completed_at = datetime.now(UTC).isoformat()
        return CheckResult(
            check_id=request.check_id,
            verdict=CheckVerdict.PASS if completed.returncode == 0 else CheckVerdict.FAIL,
            started_at=started_at,
            completed_at=completed_at,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            details="command completed",
        )


def _ensure_text(value: str | bytes | None) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return value.decode("utf-8", errors="replace")
