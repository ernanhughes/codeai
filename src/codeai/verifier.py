from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime

from .adapters import CheckRequest, CheckResult, CheckVerdict, VerificationAdapter
from .verification import ExitCodePolicy


class LocalCommandVerifier(VerificationAdapter):
    """Deterministic local command verifier with explicit argv/cwd/timeout inputs.

    The exit-code mapping is a property of the command, not of the runtime, so
    it is declared rather than assumed. The default keeps the ordinary shell
    reading -- 0 passes, anything else fails -- and a command that can say "I
    cannot tell" declares which codes mean that:

        LocalCommandVerifier(pass_exit_codes=(0,), fail_exit_codes=(1,),
                             inconclusive_exit_codes=(2,))

    A request may declare the mapping instead (``CheckRequest.verdict_policy``),
    which is the form that ends up in the ledger: declared before execution,
    recorded, and not inferred afterwards from whatever came back. When both are
    present the request wins, because that is the one a later reader can see.

    Not obtaining a result at all -- a missing command, a timeout, an OS error,
    an exit code the policy does not map -- is ERROR, never FAIL.
    """

    version = "local-command-v2"

    def __init__(
        self,
        *,
        pass_exit_codes: tuple[int, ...] = (0,),
        fail_exit_codes: tuple[int, ...] | None = None,
        inconclusive_exit_codes: tuple[int, ...] = (),
    ) -> None:
        self.policy = ExitCodePolicy(
            pass_codes=tuple(pass_exit_codes),
            fail_codes=None if fail_exit_codes is None else tuple(fail_exit_codes),
            inconclusive_codes=tuple(inconclusive_exit_codes),
        )

    def policy_for(self, request: CheckRequest) -> ExitCodePolicy:
        """The declared mapping this check runs under; the request takes precedence."""
        return ExitCodePolicy.from_payload(request.verdict_policy) or self.policy

    def run(self, request: CheckRequest) -> CheckResult:
        started_at = datetime.now(UTC).isoformat()
        policy = self.policy_for(request)
        if not request.command:
            return CheckResult(
                check_id=request.check_id,
                verdict=CheckVerdict.ERROR,
                started_at=started_at,
                completed_at=datetime.now(UTC).isoformat(),
                error="verification command is required",
                verdict_policy_id=policy.policy_id,
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
            return CheckResult(
                check_id=request.check_id,
                verdict=CheckVerdict.ERROR,
                started_at=started_at,
                completed_at=datetime.now(UTC).isoformat(),
                stdout=_ensure_text(exc.stdout),
                stderr=_ensure_text(exc.stderr),
                error=f"verification timed out after {request.timeout_seconds} seconds",
                verdict_policy_id=policy.policy_id,
            )
        except OSError as exc:
            return CheckResult(
                check_id=request.check_id,
                verdict=CheckVerdict.ERROR,
                started_at=started_at,
                completed_at=datetime.now(UTC).isoformat(),
                error=str(exc),
                verdict_policy_id=policy.policy_id,
            )

        completed_at = datetime.now(UTC).isoformat()
        verdict = policy.verdict_for(completed.returncode)
        if verdict is None:
            # The command produced an outcome the declared policy does not cover.
            # Choosing a verdict here would be a guess wearing a verdict's clothes.
            return CheckResult(
                check_id=request.check_id,
                verdict=CheckVerdict.ERROR,
                started_at=started_at,
                completed_at=completed_at,
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                error=(
                    f"exit code {completed.returncode} is not mapped by "
                    f"{policy.policy_id}: pass={list(policy.pass_codes)} "
                    f"fail={list(policy.fail_codes or [])} "
                    f"inconclusive={list(policy.inconclusive_codes)}"
                ),
                verdict_policy_id=policy.policy_id,
            )

        inconclusive_reason = None
        if verdict == CheckVerdict.INCONCLUSIVE:
            # An inconclusive result always says why, and says it in terms of the
            # mapping that was declared before the command ran.
            detail = (completed.stderr or completed.stdout or "").strip().splitlines()
            inconclusive_reason = (
                f"exit code {completed.returncode} is declared inconclusive by "
                f"{policy.policy_id}"
            )
            if detail:
                inconclusive_reason += f": {detail[-1][:200]}"

        return CheckResult(
            check_id=request.check_id,
            verdict=verdict,
            started_at=started_at,
            completed_at=completed_at,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            details="command completed",
            inconclusive_reason=inconclusive_reason,
            verdict_policy_id=policy.policy_id,
        )


def _ensure_text(value: str | bytes | None) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return value.decode("utf-8", errors="replace")
