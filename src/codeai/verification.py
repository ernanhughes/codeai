"""What a check established, and what it was checking (the Chapter 21 seam).

A verification answers two questions that are easy to run together and must not
be:

    what did the verifier say?
    what state was it actually looking at?

The second is the runtime's to answer, never the verifier's. A verifier that
reports which state it examined is reporting its own opinion of its own subject;
the reading that counts is the one the runtime took before calling it.

Four outcomes, kept semantically distinct, because collapsing them is how a
process starts treating a broken measurement as a negative result:

    PASS          the check ran against the intended target and established
                  its criterion
    FAIL          the check ran against the intended target and established
                  that the criterion was not satisfied
    INCONCLUSIVE  the check ran legitimately, but the available evidence
                  cannot establish PASS or FAIL
    ERROR         no trustworthy verification result was obtained at all:
                  binding failure, verifier crash, malformed result, an
                  unmapped outcome, an unavailable target

The engineering response to each differs, which is the whole point:

    FAIL         -> investigate the work
    INCONCLUSIVE -> gather better evidence
    ERROR        -> repair the measurement

Two rules protect that distinction.

**INCONCLUSIVE may never come from a binding failure.** Infrastructure
uncertainty is ERROR. Dressing it as INCONCLUSIVE would let a process that
cannot measure anything report that the world is merely unclear.

**An INCONCLUSIVE result must carry a reason.** An unexplained "I do not know"
is not evidence, so it is treated as a malformed result: ERROR.

What this establishes is narrow and worth stating plainly: *given this declared
check and this bound target, what result did the verification mechanism
produce?* It does not establish that the check was the right test, that the
criterion was sufficient, that coverage was adequate, that the verifier was
independent, or that the environment was hermetic.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from .adapters import CheckRequest, CheckResult, CheckVerdict, VerificationAdapter
from .artifacts import ArtifactCorruptionError
from .domain import ClaimStatus, EvidenceClass
from .interpretation import now_utc
from .ledger import Event

if TYPE_CHECKING:
    from .runtime import Runtime

VERIFICATION_V1 = "verification-v1"

ARTIFACT_PLACEHOLDER = "{artifact}"
ARTIFACT_TARGET = re.compile(r"^artifact:sha256:([0-9a-f]{64})$")

CHECK_REQUESTED = "check.requested"
CHECK_COMPLETED = "check.completed"
CLAIM_CHECK_INCONCLUSIVE = "claim.check_inconclusive"

_VERDICTS = frozenset(str(value) for value in CheckVerdict)


class BindingStatus(StrEnum):
    """What the runtime could establish about the state the check examined."""

    BOUND = "bound"              # a hash was requested, read, and matched
    UNBOUND = "unbound"          # no binding was requested; nothing pins the subject
    MISMATCH = "mismatch"        # the state moved, or was never what was expected
    UNAVAILABLE = "unavailable"  # no reading could be obtained at all


@dataclass(frozen=True, slots=True)
class VerificationBinding:
    """The subject of a check, as the runtime observed it before running one."""

    requested_state_hash: str | None
    observed_state_hash: str | None
    status: str
    reason: str | None = None
    version: str = VERIFICATION_V1

    @property
    def permits_verification(self) -> bool:
        """UNBOUND permits a check; it simply does not pin what was checked."""
        return self.status in (BindingStatus.BOUND, BindingStatus.UNBOUND)

    def as_payload(self) -> dict[str, Any]:
        return {
            "requested_state_hash": self.requested_state_hash,
            "observed_state_hash": self.observed_state_hash,
            "status": str(self.status),
            "reason": self.reason,
            "version": self.version,
        }


class ArtifactBindingStatus(StrEnum):
    """What the runtime could establish about the bytes the check was given."""

    BOUND = "bound"            # resolved from the store, digest verified, supplied
    UNBOUND = "unbound"        # the check names no artifact
    MISSING = "missing"        # named, and not retrievable
    MISMATCH = "mismatch"      # the stored bytes do not hash to the named digest
    UNCONSUMED = "unconsumed"  # a command check that was never given the artifact


@dataclass(frozen=True, slots=True)
class ArtifactBinding:
    """The bytes a check was given, as the runtime resolved them.

    Acceptance used to compare the check request target string with the artifact
    it was accepting: a label the caller wrote, checked against another label.
    This resolves the artifact from the store, verifies its digest and hands the
    bytes to the check, so the identity is a reading rather than an assertion
    (composition audit gap 3).

    The honest limit, and it is the same shape as the state binding: the runtime
    establishes what the check was *given*, never what it read.
    """

    requested_artifact_sha256: str | None
    resolved_artifact_sha256: str | None
    materialized_path: str | None
    status: str
    reason: str | None = None
    version: str = VERIFICATION_V1

    @property
    def permits_verification(self) -> bool:
        return self.status in (ArtifactBindingStatus.BOUND, ArtifactBindingStatus.UNBOUND)

    def as_payload(self) -> dict[str, Any]:
        return {
            "requested_artifact_sha256": self.requested_artifact_sha256,
            "resolved_artifact_sha256": self.resolved_artifact_sha256,
            "materialized": self.materialized_path is not None,
            "status": str(self.status),
            "reason": self.reason,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class ExitCodePolicy:
    """How a command's exit code maps to a verdict, declared before it runs.

    Declaring ``2`` inconclusive for one check is a statement about that
    command. Baking it into the runtime would smuggle a convention into every
    command that ever exits 2, which is exactly the collapse this seam refuses.

    ``fail_codes=None`` keeps the ordinary shell reading: anything not declared
    otherwise is a failure. Declaring ``fail_codes`` explicitly makes the policy
    total, and an exit code outside it is then an unmapped outcome: ERROR, not a
    guess.
    """

    policy_id: str = "exit-code-v1"
    pass_codes: tuple[int, ...] = (0,)
    fail_codes: tuple[int, ...] | None = None
    inconclusive_codes: tuple[int, ...] = ()

    def verdict_for(self, exit_code: int) -> str | None:
        """The declared verdict, or None when the code is unmapped (an ERROR)."""
        if exit_code in self.pass_codes:
            return CheckVerdict.PASS
        if exit_code in self.inconclusive_codes:
            return CheckVerdict.INCONCLUSIVE
        if self.fail_codes is None or exit_code in self.fail_codes:
            return CheckVerdict.FAIL
        return None

    def as_payload(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "pass_codes": list(self.pass_codes),
            "fail_codes": None if self.fail_codes is None else list(self.fail_codes),
            "inconclusive_codes": list(self.inconclusive_codes),
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, Any] | None) -> ExitCodePolicy | None:
        if not value:
            return None
        fail = value.get("fail_codes")
        return cls(
            policy_id=str(value.get("policy_id") or "exit-code-v1"),
            pass_codes=tuple(int(code) for code in value.get("pass_codes") or ()),
            fail_codes=None if fail is None else tuple(int(code) for code in fail),
            inconclusive_codes=tuple(int(code) for code in value.get("inconclusive_codes") or ()),
        )


def _error(
    request: CheckRequest, reason: str, *, started_at: str, completed_at: str
) -> CheckResult:
    return CheckResult(
        check_id=request.check_id,
        verdict=CheckVerdict.ERROR,
        started_at=started_at,
        completed_at=completed_at,
        error=reason,
    )


def bind_check_target(runtime: Runtime, request: CheckRequest) -> VerificationBinding:
    """Read the target state once, before the verifier runs. Appends nothing.

    One reading. Re-reading to describe a mismatch would describe a third state,
    not the one the decision rested on.
    """
    if request.target_state_hash is None:
        return VerificationBinding(
            requested_state_hash=None,
            observed_state_hash=None,
            status=BindingStatus.UNBOUND.value,
            reason="no target state was requested, so the check is not pinned to one",
        )

    try:
        observed = runtime._current_state_hash()
    except Exception as exc:  # noqa: BLE001 - an unreadable target must not run the verifier
        return VerificationBinding(
            requested_state_hash=request.target_state_hash,
            observed_state_hash=None,
            status=BindingStatus.UNAVAILABLE.value,
            reason=f"target state observation failed: {type(exc).__name__}: {exc}",
        )

    if observed is None:
        return VerificationBinding(
            requested_state_hash=request.target_state_hash,
            observed_state_hash=None,
            status=BindingStatus.UNAVAILABLE.value,
            reason="target state unavailable: requested binding cannot be checked",
        )
    if observed != request.target_state_hash:
        return VerificationBinding(
            requested_state_hash=request.target_state_hash,
            observed_state_hash=observed,
            status=BindingStatus.MISMATCH.value,
            reason=(
                f"target state mismatch: expected {request.target_state_hash}, "
                f"observed {observed}"
            ),
        )
    return VerificationBinding(
        requested_state_hash=request.target_state_hash,
        observed_state_hash=observed,
        status=BindingStatus.BOUND.value,
        reason=None,
    )


def bind_check_artifact(
    runtime: Runtime, request: CheckRequest, workdir: str | None
) -> tuple[ArtifactBinding, CheckRequest]:
    """Resolve the named artifact, verify it, and give it to the check.

    Returns the binding and the request the verifier should actually receive:
    ARTIFACT_PLACEHOLDER in the command is substituted with the path the runtime
    wrote, and ``materialized_artifact_path`` is set from the runtime, never from
    the caller.
    """
    prepared = replace(request, materialized_artifact_path=None)
    target = request.target or ""
    match = ARTIFACT_TARGET.match(target)
    if match is None:
        reason = (
            "the check names no artifact"
            if not target
            else f"target {target!r} does not name an artifact"
        )
        return (
            ArtifactBinding(None, None, None, ArtifactBindingStatus.UNBOUND.value, reason),
            prepared,
        )

    requested = match.group(1)
    store = getattr(runtime, "artifact_store", None)
    if store is None or workdir is None:
        return (
            ArtifactBinding(
                requested, None, None, ArtifactBindingStatus.MISSING.value,
                "no artifact store is configured, so the bytes cannot be established",
            ),
            prepared,
        )

    try:
        content = store.read_bytes(requested)
    except ArtifactCorruptionError as exc:
        return (
            ArtifactBinding(requested, None, None, ArtifactBindingStatus.MISMATCH.value, str(exc)),
            prepared,
        )
    except Exception as exc:  # noqa: BLE001 - an unretrievable artifact is not a verdict
        return (
            ArtifactBinding(
                requested, None, None, ArtifactBindingStatus.MISSING.value,
                f"artifact {requested} could not be read: {type(exc).__name__}: {exc}",
            ),
            prepared,
        )

    observed = hashlib.sha256(content).hexdigest()
    if observed != requested:
        return (
            ArtifactBinding(
                requested, observed, None, ArtifactBindingStatus.MISMATCH.value,
                f"artifact digest mismatch: expected {requested}, observed {observed}",
            ),
            prepared,
        )

    path = Path(workdir) / f"artifact-{requested[:16]}"
    path.write_bytes(content)
    materialized = str(path)

    if request.command:
        if not any(ARTIFACT_PLACEHOLDER in part for part in request.command):
            # A command check that never receives the artifact cannot have
            # examined it. That is a broken measurement, not a verdict.
            return (
                ArtifactBinding(
                    requested, observed, None, ArtifactBindingStatus.UNCONSUMED.value,
                    f"the command never references {ARTIFACT_PLACEHOLDER}, so the artifact it "
                    f"claims to check was never given to it",
                ),
                prepared,
            )
        prepared = replace(
            prepared,
            command=tuple(part.replace(ARTIFACT_PLACEHOLDER, materialized)
                          for part in request.command),
            materialized_artifact_path=materialized,
        )
    else:
        prepared = replace(prepared, materialized_artifact_path=materialized)

    return (
        ArtifactBinding(requested, observed, materialized, ArtifactBindingStatus.BOUND.value, None),
        prepared,
    )


def execute_verification(
    runtime: Runtime,
    request: CheckRequest,
    binding: VerificationBinding,
    verifier: VerificationAdapter,
    artifact_binding: ArtifactBinding | None = None,
) -> CheckResult:
    """Run the verifier if the binding permits it, and vet what comes back.

    Every way of not obtaining a trustworthy answer lands on ERROR, including a
    result that is merely unusable: a foreign check id, an unknown verdict, an
    unexplained INCONCLUSIVE, or a policy other than the one declared.
    """
    now = now_utc()
    if not binding.permits_verification:
        # Infrastructure uncertainty is never epistemic uncertainty.
        return _error(request, binding.reason or "target binding failed", started_at=now,
                      completed_at=now_utc())
    if artifact_binding is not None and not artifact_binding.permits_verification:
        return _error(request, artifact_binding.reason or "artifact binding failed",
                      started_at=now, completed_at=now_utc())

    started_at = now
    try:
        result = verifier.run(request)
    except Exception as exc:  # noqa: BLE001 - a raising verifier is an ERROR, never a pass
        return _error(request, f"verifier raised {type(exc).__name__}: {exc}",
                      started_at=started_at, completed_at=now_utc())

    problem = _unusable(request, result)
    if problem is not None:
        return _error(request, problem, started_at=started_at, completed_at=now_utc())
    return result


def _unusable(request: CheckRequest, result: object) -> str | None:
    """Why this result cannot be trusted as a verdict, or None when it can."""
    if not isinstance(result, CheckResult):
        return f"verifier returned {type(result).__name__}, not a CheckResult"
    if result.check_id != request.check_id:
        return f"verifier returned a result for check {result.check_id!r}"
    verdict = str(result.verdict)
    if verdict not in _VERDICTS:
        return f"verifier returned an unknown verdict: {verdict!r}"
    if verdict == CheckVerdict.INCONCLUSIVE and not result.inconclusive_reason:
        return "inconclusive result carries no reason"
    declared = ExitCodePolicy.from_payload(request.verdict_policy)
    if (
        declared is not None
        and result.verdict_policy_id is not None
        and result.verdict_policy_id != declared.policy_id
    ):
        return (
            f"verifier applied policy {result.verdict_policy_id!r}, "
            f"but {declared.policy_id!r} was declared"
        )
    return None


def finalize_verification(
    runtime: Runtime,
    result: CheckResult,
    binding: VerificationBinding,
    artifact_binding: ArtifactBinding | None = None,
) -> CheckResult:
    """Attach artifacts, then overwrite the verifier's account of its subject.

    A verifier cannot say what state it checked. Whatever it put in
    ``observed_target_state_hash`` is discarded in favour of the runtime's own
    reading, so a buggy or hostile verifier cannot describe its own target.
    """
    stored = runtime._store_check_artifacts(result)
    return replace(
        stored,
        observed_target_state_hash=binding.observed_state_hash,
        binding_status=str(binding.status),
        artifact_binding_status=(
            str(artifact_binding.status) if artifact_binding is not None else None
        ),
    )


def record_verification(
    runtime: Runtime,
    request: CheckRequest,
    binding: VerificationBinding,
    result: CheckResult,
    request_event: Event,
    verifier: VerificationAdapter,
    artifact_binding: ArtifactBinding | None = None,
) -> Event:
    """Append what was verified, against what, by whom, under which policy."""
    event = Event.create(
        stream_id=request.check_id,
        kind=CHECK_COMPLETED,
        actor_id="verifier",
        payload={
            **asdict(result),
            "binding": binding.as_payload(),
            "artifact_binding": (
                artifact_binding.as_payload() if artifact_binding is not None else None
            ),
            "declared_verdict_policy": dict(request.verdict_policy)
            if request.verdict_policy
            else None,
            "verifier_identity": _verifier_identity(verifier),
            "verification_version": VERIFICATION_V1,
        },
        causation_id=request_event.event_id,
        correlation_id=request.task_id,
    )
    runtime.ledger.append(event)
    return event


def _verifier_identity(verifier: object) -> dict[str, Any]:
    """Who ran it, as the runtime knows it: the object it actually called."""
    return {
        "name": type(verifier).__name__,
        "module": type(verifier).__module__,
        "version": getattr(verifier, "version", None),
    }


def apply_verification_to_claims(
    runtime: Runtime, request: CheckRequest, result: CheckResult
) -> None:
    """Scoped promotion: only claims the check named, and only what it settled.

    PASS and FAIL move the claims the check targeted. INCONCLUSIVE records that
    the attempt happened and settled nothing, which is a fact worth keeping and
    not a change of status. ERROR records nothing against the claim at all:

        verification failed  !=  claim disproved
    """
    if not request.claim_ids:
        return
    verdict = str(result.verdict)
    if verdict == CheckVerdict.PASS:
        for claim_id in request.claim_ids:
            runtime.ledger.append(
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
    elif verdict == CheckVerdict.FAIL:
        for claim_id in request.claim_ids:
            runtime.ledger.append(
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
    elif verdict == CheckVerdict.INCONCLUSIVE:
        for claim_id in request.claim_ids:
            runtime.ledger.append(
                Event.create(
                    stream_id=str(claim_id),
                    kind=CLAIM_CHECK_INCONCLUSIVE,
                    actor_id="verifier",
                    payload={
                        "claim_id": str(claim_id),
                        "check_id": result.check_id,
                        "verdict": verdict,
                        "inconclusive_reason": result.inconclusive_reason,
                        "details": (
                            "the check ran and could not settle this claim; "
                            "its status is unchanged"
                        ),
                        "version": VERIFICATION_V1,
                    },
                    correlation_id=request.task_id,
                )
            )


def inconclusive_checks_for_claim(runtime: Runtime, claim_id: str) -> tuple[dict[str, Any], ...]:
    """Every check that ran against this claim and could not settle it."""
    return tuple(
        event.payload
        for event in runtime.ledger.events_by_kind((CLAIM_CHECK_INCONCLUSIVE,))
        if str(event.payload.get("claim_id")) == str(claim_id)
    )


def run_check(
    runtime: Runtime, request: CheckRequest, *, verifier: VerificationAdapter
) -> CheckResult:
    """Bind the target, run the verifier, vet the result, record it, apply it."""
    request_event = Event.create(
        stream_id=request.check_id,
        kind=CHECK_REQUESTED,
        actor_id="verifier",
        payload=asdict(request),
        correlation_id=request.task_id,
    )
    runtime.ledger.append(request_event)

    binding = bind_check_target(runtime, request)
    with TemporaryDirectory(prefix="codeai-check-") as workdir:
        artifact_binding, prepared = bind_check_artifact(runtime, request, workdir)
        result = execute_verification(runtime, prepared, binding, verifier, artifact_binding)
        result = finalize_verification(runtime, result, binding, artifact_binding)
    record_verification(
        runtime, request, binding, result, request_event, verifier, artifact_binding
    )
    apply_verification_to_claims(runtime, request, result)
    return result
