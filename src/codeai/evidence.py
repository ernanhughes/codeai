"""Claims, evidence and decisions (Stage 18).

A claim is attributed, not supported. ``claim.extracted`` records who said
what, and where: an exact span of the output text read from a preserved
provider response, with its hash. Support is a separate, validated record,
``claim.evidence_recorded``: a passage in a preserved source artifact, or a
completed check that targeted the claim. A claim's status and evidence class
are projected from those records. Nothing, including the claim's own author,
can assert them.

A decision names the claims it relies on and records a snapshot of their
standing (``decision.recorded``). ``project_decision_standing`` re-derives
that standing later and reports what changed. It never revokes or edits the
decision.

Refusals are recorded (``claim.refused``, ``claim.evidence_refused``,
``decision.refused``) and raised. Nothing else is appended when a request is
refused.

The legacy ``claim.recorded`` / ``claim.evidence`` / ``claim.status`` path is
unchanged; its events are ignored here and counted on the claim's standing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any, NoReturn

from .artifacts import ArtifactCorruptionError
from .domain import ClaimStatus, EvidenceClass, LogicalCallStatus
from .interpretation import ObservationUnavailable
from .ledger import Event
from .providers import output_text_for

if TYPE_CHECKING:
    from .runtime import Runtime

CLAIM_EVIDENCE_V1 = "claim-evidence-v1"
DECISION_BASIS_V1 = "decision-basis-v1"
KNOWN_CLAIM_POLICIES = frozenset({CLAIM_EVIDENCE_V1})
KNOWN_DECISION_POLICIES = frozenset({DECISION_BASIS_V1})

CLAIM_EXTRACTED = "claim.extracted"
CLAIM_REFUSED = "claim.refused"
EVIDENCE_RECORDED = "claim.evidence_recorded"
EVIDENCE_REFUSED = "claim.evidence_refused"
DECISION_RECORDED = "decision.recorded"
DECISION_REFUSED = "decision.refused"
LEGACY_CLAIM_KINDS = ("claim.recorded", "claim.evidence", "claim.status")

SOURCE_PASSAGE = "source_passage"
CHECK = "check"
SUPPORTS = "supports"
REFUTES = "refutes"

_RANK = {
    EvidenceClass.ASSERTED: 0,
    EvidenceClass.ATTRIBUTED: 1,
    EvidenceClass.SOURCE_CHECKED: 2,
    EvidenceClass.REPRODUCED: 3,
    EvidenceClass.ROBUST: 4,
}
_CLASS_FOR_KIND = {SOURCE_PASSAGE: EvidenceClass.SOURCE_CHECKED, CHECK: EvidenceClass.REPRODUCED}
# Under decision-basis-v1 a decision may rely only on claims supported at least this well.
MINIMUM_DECISION_EVIDENCE = EvidenceClass.SOURCE_CHECKED
# Fields of a basis snapshot whose change alters a decision's standing.
TRACKED_BASIS_FIELDS = (
    "status",
    "evidence_class",
    "supporting_evidence_ids",
    "refuting_evidence_ids",
    "source_call_status",
    "source_status_event_id",
)


class _Refusal(RuntimeError):
    subject = "request"

    def __init__(self, subject_id: str, reasons: Iterable[str], *, event_id: str | None = None) -> None:
        self.subject_id = subject_id
        self.reasons = tuple(reasons)
        self.event_id = event_id
        super().__init__(f"{self.subject} {subject_id} refused: " + ", ".join(self.reasons))


class ClaimRefused(_Refusal):
    subject = "claim"


class EvidenceRefused(_Refusal):
    subject = "evidence"


class DecisionRefused(_Refusal):
    subject = "decision"


@dataclass(frozen=True, slots=True)
class ClaimExtraction:
    """An exact span of a preserved attempt's output text, and the statement it carries."""

    claim_id: str
    task_id: str
    call_id: str
    attempt_id: str
    span_start: int
    span_end: int
    quote: str
    statement: str
    extracted_by: str
    scope: str | None = None
    conditions: str | None = None
    policy_version: str = CLAIM_EVIDENCE_V1


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Support or refutation for one claim: a source passage, or a check that targeted it."""

    evidence_id: str
    claim_id: str
    kind: str
    verdict: str
    actor_id: str
    source_artifact_id: str | None = None
    passage_start: int | None = None
    passage_end: int | None = None
    passage: str | None = None
    check_id: str | None = None
    note: str | None = None
    policy_version: str = CLAIM_EVIDENCE_V1


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    decision_id: str
    task_id: str
    actor_id: str
    statement: str
    relied_on_claim_ids: tuple[str, ...]
    acknowledged_unresolved_claim_ids: tuple[str, ...] = ()
    policy_version: str = DECISION_BASIS_V1


@dataclass(frozen=True, slots=True)
class ClaimStanding:
    claim_id: str
    task_id: str
    statement: str
    quote: str
    quote_sha256: str
    call_id: str
    attempt_id: str
    observation_sha256: str | None
    status: str
    evidence_class: str
    supporting_evidence_ids: tuple[str, ...]
    refuting_evidence_ids: tuple[str, ...]
    source_call_status: str | None
    source_status_event_id: str | None
    source_status_basis: str | None
    extracted_event_id: str
    ignored_legacy_event_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DecisionStanding:
    decision_id: str
    standing: str  # "basis_intact" | "basis_changed" | "not_found"
    decision_event_id: str | None = None
    relied_on_claim_ids: tuple[str, ...] = ()
    changes: tuple[dict[str, Any], ...] = ()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------- claims ----------------


def extract_claim(runtime: Runtime, request: ClaimExtraction) -> ClaimStanding:
    """Record a claim as attributed to an exact span of preserved output.

    Attribution establishes who said what, and where. It is not support: the
    claim starts unresolved at E1_ATTRIBUTED whatever it says. An identical
    repeat returns the existing claim; a different claim under the same id is
    refused.
    """
    prior = _extracted(runtime, request.claim_id)
    if prior is not None:
        if _identity(prior.payload, ClaimExtraction) != _plain(asdict(request)):
            _refuse(runtime, ClaimRefused, CLAIM_REFUSED, request.claim_id, request,
                    ["conflicting_claim"], actor_id=request.extracted_by, task_id=request.task_id)
        return _standing(runtime, prior)

    reasons: list[str] = []
    detail: dict[str, Any] = {}
    if request.policy_version not in KNOWN_CLAIM_POLICIES:
        reasons.append("unknown_policy")
    if not request.extracted_by:
        reasons.append("missing_actor")
    if not request.statement.strip():
        reasons.append("empty_statement")
    if not _task_exists(runtime, request.task_id):
        reasons.append("unknown_task")

    observed: Event | None = None
    recorded = runtime.get_recorded_call(request.call_id)
    if recorded is None:
        reasons.append("source_call_not_found")
    else:
        if recorded.task_id != request.task_id:
            reasons.append("source_call_wrong_task")
        if not any(a.attempt_id == request.attempt_id for a in recorded.attempts):
            reasons.append("source_attempt_not_in_call")
        else:
            try:
                text, observed = _observation_text(runtime, request.attempt_id)
            except ObservationUnavailable as exc:
                reasons.append("observation_unavailable")
                detail["observation"] = exc.reason
            else:
                if not 0 <= request.span_start < request.span_end <= len(text):
                    reasons.append("span_out_of_range")
                elif text[request.span_start : request.span_end] != request.quote:
                    reasons.append("quote_not_in_source")
    if reasons or observed is None:
        _refuse(runtime, ClaimRefused, CLAIM_REFUSED, request.claim_id, request,
                reasons or ["observation_unavailable"], actor_id=request.extracted_by,
                task_id=request.task_id, detail=detail)

    ref = observed.payload["response_body_artifact"]
    event = Event.create(
        stream_id=request.claim_id,
        kind=CLAIM_EXTRACTED,
        actor_id=request.extracted_by,
        payload={
            **_plain(asdict(request)),
            "quote_sha256": text_sha256(request.quote),
            "observation_event_id": observed.event_id,
            "observation_sha256": str(ref.get("sha256")),
            "protocol": observed.payload.get("protocol"),
            "text_basis": "providers.output_text_for over the preserved response body",
            "evidence_class": EvidenceClass.ATTRIBUTED.value,
        },
        causation_id=observed.event_id,
        correlation_id=request.task_id,
    )
    runtime.ledger.append(event)
    return _standing(runtime, event)


def record_evidence(runtime: Runtime, request: EvidenceRecord) -> ClaimStanding:
    """Validate and record support or refutation for one claim.

    ``source_passage`` (E2_SOURCE_CHECKED): an exact passage of a preserved
    source artifact, which may not be the claim's own source response.
    ``check`` (E3_REPRODUCED): a completed check whose request named the claim,
    with PASS supporting and FAIL refuting; any other verdict is inconclusive.
    Whether a passage bears on the claim is the recording actor's judgment; the
    actor is recorded and may not be the actor that produced the claim.
    """
    claim = _extracted(runtime, request.claim_id)
    task_id = str(claim.payload["task_id"]) if claim is not None else None
    prior = [e for e in runtime.ledger.events_by_kind((EVIDENCE_RECORDED,))
             if e.stream_id == request.evidence_id]
    if prior:
        if _identity(prior[0].payload, EvidenceRecord) != _plain(asdict(request)):
            _refuse(runtime, EvidenceRefused, EVIDENCE_REFUSED, request.evidence_id, request,
                    ["conflicting_evidence"], actor_id=request.actor_id, task_id=task_id)
        assert claim is not None
        return _standing(runtime, claim)

    reasons: list[str] = []
    extra: dict[str, Any] = {}
    if request.policy_version not in KNOWN_CLAIM_POLICIES:
        reasons.append("unknown_policy")
    if not request.actor_id:
        reasons.append("missing_actor")
    if claim is None:
        reasons.append("claim_not_found")
    elif request.actor_id in _producers(runtime, str(claim.payload["call_id"])):
        reasons.append("self_evidence")
    if request.verdict not in (SUPPORTS, REFUTES):
        reasons.append("unknown_verdict")

    if request.kind == SOURCE_PASSAGE:
        reasons.extend(_validate_passage(runtime, request, claim, extra))
    elif request.kind == CHECK:
        reasons.extend(_validate_check(runtime, request, task_id, extra))
    else:
        reasons.append("unknown_kind")
    if reasons or claim is None:
        _refuse(runtime, EvidenceRefused, EVIDENCE_REFUSED, request.evidence_id, request,
                reasons or ["claim_not_found"], actor_id=request.actor_id, task_id=task_id)

    event = Event.create(
        stream_id=request.evidence_id,
        kind=EVIDENCE_RECORDED,
        actor_id=request.actor_id,
        payload={
            **_plain(asdict(request)),
            **extra,
            "task_id": task_id,
            "evidence_class": _CLASS_FOR_KIND[request.kind].value,
        },
        causation_id=claim.event_id,
        correlation_id=task_id,
    )
    runtime.ledger.append(event)
    return _standing(runtime, claim)


def project_claim_standing(runtime: Runtime, claim_id: str) -> ClaimStanding | None:
    """Derive a claim's status and evidence class from recorded evidence. Appends nothing."""
    claim = _extracted(runtime, claim_id)
    return None if claim is None else _standing(runtime, claim)


def _standing(runtime: Runtime, claim: Event) -> ClaimStanding:
    payload = claim.payload
    claim_id = str(payload["claim_id"])
    evidence = [e for e in runtime.ledger.events_by_kind((EVIDENCE_RECORDED,))
                if e.payload.get("claim_id") == claim_id]
    supports = [e for e in evidence if e.payload.get("verdict") == SUPPORTS]
    refutes = [e for e in evidence if e.payload.get("verdict") == REFUTES]
    if supports and refutes:
        status = ClaimStatus.CONTESTED
    elif supports:
        status = ClaimStatus.SUPPORTED
    elif refutes:
        status = ClaimStatus.REFUTED
    else:
        status = ClaimStatus.UNRESOLVED
    evidence_class = max(
        (EvidenceClass(str(e.payload["evidence_class"])) for e in supports),
        key=_RANK.__getitem__,
        default=EvidenceClass.ATTRIBUTED,
    )
    adopted = _adopted_status(runtime, str(payload["call_id"]))
    legacy = tuple(
        e.event_id for e in runtime.ledger.events_by_kind(LEGACY_CLAIM_KINDS)
        if e.stream_id == claim_id or str(e.payload.get("claim_id", "")) == claim_id
    )
    return ClaimStanding(
        claim_id=claim_id,
        task_id=str(payload["task_id"]),
        statement=str(payload["statement"]),
        quote=str(payload["quote"]),
        quote_sha256=str(payload["quote_sha256"]),
        call_id=str(payload["call_id"]),
        attempt_id=str(payload["attempt_id"]),
        observation_sha256=payload.get("observation_sha256"),
        status=status.value,
        evidence_class=evidence_class.value,
        supporting_evidence_ids=tuple(e.stream_id for e in supports),
        refuting_evidence_ids=tuple(e.stream_id for e in refutes),
        source_call_status=str(adopted.payload.get("status")) if adopted is not None else None,
        source_status_event_id=adopted.event_id if adopted is not None else None,
        source_status_basis=_status_basis(adopted),
        extracted_event_id=claim.event_id,
        ignored_legacy_event_ids=legacy,
    )


# ---------------- decisions ----------------


def record_decision(runtime: Runtime, request: DecisionRequest) -> DecisionStanding:
    """Record a decision against the claims it relies on, with a snapshot of their standing.

    Every relied-on claim must be supported, at MINIMUM_DECISION_EVIDENCE or
    better, from a source call whose adopted status is succeeded. Claims the
    decision knowingly leaves open are named separately and never relied on.
    """
    prior = [e for e in runtime.ledger.events_by_kind((DECISION_RECORDED,))
             if e.stream_id == request.decision_id]
    if prior:
        if _identity(prior[0].payload, DecisionRequest) != _plain(asdict(request)):
            _refuse(runtime, DecisionRefused, DECISION_REFUSED, request.decision_id, request,
                    ["conflicting_decision"], actor_id=request.actor_id, task_id=request.task_id)
        return project_decision_standing(runtime, request.decision_id)

    reasons: list[str] = []
    if request.policy_version not in KNOWN_DECISION_POLICIES:
        reasons.append("unknown_policy")
    if not request.actor_id:
        reasons.append("missing_actor")
    if not request.statement.strip():
        reasons.append("empty_statement")
    if not _task_exists(runtime, request.task_id):
        reasons.append("unknown_task")
    relied = list(request.relied_on_claim_ids)
    acknowledged = list(request.acknowledged_unresolved_claim_ids)
    if not relied:
        reasons.append("no_claims_relied_on")
    if len(set(relied)) != len(relied):
        reasons.append("duplicate_relied_on_claim")
    reasons.extend(f"claim_relied_on_and_acknowledged_unresolved:{claim_id}"
                   for claim_id in sorted(set(relied) & set(acknowledged)))

    basis: list[dict[str, Any]] = []
    seen: dict[str, Any] = {}
    for claim_id in relied:
        standing = project_claim_standing(runtime, claim_id)
        if standing is None:
            reasons.append(f"claim_not_found:{claim_id}")
            continue
        seen[claim_id] = _summary(standing)
        if standing.task_id != request.task_id:
            reasons.append(f"claim_wrong_task:{claim_id}")
        if standing.status != ClaimStatus.SUPPORTED.value:
            reasons.append(f"claim_not_supported:{claim_id}")
        if _RANK[EvidenceClass(standing.evidence_class)] < _RANK[MINIMUM_DECISION_EVIDENCE]:
            reasons.append(f"evidence_below_policy:{claim_id}")
        if standing.source_call_status != LogicalCallStatus.SUCCEEDED.value:
            reasons.append(f"source_call_not_succeeded:{claim_id}")
        basis.append(_snapshot(standing))
    open_claims: list[dict[str, Any]] = []
    for claim_id in acknowledged:
        standing = project_claim_standing(runtime, claim_id)
        if standing is None:
            reasons.append(f"claim_not_found:{claim_id}")
            continue
        open_claims.append(_summary(standing))
    if reasons:
        _refuse(runtime, DecisionRefused, DECISION_REFUSED, request.decision_id, request, reasons,
                actor_id=request.actor_id, task_id=request.task_id, detail={"claims": seen})

    event = Event.create(
        stream_id=request.decision_id,
        kind=DECISION_RECORDED,
        actor_id=request.actor_id,
        payload={
            **_plain(asdict(request)),
            "basis": basis,
            "acknowledged_unresolved": open_claims,
            "minimum_evidence_class": MINIMUM_DECISION_EVIDENCE.value,
        },
        correlation_id=request.task_id,
    )
    runtime.ledger.append(event)
    return project_decision_standing(runtime, request.decision_id)


def project_decision_standing(runtime: Runtime, decision_id: str) -> DecisionStanding:
    """Compare a decision's recorded basis with the claims' standing now. Appends nothing."""
    recorded = [e for e in runtime.ledger.events_by_kind((DECISION_RECORDED,))
                if e.stream_id == decision_id]
    if not recorded:
        return DecisionStanding(decision_id, "not_found")
    event = recorded[0]
    changes: list[dict[str, Any]] = []
    for snapshot in event.payload.get("basis") or ():
        claim_id = str(snapshot["claim_id"])
        standing = project_claim_standing(runtime, claim_id)
        current = _snapshot(standing) if standing is not None else {}
        for name in TRACKED_BASIS_FIELDS:
            if current.get(name) != snapshot.get(name):
                changes.append({"claim_id": claim_id, "field": name,
                                "recorded": snapshot.get(name), "current": current.get(name)})
    return DecisionStanding(
        decision_id=decision_id,
        standing="basis_changed" if changes else "basis_intact",
        decision_event_id=event.event_id,
        relied_on_claim_ids=tuple(str(c) for c in event.payload.get("relied_on_claim_ids") or ()),
        changes=tuple(changes),
    )


DECISION_EVIDENCE_V1 = "decision-evidence-v1"


@dataclass(frozen=True, slots=True)
class BasisClaimStanding:
    """One claim a decision relied on, and whether it still bears the weight."""

    claim_id: str
    verdict: str          # intact | defeated | unknown
    reason: str
    recorded_status: str | None = None
    current_status: str | None = None
    recorded_evidence_class: str | None = None
    current_evidence_class: str | None = None
    new_refuting_evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DecisionEvidenceStanding:
    """Whether a decision's recorded evidentiary basis is still admissible.

    Deliberately narrower than ``project_decision_standing``, which reports any
    difference from the recorded snapshot. A decision that has *gained*
    supporting evidence has changed and has not been defeated, and gating on
    change rather than defeat would revoke decisions for becoming better founded.
    """

    decision_id: str
    admissible: bool
    found: bool
    claims: tuple[BasisClaimStanding, ...] = ()
    decision_event_id: str | None = None
    version: str = DECISION_EVIDENCE_V1

    @property
    def defeated(self) -> tuple[BasisClaimStanding, ...]:
        return tuple(c for c in self.claims if c.verdict == "defeated")

    @property
    def unknown(self) -> tuple[BasisClaimStanding, ...]:
        return tuple(c for c in self.claims if c.verdict == "unknown")

    def as_payload(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "admissible": self.admissible,
            "found": self.found,
            "decision_event_id": self.decision_event_id,
            "claims": [
                {
                    "claim_id": c.claim_id,
                    "verdict": c.verdict,
                    "reason": c.reason,
                    "recorded_status": c.recorded_status,
                    "current_status": c.current_status,
                    "recorded_evidence_class": c.recorded_evidence_class,
                    "current_evidence_class": c.current_evidence_class,
                    "new_refuting_evidence_ids": list(c.new_refuting_evidence_ids),
                }
                for c in self.claims
            ],
            "version": DECISION_EVIDENCE_V1,
        }


DEFEATING_STATUSES = frozenset({ClaimStatus.REFUTED.value, ClaimStatus.CONTESTED.value})


def _basis_claim_standing(runtime: Runtime, snapshot: dict[str, Any]) -> BasisClaimStanding:
    """Judge one recorded basis claim against its standing now.

    The rules are frozen in experiments/W2-3-prereg.md. Attempts are not
    standing: an INCONCLUSIVE or ERROR verification moved no claim, so it cannot
    defeat a decision that relied on one.
    """
    claim_id = str(snapshot.get("claim_id"))
    standing = project_claim_standing(runtime, claim_id)
    recorded_status = snapshot.get("status")
    recorded_class = snapshot.get("evidence_class")
    if standing is None:
        return BasisClaimStanding(
            claim_id=claim_id, verdict="unknown",
            reason="the claim can no longer be projected, so its support cannot be established",
            recorded_status=recorded_status, recorded_evidence_class=recorded_class,
        )

    current = _snapshot(standing)
    recorded_refuting = tuple(str(i) for i in (snapshot.get("refuting_evidence_ids") or ()))
    current_refuting = tuple(str(i) for i in (current.get("refuting_evidence_ids") or ()))
    new_refuting = tuple(i for i in current_refuting if i not in set(recorded_refuting))
    common = {
        "claim_id": claim_id,
        "recorded_status": recorded_status,
        "current_status": current.get("status"),
        "recorded_evidence_class": recorded_class,
        "current_evidence_class": current.get("evidence_class"),
        "new_refuting_evidence_ids": new_refuting,
    }

    if new_refuting:
        # Refutation arrived after the decision. Later re-support does not
        # resurrect it: the judgment was made against a state that has since
        # been overturned, and the honest repair is a new decision.
        return BasisClaimStanding(
            verdict="defeated",
            reason=f"refuting evidence was recorded after the decision: {', '.join(new_refuting)}",
            **common,
        )
    if str(current.get("status")) in DEFEATING_STATUSES:
        return BasisClaimStanding(
            verdict="defeated",
            reason=f"the claim now stands as {current.get('status')}",
            **common,
        )
    current_class = current.get("evidence_class")
    if current_class is not None:
        rank = _RANK[EvidenceClass(str(current_class))]
        if rank < _RANK[MINIMUM_DECISION_EVIDENCE]:
            return BasisClaimStanding(
                verdict="defeated",
                reason=f"evidence fell to {current_class}, below the decision minimum",
                **common,
            )
        if recorded_class is not None and rank < _RANK[EvidenceClass(str(recorded_class))]:
            return BasisClaimStanding(
                verdict="defeated",
                reason=f"evidence fell from {recorded_class} to {current_class}",
                **common,
            )
    if (
        snapshot.get("source_call_status") == "succeeded"
        and current.get("source_call_status") != "succeeded"
    ):
        return BasisClaimStanding(
            verdict="defeated",
            reason=(
                f"the source call's adopted status is now "
                f"{current.get('source_call_status')}, not succeeded"
            ),
            **common,
        )
    return BasisClaimStanding(
        verdict="intact",
        reason="the claim still bears the weight the decision put on it",
        **common,
    )


def project_decision_evidence(runtime: Runtime, decision_id: str) -> DecisionEvidenceStanding:
    """Is this decision's recorded evidentiary basis still admissible? Appends nothing."""
    recorded = [
        event for event in runtime.ledger.events_by_kind((DECISION_RECORDED,))
        if event.stream_id == str(decision_id)
    ]
    if len(recorded) != 1:
        return DecisionEvidenceStanding(
            decision_id=str(decision_id), admissible=False, found=False
        )
    event = recorded[0]
    claims = tuple(
        _basis_claim_standing(runtime, dict(snapshot))
        for snapshot in (event.payload.get("basis") or ())
    )
    return DecisionEvidenceStanding(
        decision_id=str(decision_id),
        admissible=all(c.verdict == "intact" for c in claims),
        found=True,
        claims=claims,
        decision_event_id=event.event_id,
    )


def decisions_resting_on(runtime: Runtime, claim_id: str) -> tuple[DecisionStanding, ...]:
    """Every recorded decision that relied on the claim, with its standing now."""
    return tuple(
        project_decision_standing(runtime, event.stream_id)
        for event in runtime.ledger.events_by_kind((DECISION_RECORDED,))
        if claim_id in (event.payload.get("relied_on_claim_ids") or ())
    )


# ---------------- validation helpers ----------------


def _validate_passage(
    runtime: Runtime, request: EvidenceRecord, claim: Event | None, extra: dict[str, Any]
) -> list[str]:
    if (request.source_artifact_id is None or request.passage is None
            or request.passage_start is None or request.passage_end is None):
        return ["incomplete_source_passage"]
    if runtime.artifact_store is None:
        return ["source_unavailable"]
    metadata = runtime.ledger.read_artifact(request.source_artifact_id)
    try:
        source = runtime.artifact_store.read_bytes(request.source_artifact_id).decode("utf-8")
    except FileNotFoundError:
        return ["source_unavailable"]
    except ArtifactCorruptionError:
        return ["source_corrupted"]
    except UnicodeDecodeError:
        return ["source_not_text"]
    reasons: list[str] = []
    if claim is not None and metadata is not None and (
        metadata.sha256 == claim.payload.get("observation_sha256")
    ):
        reasons.append("source_is_claim_origin")
    if not 0 <= request.passage_start < request.passage_end <= len(source):
        reasons.append("passage_out_of_range")
    elif source[request.passage_start : request.passage_end] != request.passage:
        reasons.append("passage_not_in_source")
    extra.update({
        "source_sha256": metadata.sha256 if metadata is not None else None,
        "passage_sha256": text_sha256(request.passage),
    })
    return reasons


def _validate_check(
    runtime: Runtime, request: EvidenceRecord, task_id: str | None, extra: dict[str, Any]
) -> list[str]:
    if not request.check_id:
        return ["missing_check"]
    requested = [e for e in runtime.ledger.events_by_kind(("check.requested",))
                 if e.stream_id == request.check_id]
    completed = [e for e in runtime.ledger.events_by_kind(("check.completed",))
                 if e.stream_id == request.check_id]
    if not requested:
        return ["check_not_found"]
    if len(requested) > 1 or len(completed) > 1:
        return ["check_ambiguous"]
    if not completed:
        return ["check_not_completed"]
    reasons: list[str] = []
    check = requested[0].payload
    verdict = str(completed[0].payload.get("verdict"))
    if task_id is not None and str(check.get("task_id")) != task_id:
        reasons.append("check_wrong_task")
    if request.claim_id not in (check.get("claim_ids") or ()):
        reasons.append("check_not_targeting_claim")
    if verdict == "ERROR":
        # No trustworthy result was obtained, so this check bears on nothing.
        # Reading it as evidence either way would turn a broken measurement into
        # a finding about the claim.
        reasons.append("check_errored")
    elif verdict not in ("PASS", "FAIL"):
        reasons.append("check_inconclusive")
    elif (verdict == "PASS") != (request.verdict == SUPPORTS):
        reasons.append("verdict_contradicts_check")
    extra.update({
        "check_requested_event_id": requested[0].event_id,
        "check_completed_event_id": completed[0].event_id,
        "check_verdict": verdict,
    })
    return reasons


def preserved_output_text(runtime: Runtime, attempt_id: str) -> str:
    """Output text read from an attempt's preserved response bytes.

    Raises ObservationUnavailable when the observation or its bytes cannot be read.
    """
    return _observation_text(runtime, attempt_id)[0]


def _observation_text(runtime: Runtime, attempt_id: str) -> tuple[str, Event]:
    observed = next(
        (e for e in runtime.ledger.events_by_kind(("attempt.observed",))
         if e.stream_id == attempt_id or str(e.payload.get("attempt_id", "")) == attempt_id),
        None,
    )
    if observed is None:
        raise ObservationUnavailable(attempt_id, "no attempt.observed event: nothing was preserved")
    ref = observed.payload.get("response_body_artifact")
    if not isinstance(ref, dict) or runtime.artifact_store is None:
        raise ObservationUnavailable(attempt_id, "no preserved response body")
    try:
        body = runtime.artifact_store.read_bytes(str(ref["artifact_id"]))
    except FileNotFoundError as exc:
        raise ObservationUnavailable(attempt_id, "response body bytes are missing") from exc
    except ArtifactCorruptionError as exc:
        raise ObservationUnavailable(
            attempt_id, "response body bytes do not match their recorded sha256"
        ) from exc
    except KeyError as exc:
        raise ObservationUnavailable(attempt_id, "response body reference is malformed") from exc
    try:
        loaded = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        loaded = None
    text = output_text_for(observed.payload.get("protocol"), loaded) if isinstance(loaded, dict) else ""
    return text, observed


def _extracted(runtime: Runtime, claim_id: str) -> Event | None:
    return next((e for e in runtime.ledger.events_by_kind((CLAIM_EXTRACTED,))
                 if e.stream_id == claim_id), None)


def _adopted_status(runtime: Runtime, call_id: str) -> Event | None:
    records = [e for e in runtime.ledger.events_by_kind(("call.status_decided", "call.reinterpreted"))
               if e.stream_id == call_id]
    return records[-1] if records else None


def _status_basis(event: Event | None) -> str | None:
    if event is None:
        return None
    if event.kind == "call.reinterpreted":
        return (f"reinterpretation:{event.payload.get('interpreter_version')}"
                f"/{event.payload.get('policy_version')}")
    return f"execution:{event.payload.get('policy_version')}"


def _producers(runtime: Runtime, call_id: str) -> set[str]:
    producers: set[str] = set()
    for event in runtime.ledger.events_by_kind(("call.requested",)):
        if str(event.payload.get("call_id") or event.stream_id) == call_id:
            producers.add(event.actor_id)
            actor = event.payload.get("actor")
            if isinstance(actor, dict) and actor.get("actor_id"):
                producers.add(str(actor["actor_id"]))
    return producers


def _task_exists(runtime: Runtime, task_id: str) -> bool:
    return any(
        str(e.payload.get("task_id", e.stream_id)) == task_id
        for e in runtime.ledger.events_by_kind(("task.created",))
    )


def _snapshot(standing: ClaimStanding) -> dict[str, Any]:
    return _plain({
        "claim_id": standing.claim_id,
        "status": standing.status,
        "evidence_class": standing.evidence_class,
        "supporting_evidence_ids": list(standing.supporting_evidence_ids),
        "refuting_evidence_ids": list(standing.refuting_evidence_ids),
        "source_call_status": standing.source_call_status,
        "source_status_event_id": standing.source_status_event_id,
        "source_status_basis": standing.source_status_basis,
        "quote_sha256": standing.quote_sha256,
        "observation_sha256": standing.observation_sha256,
    })


def _summary(standing: ClaimStanding) -> dict[str, Any]:
    return {"claim_id": standing.claim_id, "status": standing.status,
            "evidence_class": standing.evidence_class,
            "source_call_status": standing.source_call_status}


def _identity(payload: dict[str, Any], cls: type) -> dict[str, Any]:
    return {f.name: payload.get(f.name) for f in fields(cls)}


def _plain(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _refuse(
    runtime: Runtime,
    error: type[_Refusal],
    kind: str,
    subject_id: str,
    request: Any,
    reasons: list[str],
    *,
    actor_id: str,
    task_id: str | None,
    detail: dict[str, Any] | None = None,
) -> NoReturn:
    event = Event.create(
        stream_id=subject_id,
        kind=kind,
        actor_id=actor_id or "unknown",
        payload={"request": _plain(asdict(request)), "reasons": list(reasons),
                 "detail": _plain(detail or {})},
        correlation_id=task_id,
    )
    runtime.ledger.append(event)
    raise error(subject_id, reasons, event_id=event.event_id)
