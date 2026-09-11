from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .domain import ActorRef, ContextPackage, Seal
from .ledger import Event


class ContextSealViolation(ValueError):
    pass


class RequiredContextMissing(ContextSealViolation):
    """A required item was excluded by seal (or absent). Never silently degrade."""



class ContextBudgetUnsatisfiable(ValueError):
    """Required context cannot fit within the token budget."""



@dataclass(frozen=True, slots=True)
class ContextCandidate:
    """One compilable unit: an event, artifact, claim, or decision excerpt."""

    candidate_id: str
    kind: str = "event"
    size_tokens: int = 0
    required: bool = False
    lineage_ids: tuple[str, ...] = ()
    label: str = ""


@dataclass(frozen=True, slots=True)
class TraceEntry:
    candidate_id: str
    decision: str  # "included" | "excluded"
    reason: str


@dataclass(frozen=True, slots=True)
class CompilationTrace:
    trace_id: str
    task_id: str
    package_id: str
    entries: tuple[TraceEntry, ...] = ()
    included_ids: tuple[str, ...] = ()
    excluded_ids: tuple[str, ...] = ()
    budget_tokens: int | None = None
    required_tokens: int = 0
    total_tokens: int = 0

    def reason_for(self, candidate_id: str) -> str | None:
        for entry in self.entries:
            if entry.candidate_id == candidate_id:
                return entry.reason
        return None


def event_lineage(event: Event) -> frozenset[str]:
    """Lineage identifiers through which a seal can exclude derived content."""
    lineage: set[str] = {event.event_id}
    payload = event.payload or {}
    for key in ("call_id", "source_call_id", "claim_id", "source_artifact_id", "artifact_id"):
        value = payload.get(key)
        if value is not None:
            lineage.add(str(value))
    claim_ids = payload.get("claim_ids")
    if isinstance(claim_ids, (list, tuple)):
        lineage.update(str(item) for item in claim_ids)
    raw_lineage = payload.get("lineage_ids")
    if isinstance(raw_lineage, (list, tuple)):
        lineage.update(str(item) for item in raw_lineage)
    # call.completed events expose the call identity in stream_id as well.
    if event.kind == "call.completed":
        lineage.add(event.stream_id)
    if event.kind in ("claim.recorded", "claim.created"):
        claim_id = payload.get("claim_id")
        if claim_id is not None:
            lineage.add(str(claim_id))
    return frozenset(lineage)


def candidate_blocked_by_seal(
    candidate_id: str,
    lineage_ids: Iterable[str],
    seal: Seal,
) -> str | None:
    """Return a human-readable seal reason, or None if allowed."""
    if candidate_id in seal.forbidden_event_ids:
        return f"excluded because seal forbids event {candidate_id}"
    if candidate_id in seal.forbidden_artifact_ids:
        return f"excluded because seal forbids artifact {candidate_id}"
    lineage = set(lineage_ids) | {candidate_id}
    blocked_calls = lineage.intersection(seal.forbidden_call_ids)
    if blocked_calls:
        return f"excluded because seal forbids call lineage {sorted(blocked_calls)}"
    blocked_lineage = lineage.intersection(seal.forbidden_lineage_ids)
    if blocked_lineage:
        return f"excluded because seal forbids lineage {sorted(blocked_lineage)}"
    return None


def _estimate_tokens(text: str) -> int:
    # Cheap deterministic estimator: ~4 chars per token, minimum 1 for nonempty.
    if not text:
        return 0
    return max(1, len(text) // 4)


class ContextCompiler:
    """Builds replayable, immutable, hashed context packages with isolation seals."""

    def compile(
        self,
        *,
        task_id: str,
        actor: ActorRef,
        prompt: str,
        events: Iterable[Event] = (),
        artifact_ids: Iterable[str] = (),
        seal: Seal | None = None,
        objective: str | None = None,
        claim_ids: Iterable[str] = (),
        budget_tokens: int | None = None,
        prompt_version: str | None = None,
        required_event_ids: Iterable[str] = (),
        required_artifact_ids: Iterable[str] = (),
        required_claim_ids: Iterable[str] = (),
        event_token_sizes: Mapping[str, int] | None = None,
        claim_lineage: Mapping[str, Iterable[str]] | None = None,
        artifact_lineage: Mapping[str, Iterable[str]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContextPackage:
        package, _ = self.compile_with_trace(
            task_id=task_id,
            actor=actor,
            prompt=prompt,
            events=events,
            artifact_ids=artifact_ids,
            seal=seal,
            objective=objective,
            claim_ids=claim_ids,
            budget_tokens=budget_tokens,
            prompt_version=prompt_version,
            required_event_ids=required_event_ids,
            required_artifact_ids=required_artifact_ids,
            required_claim_ids=required_claim_ids,
            event_token_sizes=event_token_sizes,
            claim_lineage=claim_lineage,
            artifact_lineage=artifact_lineage,
            metadata=metadata,
        )
        return package

    def compile_with_trace(
        self,
        *,
        task_id: str,
        actor: ActorRef,
        prompt: str,
        events: Iterable[Event] = (),
        artifact_ids: Iterable[str] = (),
        seal: Seal | None = None,
        objective: str | None = None,
        claim_ids: Iterable[str] = (),
        budget_tokens: int | None = None,
        prompt_version: str | None = None,
        required_event_ids: Iterable[str] = (),
        required_artifact_ids: Iterable[str] = (),
        required_claim_ids: Iterable[str] = (),
        event_token_sizes: Mapping[str, int] | None = None,
        claim_lineage: Mapping[str, Iterable[str]] | None = None,
        artifact_lineage: Mapping[str, Iterable[str]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[ContextPackage, CompilationTrace]:
        seal = seal or Seal()
        selected_events = tuple(events)
        event_by_id = {e.event_id: e for e in selected_events}
        required_events = set(required_event_ids)
        required_artifacts = set(required_artifact_ids)
        required_claims = set(required_claim_ids)
        sizes = dict(event_token_sizes or {})
        claim_lineage_map = {k: tuple(v) for k, v in dict(claim_lineage or {}).items()}
        artifact_lineage_map = {k: tuple(v) for k, v in dict(artifact_lineage or {}).items()}

        # Legacy behavior: explicitly passed events/artifacts are required candidates.
        # This keeps seal violations loud (ContextSealViolation) instead of silent.
        candidates: list[ContextCandidate] = []
        for event in selected_events:
            lineage = tuple(sorted(event_lineage(event)))
            size = sizes.get(event.event_id, _estimate_tokens(json.dumps(event.payload, sort_keys=True, default=str)))
            required = True if not required_events else event.event_id in required_events
            # When the caller did not name required subsets, every explicitly
            # passed event is required (loud seal semantics, backward compatible).
            if not required_events and not required_artifacts and not required_claims:
                required = True
            candidates.append(
                ContextCandidate(
                    candidate_id=event.event_id,
                    kind="event",
                    size_tokens=size,
                    required=required,
                    lineage_ids=lineage,
                    label=event.kind,
                )
            )
        for artifact_id in artifact_ids:
            artifact_id = str(artifact_id)
            lineage = artifact_lineage_map.get(artifact_id, ())
            required = True if not required_events and not required_artifacts and not required_claims else artifact_id in required_artifacts
            candidates.append(
                ContextCandidate(
                    candidate_id=artifact_id,
                    kind="artifact",
                    size_tokens=0,
                    required=required,
                    lineage_ids=tuple(lineage),
                    label="artifact",
                )
            )
        for claim_id in claim_ids:
            claim_id = str(claim_id)
            lineage = claim_lineage_map.get(claim_id, ())
            required = claim_id in required_claims
            candidates.append(
                ContextCandidate(
                    candidate_id=claim_id,
                    kind="claim",
                    size_tokens=0,
                    required=required,
                    lineage_ids=tuple(lineage),
                    label="claim",
                )
            )

        # Explicit required ids that were never supplied as candidates are missing.
        supplied_ids = {c.candidate_id for c in candidates}
        for missing in sorted((required_events | required_artifacts | required_claims) - supplied_ids):
            raise RequiredContextMissing(f"REQUIRED_CONTEXT_MISSING: {missing} was required but not supplied")

        # Deterministic ordering for replayability.
        candidates.sort(key=lambda c: (c.kind, c.candidate_id))

        entries: list[TraceEntry] = []
        included: list[ContextCandidate] = []
        required_tokens = 0
        for candidate in candidates:
            seal_reason = candidate_blocked_by_seal(candidate.candidate_id, candidate.lineage_ids, seal)
            if seal_reason is not None:
                if candidate.required:
                    raise RequiredContextMissing(
                        f"REQUIRED_CONTEXT_MISSING: {candidate.candidate_id} required but {seal_reason}"
                    )
                entries.append(TraceEntry(candidate.candidate_id, "excluded", seal_reason))
                continue
            if candidate.required:
                required_tokens += candidate.size_tokens
                entries.append(
                    TraceEntry(candidate.candidate_id, "included", "included because required")
                )
                included.append(candidate)
            else:
                entries.append(TraceEntry(candidate.candidate_id, "pending", "optional candidate"))

        if budget_tokens is not None and required_tokens > budget_tokens:
            raise ContextBudgetUnsatisfiable(
                f"CONTEXT_BUDGET_UNSATISFIABLE: required {required_tokens} tokens exceed budget {budget_tokens}"
            )

        # Fill optional candidates deterministically within budget.
        running = required_tokens
        finalized_entries: list[TraceEntry] = []
        final_included: list[ContextCandidate] = [c for c in included]
        for candidate, entry in zip(candidates, entries):
            if entry.decision != "pending":
                finalized_entries.append(entry)
                continue
            if budget_tokens is not None and running + candidate.size_tokens > budget_tokens:
                finalized_entries.append(
                    TraceEntry(candidate.candidate_id, "excluded", "excluded because budget")
                )
                continue
            running += candidate.size_tokens
            final_included.append(candidate)
            finalized_entries.append(
                TraceEntry(candidate.candidate_id, "included", "included within budget")
            )

        event_ids = tuple(c.candidate_id for c in sorted(final_included, key=lambda c: c.candidate_id) if c.kind == "event")
        artifact_ids_out = tuple(
            c.candidate_id for c in sorted(final_included, key=lambda c: c.candidate_id) if c.kind == "artifact"
        )
        claim_ids_out = tuple(
            c.candidate_id for c in sorted(final_included, key=lambda c: c.candidate_id) if c.kind == "claim"
        )

        canonical = json.dumps(
            {
                "task_id": task_id,
                "actor": asdict(actor),
                "prompt": prompt,
                "prompt_version": prompt_version,
                "objective": objective,
                "event_ids": sorted(event_ids),
                "artifact_ids": sorted(artifact_ids_out),
                "claim_ids": sorted(claim_ids_out),
                "budget_tokens": budget_tokens,
                "seal": {
                    "forbidden_event_ids": sorted(seal.forbidden_event_ids),
                    "forbidden_call_ids": sorted(seal.forbidden_call_ids),
                    "forbidden_artifact_ids": sorted(seal.forbidden_artifact_ids),
                    "forbidden_lineage_ids": sorted(seal.forbidden_lineage_ids),
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        package_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        trace_canonical = json.dumps(
            {
                "package_id": package_id,
                "entries": [[e.candidate_id, e.decision, e.reason] for e in finalized_entries],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        trace_id = hashlib.sha256(trace_canonical.encode("utf-8")).hexdigest()

        # Backward-compat check: legacy callers passed events positionally and expect
        # event order preserved? We preserve deterministic sorted order in the hash,
        # but keep the package event_ids in sorted order for replayability.
        _ = event_by_id  # (kept for future provenance joins)

        package = ContextPackage(
            package_id=package_id,
            task_id=task_id,
            actor=actor,
            prompt=prompt,
            event_ids=event_ids,
            artifact_ids=artifact_ids_out,
            seal=seal,
            metadata=dict(metadata or {}),
            objective=objective,
            claim_ids=claim_ids_out,
            budget_tokens=budget_tokens,
            prompt_version=prompt_version,
            trace_hash=trace_id,
            provenance={"trace_id": trace_id},
        )
        trace = CompilationTrace(
            trace_id=trace_id,
            task_id=task_id,
            package_id=package_id,
            entries=tuple(finalized_entries),
            included_ids=tuple(e.candidate_id for e in finalized_entries if e.decision == "included"),
            excluded_ids=tuple(e.candidate_id for e in finalized_entries if e.decision == "excluded"),
            budget_tokens=budget_tokens,
            required_tokens=required_tokens,
            total_tokens=running,
        )
        return package, trace

    def compile_candidates(
        self,
        *,
        task_id: str,
        actor: ActorRef,
        prompt: str,
        candidates: Iterable[ContextCandidate],
        seal: Seal | None = None,
        budget_tokens: int | None = None,
        objective: str | None = None,
        prompt_version: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[ContextPackage, CompilationTrace]:
        """Lower-level entry point for explicit candidate lists (used by fanout/tests)."""
        seal = seal or Seal()
        ordered = sorted(candidates, key=lambda c: (c.kind, c.candidate_id))
        entries: list[TraceEntry] = []
        required_tokens = 0
        pending: list[ContextCandidate] = []
        for candidate in ordered:
            seal_reason = candidate_blocked_by_seal(candidate.candidate_id, candidate.lineage_ids, seal)
            if seal_reason is not None:
                if candidate.required:
                    raise RequiredContextMissing(
                        f"REQUIRED_CONTEXT_MISSING: {candidate.candidate_id} required but {seal_reason}"
                    )
                entries.append(TraceEntry(candidate.candidate_id, "excluded", seal_reason))
                continue
            if candidate.required:
                required_tokens += candidate.size_tokens
                entries.append(TraceEntry(candidate.candidate_id, "included", "included because required"))
            else:
                entries.append(TraceEntry(candidate.candidate_id, "pending", "optional candidate"))
                pending.append(candidate)
        if budget_tokens is not None and required_tokens > budget_tokens:
            raise ContextBudgetUnsatisfiable(
                f"CONTEXT_BUDGET_UNSATISFIABLE: required {required_tokens} tokens exceed budget {budget_tokens}"
            )
        running = required_tokens
        finalized: list[TraceEntry] = []
        included: list[ContextCandidate] = [c for c in ordered if c.required and candidate_blocked_by_seal(c.candidate_id, c.lineage_ids, seal) is None]
        for candidate, entry in zip(ordered, entries):
            if entry.decision != "pending":
                finalized.append(entry)
                continue
            if budget_tokens is not None and running + candidate.size_tokens > budget_tokens:
                finalized.append(TraceEntry(candidate.candidate_id, "excluded", "excluded because budget"))
                continue
            running += candidate.size_tokens
            included.append(candidate)
            finalized.append(TraceEntry(candidate.candidate_id, "included", "included within budget"))

        event_ids = tuple(sorted(c.candidate_id for c in included if c.kind == "event"))
        artifact_ids = tuple(sorted(c.candidate_id for c in included if c.kind == "artifact"))
        claim_ids = tuple(sorted(c.candidate_id for c in included if c.kind == "claim"))
        canonical = json.dumps(
            {
                "task_id": task_id,
                "actor": asdict(actor),
                "prompt": prompt,
                "prompt_version": prompt_version,
                "objective": objective,
                "event_ids": sorted(event_ids),
                "artifact_ids": sorted(artifact_ids),
                "claim_ids": sorted(claim_ids),
                "budget_tokens": budget_tokens,
                "seal": {
                    "forbidden_event_ids": sorted(seal.forbidden_event_ids),
                    "forbidden_call_ids": sorted(seal.forbidden_call_ids),
                    "forbidden_artifact_ids": sorted(seal.forbidden_artifact_ids),
                    "forbidden_lineage_ids": sorted(seal.forbidden_lineage_ids),
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        package_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        trace_id = hashlib.sha256(
            json.dumps(
                {"package_id": package_id, "entries": [[e.candidate_id, e.decision, e.reason] for e in finalized]},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        package = ContextPackage(
            package_id=package_id,
            task_id=task_id,
            actor=actor,
            prompt=prompt,
            event_ids=event_ids,
            artifact_ids=artifact_ids,
            seal=seal,
            metadata=dict(metadata or {}),
            objective=objective,
            claim_ids=claim_ids,
            budget_tokens=budget_tokens,
            prompt_version=prompt_version,
            trace_hash=trace_id,
            provenance={"trace_id": trace_id},
        )
        trace = CompilationTrace(
            trace_id=trace_id,
            task_id=task_id,
            package_id=package_id,
            entries=tuple(finalized),
            included_ids=tuple(e.candidate_id for e in finalized if e.decision == "included"),
            excluded_ids=tuple(e.candidate_id for e in finalized if e.decision == "excluded"),
            budget_tokens=budget_tokens,
            required_tokens=required_tokens,
            total_tokens=running,
        )
        return package, trace
