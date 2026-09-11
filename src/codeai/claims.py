from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict

from .domain import Claim, ClaimRelationship, ClaimStatus, EvidenceClass
from .ledger import Event


def _claim_from_payload(payload: dict) -> Claim:
    return Claim(
        claim_id=str(payload["claim_id"]),
        task_id=str(payload["task_id"]),
        statement=str(payload["statement"]),
        source_call_id=str(payload.get("source_call_id", "")),
        evidence_class=EvidenceClass(str(payload.get("evidence_class", EvidenceClass.ASSERTED.value))),
        status=ClaimStatus(str(payload.get("status", ClaimStatus.ASSERTED.value))),
        scope=payload.get("scope"),
        anchors=tuple(payload.get("anchors", ()) or ()),
        depends_on=tuple(payload.get("depends_on", ()) or ()),
        run_id=payload.get("run_id"),
        source_artifact_id=payload.get("source_artifact_id"),
        source_span=payload.get("source_span"),
        conditions=payload.get("conditions"),
    )


def project_claims(events: tuple[Event, ...]) -> dict[str, Claim]:
    """Fold ledgered claim facts into current claim states.

    Sources of truth remain the events; this is a projection.
    Applies claim.recorded, claim.evidence (promotion), claim.status (refute/etc).
    """
    claims: dict[str, Claim] = {}
    for event in events:
        if event.kind == "claim.recorded":
            claim = _claim_from_payload(event.payload)
            claims[claim.claim_id] = claim
        elif event.kind == "claim.evidence":
            claim_id = str(event.payload.get("claim_id", ""))
            if claim_id in claims:
                current = claims[claim_id]
                evidence = event.payload.get("evidence_class")
                status = event.payload.get("status")
                claims[claim_id] = Claim(
                    claim_id=current.claim_id,
                    task_id=current.task_id,
                    statement=current.statement,
                    source_call_id=current.source_call_id,
                    evidence_class=EvidenceClass(str(evidence)) if evidence else current.evidence_class,
                    status=ClaimStatus(str(status)) if status else current.status,
                    scope=current.scope,
                    anchors=current.anchors,
                    depends_on=current.depends_on,
                    run_id=current.run_id,
                    source_artifact_id=current.source_artifact_id,
                    source_span=current.source_span,
                    conditions=current.conditions,
                )
        elif event.kind == "claim.status":
            claim_id = str(event.payload.get("claim_id", ""))
            if claim_id in claims:
                current = claims[claim_id]
                status = event.payload.get("status")
                claims[claim_id] = Claim(
                    claim_id=current.claim_id,
                    task_id=current.task_id,
                    statement=current.statement,
                    source_call_id=current.source_call_id,
                    evidence_class=current.evidence_class,
                    status=ClaimStatus(str(status)) if status else current.status,
                    scope=current.scope,
                    anchors=current.anchors,
                    depends_on=current.depends_on,
                    run_id=current.run_id,
                    source_artifact_id=current.source_artifact_id,
                    source_span=current.source_span,
                    conditions=current.conditions,
                )
    return claims


def project_relationships(events: tuple[Event, ...]) -> tuple[ClaimRelationship, ...]:
    relationships: list[ClaimRelationship] = []
    for event in events:
        if event.kind == "claim.linked":
            payload = event.payload
            from .domain import ClaimRelationshipType

            relationships.append(
                ClaimRelationship(
                    relationship_id=str(payload["relationship_id"]),
                    from_claim_id=str(payload["from_claim_id"]),
                    to_claim_id=str(payload["to_claim_id"]),
                    relationship_type=ClaimRelationshipType(str(payload["relationship_type"])),
                    run_id=payload.get("run_id"),
                    created_by=payload.get("created_by"),
                )
            )
    return tuple(relationships)


def normalize_statement(statement: str) -> str:
    return " ".join(statement.strip().lower().split())


def disagreement_report(
    claims: dict[str, Claim],
    relationships: tuple[ClaimRelationship, ...] = (),
) -> dict[str, object]:
    """Basic report: concurrence groups, unique, contradicted, unresolved.

    Concurrence is explicitly NOT evidence: agreeing calls never upgrade
    evidence_class here.
    """
    by_statement: dict[str, list[Claim]] = defaultdict(list)
    for claim in claims.values():
        by_statement[normalize_statement(claim.statement)].append(claim)

    concurrence: list[dict[str, object]] = []
    unique: list[dict[str, object]] = []
    for normalized, group in by_statement.items():
        sources = sorted({c.source_call_id for c in group})
        entry: dict[str, object] = {
            "statement": group[0].statement,
            "normalized": normalized,
            "claim_ids": sorted(c.claim_id for c in group),
            "sources": sources,
            "concurrence": len(sources),
            "evidence": sorted({c.evidence_class.value for c in group}),
        }
        if len(sources) > 1:
            concurrence.append(entry)
        else:
            unique.append(entry)

    contradicted_ids: set[str] = set()
    for rel in relationships:
        if rel.relationship_type.value == "CONTRADICTS":
            contradicted_ids.add(rel.from_claim_id)
            contradicted_ids.add(rel.to_claim_id)
    contradicted = [asdict(claims[cid]) for cid in sorted(contradicted_ids) if cid in claims]
    unresolved = [
        asdict(c)
        for c in claims.values()
        if c.status in (ClaimStatus.ASSERTED, ClaimStatus.UNRESOLVED, ClaimStatus.OPEN, ClaimStatus.CONTESTED)
    ]
    return {
        "concurrence": concurrence,
        "unique": unique,
        "contradicted": contradicted,
        "unresolved": unresolved,
    }
