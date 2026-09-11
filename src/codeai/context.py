from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict

from .domain import ActorRef, ContextPackage, Seal
from .ledger import Event


class ContextSealViolation(ValueError):
    pass


class ContextCompiler:
    """Builds replayable context packages and enforces cognitive isolation."""

    def compile(
        self,
        *,
        task_id: str,
        actor: ActorRef,
        prompt: str,
        events: Iterable[Event],
        artifact_ids: Iterable[str] = (),
        seal: Seal | None = None,
    ) -> ContextPackage:
        seal = seal or Seal()
        selected = tuple(events)
        forbidden_events = seal.forbidden_event_ids.intersection(e.event_id for e in selected)
        forbidden_calls = {
            str(e.payload.get("call_id"))
            for e in selected
            if e.payload.get("call_id") is not None
        }.intersection(seal.forbidden_call_ids)
        if forbidden_events or forbidden_calls:
            raise ContextSealViolation(
                f"context violates seal: events={sorted(forbidden_events)}, "
                f"calls={sorted(forbidden_calls)}"
            )

        event_ids = tuple(e.event_id for e in selected)
        artifact_ids = tuple(artifact_ids)
        canonical = json.dumps(
            {
                "task_id": task_id,
                "actor": asdict(actor),
                "prompt": prompt,
                "event_ids": event_ids,
                "artifact_ids": artifact_ids,
                "seal": {
                    "forbidden_event_ids": sorted(seal.forbidden_event_ids),
                    "forbidden_call_ids": sorted(seal.forbidden_call_ids),
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        package_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return ContextPackage(
            package_id=package_id,
            task_id=task_id,
            actor=actor,
            prompt=prompt,
            event_ids=event_ids,
            artifact_ids=artifact_ids,
            seal=seal,
        )
