"""Render a recorded context selection into request bytes (Stage 15B).

A ContextPackage is a selection identity: it names which events, artifacts and
claims were chosen, not their bytes. Rendering resolves every selected ID to
bytes, lays them out in a canonical, versioned format, and returns a
RenderedContext whose SHA-256 names exactly what was rendered. The runtime,
not the caller, renders; the adapter builds the request text from the result;
the call manifest binds the rendered hash to the prepared request before any
provider effect.

Resolution under ``context-render-v1``:

- artifact: the artifact store's bytes for that ID, integrity-checked against
  the content address; the bytes must be UTF-8 text.
- event: the ledger event with that ID; its bytes are the payload in the
  ledger's canonical serialization (sorted keys, compact separators).
- claim: the single ``claim.recorded`` event for that claim ID, serialized the
  same way. Later ``claim.status`` / ``claim.evidence`` events are not rendered.

Any selected ID that cannot be resolved fails the whole rendering: no partial
context, no silent skip, no degradation policy in v1.

Layout: items in the compiler's canonical order (kind, then ID). Each item is
a header line ``<context-item kind=K id=I sha256=H bytes=N>``, the content
bytes, and a closing ``</context-item>`` line. The byte offsets of every item's
content are recorded, so provenance never depends on parsing separators.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from .artifacts import ArtifactCorruptionError
from .domain import ContextPackage, RenderedContext, RenderedItem

if TYPE_CHECKING:
    from .artifacts import FileArtifactStore
    from .ledger import SQLiteLedger

CONTEXT_RENDER_V1 = "context-render-v1"
KNOWN_RENDERERS = frozenset({CONTEXT_RENDER_V1})
INPUT_SEPARATOR = b"\n\n"


class ContextResolutionError(RuntimeError):
    """A selected context item could not be resolved to bytes."""

    def __init__(self, unresolved: tuple[dict[str, str], ...]) -> None:
        self.unresolved = tuple(unresolved)
        super().__init__(
            "context rendering failed: "
            + "; ".join(f"{u['kind']} {u['id']}: {u['reason']}" for u in self.unresolved)
        )


class RenderBindingError(RuntimeError):
    """The prepared request does not carry the rendered model input."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ledger_payload_bytes(payload: Mapping[str, Any]) -> bytes:
    """A payload in the ledger's canonical serialization (as SQLiteLedger.append stores it)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def render_context(
    package: ContextPackage,
    *,
    ledger: SQLiteLedger,
    artifact_store: FileArtifactStore | None,
    version: str = CONTEXT_RENDER_V1,
) -> RenderedContext:
    """Resolve and render the package's selected items. Pure: appends nothing."""
    if version not in KNOWN_RENDERERS:
        raise ValueError(f"unknown context renderer version: {version!r}")
    selected = sorted(
        [("artifact", str(i)) for i in package.artifact_ids]
        + [("claim", str(i)) for i in package.claim_ids]
        + [("event", str(i)) for i in package.event_ids]
    )

    events: dict[str, Any] = {}
    claims: dict[str, list[Any]] = {}
    if any(kind != "artifact" for kind, _ in selected):
        for event in ledger.read_all():
            events[event.event_id] = event
            if event.kind == "claim.recorded":
                claims.setdefault(str(event.payload.get("claim_id")), []).append(event)

    unresolved: list[dict[str, str]] = []
    resolved: list[tuple[str, str, bytes]] = []
    for kind, item_id in selected:
        content: bytes | None = None
        reason = ""
        if kind == "artifact":
            if artifact_store is None:
                reason = "no artifact store available"
            else:
                try:
                    data = artifact_store.read_bytes(item_id)
                except FileNotFoundError:
                    reason = "artifact not found"
                except ArtifactCorruptionError:
                    reason = "artifact bytes do not match their content address"
                else:
                    try:
                        data.decode("utf-8")
                    except UnicodeDecodeError:
                        reason = "artifact is not UTF-8 text"
                    else:
                        content = data
        elif kind == "event":
            event = events.get(item_id)
            if event is None:
                reason = "event not found in ledger"
            else:
                content = ledger_payload_bytes(event.payload)
        else:
            found = claims.get(item_id, [])
            if not found:
                reason = "no claim.recorded event"
            elif len(found) > 1:
                reason = "multiple claim.recorded events"
            else:
                content = ledger_payload_bytes(found[0].payload)
        if content is None:
            unresolved.append({"kind": kind, "id": item_id, "reason": reason})
        else:
            resolved.append((kind, item_id, content))
    if unresolved:
        raise ContextResolutionError(tuple(unresolved))

    buffer = bytearray()
    items: list[RenderedItem] = []
    for kind, item_id, content in resolved:
        digest = _sha256(content)
        buffer += f"<context-item kind={kind} id={item_id} sha256={digest} bytes={len(content)}>\n".encode()
        start = len(buffer)
        buffer += content
        items.append(RenderedItem(kind, item_id, digest, start, len(buffer)))
        buffer += b"\n</context-item>\n"
    data = bytes(buffer)
    return RenderedContext(
        version=version,
        package_id=package.package_id,
        text=data.decode("utf-8"),
        sha256=_sha256(data),
        items=tuple(items),
    )


def compose_model_input(
    instruction: str, rendered: RenderedContext | None, query: str
) -> tuple[str, tuple[dict[str, Any], ...]]:
    """Model input text plus the byte layout of its parts.

    Parts, in order: instruction, context, query. Empty parts are omitted;
    present parts are joined by a blank line, which belongs to no part.
    """
    buffer = bytearray()
    layout: list[dict[str, Any]] = []
    for part, text in (
        ("instruction", instruction or ""),
        ("context", rendered.text if rendered is not None else ""),
        ("query", query or ""),
    ):
        if not text:
            continue
        if buffer:
            buffer += INPUT_SEPARATOR
        data = text.encode("utf-8")
        start = len(buffer)
        buffer += data
        layout.append({"part": part, "sha256": _sha256(data), "start": start, "end": len(buffer)})
    return bytes(buffer).decode("utf-8"), tuple(layout)


def prepared_input_text(body: Mapping[str, Any]) -> str | None:
    """The single model-input string of a prepared chat/messages/responses body."""
    messages = body.get("messages")
    if isinstance(messages, list):
        if len(messages) == 1 and isinstance(messages[0], Mapping):
            content = messages[0].get("content")
            return content if isinstance(content, str) else None
        return None
    value = body.get("input")
    return value if isinstance(value, str) else None
