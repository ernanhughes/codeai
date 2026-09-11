from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from .domain import ArtifactRef
from .ledger import SQLiteLedger


class ArtifactCorruptionError(RuntimeError):
    pass


class FileArtifactStore:
    def __init__(self, base_dir: str | Path, ledger: SQLiteLedger) -> None:
        self.base_dir = Path(base_dir)
        self.ledger = ledger
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def store_bytes(
        self,
        content: bytes,
        *,
        media_type: str,
        artifact_type: str,
    ) -> ArtifactRef:
        sha256 = hashlib.sha256(content).hexdigest()
        path = self._path_for(sha256)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = path.read_bytes()
            if hashlib.sha256(existing).hexdigest() != sha256:
                raise ArtifactCorruptionError(f"artifact corruption detected while writing {sha256}")
        else:
            temp_path = path.with_name(f".{path.name}.tmp")
            temp_path.write_bytes(content)
            temp_path.replace(path)

        ref = ArtifactRef(
            artifact_id=sha256,
            sha256=sha256,
            media_type=media_type,
            uri=str(path),
        )
        self.ledger.register_artifact(
            ref,
            artifact_type=artifact_type,
            byte_length=len(content),
            created_at=datetime.now(UTC).isoformat(),
        )
        return ref

    def store_text(
        self,
        content: str,
        *,
        media_type: str = "text/plain",
        artifact_type: str,
    ) -> ArtifactRef:
        return self.store_bytes(
            content.encode("utf-8"),
            media_type=media_type,
            artifact_type=artifact_type,
        )

    def read_bytes(self, artifact_id: str) -> bytes:
        metadata = self.ledger.read_artifact(artifact_id)
        if metadata is None:
            raise FileNotFoundError(artifact_id)
        path = self._path_for(metadata.sha256)
        content = path.read_bytes()
        observed = hashlib.sha256(content).hexdigest()
        if observed != metadata.sha256:
            raise ArtifactCorruptionError(
                f"artifact corruption detected: expected {metadata.sha256}, observed {observed}"
            )
        return content

    def read_text(self, artifact_id: str) -> str:
        return self.read_bytes(artifact_id).decode("utf-8")

    def _path_for(self, sha256: str) -> Path:
        return self.base_dir / sha256[:2] / sha256[2:]
