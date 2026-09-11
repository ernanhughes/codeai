from pathlib import Path

import pytest

from codeai.artifacts import ArtifactCorruptionError, FileArtifactStore
from codeai.ledger import SQLiteLedger


def test_same_content_reuses_same_artifact(tmp_path: Path):
    store = FileArtifactStore(tmp_path / "artifacts", SQLiteLedger())

    first = store.store_text("same", artifact_type="prompt")
    second = store.store_text("same", artifact_type="prompt")

    assert first.artifact_id == second.artifact_id


def test_changed_content_creates_new_artifact(tmp_path: Path):
    store = FileArtifactStore(tmp_path / "artifacts", SQLiteLedger())

    first = store.store_text("first", artifact_type="prompt")
    second = store.store_text("second", artifact_type="prompt")

    assert first.artifact_id != second.artifact_id


def test_corruption_is_detected_on_read(tmp_path: Path):
    ledger = SQLiteLedger()
    store = FileArtifactStore(tmp_path / "artifacts", ledger)
    artifact = store.store_text("safe", artifact_type="prompt")

    path = Path(artifact.uri or "")
    path.write_text("tampered", encoding="utf-8")

    with pytest.raises(ArtifactCorruptionError):
        store.read_text(artifact.artifact_id)


def test_round_trip_text_and_metadata(tmp_path: Path):
    ledger = SQLiteLedger()
    store = FileArtifactStore(tmp_path / "artifacts", ledger)

    artifact = store.store_text("hello", artifact_type="raw_model_output")

    assert store.read_text(artifact.artifact_id) == "hello"
    metadata = ledger.read_artifact(artifact.artifact_id)
    assert metadata is not None
    assert metadata.artifact_type == "raw_model_output"
