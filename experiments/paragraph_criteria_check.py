"""Deterministic criteria for the Stage 14 paragraph repair.

Checks exact bytes: the file must hash to --sha256, then each declared
criterion is evaluated mechanically. Exits 0 only when the bytes match and
every criterion passes. It checks the declared mechanical criteria; it does
not establish that the prose is true. Imports nothing from CodeAI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

REQUIRED_SENTENCE = "It stores rendered fragments close to readers."
MARKER = "[S1]"

CRITERIA = (
    ("no-percentage", "The paragraph contains no percentage figure (a digit followed by %)."),
    ("marker-once", "The source marker [S1] appears exactly once."),
    ("sentence-kept", f"The sentence '{REQUIRED_SENTENCE}' is retained verbatim."),
)


def evaluate(text: str) -> list[dict[str, object]]:
    outcomes = {
        "no-percentage": re.search(r"\d\s*%", text) is None,
        "marker-once": text.count(MARKER) == 1,
        "sentence-kept": REQUIRED_SENTENCE in text,
    }
    return [
        {"id": key, "criterion": description, "passed": outcomes[key]}
        for key, description in CRITERIA
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    data = args.path.read_bytes()
    observed = hashlib.sha256(data).hexdigest()
    report: dict[str, object] = {
        "artifact_sha256_expected": args.sha256,
        "artifact_sha256_observed": observed,
        "bytes_match": observed == args.sha256,
        "criteria": [],
    }
    if report["bytes_match"]:
        report["criteria"] = evaluate(data.decode("utf-8"))
    passed = bool(report["bytes_match"]) and all(
        item["passed"] for item in report["criteria"]  # type: ignore[union-attr]
    )
    report["passed"] = passed
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
