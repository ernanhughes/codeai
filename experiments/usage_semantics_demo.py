"""Stage 13: usage semantics over immutable preserved responses.

Offline only; outbound sockets are refused. Reads the Stage 12 live
response bytes and the Chapter 11 legacy decoded envelope from the book's
evidence tree without writing to them, interprets each usage member under
usage-semantics-v1 and -v2, runs a labelled synthetic specification corpus,
and compares every result with separately authored expectations.

Usage:
    python experiments/usage_semantics_demo.py \
        --evidence C:/Projects/new-books/experiments/applied-ai/evidence \
        --expected C:/Projects/new-books/experiments/applied-ai/evidence/usage-semantics/expected.json \
        --output   <new directory>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from codeai.usage_semantics import (
    USAGE_SEMANTICS_V1,
    USAGE_SEMANTICS_V2,
    interpret_usage,
)

CH11_ARTIFACT = "ch11-live-opencode/artifacts/1a459bea7ba6b4726f7387cbb0fa8482e66786b82aa2b2799276f4a1127bff05.json"

REAL_CASES = [
    ("stage12-responses-luna", "responses",
     "protocol-conformance/live/responses/response.bin", "transport_body"),
    ("stage12-chat-mimo", "chat_completions",
     "protocol-conformance/live/chat_completions/response.bin", "transport_body"),
    ("stage12-messages-minimax", "messages",
     "protocol-conformance/live/messages/response.bin", "transport_body"),
    ("ch11-run4-chat-mimo", "chat_completions", CH11_ARTIFACT, "legacy_decoded_envelope"),
]

ESTIMATE = {"input_tokens": 100, "output_tokens": 20, "method": "fixture-estimator-v1"}

SYNTHETIC_CASES = [
    ("U01-chat-absent", "chat_completions", {"choices": []}, None),
    ("U02-chat-measured", "chat_completions",
     {"usage": {"prompt_tokens": 100, "completion_tokens": 20}}, None),
    ("U03-responses-total", "responses",
     {"usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}}, None),
    ("U04-responses-conflict", "responses",
     {"usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 150}}, None),
    ("U05-chat-cache-subset", "chat_completions",
     {"usage": {"prompt_tokens": 100, "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 80}}}, None),
    ("U06-responses-reasoning-subset", "responses",
     {"usage": {"input_tokens": 100, "output_tokens": 20,
                "output_tokens_details": {"reasoning_tokens": 5}}}, None),
    ("U07-messages-additive", "messages",
     {"usage": {"input_tokens": 20, "output_tokens": 6,
                "cache_read_input_tokens": 15, "cache_creation_input_tokens": 2}}, None),
    ("U08-messages-creation-only", "messages",
     {"usage": {"input_tokens": 20, "output_tokens": 6, "cache_creation_input_tokens": 2}}, None),
    ("U09-chat-partial", "chat_completions", {"usage": {"prompt_tokens": 100}}, None),
    ("U10-responses-zeros", "responses",
     {"usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}, None),
    ("U11-estimate-only", "responses", {"output": []}, ESTIMATE),
    ("U12-gateway-cost", "chat_completions",
     {"usage": {"prompt_tokens": 100, "completion_tokens": 20}, "cost": "0"}, None),
    ("U13-unrecognized-detail", "responses",
     {"usage": {"input_tokens": 20, "output_tokens": 6, "cached_tokens": 15}}, None),
    ("OVERLAP-draft-100-20-80-5", "chat_completions",
     {"usage": {"prompt_tokens": 100, "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 80},
                "completion_tokens_details": {"reasoning_tokens": 5}}}, None),
    ("ADV-negative", "chat_completions",
     {"usage": {"prompt_tokens": -5, "completion_tokens": 20}}, None),
    ("ADV-string", "chat_completions",
     {"usage": {"prompt_tokens": "100", "completion_tokens": 20}}, None),
    ("ADV-null", "chat_completions",
     {"usage": {"prompt_tokens": None, "completion_tokens": 20}}, None),
    ("ADV-subset-exceeds-parent", "chat_completions",
     {"usage": {"prompt_tokens": 100, "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 150, "cache_write_tokens": 0}}}, None),
    ("ADV-reasoning-exceeds-output", "responses",
     {"usage": {"input_tokens": 100, "output_tokens": 20,
                "output_tokens_details": {"reasoning_tokens": 30}}}, None),
    ("UNKNOWN-protocol", "gemini",
     {"usage": {"promptTokenCount": 10, "candidatesTokenCount": 3}}, None),
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")


def load_real(evidence: Path, relpath: str, evidence_class: str) -> tuple[dict, bytes]:
    raw = (evidence / relpath).read_bytes()
    loaded = json.loads(raw)
    if evidence_class == "legacy_decoded_envelope":
        loaded = loaded.get("provider_response") or {}
    return loaded, raw


def summarize(v1, v2) -> dict:
    def pair(name):
        d = v2.derived_value(name)
        return None if d is None else [d.value, d.lower_bound]

    return {
        "v1": [v1.component("input").value, v1.component("output").value],
        "total_input": pair("total_input"),
        "fresh_input": pair("fresh_input"),
        "processed_total": pair("processed_total"),
        "conflicts": list(v2.conflicts),
        "diagnostics": list(v2.diagnostics),
        "unrecognized_paths": list(v2.unrecognized_paths),
        "provider_cost_raw": v2.provider_cost.get("raw") if v2.provider_cost else None,
    }


def compare(expected: dict, actual: dict) -> list[str]:
    mismatches = []
    for key, want in expected.items():
        got = actual.get(key)
        if isinstance(want, list) and key in ("conflicts", "diagnostics", "unrecognized_paths"):
            ok = sorted(want) == sorted(got or [])
        else:
            ok = want == got
        if not ok:
            mismatches.append(f"{key}: expected {want!r}, got {got!r}")
    return mismatches


def run(evidence: Path, expected_path: Path, output: Path) -> int:
    expected = json.loads(expected_path.read_text(encoding="utf-8"))["cases"]
    rows, before, after = [], {}, {}
    for case_id, _protocol, relpath, _cls in REAL_CASES:
        before[relpath] = sha256((evidence / relpath).read_bytes())

    with patch.object(socket.socket, "connect", side_effect=AssertionError("offline: network denied")):
        for case_id, protocol, relpath, evidence_class in REAL_CASES:
            parsed, raw = load_real(evidence, relpath, evidence_class)
            v1 = interpret_usage(protocol, parsed, version=USAGE_SEMANTICS_V1)
            v2 = interpret_usage(protocol, parsed, version=USAGE_SEMANTICS_V2)
            summary = summarize(v1, v2)
            rows.append({
                "case": case_id, "evidence_class": evidence_class, "protocol": protocol,
                "source": relpath, "source_sha256": sha256(raw),
                "raw_usage": parsed.get("usage"), "summary": summary,
                "v1": asdict(v1), "v2": asdict(v2),
                "mismatches": compare(expected[case_id], summary),
            })
        for case_id, protocol, parsed, estimate in SYNTHETIC_CASES:
            v1 = interpret_usage(protocol, parsed, version=USAGE_SEMANTICS_V1)
            v2 = interpret_usage(protocol, parsed, version=USAGE_SEMANTICS_V2, estimate=estimate)
            summary = summarize(v1, v2)
            rows.append({
                "case": case_id, "evidence_class": "synthetic_specification", "protocol": protocol,
                "raw_usage": parsed.get("usage"), "estimate": estimate, "summary": summary,
                "v1": asdict(v1), "v2": asdict(v2),
                "mismatches": compare(expected[case_id], summary),
            })

    for case_id, _protocol, relpath, _cls in REAL_CASES:
        after[relpath] = sha256((evidence / relpath).read_bytes())

    sources_unchanged = before == after
    mismatch_count = sum(len(r["mismatches"]) for r in rows)
    write(output / "synthetic-corpus.json",
          [{"case": c, "protocol": p, "parsed": d, "estimate": e} for c, p, d, e in SYNTHETIC_CASES])
    write(output / "report.json", {
        "network_calls": 0,
        "versions": [USAGE_SEMANTICS_V1, USAGE_SEMANTICS_V2],
        "real_cases": len(REAL_CASES), "synthetic_cases": len(SYNTHETIC_CASES),
        "source_hashes_before": before, "source_hashes_after": after,
        "sources_unchanged": sources_unchanged,
        "expectations_file_sha256": sha256(expected_path.read_bytes()),
        "mismatch_count": mismatch_count, "rows": rows,
    })

    lines = [
        "# Usage semantics: v1 vs v2 over preserved responses",
        "",
        "Derived values are shown as value (or UNKNOWN, lower bound). Real cases are preserved provider",
        "responses; synthetic cases are specifications, not observations.",
        "",
        "| Case | Evidence | v1 input / output | total input | fresh input | processed total | Conflicts / diagnostics | Matches expected |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def fmt(pair):
        if pair is None:
            return "—"
        value, lower = pair
        return str(value) if value is not None else ("UNKNOWN" if lower is None else f"UNKNOWN (≥ {lower})")

    for r in rows:
        s = r["summary"]
        notes = ", ".join(s["conflicts"] + s["diagnostics"]) or "—"
        lines.append(
            f"| {r['case']} | {r['evidence_class']} | {s['v1'][0]} / {s['v1'][1]} | "
            f"{fmt(s['total_input'])} | {fmt(s['fresh_input'])} | {fmt(s['processed_total'])} | "
            f"{notes} | {'yes' if not r['mismatches'] else 'NO'} |"
        )
    lines += ["", f"Sources unchanged: {sources_unchanged}. Mismatches against expectations: {mismatch_count}."]
    (output / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if sources_unchanged and mismatch_count == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                            capture_output=True, text=True, check=False).stdout.splitlines()
    write(args.output / "manifest.json", {
        "question": "Which usage quantities are equivalent enough to normalize?",
        "mode": "offline",
        "command": sys.argv,
        "codeai_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "codeai_dirty_paths": status,
        "source_hashes": {str(p.relative_to(ROOT)): sha256(p.read_bytes()) for p in [
            ROOT / "src/codeai/usage_semantics.py", ROOT / "src/codeai/runtime.py",
            ROOT / "tests/test_usage_semantics.py", Path(__file__).resolve()]},
        "expected_file": str(args.expected),
    })
    code = run(args.evidence.resolve(), args.expected.resolve(), args.output)
    write(args.output / "hashes.json", {
        str(p.relative_to(args.output)): sha256(p.read_bytes())
        for p in sorted(args.output.rglob("*")) if p.is_file() and p.name != "hashes.json"})
    print((args.output / "table.md").read_text(encoding="utf-8"))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
