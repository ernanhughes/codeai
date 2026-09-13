"""Stage 14: a succeeded call is not a completed task.

Offline; outbound sockets are refused in every CodeAI phase. Phases run as
separate processes over one ledger:

  produce        synthetic transport through the recorded call path; preserve
                 the output; write a durable checkpoint; then exit
                 (clean-exit) or wait to be killed by the parent
                 (abrupt-termination)
  inspect        reopen; project the task; append nothing; run no cognition
  verify-accept  reopen; run the deterministic check on the stored artifact;
                 accept; repeat identically; attempt a conflicting acceptance

`run` orchestrates both interruption modes, then in-process negative cases,
into a new output directory.

Usage:
    python experiments/task_completion_demo.py run --output <new directory>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from collections import Counter
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent))

from paragraph_criteria_check import CRITERIA

from codeai import acceptance
from codeai.acceptance import (
    AcceptanceRejected,
    AcceptanceRequest,
    artifact_target,
    criteria_sha256,
    text_sha256,
)
from codeai.adapters import CallSpec, CheckRequest
from codeai.artifacts import FileArtifactStore
from codeai.context import ContextCompiler
from codeai.domain import ActorRef, Authority, Budget, Capability, Task
from codeai.ledger import Event, SQLiteLedger
from codeai.providers import HttpResponse, OpenCodeCognitionAdapter
from codeai.runtime import Runtime
from codeai.verifier import LocalCommandVerifier

CHECK_SCRIPT = HERE.parent / "paragraph_criteria_check.py"
MODEL = "mimo-v2.5"
PRODUCER_ID = "repairer"
ACCEPTOR_ID = "reviewer"
ACCEPTOR_NOTE = "in-process actor label; not an authenticated identity"
CRITERIA_TEXTS = tuple(description for _key, description in CRITERIA)
ORIGINAL = (
    "The new cache makes every page load 73% faster, according to the platform team [S1]. "
    "It stores rendered fragments close to readers."
)
REPAIRED = (
    "The new cache is intended to make page loads faster, according to the platform team [S1]. "
    "It stores rendered fragments close to readers."
)
INSTRUCTION = "Repair the paragraph so that it meets every criterion. Return only the paragraph."
COGNITION_KINDS = ("call.manifest", "attempt.started", "attempt.observed", "attempt.completed")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")


def atomic_write(path: Path, value) -> None:
    temp = path.with_name(path.name + ".tmp")
    write(temp, value)
    os.replace(temp, path)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def offline():
    return patch.object(socket.socket, "connect",
                        side_effect=AssertionError("offline: network denied"))


def open_runtime(directory: Path) -> Runtime:
    ledger = SQLiteLedger(directory / "ledger.sqlite")
    return Runtime(ledger, artifact_store=FileArtifactStore(directory / "artifacts", ledger))


def artifact_path(runtime: Runtime, sha: str) -> Path:
    record = runtime.ledger.read_artifact(sha)
    if record is not None and record.uri:
        return Path(record.uri)
    return runtime.artifact_store.base_dir / sha[:2] / sha[2:]


def chat_post(text: str, finish_reason: str, posts: list):
    body = json.dumps({
        "id": "synthetic-repair", "object": "chat.completion", "model": MODEL,
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": text}}],
    }).encode()

    def post(url, payload, headers, timeout):
        posts.append(url)
        return HttpResponse(200, {"request-id": "synthetic-repair"}, body, "application/json")

    return post


def produce_into(runtime: Runtime, *, task_id: str, text: str = REPAIRED,
                 finish_reason: str = "stop", directive_id: str = "stage14-repair",
                 store_candidate: bool = True) -> dict:
    runtime.create_task(Task(task_id, directive_id,
                             "Repair one paragraph against deterministic criteria",
                             CRITERIA_TEXTS, Budget(), Authority(frozenset({Capability.READ}))))
    actor = ActorRef(PRODUCER_ID, "model", provider="opencode", model=MODEL)
    prompt = (INSTRUCTION + "\n\nCriteria:\n" + "\n".join(f"- {c}" for c in CRITERIA_TEXTS)
              + "\n\nParagraph:\n" + ORIGINAL)
    context = ContextCompiler().compile(task_id=task_id, actor=actor, prompt=prompt,
                                        prompt_version="paragraph-repair-v1")
    spec = CallSpec(str(uuid.uuid4()), task_id, actor, context, str(uuid.uuid4()),
                    chamber="deep-review", parameters={"max_tokens": 512})
    posts: list = []
    adapter = OpenCodeCognitionAdapter(model=MODEL, protocol="chat_completions",
                                       gateway_plan="go", api_key="offline-decoy",
                                       http_post=chat_post(text, finish_reason, posts), timeout=60)
    call = runtime.invoke_recorded_call(spec, adapter=adapter, max_attempts=1)
    attempt = call.attempts[-1]
    interpretation = runtime.interpretations_for_attempt(attempt.attempt_id)[-1]
    envelope = json.loads(runtime.artifact_store.read_text(attempt.raw_artifact.artifact_id))
    output = str(envelope.get("output_text") or "")
    if store_candidate:
        runtime.artifact_store.store_text(output, media_type="text/plain",
                                          artifact_type="candidate_output")
    return {
        "task_id": task_id, "directive_id": directive_id, "call_id": call.call_id,
        "attempt_id": attempt.attempt_id, "interpretation_id": interpretation.interpretation_id,
        "call_status": call.status, "generation_state": interpretation.generation_state,
        "artifact_sha256": text_sha256(output), "criteria_sha256": criteria_sha256(CRITERIA_TEXTS),
        "synthetic_transport_posts": len(posts),
    }


def run_artifact_check(runtime: Runtime, ids: dict, *, sha: str | None = None,
                       task_id: str | None = None, command: tuple | None = None) -> tuple:
    sha = sha or ids["artifact_sha256"]
    check_id = f"check-{uuid.uuid4()}"
    request = CheckRequest(
        check_id=check_id, task_id=task_id or ids["task_id"], directive_id=ids["directive_id"],
        command=command or (sys.executable, str(CHECK_SCRIPT), "--path",
                            str(artifact_path(runtime, sha)), "--sha256", sha),
        cwd=str(HERE.parent), timeout_seconds=60, target=artifact_target(sha))
    return check_id, request, runtime.run_check(request, verifier=LocalCommandVerifier())


def acceptance_for(ids: dict, check_ids, **overrides) -> AcceptanceRequest:
    fields = {
        "acceptance_id": str(uuid.uuid4()), "task_id": ids["task_id"], "actor_id": ACCEPTOR_ID,
        "criteria_sha256": ids["criteria_sha256"], "artifact_sha256": ids["artifact_sha256"],
        "source_call_id": ids["call_id"], "source_attempt_id": ids["attempt_id"],
        "source_interpretation_id": ids["interpretation_id"], "check_ids": tuple(check_ids),
    }
    fields.update(overrides)
    return AcceptanceRequest(**fields)


ACCEPTOR = Authority(frozenset({Capability.ACCEPT}))


# ---------------- phases (each runs in its own process) ----------------


def phase_produce(directory: Path, mode: str) -> int:
    assert not (directory / "ledger.sqlite").exists(), "produce requires a fresh ledger"
    with offline():
        runtime = open_runtime(directory)
        ids = produce_into(runtime, task_id=f"task-{mode}")
        atomic_write(directory / "checkpoint.json", {
            **ids, "phase": "produce", "mode": mode, "pid": os.getpid(),
            "ledger_event_count": len(runtime.ledger.read_all()), "written_at": now(),
            "durability": ("written after the ledger committed the call, attempt, observation, "
                           "interpretation and decision events and after the candidate artifact "
                           "was stored; before any check or acceptance"),
        })
    print("checkpoint written", flush=True)
    if mode == "abrupt-termination":
        time.sleep(600)  # the parent terminates this process here
        return 3
    return 0


def phase_inspect(directory: Path, label: str) -> int:
    checkpoint = load(directory / "checkpoint.json")
    with offline():
        runtime = open_runtime(directory)
        before = runtime.ledger.read_all()
        recorded = runtime.get_recorded_call(checkpoint["call_id"])
        attempt = recorded.attempts[-1]
        observation = runtime.get_attempt_observation(attempt.attempt_id)
        interpretations = runtime.interpretations_for_attempt(attempt.attempt_id)
        body_ref = (observation or {}).get("response_body_artifact") or {}
        body = runtime.artifact_store.read_bytes(body_ref["artifact_id"]) if body_ref else b""
        candidate = runtime.artifact_store.read_bytes(checkpoint["artifact_sha256"])
        decisions = [asdict(e) for e in runtime.ledger.events_by_kind(
            ("attempt.retry_decided", "call.status_decided"))
            if e.payload.get("call_id") == checkpoint["call_id"]]
        completion = runtime.task_completion(checkpoint["task_id"])
        after = runtime.ledger.read_all()
    write(directory / f"projection-{label}.json", {
        "label": label, "pid": os.getpid(), "task_id": checkpoint["task_id"],
        "call_id": recorded.call_id, "attempt_id": attempt.attempt_id,
        "call_status": recorded.status, "attempt_count": len(recorded.attempts),
        "interpretations": [asdict(i) for i in interpretations], "decisions": decisions,
        "observation": observation, "response_body_sha256": sha256(body) if body else None,
        "candidate_sha256_observed": sha256(candidate),
        "task_completion": asdict(completion),
        "events_before": len(before), "events_after": len(after),
        "events_appended_by_inspect": len(after) - len(before),
        "event_kinds": dict(Counter(e.kind for e in after)),
    })
    write(directory / f"events-{label}.json", [asdict(e) for e in after])
    if label == "before":
        (directory / "transport-body.bin").write_bytes(body)
        (directory / "candidate.txt").write_bytes(candidate)
    return 0


def phase_verify_accept(directory: Path) -> int:
    ids = load(directory / "checkpoint.json")
    with offline():
        runtime = open_runtime(directory)
        count = lambda: len(runtime.ledger.read_all())
        check_id, check_request, check_result = run_artifact_check(runtime, ids)
        request = acceptance_for(ids, [check_id])
        n0 = count()
        completion = runtime.accept_task(request, authority=ACCEPTOR)
        n1 = count()
        repeat = runtime.accept_task(replace(request, acceptance_id=str(uuid.uuid4())),
                                     authority=ACCEPTOR)
        n2 = count()
        second_check_id, _req, second_result = run_artifact_check(runtime, ids)
        n3 = count()
        try:
            runtime.accept_task(acceptance_for(ids, [check_id, second_check_id]),
                                authority=ACCEPTOR)
            conflict = {"rejected": False}
        except AcceptanceRejected as exc:
            conflict = {"rejected": True, "reasons": list(exc.reasons), "event_id": exc.event_id}
        n4 = count()
        final = runtime.task_completion(ids["task_id"])
        chain = [asdict(e) for e in runtime.ledger.events_by_kind(
            ("task.accepted", "task.completed", "task.acceptance_rejected"))
            if e.payload.get("task_id") == ids["task_id"]]
    write(directory / "verify-accept.json", {
        "pid": os.getpid(),
        "check_request": asdict(check_request), "check_result": asdict(check_result),
        "check_report": json.loads(check_result.stdout) if check_result.stdout else None,
        "acceptance_request": asdict(request),
        "acceptor_authority": sorted(c.value for c in ACCEPTOR.capabilities),
        "acceptor_identity": ACCEPTOR_NOTE,
        "completion_after_accept": asdict(completion), "events_appended_by_accept": n1 - n0,
        "identical_repeat": {"completion": asdict(repeat), "events_appended": n2 - n1},
        "second_check": {"check_id": second_check_id, "verdict": second_result.verdict},
        "conflicting_repeat": {**conflict, "events_appended": n4 - n3},
        "final_completion": asdict(final), "acceptance_chain_events": chain,
    })
    return 0


# ---------------- orchestration ----------------


def run_phase(directory: Path, *args: str) -> None:
    completed = subprocess.run([sys.executable, str(HERE), *args, "--dir", str(directory)],
                               capture_output=True, text=True, check=False)
    (directory / f"{args[0]}{'-' + args[2] if len(args) > 2 else ''}.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise SystemExit(f"phase {args} failed: {completed.stderr}")


def run_chain(root: Path, mode: str) -> dict:
    directory = root / mode
    directory.mkdir(parents=True)
    started = time.monotonic()
    with (directory / "produce.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen([sys.executable, str(HERE), "produce", "--dir", str(directory),
                                  "--mode", mode], stdout=log, stderr=subprocess.STDOUT)
        checkpoint = directory / "checkpoint.json"
        while not checkpoint.exists():
            if child.poll() is not None:
                raise SystemExit(f"produce exited {child.returncode} before its checkpoint")
            if time.monotonic() - started > 120:
                child.kill()
                raise SystemExit("checkpoint timeout")
            time.sleep(0.05)
        termination = {
            "mode": mode, "child_pid": child.pid,
            "checkpoint_seen_after_seconds": round(time.monotonic() - started, 3),
        }
        if mode == "abrupt-termination":
            termination["alive_when_terminated"] = child.poll() is None
            termination["method"] = "Popen.kill() (TerminateProcess on Windows; no cleanup runs)"
            child.kill()
        else:
            termination["method"] = "child returned normally after the checkpoint"
        child.wait(timeout=60)
    termination["returncode"] = child.returncode
    termination["label"] = (
        "ABRUPT TERMINATION at a durable checkpoint; not a demonstration of arbitrary crash "
        "atomicity" if mode == "abrupt-termination" else
        "CLEAN EXIT after a durable checkpoint")
    write(directory / "termination.json", termination)

    run_phase(directory, "inspect", "--label", "before")
    run_phase(directory, "verify-accept")
    run_phase(directory, "inspect", "--label", "after")

    cp = load(directory / "checkpoint.json")
    before = load(directory / "projection-before.json")
    va = load(directory / "verify-accept.json")
    after = load(directory / "projection-after.json")
    events = load(directory / "events-after.json")
    accepted = [e for e in events if e["kind"] == "task.accepted"]
    completed = [e for e in events if e["kind"] == "task.completed"]
    cognition = {k: (before["event_kinds"].get(k), after["event_kinds"].get(k))
                 for k in COGNITION_KINDS}
    assertions = {
        "termination_as_labelled": (
            termination["returncode"] == 0 if mode == "clean-exit"
            else termination["alive_when_terminated"] and termination["returncode"] != 0),
        "call_succeeded": before["call_status"] == "succeeded",
        "generation_complete": before["interpretations"][-1]["generation_state"] == "complete",
        "task_incomplete_after_reopen": before["task_completion"]["status"] == "incomplete",
        "inspect_appended_nothing": (before["events_appended_by_inspect"] == 0
                                     and after["events_appended_by_inspect"] == 0),
        "cognition_not_rerun": all(b == a == 1 for b, a in cognition.values()),
        "candidate_bytes_match_checkpoint": before["candidate_sha256_observed"]
        == cp["artifact_sha256"],
        "check_passed_on_exact_bytes": (va["check_result"]["verdict"] == "PASS"
                                        and va["check_report"]["bytes_match"]),
        "acceptance_appended_two_events": va["events_appended_by_accept"] == 2,
        "identical_repeat_appended_nothing": va["identical_repeat"]["events_appended"] == 0,
        "conflicting_repeat_rejected": (va["conflicting_repeat"]["rejected"]
                                        and va["conflicting_repeat"]["reasons"]
                                        == ["conflicting_acceptance"]),
        "task_complete_after_reopen": after["task_completion"]["status"] == "completed",
        "exactly_one_acceptance_and_completion": len(accepted) == 1 and len(completed) == 1,
        "completion_caused_by_acceptance": (len(completed) == 1 and len(accepted) == 1
                                            and completed[0]["causation_id"]
                                            == accepted[0]["event_id"]),
        "same_evidence_chain": (
            after["task_completion"]["artifact_sha256"] == cp["artifact_sha256"]
            and after["task_completion"]["source_call_id"] == cp["call_id"]
            and after["task_completion"]["acceptance_event_id"]
            == va["final_completion"]["acceptance_event_id"]),
        "historical_interpretation_and_decisions_unchanged": (
            before["interpretations"] == after["interpretations"]
            and before["decisions"] == after["decisions"]),
    }
    summary = {"mode": mode, "cognition_event_counts_before_after": cognition,
               "assertions": assertions, "passed": all(assertions.values())}
    write(directory / "chain-summary.json", summary)
    return summary


def negative_cases(root: Path) -> list[dict]:
    rows: list[dict] = []
    base = root / "negatives"
    base.mkdir()

    def record(case, description, runtime, ids, request, authority=ACCEPTOR, expected=()):
        n0 = len(runtime.ledger.read_all())
        try:
            runtime.accept_task(request, authority=authority)
            reasons, rejected = [], False
        except AcceptanceRejected as exc:
            reasons, rejected = list(exc.reasons), True
        kinds = Counter(e.kind for e in runtime.ledger.read_all())
        completion = runtime.task_completion(ids["task_id"])
        row = {
            "case": case, "description": description, "rejected": rejected,
            "expected_reasons": list(expected), "reasons": reasons,
            "events_appended": len(runtime.ledger.read_all()) - n0,
            "task_accepted_events": kinds.get("task.accepted", 0),
            "task_completed_events": kinds.get("task.completed", 0),
            "task_completion": asdict(completion),
        }
        row["passed"] = (rejected and set(expected) <= set(reasons)
                         and completion.status == "incomplete"
                         and row["task_completed_events"] == 0)
        rows.append(row)

    def fresh(case):
        directory = base / case
        directory.mkdir()
        runtime = open_runtime(directory)
        return runtime

    with offline():
        runtime = fresh("no-acceptance")
        ids = produce_into(runtime, task_id="task-no-acceptance")
        _cid, _req, result = run_artifact_check(runtime, ids)
        completion = runtime.task_completion(ids["task_id"])
        rows.append({"case": "no-acceptance",
                     "description": "succeeded call and passing check, but nobody accepts",
                     "check_verdict": result.verdict, "task_completion": asdict(completion),
                     "passed": completion.status == "incomplete" and result.verdict == "PASS"})

        runtime = fresh("missing-check")
        ids = produce_into(runtime, task_id="task-missing-check")
        record("missing-check", "acceptance cites no check", runtime, ids,
               acceptance_for(ids, []), expected=["missing_check"])

        runtime = fresh("failed-check")
        ids = produce_into(runtime, task_id="task-failed-check", text=ORIGINAL)
        cid, _req, result = run_artifact_check(runtime, ids)
        record("failed-check", f"output kept the percentage; real check verdict {result.verdict}",
               runtime, ids, acceptance_for(ids, [cid]),
               expected=[f"check_not_passed:{cid}:FAIL"])

        runtime = fresh("error-check")
        ids = produce_into(runtime, task_id="task-error-check")
        cid, _req, result = run_artifact_check(runtime, ids,
                                               command=("codeai-missing-check-command",))
        record("error-check", f"check command could not run; verdict {result.verdict}",
               runtime, ids, acceptance_for(ids, [cid]),
               expected=[f"check_not_passed:{cid}:ERROR"])

        runtime = fresh("check-for-other-task")
        ids = produce_into(runtime, task_id="task-a")
        produce_into(runtime, task_id="task-b")
        cid, _req, _result = run_artifact_check(runtime, ids, task_id="task-b")
        record("check-for-other-task", "passing check on the right bytes, recorded for task-b",
               runtime, ids, acceptance_for(ids, [cid]), expected=[f"check_wrong_task:{cid}"])

        runtime = fresh("call-from-other-task")
        ids = produce_into(runtime, task_id="task-a")
        other = produce_into(runtime, task_id="task-b")
        cid, _req, _result = run_artifact_check(runtime, ids, sha=other["artifact_sha256"])
        record("call-from-other-task", "task-a acceptance cites task-b's call and output",
               runtime, ids,
               acceptance_for(ids, [cid], artifact_sha256=other["artifact_sha256"],
                              source_call_id=other["call_id"],
                              source_attempt_id=other["attempt_id"],
                              source_interpretation_id=other["interpretation_id"]),
               expected=["source_call_wrong_task"])

        runtime = fresh("changed-artifact")
        ids = produce_into(runtime, task_id="task-changed")
        edited = REPAIRED.replace("is intended to make", "should make")
        ref = runtime.artifact_store.store_text(edited, media_type="text/plain",
                                                artifact_type="candidate_output")
        cid, _req, result = run_artifact_check(runtime, ids, sha=ref.sha256)
        record("changed-artifact",
               f"edited text passes the criteria check ({result.verdict}) but is not the call output",
               runtime, ids, acceptance_for(ids, [cid], artifact_sha256=ref.sha256),
               expected=["artifact_not_source_output"])

        runtime = fresh("unauthorized")
        ids = produce_into(runtime, task_id="task-unauthorized")
        cid, _req, _result = run_artifact_check(runtime, ids)
        record("unauthorized", "acceptor holds every capability except accept", runtime, ids,
               acceptance_for(ids, [cid]),
               authority=Authority(frozenset(set(Capability) - {Capability.ACCEPT})),
               expected=["unauthorized"])

        runtime = fresh("self-acceptance")
        ids = produce_into(runtime, task_id="task-self")
        cid, _req, _result = run_artifact_check(runtime, ids)
        record("self-acceptance", "the producing actor label accepts its own output", runtime,
               ids, acceptance_for(ids, [cid], actor_id=PRODUCER_ID),
               expected=["self_acceptance"])

        runtime = fresh("truncated-generation")
        ids = produce_into(runtime, task_id="task-truncated", finish_reason="length")
        cid, _req, result = run_artifact_check(runtime, ids)
        record("truncated-generation",
               f"finish_reason=length; the text still passes the check ({result.verdict})",
               runtime, ids, acceptance_for(ids, [cid]),
               expected=["source_call_not_succeeded", "generation_not_complete"])

        runtime = fresh("completion-without-acceptance")
        ids = produce_into(runtime, task_id="task-forged")
        runtime.ledger.append(Event.create(
            stream_id=ids["task_id"], kind="task.completed", actor_id=PRODUCER_ID,
            payload={"task_id": ids["task_id"], "artifact_sha256": ids["artifact_sha256"]}))
        completion = runtime.task_completion(ids["task_id"])
        rows.append({"case": "completion-without-acceptance",
                     "description": "a task.completed event appended with no acceptance behind it",
                     "task_completion": asdict(completion),
                     "passed": completion.status == "incomplete"})

        runtime = fresh("interrupted-between-appends")
        ids = produce_into(runtime, task_id="task-interrupted")
        cid, _req, _result = run_artifact_check(runtime, ids)
        request = acceptance_for(ids, [cid])
        original = acceptance._append_completion

        def interrupted(_runtime, _accepted):
            raise RuntimeError("injected interruption between task.accepted and task.completed")

        acceptance._append_completion = interrupted
        try:
            runtime.accept_task(request, authority=ACCEPTOR)
        except RuntimeError as exc:
            injected = str(exc)
        finally:
            acceptance._append_completion = original
        reopened = open_runtime(base / "interrupted-between-appends")
        pending = reopened.task_completion(ids["task_id"])
        repaired = reopened.accept_task(request, authority=ACCEPTOR)
        kinds = Counter(e.kind for e in reopened.ledger.read_all())
        rows.append({
            "case": "interrupted-between-appends",
            "description": ("exception injected in-process after task.accepted; not a process "
                            "kill. Reopen, then repeat the identical acceptance"),
            "injected": injected, "pending": asdict(pending), "after_repeat": asdict(repaired),
            "task_accepted_events": kinds.get("task.accepted", 0),
            "task_completed_events": kinds.get("task.completed", 0),
            "passed": (pending.status == "incomplete" and pending.acceptance_pending_completion
                       and repaired.status == "completed"
                       and kinds.get("task.accepted") == 1 and kinds.get("task.completed") == 1),
        })
    write(base / "negatives-report.json", rows)
    return rows


def orchestrate(output: Path) -> int:
    output.mkdir(parents=True, exist_ok=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                            text=True, check=False).stdout.splitlines()
    sources = [ROOT / "src/codeai/acceptance.py", ROOT / "src/codeai/runtime.py",
               ROOT / "src/codeai/domain.py", ROOT / "src/codeai/adapters.py",
               ROOT / "src/codeai/verifier.py", ROOT / "tests/test_task_acceptance.py",
               CHECK_SCRIPT, HERE]
    write(output / "manifest.json", {
        "question": "What event may turn a task into completed work?",
        "mode": "offline", "network": "outbound sockets refused in CodeAI phases",
        "transport": "synthetic chat_completions response through the recorded call path",
        "command": sys.argv, "python": sys.version,
        "codeai_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                 cwd=ROOT).decode().strip(),
        "codeai_dirty_paths": status,
        "source_hashes": {str(p.relative_to(ROOT)): sha256(p.read_bytes()) for p in sources},
        "task": {"original_paragraph": ORIGINAL, "criteria": CRITERIA_TEXTS,
                 "criteria_sha256": criteria_sha256(CRITERIA_TEXTS),
                 "synthetic_model_output": REPAIRED},
        "acceptor": {"actor_id": ACCEPTOR_ID, "authority": ["accept"], "note": ACCEPTOR_NOTE},
    })
    chains = {mode: run_chain(output, mode) for mode in ("clean-exit", "abrupt-termination")}
    negatives = negative_cases(output)
    report = {
        "chains": chains,
        "negatives": [{"case": r["case"], "passed": r["passed"], "reasons": r.get("reasons")}
                      for r in negatives],
        "all_passed": all(c["passed"] for c in chains.values()) and all(
            r["passed"] for r in negatives),
    }
    write(output / "report.json", report)
    write(output / "hashes.json", {
        str(p.relative_to(output)).replace("\\", "/"): sha256(p.read_bytes())
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "hashes.json" and not p.name.startswith("ledger.sqlite")})
    print(json.dumps(report, indent=2))
    return 0 if report["all_passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="phase", required=True)
    run = sub.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    produce = sub.add_parser("produce")
    produce.add_argument("--dir", type=Path, required=True)
    produce.add_argument("--mode", choices=("clean-exit", "abrupt-termination"), required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--dir", type=Path, required=True)
    inspect.add_argument("--label", required=True)
    verify = sub.add_parser("verify-accept")
    verify.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "run":
        return orchestrate(args.output.resolve())
    if args.phase == "produce":
        return phase_produce(args.dir, args.mode)
    if args.phase == "inspect":
        return phase_inspect(args.dir, args.label)
    return phase_verify_accept(args.dir)


if __name__ == "__main__":
    raise SystemExit(main())
