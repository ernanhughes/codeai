"""Chapter 28: the scheduler chooses what to do; the ledger says what is true.

Run it:

    python examples/applied_ai/ch28_scheduler.py

Two tasks, identical in every way except one recorded fact: whether their
directive grants ACCEPT. The scheduler reads that fact rather than being told
it, and the two tasks diverge at the same point in their lives.

    gated task    CALL -> CHECK -> ASK_HUMAN   the record authorizes no acceptance
    granted task  CALL -> CHECK -> STOP        nothing is owed; acceptance may proceed

Then the acceptance itself, decided the same way:

    gated task    refused: acceptance_not_granted -- and a person cannot widen it
    granted task  completed, on the grant the record establishes

Then the point of recording a decision at all: decision 2 was CHECK. The world
moved on, and reprojecting now yields STOP. Decision 2 still says CHECK, because
that is what was decided, on facts that were true then -- and replaying its
recorded state through today's policy still yields CHECK.

The reduction a chapter can print:

    state    = project_process_state(runtime, task_id)        # what is true
    decision = decide_next_step(state.to_scheduler_input())   # what to do
    record(decision, state)                                   # what was decided, and why

The scheduler never reads the ledger and never appends to it. It cannot invent a
fact, because it is handed nothing but facts the projection derived.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from codeai.acceptance import (
    AcceptanceRejected,
    AcceptanceRequest,
    artifact_target,
    criteria_sha256,
    text_sha256,
)
from codeai.adapters import CallSpec, CheckRequest, CheckResult, CheckVerdict
from codeai.artifacts import FileArtifactStore
from codeai.context import ContextCompiler
from codeai.domain import ActorRef, Authority, Budget, Capability, Directive, Task
from codeai.ledger import SQLiteLedger
from codeai.process_state import state_snapshot
from codeai.providers import HttpResponse, OpenCodeCognitionAdapter
from codeai.runtime import Runtime
from codeai.scheduler import SchedulerInput, decide_next_step

CRITERIA = ("no percentage figure", "source marker [S1] retained exactly once")
REPAIRED = "The cache is intended to make page loads faster [S1]."


class PassingVerifier:
    """A static verifier. The point here is the joint, not the checker."""

    def run(self, request: CheckRequest) -> CheckResult:
        return CheckResult(check_id=request.check_id, verdict=CheckVerdict.PASS)


def synthetic_post(text: str):
    """Offline transport: no network, no credentials, no model quality claim."""
    body = json.dumps(
        {"id": "synthetic", "choices": [{"finish_reason": "stop", "message": {"content": text}}]}
    ).encode()

    def post(url, payload, headers, timeout):
        return HttpResponse(200, {"request-id": "synthetic"}, body, "application/json")

    return post


def produce_candidate(runtime: Runtime, task_id: str):
    """One recorded call through the ordinary path. Returns the ids it produced."""
    actor = ActorRef("repairer", "model", provider="opencode", model="mimo-v2.5")
    context = ContextCompiler().compile(
        task_id=task_id, actor=actor, prompt="Repair: pages load 73% faster [S1].",
        prompt_version="repair-v1",
    )
    spec = CallSpec(
        str(uuid.uuid4()), task_id, actor, context, str(uuid.uuid4()),
        chamber="deep-review", parameters={"max_tokens": 256},
    )
    adapter = OpenCodeCognitionAdapter(
        model="mimo-v2.5", protocol="chat_completions", gateway_plan="go",
        api_key="offline-decoy", http_post=synthetic_post(REPAIRED), timeout=60,
    )
    call = runtime.invoke_recorded_call(spec, adapter=adapter, max_attempts=1)
    attempt = call.attempts[-1]
    interpretation = runtime.interpretations_for_attempt(attempt.attempt_id)[-1]
    envelope = runtime.artifact_store.read_text(attempt.raw_artifact.artifact_id)
    output = json.loads(envelope)["output_text"]
    runtime.artifact_store.store_text(output, artifact_type="candidate_output")
    return call.call_id, attempt.attempt_id, interpretation.interpretation_id, text_sha256(output)


def show(label: str, runtime: Runtime, task_id: str) -> str:
    state = runtime.process_state(task_id)
    decision = runtime.decide_next_for_task(task_id)
    print(
        f"{label}: {str(decision.operation):9} <- proposals={state.proposal_count} "
        f"check_owed={state.check_required and not state.check_satisfied} "
        f"accept_grant={state.acceptance_authority_available} "
        f"complete={state.process_complete}"
    )
    return str(decision.operation)


def replay(snapshot: dict) -> object:
    """Re-decide from the facts a decision recorded, not from the world now."""
    return decide_next_step(
        SchedulerInput(
            has_required_verification=(
                snapshot["check_required"] and not snapshot["check_satisfied"]
            ),
            requests_independent_proposals=snapshot["proposal_count"] == 0,
            requires_human_authority_for_next_effect=(
                snapshot["acceptance_required"] and not snapshot["acceptance_authority_available"]
            ),
            unresolved_effect=bool(snapshot["unresolved_effects"]),
            process_complete=snapshot["process_complete"],
        )
    )


def open_directive(runtime: Runtime, directive_id: str, capabilities) -> None:
    runtime.open_directive(
        Directive(
            directive_id=directive_id, objective="repair the cache paragraph",
            success_criteria=(), budget=Budget(max_tokens=100_000),
            authority=Authority(frozenset(capabilities)),
        )
    )


def main() -> dict[str, object]:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        ledger = SQLiteLedger(root / "ledger.sqlite")
        runtime = Runtime(ledger, artifact_store=FileArtifactStore(root / "artifacts", ledger))

        # The only difference between these two directives is one capability.
        open_directive(runtime, "review-gated", {Capability.WRITE})
        open_directive(runtime, "review-full", {Capability.WRITE, Capability.ACCEPT})
        produced = {}
        for task_id, directive_id in (("t-gated", "review-gated"), ("t-full", "review-full")):
            runtime.create_task(
                Task(task_id, directive_id, "Repair the paragraph", CRITERIA, Budget(), Authority())
            )

        decisions: dict[str, list[str]] = {"t-gated": [], "t-full": []}
        for task_id in ("t-gated", "t-full"):
            decisions[task_id].append(show(f"1 nothing proposed  [{task_id:8}]", runtime, task_id))

        for task_id in ("t-gated", "t-full"):
            produced[task_id] = produce_candidate(runtime, task_id)
            decisions[task_id].append(show(f"2 candidate exists  [{task_id:8}]", runtime, task_id))
        recorded_second = [
            event.payload for event in ledger.events_by_kind(("scheduler.decision_recorded",))
            if event.payload["task_id"] == "t-full"
        ][-1]

        for task_id in ("t-gated", "t-full"):
            call_id, attempt_id, interpretation_id, artifact_sha = produced[task_id]
            runtime.run_check(
                CheckRequest(check_id=f"k-{task_id}", task_id=task_id,
                             target=artifact_target(artifact_sha)),
                verifier=PassingVerifier(),
            )
            decisions[task_id].append(show(f"3 check passed      [{task_id:8}]", runtime, task_id))

        print()
        outcomes = {}
        for task_id in ("t-gated", "t-full"):
            call_id, attempt_id, interpretation_id, artifact_sha = produced[task_id]
            request = AcceptanceRequest(
                acceptance_id=str(uuid.uuid4()), task_id=task_id, actor_id="reviewer",
                criteria_sha256=criteria_sha256(CRITERIA), artifact_sha256=artifact_sha,
                source_call_id=call_id, source_attempt_id=attempt_id,
                source_interpretation_id=interpretation_id, check_ids=(f"k-{task_id}",),
            )
            try:
                # Nothing is passed in: the grant comes from the task's directive.
                outcomes[task_id] = str(runtime.accept_task(request).status)
            except AcceptanceRejected as exc:
                outcomes[task_id] = f"refused ({', '.join(exc.reasons)})"
            print(f"acceptance [{task_id:8}]: {outcomes[task_id]}")

        for task_id in ("t-gated", "t-full"):
            decisions[task_id].append(show(f"4 after acceptance  [{task_id:8}]", runtime, task_id))

        # History, not a live recomputation. Decision 2 said CHECK on facts that
        # have since changed. It still says CHECK, and still replays to CHECK.
        now = str(decide_next_step(runtime.process_state("t-full").to_scheduler_input()).operation)
        snapshot = recorded_second["state"]
        replayed = replay(snapshot)

        print()
        print(f"decision 2 as recorded : {recorded_second['operation']}")
        print(f"reprojecting the task  : {now}")
        print(f"replaying decision 2   : {str(replayed.operation)} (policy {replayed.policy_version})")
        print(
            f"facts it rested on     : sha256 {recorded_second['process_state_sha256'][:12]}, "
            f"{len(recorded_second['basis_event_ids'])} events"
        )

        summary = {
            "gated": decisions["t-gated"],
            "granted": decisions["t-full"],
            "acceptance": outcomes,
            "recorded_second": recorded_second["operation"],
            "reprojected_now": now,
            "replayed_second": str(replayed.operation),
            "second_state_check_satisfied": snapshot["check_satisfied"],
            "final_state": state_snapshot(runtime.process_state("t-full")),
            "basis_events": len(recorded_second["basis_event_ids"]),
        }
        ledger.close()
        return summary


if __name__ == "__main__":
    main()
