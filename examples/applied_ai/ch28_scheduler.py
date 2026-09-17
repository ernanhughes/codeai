"""Chapter 28: the scheduler chooses what to do; the ledger says what is true.

Run it:

    python examples/applied_ai/ch28_scheduler.py

One task, four decisions, each taken from facts projected out of the record:

    1  nothing proposed yet                          -> CALL
    2  a candidate exists, a declared check is owed   -> CHECK
    3  the check passed, no recorded accept grant     -> ASK_HUMAN
    4  a human with that authority accepted           -> STOP

Then the point of recording a decision at all: decision 2 was CHECK. The world
moved on, and reprojecting the task now yields STOP. Decision 2 still says
CHECK, because that is what was decided, on facts that were true then -- and
replaying its recorded state through today's policy still yields CHECK.

The reduction a chapter can print:

    state    = project_process_state(runtime, task_id)        # what is true
    decision = decide_next_step(state.to_scheduler_input())   # what to do
    record(decision, state)                                   # what was decided, and why

The scheduler never reads the ledger and never appends to it. It cannot invent
a fact, because it is handed nothing but facts the projection derived.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from codeai.acceptance import (
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
        return CheckResult(check_id=request.check_id, verdict=CheckVerdict.PASS, exit_code=0)


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
        task_id=task_id,
        actor=actor,
        prompt="Repair: pages load 73% faster [S1].",
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
        f"check_required={state.check_required} check_satisfied={state.check_satisfied} "
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


def main() -> dict[str, object]:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        ledger = SQLiteLedger(root / "ledger.sqlite")
        runtime = Runtime(ledger, artifact_store=FileArtifactStore(root / "artifacts", ledger))

        # The directive grants WRITE. It never grants ACCEPT, so acceptance is a
        # question this process is not authorized to answer for itself.
        runtime.open_directive(
            Directive(
                directive_id="run-repair",
                objective="repair the cache paragraph",
                success_criteria=(),
                budget=Budget(max_tokens=100_000),
                authority=Authority(frozenset({Capability.WRITE})),
            )
        )
        runtime.create_task(
            Task(
                "task-repair", "run-repair", "Repair the paragraph", CRITERIA, Budget(), Authority()
            )
        )

        first = show("1 nothing proposed  ", runtime, "task-repair")

        call_id, attempt_id, interpretation_id, artifact_sha = produce_candidate(
            runtime, "task-repair"
        )
        second = show("2 candidate exists  ", runtime, "task-repair")
        recorded_second = ledger.events_by_kind(("scheduler.decision_recorded",))[-1].payload

        check_id = f"check-{uuid.uuid4()}"
        runtime.run_check(
            CheckRequest(
                check_id=check_id, task_id="task-repair", target=artifact_target(artifact_sha)
            ),
            verifier=PassingVerifier(),
        )
        third = show("3 check passed      ", runtime, "task-repair")

        # The human the scheduler asked for answers: an acceptor whose authority
        # grants ACCEPT accepts these exact bytes against these exact criteria.
        runtime.accept_task(
            AcceptanceRequest(
                acceptance_id=str(uuid.uuid4()),
                task_id="task-repair",
                actor_id="reviewer",
                criteria_sha256=criteria_sha256(CRITERIA),
                artifact_sha256=artifact_sha,
                source_call_id=call_id,
                source_attempt_id=attempt_id,
                source_interpretation_id=interpretation_id,
                check_ids=(check_id,),
            ),
            authority=Authority(frozenset({Capability.ACCEPT})),
        )
        fourth = show("4 accepted          ", runtime, "task-repair")

        # History, not a live recomputation. Decision 2 said CHECK on facts that
        # have since changed. It still says CHECK, and still replays to CHECK.
        now = str(decide_next_step(runtime.process_state("task-repair").to_scheduler_input()).operation)
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
            "decisions": [first, second, third, fourth],
            "recorded_second": recorded_second["operation"],
            "reprojected_now": now,
            "replayed_second": str(replayed.operation),
            "second_state_check_satisfied": snapshot["check_satisfied"],
            "final_state": state_snapshot(runtime.process_state("task-repair")),
            "basis_events": len(recorded_second["basis_event_ids"]),
        }
        ledger.close()
        return summary


if __name__ == "__main__":
    main()
