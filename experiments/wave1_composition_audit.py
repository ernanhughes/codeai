"""Wave 1 composition audit: do four correct seams compose into one enforced process?

Protocol: experiments/W1-composition-prereg.md (registered before this file existed).

Run it:

    PYTHONPATH=src python experiments/wave1_composition_audit.py

Every probe exercises the public runtime API and records what it *observed*, not
what the design intends. Classification vocabulary is fixed by the protocol:

    ENFORCED      derived or checked from durable facts, and a violation is refused
    DERIVED       reconstructable from durable facts, but not an execution gate
    RECORDED      persisted, but trusted from the caller or producer
    CONVENTIONAL  holds only because callers cooperate
    ABSENT        not represented at all
    UNTESTABLE    cannot be expressed through the public API

No runtime behaviour is modified by this file.
"""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codeai.acceptance import (  # noqa: E402
    AcceptanceRejected,
    AcceptanceRequest,
    artifact_target,
    criteria_sha256,
    text_sha256,
)
from codeai.actions import (  # noqa: E402
    ActionNextOperation,
    EffectState,
    ReconciliationVerdict,
)
from codeai.adapters import (  # noqa: E402
    ActionRequest,
    ActionResult,
    ActionStatus,
    CallSpec,
    CheckRequest,
    CheckResult,
    CheckVerdict,
)
from codeai.artifacts import FileArtifactStore  # noqa: E402
from codeai.context import ContextCompiler  # noqa: E402
from codeai.domain import (  # noqa: E402
    ActorRef,
    Authority,
    Budget,
    Capability,
    Claim,
    ClaimStatus,
    Directive,
    Task,
)
from codeai.ledger import SQLiteLedger  # noqa: E402
from codeai.providers import HttpResponse, OpenCodeCognitionAdapter  # noqa: E402
from codeai.runtime import Runtime  # noqa: E402
from codeai.scheduler import Operation  # noqa: E402
from codeai.verification import BindingStatus  # noqa: E402
from codeai.verifier import LocalCommandVerifier  # noqa: E402

CRITERIA = ("no percentage figure", "source marker [S1] retained exactly once")
REPAIRED = "The cache is intended to make page loads faster [S1]."
EXIT_POLICY = {
    "policy_id": "exit-code-v1",
    "pass_codes": [0],
    "fail_codes": [1],
    "inconclusive_codes": [2],
}
MARKER_SCRIPT = """
import sys, pathlib
text = pathlib.Path(sys.argv[1]).read_text(encoding='utf-8', errors='replace')
sys.exit(1 if 'TODO(marker)' in text else 0)
"""


# ---------------------------------------------------------------- scaffolding


@dataclass
class Probe:
    probe_id: str
    joint: str
    question: str
    expected_property: str
    observed: list[str] = field(default_factory=list)
    classification: str = "UNCLASSIFIED"
    note: str = ""

    def see(self, fact: str) -> None:
        self.observed.append(fact)

    def as_dict(self) -> dict[str, object]:
        return {
            "probe_id": self.probe_id,
            "joint": self.joint,
            "question": self.question,
            "expected_property": self.expected_property,
            "observed": self.observed,
            "classification": self.classification,
            "note": self.note,
        }


class Bench:
    """One runtime over one temp directory, with a file as the world state."""

    def __init__(self, directory: Path, *, target_text: str = "clean paragraph\n") -> None:
        self.root = directory
        self.target = directory / "target.txt"
        self.target.write_text(target_text, encoding="utf-8")
        self.ledger_path = directory / "ledger.sqlite"
        self.ledger = SQLiteLedger(self.ledger_path)
        self.runtime = Runtime(
            self.ledger,
            artifact_store=FileArtifactStore(directory / "artifacts", self.ledger),
            state_resolver=self.state_hash,
        )

    def state_hash(self) -> str:
        return sha256(self.target.read_bytes()).hexdigest()

    def directive(self, directive_id="d-root", capabilities=(Capability.WRITE,), parent=None,
                  max_tokens=100_000):
        return self.runtime.open_directive(
            Directive(
                directive_id=directive_id,
                parent_directive_id=parent,
                objective="repair the paragraph",
                success_criteria=(),
                budget=Budget(max_tokens=max_tokens),
                authority=Authority(frozenset(capabilities)),
            )
        )

    def task(self, task_id="t1", directive_id="d-root", criteria=CRITERIA):
        return self.runtime.create_task(
            Task(task_id, directive_id, "Repair the paragraph", criteria, Budget(), Authority())
        )

    def action(self, action_id, *, capability="write", key=None, directive_id="d-root",
               precondition=None, task_id="t1"):
        return ActionRequest(
            action_id=action_id,
            task_id=task_id,
            directive_id=directive_id,
            capability=capability,
            instruction="rewrite the paragraph",
            precondition_hash=precondition,
            idempotency_key=key or f"key:{action_id}",
            requested_by="human",
            actor_id="worker",
            adapter="fixture",
        )

    def check(self, check_id, *, script=MARKER_SCRIPT, claims=(), target_hash=None, target=None,
              task_id="t1", artifact_arg=False):
        """A check over the world file, or -- with artifact_arg -- over the bytes.

        ``{artifact}`` is the placeholder the runtime substitutes with the path
        it materialized after verifying the digest.
        """
        subject = "{artifact}" if artifact_arg else str(self.target)
        return CheckRequest(
            check_id=check_id,
            task_id=task_id,
            claim_ids=claims,
            command=(sys.executable, "-c", script, subject),
            cwd=str(self.root),
            target=target,
            target_state_hash=target_hash,
            verdict_policy=EXIT_POLICY,
        )

    def produce_candidate(self, task_id="t1", text=REPAIRED):
        """One recorded cognition call through the ordinary path, offline."""
        body = json.dumps(
            {"id": "synthetic",
             "choices": [{"finish_reason": "stop", "message": {"content": text}}]}
        ).encode()

        def post(url, payload, headers, timeout):
            return HttpResponse(200, {"request-id": "synthetic"}, body, "application/json")

        actor = ActorRef("repairer", "model", provider="opencode", model="mimo-v2.5")
        context = ContextCompiler().compile(
            task_id=task_id, actor=actor, prompt="Repair: pages load 73% faster [S1].",
            prompt_version="repair-v1",
        )
        spec = CallSpec(str(uuid.uuid4()), task_id, actor, context, str(uuid.uuid4()),
                        chamber="deep-review", parameters={"max_tokens": 256})
        adapter = OpenCodeCognitionAdapter(
            model="mimo-v2.5", protocol="chat_completions", gateway_plan="go",
            api_key="offline-decoy", http_post=post, timeout=60,
        )
        call = adapter_call = self.runtime.invoke_recorded_call(spec, adapter=adapter,
                                                               max_attempts=1)
        attempt = adapter_call.attempts[-1]
        interpretation = self.runtime.interpretations_for_attempt(attempt.attempt_id)[-1]
        envelope = self.runtime.artifact_store.read_text(attempt.raw_artifact.artifact_id)
        output = json.loads(envelope)["output_text"]
        self.runtime.artifact_store.store_text(output, artifact_type="candidate_output")
        return {
            "call_id": call.call_id,
            "attempt_id": attempt.attempt_id,
            "interpretation_id": interpretation.interpretation_id,
            "artifact_sha256": text_sha256(output),
            "output": output,
        }

    def kinds(self) -> list[str]:
        return [event.kind for event in self.ledger.read_all()]

    def close(self) -> None:
        self.ledger.close()


class HonestWriter:
    """Does the work, then reports it."""

    def __init__(self, bench: Bench, text: str = "TODO(marker) rewritten\n") -> None:
        self.bench, self.text, self.calls = bench, text, 0

    def execute(self, request: ActionRequest) -> ActionResult:
        self.calls += 1
        self.bench.target.write_text(self.text, encoding="utf-8")
        return ActionResult(action_id=request.action_id, status=ActionStatus.SUCCEEDED)


class LyingWorker:
    """Reports success and changes nothing."""

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ActionRequest) -> ActionResult:
        self.calls += 1
        return ActionResult(action_id=request.action_id, status=ActionStatus.SUCCEEDED)


class CrashAfterEffect:
    """Writes, then the connection dies before any completion is reported."""

    def __init__(self, bench: Bench, text: str = "written by the crashed worker\n") -> None:
        self.bench, self.text, self.calls = bench, text, 0

    def execute(self, request: ActionRequest) -> ActionResult:
        self.calls += 1
        self.bench.target.write_text(self.text, encoding="utf-8")
        raise ConnectionError("connection reset after the write")


class ForgingVerifier:
    def __init__(self, verdict=CheckVerdict.PASS) -> None:
        self.calls, self.verdict = 0, verdict

    def run(self, request: CheckRequest) -> CheckResult:
        self.calls += 1
        return CheckResult(
            check_id=request.check_id,
            verdict=self.verdict,
            observed_target_state_hash="whatever-I-like",
        )


def bench(stack: list[TemporaryDirectory], **kwargs) -> Bench:
    directory = TemporaryDirectory()
    stack.append(directory)
    return Bench(Path(directory.name), **kwargs)


# ---------------------------------------------------------------- happy path


def happy_path(stack) -> dict[str, object]:
    """One lifecycle crossing every Wave 1 seam, with durable identities."""
    b = bench(stack)
    trace: list[dict[str, object]] = []

    def step(name: str, **facts):
        trace.append({"step": name, **facts})

    b.directive("d-root", (Capability.WRITE, Capability.ACCEPT))
    b.directive("d-child", (Capability.WRITE, Capability.ACCEPT), parent="d-root")
    b.task("t1", "d-child")
    step("directive recorded",
         chain=[link.directive_id for link in b.runtime.directive_authority("d-child").grant_chain],
         effective=list(b.runtime.directive_authority("d-child").effective_capabilities))

    first = b.runtime.decide_next_for_task("t1")
    step("decision 1", operation=str(first.operation), reason=first.reason,
         policy=first.policy_version)

    produced = b.produce_candidate()
    step("recorded call", call_id=produced["call_id"],
         artifact_sha256=produced["artifact_sha256"][:16])

    second = b.runtime.decide_next_for_task("t1")
    step("decision 2", operation=str(second.operation), reason=second.reason)

    precondition = b.state_hash()
    worker = HonestWriter(b, text="The cache is intended to make page loads faster [S1].\n")
    result = b.runtime.execute_action(
        b.action("a1", directive_id="d-child", precondition=precondition), adapter=worker
    )
    state = b.runtime.action_state("a1")
    step("effect", status=str(result.status), effect_state=str(state.effect_state),
         next_operation=str(state.next_operation), adapter_calls=worker.calls)

    verification = b.runtime.run_check(
        b.check("k1", target_hash=b.state_hash(), artifact_arg=True,
                target=artifact_target(produced["artifact_sha256"])),
        verifier=LocalCommandVerifier(),
    )
    step("verification", verdict=str(verification.verdict), binding=verification.binding_status,
         artifact_binding=verification.artifact_binding_status,
         observed=(verification.observed_target_state_hash or "")[:16])

    third = b.runtime.decide_next_for_task("t1")
    step("decision 3", operation=str(third.operation), reason=third.reason)

    acceptance = AcceptanceRequest(
        acceptance_id=str(uuid.uuid4()), task_id="t1", actor_id="reviewer",
        criteria_sha256=criteria_sha256(CRITERIA), artifact_sha256=produced["artifact_sha256"],
        source_call_id=produced["call_id"], source_attempt_id=produced["attempt_id"],
        source_interpretation_id=produced["interpretation_id"], check_ids=("k1",),
    )
    accepted = None
    try:
        completion = b.runtime.accept_task(acceptance)
        accepted = str(completion.status)
    except AcceptanceRejected as exc:
        accepted = f"rejected: {', '.join(exc.reasons)}"
    standing = b.runtime.acceptance_authority("t1")
    step("acceptance", outcome=accepted,
         acceptance_authority=f"resolved from the record: {standing.status}")

    fourth = b.runtime.decide_next_for_task("t1")
    step("decision 4", operation=str(fourth.operation), reason=fourth.reason)

    events = b.ledger.read_all()
    b.close()
    reopened = SQLiteLedger(b.ledger_path)
    same = reopened.read_all() == events
    reprojected = Runtime(reopened, state_resolver=b.state_hash)
    status_again = str(reprojected.task_completion("t1").status)
    decision_again = str(reprojected.decide_next("t1").operation)
    reopened.close()

    return {
        "trace": trace,
        "event_count": len(events),
        "kinds": sorted({event.kind for event in events}),
        "reopen_identical": same,
        "reprojected_status": status_again,
        "reprojected_decision": decision_again,
    }


# ---------------------------------------------------------------- probes


def probe_a_action_authority(stack) -> Probe:
    p = Probe("A", "directive -> action authority",
              "Can caller-supplied authority widen the recorded grant?",
              "an action the record does not grant must not execute")
    b = bench(stack)
    b.directive("d-root", (Capability.READ,))
    b.task("t1")
    worker = HonestWriter(b)
    denied = b.runtime.execute_action(
        b.action("a-write", capability="write"), adapter=worker,
        authority=Authority(frozenset({Capability.WRITE, Capability.DESTRUCTIVE})),
    )
    p.see(f"record grants READ only; caller passed WRITE+DESTRUCTIVE -> {denied.status}")
    p.see(f"adapter invocations: {worker.calls}")
    p.see(f"events: {[k for k in b.kinds() if k.startswith('action.')]}")
    refusal = b.ledger.events_by_kind(("action.authorization_refused",))
    p.see(f"refusal carries basis: {sorted(refusal[0].payload) if refusal else 'none'}")
    p.classification = "ENFORCED" if denied.status == ActionStatus.DENIED and worker.calls == 0 else "NOT ENFORCED"
    b.close()
    return p


def probe_b_acceptance_authority(stack) -> Probe:
    p = Probe("B", "directive -> acceptance authority",
              "Can a caller accept a task when the recorded directive grants no ACCEPT?",
              "acceptance authority should come from the same recorded chain as action authority")
    # W1-R2 repaired this joint; the probe still asks the same question.
    b = bench(stack)
    b.directive("d-root", (Capability.WRITE,))  # deliberately no ACCEPT
    b.task("t1")
    produced = b.produce_candidate()
    b.runtime.run_check(
        b.check("k1", artifact_arg=True, target=artifact_target(produced["artifact_sha256"])),
        verifier=LocalCommandVerifier(),
    )
    standing = b.runtime.directive_authority("d-root")
    p.see(f"recorded grant: {list(standing.effective_capabilities)}; allows accept: {standing.allows('accept')}")

    request = AcceptanceRequest(
        acceptance_id=str(uuid.uuid4()), task_id="t1", actor_id="reviewer",
        criteria_sha256=criteria_sha256(CRITERIA), artifact_sha256=produced["artifact_sha256"],
        source_call_id=produced["call_id"], source_attempt_id=produced["attempt_id"],
        source_interpretation_id=produced["interpretation_id"], check_ids=("k1",),
    )
    try:
        completion = b.runtime.accept_task(
            request, authority=Authority(frozenset({Capability.ACCEPT}))
        )
        outcome = str(completion.status)
    except AcceptanceRejected as exc:
        outcome = f"rejected: {', '.join(exc.reasons)}"
    p.see(f"caller passed Authority({{ACCEPT}}) against a record that grants none -> {outcome}")
    p.see(f"task completion projects: {b.runtime.task_completion('t1').status}")
    accepted = b.ledger.events_by_kind(("task.accepted",))
    refused = b.ledger.events_by_kind(("task.acceptance_rejected",))
    p.see(f"task.accepted events: {len(accepted)}; durable refusals: {len(refused)}")
    if refused:
        payload = refused[-1].payload
        p.see(f"refusal names the directive it resolved: {payload.get('directive_id')}")
        p.see(f"refusal basis status: {(payload.get('authority_basis') or {}).get('status')}")
        p.see(f"what the caller claimed, recorded as a claim: "
              f"{payload.get('caller_claimed_capabilities')}")
    p.classification = "RECORDED" if accepted else "ENFORCED"
    p.note = ("Baseline (f0c730b): the grant behind an acceptance was the caller's argument. After "
              "W1-R2 it is resolved from the task's own recorded directive chain, and what the "
              "caller passes is recorded as a claim that changes nothing.")
    b.close()
    return p


def probe_c_decision_to_execution(stack) -> Probe:
    p = Probe("C", "decision -> execution",
              "Does the runtime enforce that the executed operation is the decided one?",
              "a governed operation must match its decision; an ungoverned one must say so")
    b = bench(stack)
    b.directive("d-root", (Capability.WRITE,))
    b.task("t1")
    b.produce_candidate()
    decision = b.runtime.decide_next_for_task("t1")
    recorded = [
        event for event in b.ledger.events_by_kind(("scheduler.decision_recorded",))
        if event.payload["task_id"] == "t1"
    ][-1].payload
    p.see(f"recorded decision: {decision.operation} ({recorded['decision_id'][:8]}...)")

    # 1. The action claims the CHECK decision.
    claimed = HonestWriter(b)
    refused = b.runtime.execute_action(
        b.action("a-claimed"), adapter=claimed,
        decision_id=recorded["decision_id"], source="scheduler",
    )
    p.see(f"action claiming that CHECK decision -> {refused.status}, "
          f"adapter calls {claimed.calls}")

    # 2. The action claims the scheduler without naming a decision.
    masquerade = HonestWriter(b)
    pretending = b.runtime.execute_action(
        b.action("a-pretend"), adapter=masquerade, source="scheduler"
    )
    p.see(f"action claiming scheduler governance with no decision -> {pretending.status}, "
          f"adapter calls {masquerade.calls}")

    # 3. The action claims nothing: still allowed, and recorded as what it is.
    external = HonestWriter(b)
    allowed = b.runtime.execute_action(b.action("a-external"), adapter=external)
    governance = [
        event for event in b.ledger.events_by_kind(("operation.governance_recorded",))
        if event.payload["subject_id"] == "a-external"
    ][0].payload
    p.see(f"action claiming nothing -> {allowed.status}, adapter calls {external.calls}, "
          f"recorded as {governance['status']}/{governance['source']}")

    # 4. A check under a decision taken on the state as it stands now. The
    #    earlier decision is already stale: the external action moved the basis.
    fresh = b.runtime.decide_next_for_task("t1")
    fresh_id = [
        event for event in b.ledger.events_by_kind(("scheduler.decision_recorded",))
        if event.payload["task_id"] == "t1"
    ][-1].payload["decision_id"]
    p.see(f"a decision taken now says: {fresh.operation}")
    governed = b.runtime.run_check(
        b.check("k-governed", script="import sys; sys.exit(0)"),
        verifier=LocalCommandVerifier(),
        decision_id=fresh_id, source="scheduler",
    )
    p.see(f"check under that fresh CHECK decision -> {governed.verdict}")

    # 5. The first decision, long since overtaken by the world.
    stale = b.runtime.run_check(
        b.check("k-stale", script="import sys; sys.exit(0)"),
        verifier=LocalCommandVerifier(),
        decision_id=recorded["decision_id"], source="scheduler",
    )
    p.see(f"the first decision, after the state moved -> {stale.verdict} "
          f"({(stale.error or '')[:52]}...)")

    mismatch_refused = refused.status != ActionStatus.SUCCEEDED and claimed.calls == 0
    masquerade_refused = pretending.status != ActionStatus.SUCCEEDED and masquerade.calls == 0
    external_visible = governance["status"] == "ungoverned"
    p.classification = (
        "ENFORCED" if (mismatch_refused and masquerade_refused and external_visible)
        else "CONVENTIONAL"
    )
    p.note = ("Baseline (f0c730b): the decision and the effect coexisted with no link in either "
              "direction. After W1-R4 an operation that claims a decision is checked against it "
              "for task, operation class and freshness, and an operation that claims none is "
              "recorded as external rather than silently passing for governed. The scheduler "
              "still cannot select ACTION, so no effect can be scheduler-governed: that is the "
              "book's position, now enforced rather than assumed.")
    b.close()
    return p


def probe_d_report_vs_observation(stack) -> Probe:
    p = Probe("D", "action request -> effect",
              "Does a worker's report of success establish that the effect happened?",
              "the worker's report and the runtime's observation must stay separable")
    b = bench(stack)
    b.directive("d-root", (Capability.WRITE,))
    b.task("t1")
    before = b.state_hash()
    liar = LyingWorker()
    result = b.runtime.execute_action(b.action("a1"), adapter=liar)
    after = b.state_hash()
    p.see(f"worker reported {result.status} and changed nothing: state hash moved = {before != after}")
    state = b.runtime.action_state("a1")
    p.see(f"projected effect state: {state.effect_state} (basis {state.basis})")
    p.see(f"the projection's stated reason: {state.reason!r}")
    p.see(f"runtime's own readings: pre {before[:12]}, recorded resulting "
          f"{(result.resulting_state_hash or '')[:12]} -- identical: "
          f"{result.resulting_state_hash == before}")

    # The intended effect was that the paragraph now carries the source marker.
    marker_script = (
        "import sys, pathlib; "
        "sys.exit(0 if '[S1]' in pathlib.Path(sys.argv[1]).read_text() else 1)"
    )
    verdict = b.runtime.run_check(
        b.check("k1", script=marker_script, target_hash=after), verifier=LocalCommandVerifier()
    )
    p.see(f"a check of the intended post-condition: {verdict.verdict} "
          f"(binding {verdict.binding_status})")
    trusted_the_report = str(state.effect_state) == "observed"
    p.see(f"effect_state trusts the report alone: {trusted_the_report}")
    p.classification = "RECORDED" if trusted_the_report else "DERIVED"
    p.note = (
        "Baseline (f0c730b): seam 1 separated result status from effect state everywhere except the "
        "SUCCEEDED branch, where OBSERVED came from the worker's report while the runtime held two "
        "identical readings it never compared. After W1-R1 the effect state is derived from those "
        "readings; the report itself is still RECORDED, which is correct -- it is the actor's claim, "
        "and it is kept as one."
    )
    b.close()
    return p


def probe_e_effect_uncertainty(stack) -> Probe:
    p = Probe("E", "effect -> recovery / reconciliation",
              "Does effect-unknown survive composition without becoming retry-safe?",
              "FAILED != INEFFECTUAL, and unknown must not imply permission to act again")
    b = bench(stack)
    b.directive("d-root", (Capability.WRITE,))
    b.task("t1")
    crasher = CrashAfterEffect(b)
    failed = b.runtime.execute_action(b.action("a1"), adapter=crasher)
    state = b.runtime.action_state("a1")
    p.see(f"reported status {failed.status}; effect state {state.effect_state}; "
          f"next operation {state.next_operation}")
    p.see(f"open effects for the task: {[s.action_id for s in b.runtime.open_effects(task_id="t1")]}")

    decision = b.runtime.decide_next_for_task("t1")
    p.see(f"scheduler with an unresolved effect: {decision.operation} ({decision.reason})")

    retry = b.runtime.execute_action(b.action("a1-retry", key="key:a1"), adapter=crasher)
    p.see(f"same-key retry -> {retry.status}; adapter invocations total {crasher.calls}")

    # A refused precondition proves the other side: no effect was possible.
    b.runtime.execute_action(
        b.action("a2", key="key:a2", precondition="a-state-that-is-not-current"),
        adapter=HonestWriter(b),
    )
    clean = b.runtime.action_state("a2")
    p.see(f"precondition mismatch -> effect {clean.effect_state}, next {clean.next_operation}")

    b.runtime.reconcile_action("a1", verdict=ReconciliationVerdict.EFFECT_CONFIRMED,
                               actor_id="operator", evidence_refs=("file:target#written",))
    after = b.runtime.action_state("a1")
    p.see(f"after reconciliation: effect {after.effect_state}, "
          f"reported status still {after.result_status}")
    p.see(f"scheduler after reconciliation: {b.runtime.decide_next_for_task('t1').operation}")
    p.classification = "ENFORCED"
    b.close()
    return p


def probe_f_binding(stack) -> Probe:
    p = Probe("F", "observed state -> verification binding",
              "Can a verifier decide, or misreport, what it was checking?",
              "binding failure is ERROR; the runtime's reading is the only account of the subject")
    b = bench(stack)
    forger = ForgingVerifier()
    bound = b.runtime.run_check(
        CheckRequest("k-bound", "t1", target_state_hash=b.state_hash()), verifier=forger
    )
    p.see(f"BOUND: verdict {bound.verdict}, binding {bound.binding_status}, "
          f"recorded observation {bound.observed_target_state_hash[:12]}... "
          f"(verifier claimed 'whatever-I-like')")

    unbound = b.runtime.run_check(CheckRequest("k-unbound", "t1"), verifier=forger)
    p.see(f"UNBOUND: verdict {unbound.verdict}, binding {unbound.binding_status}, "
          f"observation {unbound.observed_target_state_hash}")

    stale = b.state_hash()
    b.target.write_text("the world moved on\n", encoding="utf-8")
    mismatch = b.runtime.run_check(
        CheckRequest("k-stale", "t1", target_state_hash=stale), verifier=forger
    )
    p.see(f"MISMATCH: verdict {mismatch.verdict}, binding {mismatch.binding_status}, "
          f"verifier invocations so far {forger.calls}")

    b2 = bench(stack)
    b2.runtime.state_resolver = None
    unavailable = b2.runtime.run_check(
        CheckRequest("k-gone", "t1", target_state_hash="expected"), verifier=forger
    )
    p.see(f"UNAVAILABLE: verdict {unavailable.verdict}, binding {unavailable.binding_status}")
    p.see("binding failure never produced INCONCLUSIVE: "
          f"{unavailable.verdict == CheckVerdict.ERROR and mismatch.verdict == CheckVerdict.ERROR}")
    completed = {e.stream_id: e.payload for e in b.ledger.events_by_kind(("check.completed",))}
    p.see(f"binding is durable in the record: {sorted(completed['k-bound']['binding'])}")
    p.classification = "ENFORCED"
    p.note = "The binding is a reading, not a lock: nothing holds the state still during the run."
    b.close()
    b2.close()
    return p


def probe_g_verdict_semantics(stack) -> Probe:
    p = Probe("G", "verification command -> verdict semantics",
              "Do all four verdicts survive composition without collapsing?",
              "PASS/FAIL/INCONCLUSIVE/ERROR stay four different answers")
    b = bench(stack)
    verifier = LocalCommandVerifier()
    outcomes = {}
    for check_id, script in (
        ("k-pass", "import sys; sys.exit(0)"),
        ("k-fail", "import sys; sys.exit(1)"),
        ("k-maybe", "import sys; sys.stderr.write('cannot read as prose\\n'); sys.exit(2)"),
        ("k-unmapped", "import sys; sys.exit(7)"),
    ):
        result = b.runtime.run_check(b.check(check_id, script=script), verifier=verifier)
        outcomes[check_id] = str(result.verdict)
        detail = result.inconclusive_reason or result.error or f"exit {result.exit_code}"
        p.see(f"{check_id}: {result.verdict} -- {detail}")
    malformed = b.runtime.run_check(
        b.check("k-bare"), verifier=ForgingVerifier(CheckVerdict.INCONCLUSIVE)
    )
    p.see(f"INCONCLUSIVE with no reason -> {malformed.verdict} ({malformed.error})")
    p.classification = "ENFORCED" if len(set(outcomes.values())) == 4 else "COLLAPSED"
    b.close()
    return p


def probe_h_verification_to_claim(stack) -> Probe:
    p = Probe("H", "verification -> claim evidence",
              "Is an attempted-and-errored verification discoverable from the claim?",
              "ERROR must not refute, but the attempt must not become invisible either")
    b = bench(stack)
    for claim_id in ("c-pass", "c-fail", "c-maybe", "c-error"):
        b.runtime.record_claim(Claim(claim_id=claim_id, task_id="t1", statement="the marker is gone",
                                     source_call_id="m"))
    verifier = LocalCommandVerifier()
    b.runtime.run_check(b.check("k-pass", script="import sys; sys.exit(0)", claims=("c-pass",)),
                        verifier=verifier)
    b.runtime.run_check(b.check("k-fail", script="import sys; sys.exit(1)", claims=("c-fail",)),
                        verifier=verifier)
    b.runtime.run_check(b.check("k-maybe", script="import sys; sys.exit(2)", claims=("c-maybe",)),
                        verifier=verifier)
    errored = b.runtime.run_check(
        b.check("k-error", script="import sys; sys.exit(7)", claims=("c-error",)), verifier=verifier
    )
    claims = b.runtime.claims_for_task("t1")
    p.see(f"PASS -> {claims['c-pass'].evidence_class}; FAIL -> {claims['c-fail'].status}")
    p.see(f"INCONCLUSIVE -> status {claims['c-maybe'].status}, "
          f"durable attempts {len(b.runtime.inconclusive_checks_for_claim('c-maybe'))}")
    p.see(f"ERROR ({errored.error[:40]}...) -> status {claims['c-error'].status}")

    # The registered question: can a later reader find the errored attempt from the claim?
    claim_side = {
        "claim-scoped inconclusive index": len(b.runtime.inconclusive_checks_for_claim("c-error")),
        "events in the claim stream": len([
            e for e in b.ledger.read_all() if e.stream_id == "c-error"
        ]),
    }
    ledger_scan = [
        e.stream_id for e in b.ledger.events_by_kind(("check.requested",))
        if "c-error" in (e.payload.get("claim_ids") or ())
    ]
    p.see(f"from the claim side: {claim_side}")
    p.see(f"by scanning check.requested for the claim id: {ledger_scan}")
    p.classification = "DERIVED"
    p.note = ("The attempt is reconstructable only by scanning every check.requested for the claim "
              "id. Nothing on the claim records that a verification of it errored, so 'no negative "
              "evidence' and 'no claim-side trace' are currently the same thing.")
    b.close()
    return p


def probe_i_verification_to_acceptance(stack) -> Probe:
    p = Probe("I", "verification -> acceptance",
              "Which verification records can acceptance rely on, and are they bound to the bytes?",
              "acceptance must not rest on a stale, unbound, inconclusive or errored check")
    b = bench(stack)
    b.directive("d-root", (Capability.WRITE, Capability.ACCEPT))
    b.task("t1")
    produced = b.produce_candidate()
    other = b.produce_candidate(text="A different paragraph entirely [S1].")
    verifier = LocalCommandVerifier()

    def accept(check_ids, artifact=None):
        request = AcceptanceRequest(
            acceptance_id=str(uuid.uuid4()), task_id="t1", actor_id="reviewer",
            criteria_sha256=criteria_sha256(CRITERIA),
            artifact_sha256=artifact or produced["artifact_sha256"],
            source_call_id=produced["call_id"], source_attempt_id=produced["attempt_id"],
            source_interpretation_id=produced["interpretation_id"], check_ids=tuple(check_ids),
        )
        try:
            return str(b.runtime.accept_task(request).status)
        except AcceptanceRejected as exc:
            return f"rejected: {', '.join(exc.reasons)}"

    b.runtime.run_check(b.check("k-other", script="import sys; sys.exit(0)", artifact_arg=True,
                                target=artifact_target(other["artifact_sha256"])), verifier=verifier)
    p.see(f"PASS on a different artifact -> {accept(['k-other'])}")

    b.runtime.run_check(b.check("k-maybe", script="import sys; sys.exit(2)", artifact_arg=True,
                                target=artifact_target(produced["artifact_sha256"])),
                        verifier=verifier)
    p.see(f"INCONCLUSIVE on the right artifact -> {accept(['k-maybe'])}")

    b.runtime.run_check(b.check("k-err", script="import sys; sys.exit(7)", artifact_arg=True,
                                target=artifact_target(produced["artifact_sha256"])),
                        verifier=verifier)
    p.see(f"ERROR on the right artifact -> {accept(['k-err'])}")

    b.runtime.run_check(b.check("k-unlabelled", script="import sys; sys.exit(0)"),
                        verifier=verifier)
    p.see(f"PASS with no artifact label -> {accept(['k-unlabelled'])}")

    b.runtime.run_check(b.check("k-good", script="import sys; sys.exit(0)", artifact_arg=True,
                                target=artifact_target(produced["artifact_sha256"])),
                        verifier=verifier)
    p.see(f"PASS labelled with the accepted artifact -> {accept(['k-good'])}")

    completed = {e.stream_id: e.payload for e in b.ledger.events_by_kind(("check.completed",))}
    artifact_binding = completed["k-good"].get("artifact_binding") or {}
    p.see(f"the artifact identity acceptance rests on: status={artifact_binding.get('status')}, "
          f"materialized={artifact_binding.get('materialized')}")
    p.see(f"resolved digest: {str(artifact_binding.get('resolved_artifact_sha256'))[:16]}...")
    p.classification = "RECORDED" if artifact_binding.get("status") != "bound" else "ENFORCED"
    p.note = ("Baseline (f0c730b): acceptance matched a caller-written label against another label. "
              "After W1-R3 the runtime resolves the artifact from the store, verifies its digest and "
              "materializes it for the check, and acceptance compares that reading.")
    b.close()
    return p


def probe_j_acceptance_to_completion(stack) -> Probe:
    p = Probe("J", "acceptance -> completion",
              "Can completion occur without a valid acceptance path?",
              "model completed != check passed != accepted != completed")
    b = bench(stack)
    b.directive("d-root", (Capability.WRITE,))
    b.task("t1")
    produced = b.produce_candidate()
    p.see(f"after a succeeded call, completion projects: {b.runtime.task_completion('t1').status}")
    b.runtime.run_check(b.check("k1", script="import sys; sys.exit(0)",
                                target=artifact_target(produced["artifact_sha256"])),
                        verifier=LocalCommandVerifier())
    p.see(f"after a PASS, completion projects: {b.runtime.task_completion('t1').status}")

    from codeai.ledger import Event
    b.ledger.append(Event.create(stream_id="t1", kind="task.completed", actor_id="whoever",
                                 payload={"task_id": "t1"}, correlation_id="t1"))
    p.see(f"after a hand-appended task.completed: {b.runtime.task_completion('t1').status}")
    p.see(f"scheduler still says: {b.runtime.decide_next_for_task('t1').operation}")
    p.classification = "ENFORCED"
    b.close()
    return p


def probe_k_replay_authority(stack) -> Probe:
    p = Probe("K", "completion / replay -> current authority",
              "Which authority governs a replay of a previously authorized operation?",
              "a past success must not authorize a future replay by itself")
    b = bench(stack)
    b.directive("d-root", (Capability.WRITE, Capability.DESTRUCTIVE))
    b.directive("d-child", (Capability.WRITE,), parent="d-root")
    b.task("t1", "d-child")
    worker = HonestWriter(b)
    first = b.runtime.execute_action(b.action("a1", directive_id="d-child"), adapter=worker)
    p.see(f"authorized effect under d-child: {first.status}, adapter calls {worker.calls}")

    replay = b.runtime.execute_action(
        b.action("a1-again", key="key:a1", directive_id="d-child"), adapter=worker
    )
    p.see(f"same-key replay under the same grant: {replay.status}, adapter calls {worker.calls}")

    # Attempt to narrow the recorded grant after the fact.
    narrowed = None
    try:
        b.directive("d-child", (Capability.READ,), parent="d-root")
        narrowed = "second registration accepted"
    except Exception as exc:  # noqa: BLE001 - the audit records whatever happens
        narrowed = f"{type(exc).__name__}: {exc}"
    p.see(f"attempt to re-register d-child with a narrower grant -> {narrowed}")
    standing = b.runtime.directive_authority("d-child")
    p.see(f"resolved standing now: resolvable={standing.resolvable}, reason={standing.reason}")

    after = b.runtime.execute_action(
        b.action("a1-third", key="key:a1", directive_id="d-child"), adapter=worker
    )
    p.see(f"same-key replay after the attempted narrowing: {after.status} ({after.error})")
    p.see(f"adapter calls total: {worker.calls}")

    unauthorized = b.runtime.execute_action(
        b.action("a-destroy", capability="destructive", key="key:a1", directive_id="d-child"),
        adapter=worker,
    )
    p.see(f"replaying the same key while asking for a capability the child lacks: "
          f"{unauthorized.status}")
    p.classification = "ENFORCED"
    p.note = ("Replay is authorized against the record as it stands now, not against the grant the "
              "original effect ran under. Revocation itself is absent: a directive cannot be "
              "narrowed, only invalidated by a second registration.")
    b.close()
    return p


def probe_l_decision_provenance(stack) -> Probe:
    p = Probe("L", "durable state -> next-operation decision",
              "Is every decision derived from durable facts, and is every decision recorded?",
              "recomputable != recorded != causally binding")
    b = bench(stack)
    b.directive("d-root", (Capability.WRITE,))
    b.task("t1")
    b.produce_candidate()
    recorded = b.runtime.decide_next_for_task("t1")
    p.see(f"task-level API: {recorded.operation}, decisions recorded: "
          f"{len(b.ledger.events_by_kind(('scheduler.decision_recorded',)))}")

    from codeai.scheduler import SchedulerInput
    fabricated = b.runtime.decide_next(SchedulerInput(has_required_verification=False,
                                                      requests_independent_proposals=False))
    p.see(f"raw SchedulerInput API: {fabricated.operation} from caller-supplied flags")
    p.see(f"decisions recorded after that call: "
          f"{len(b.ledger.events_by_kind(('scheduler.decision_recorded',)))}")
    p.see(f"projection contradicts the fabricated flags: "
          f"check_required={b.runtime.process_state('t1').check_required}")
    p.classification = "DERIVED"
    p.note = ("The task-level path projects and records. The raw path remains a pure function a "
              "caller may call with any flags; it records nothing, so it creates no false basis, "
              "but it also means a decision can exist outside the record.")
    b.close()
    return p


def probe_m_artifact_label(stack) -> Probe:
    p = Probe("M", "check -> artifact identity",
              "Is the artifact a check claims to have examined established, or merely labelled?",
              "acceptance should rest on a check that demonstrably examined those bytes")
    b = bench(stack)
    b.directive("d-root", (Capability.WRITE, Capability.ACCEPT))
    b.task("t1")
    produced = b.produce_candidate()

    # A command that never reads the artifact, wearing the artifact's label.
    blind = "import sys; sys.exit(0)"
    b.runtime.run_check(
        b.check("k-blind", script=blind, target=artifact_target(produced["artifact_sha256"])),
        verifier=LocalCommandVerifier(),
    )
    request = AcceptanceRequest(
        acceptance_id=str(uuid.uuid4()), task_id="t1", actor_id="reviewer",
        criteria_sha256=criteria_sha256(CRITERIA), artifact_sha256=produced["artifact_sha256"],
        source_call_id=produced["call_id"], source_attempt_id=produced["attempt_id"],
        source_interpretation_id=produced["interpretation_id"], check_ids=("k-blind",),
    )
    try:
        outcome = str(b.runtime.accept_task(request).status)
    except AcceptanceRejected as exc:
        outcome = f"rejected: {', '.join(exc.reasons)}"
    p.see("check command: 'sys.exit(0)' -- it never opened the artifact")
    p.see(f"check request labelled target={artifact_target(produced['artifact_sha256'])[:28]}...")
    p.see(f"acceptance citing that check -> {outcome}")
    completed = {e.stream_id: e.payload for e in b.ledger.events_by_kind(("check.completed",))}
    artifact_binding = completed["k-blind"].get("artifact_binding") or {}
    p.see(f"the check's own verdict: {completed['k-blind']['verdict']}")
    p.see(f"artifact binding: {artifact_binding.get('status')} -- "
          f"{artifact_binding.get('reason')}")
    p.classification = "CONVENTIONAL" if "completed" in outcome else "ENFORCED"
    p.note = ("Baseline (f0c730b): a command that never opened the artifact carried its label, "
              "passed, and completed an acceptance. After W1-R3 a command check that never "
              "references the materialized artifact is UNCONSUMED, which is ERROR rather than a "
              "verdict. What the command does with the bytes it is given remains its own business.")
    b.close()
    return p


def probe_n_human_gate_and_acceptance_grant(stack) -> Probe:
    p = Probe("N", "scheduler human gate -> acceptance authority",
              "When the scheduler asks for a human, can a human answer?",
              "ASK_HUMAN should name a gate somebody can pass")
    b = bench(stack)
    b.directive("d-root", (Capability.WRITE,))  # no ACCEPT anywhere in the chain
    b.task("t1")
    produced = b.produce_candidate()
    b.runtime.run_check(
        b.check("k1", script="import sys; sys.exit(0)", artifact_arg=True,
                target=artifact_target(produced["artifact_sha256"])),
        verifier=LocalCommandVerifier(),
    )
    decision = b.runtime.decide_next_for_task("t1")
    p.see(f"scheduler: {decision.operation} ({decision.reason})")

    request = AcceptanceRequest(
        acceptance_id=str(uuid.uuid4()), task_id="t1", actor_id="a-human-reviewer",
        criteria_sha256=criteria_sha256(CRITERIA), artifact_sha256=produced["artifact_sha256"],
        source_call_id=produced["call_id"], source_attempt_id=produced["attempt_id"],
        source_interpretation_id=produced["interpretation_id"], check_ids=("k1",),
    )
    try:
        outcome = str(b.runtime.accept_task(request).status)
    except AcceptanceRejected as exc:
        outcome = f"rejected: {', '.join(exc.reasons)}"
    p.see(f"a human answering that gate -> {outcome}")
    p.see(f"the task ends at: {b.runtime.decide_next_for_task('t1').operation}")
    p.see("the gate can only be opened by recording a grant, which this task cannot acquire")
    p.classification = "ABSENT"
    p.note = ("Not a stuck scheduler: the durable process has reached a state from which completion "
              "is unauthorized, which is the correct result. What is absent is a legal transition "
              "out of it. Human intervention should change authority durably and then re-project, "
              "never bypass it -- so the missing capability is authority transition (supersession), "
              "not an acceptance override. Delegation may only narrow; altering authority is a "
              "different operation. Queued as BOOK-CORE/authority-transition.")
    b.close()
    return p


PROBES = (
    probe_a_action_authority,
    probe_b_acceptance_authority,
    probe_c_decision_to_execution,
    probe_d_report_vs_observation,
    probe_e_effect_uncertainty,
    probe_f_binding,
    probe_g_verdict_semantics,
    probe_h_verification_to_claim,
    probe_i_verification_to_acceptance,
    probe_j_acceptance_to_completion,
    probe_k_replay_authority,
    probe_l_decision_provenance,
    probe_m_artifact_label,
    probe_n_human_gate_and_acceptance_grant,
)


def subject_identity() -> dict[str, str]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[1]).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        # The runtime under audit: the audit branch adds tests and experiments,
        # never src/, so this is the commit whose behaviour was observed.
        "runtime_commit": git("log", "-1", "--format=%H", "--", "src"),
        "python": sys.version.split()[0],
    }


def main() -> dict[str, object]:
    stack: list[TemporaryDirectory] = []
    try:
        lifecycle = happy_path(stack)
        probes = [probe(stack) for probe in PROBES]
        result = {
            "subject": subject_identity(),
            "protocol": "experiments/W1-composition-prereg.md",
            "happy_path": lifecycle,
            "probes": [p.as_dict() for p in probes],
        }
    finally:
        for directory in stack:
            try:
                directory.cleanup()
            except OSError:
                pass

    print("=" * 78)
    print("HAPPY PATH")
    for step in lifecycle["trace"]:
        print(f"  {step['step']:22} {({k: v for k, v in step.items() if k != 'step'})}")
    print(f"  reopen identical: {lifecycle['reopen_identical']}, "
          f"reprojected status {lifecycle['reprojected_status']}, "
          f"decision {lifecycle['reprojected_decision']}")
    print()
    for probe in result["probes"]:
        print("=" * 78)
        print(f"{probe['probe_id']}  {probe['joint']}  -> {probe['classification']}")
        print(f"    Q: {probe['question']}")
        for fact in probe["observed"]:
            print(f"     - {fact}")
        if probe["note"]:
            print(f"    note: {probe['note']}")
    return result


if __name__ == "__main__":
    output = main()
    destination = Path(__file__).resolve().parent / (
        sys.argv[1] if len(sys.argv) > 1 else "W1-composition-results.json"
    )
    destination.write_text(json.dumps(output, indent=2, sort_keys=False), encoding="utf-8")
    print(f"\nfrozen: {destination}")
