"""Chapter 21 seam: four verification outcomes that must not collapse into two.

    PASS          ran against the intended target, criterion established
    FAIL          ran against the intended target, criterion violated
    INCONCLUSIVE  ran legitimately, cannot establish PASS or FAIL
    ERROR         no trustworthy result was obtained at all

The two rules that keep them apart: a binding failure is never INCONCLUSIVE,
and an unexplained INCONCLUSIVE is a malformed result. Nothing here claims a
PASS means the system is correct; it means this declared check, against this
bound target, produced this result.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from codeai.adapters import CheckRequest, CheckResult, CheckVerdict
from codeai.artifacts import FileArtifactStore
from codeai.domain import Claim, ClaimStatus, EvidenceClass
from codeai.ledger import SQLiteLedger
from codeai.runtime import Runtime
from codeai.verification import (
    VERIFICATION_V1,
    BindingStatus,
    ExitCodePolicy,
    bind_check_target,
)
from codeai.verifier import LocalCommandVerifier

EXIT_POLICY = {
    "policy_id": "exit-code-v1",
    "pass_codes": [0],
    "fail_codes": [1],
    "inconclusive_codes": [2],
}


def make_runtime(tmp_path: Path, *, state_resolver=None) -> Runtime:
    ledger = SQLiteLedger(":memory:")
    return Runtime(
        ledger,
        artifact_store=FileArtifactStore(tmp_path / "artifacts", ledger),
        state_resolver=state_resolver,
    )


def command(script: str) -> tuple[str, ...]:
    return (sys.executable, "-c", script)


def declared_verifier() -> LocalCommandVerifier:
    return LocalCommandVerifier(
        pass_exit_codes=(0,), fail_exit_codes=(1,), inconclusive_exit_codes=(2,)
    )


def request(
    check_id: str, script: str, *, claims: tuple[str, ...] = (), target_hash: str | None = None
) -> CheckRequest:
    return CheckRequest(
        check_id=check_id,
        task_id="t1",
        claim_ids=claims,
        command=command(script),
        cwd=".",
        target_state_hash=target_hash,
        verdict_policy=EXIT_POLICY,
    )


class ForgingVerifier:
    """Reports a verdict, and lies about what it was looking at."""

    def __init__(self, verdict=CheckVerdict.PASS) -> None:
        self.calls = 0
        self.verdict = verdict

    def run(self, request: CheckRequest) -> CheckResult:
        self.calls += 1
        return CheckResult(
            check_id=request.check_id,
            verdict=self.verdict,
            observed_target_state_hash="whatever-I-like",
            inconclusive_reason="stated" if self.verdict == CheckVerdict.INCONCLUSIVE else None,
        )


# ---------------- the matrix ----------------


def test_a_bound_target_and_an_established_criterion_passes(tmp_path):
    runtime = make_runtime(tmp_path, state_resolver=lambda: "state-A")
    result = runtime.run_check(
        request("k-pass", "print('ok')", target_hash="state-A"), verifier=declared_verifier()
    )
    assert result.verdict == CheckVerdict.PASS
    assert result.binding_status == BindingStatus.BOUND
    assert result.observed_target_state_hash == "state-A"


def test_a_bound_target_and_a_violated_criterion_fails(tmp_path):
    runtime = make_runtime(tmp_path, state_resolver=lambda: "state-A")
    result = runtime.run_check(
        request("k-fail", "import sys; sys.exit(1)", target_hash="state-A"),
        verifier=declared_verifier(),
    )
    assert result.verdict == CheckVerdict.FAIL
    assert result.binding_status == BindingStatus.BOUND


def test_a_check_that_legitimately_cannot_decide_is_inconclusive(tmp_path):
    # A real command, a real exit code, and a mapping declared before it ran.
    runtime = make_runtime(tmp_path, state_resolver=lambda: "state-A")
    result = runtime.run_check(
        request(
            "k-maybe",
            "import sys; sys.stderr.write('no signature block to compare\\n'); sys.exit(2)",
            target_hash="state-A",
        ),
        verifier=declared_verifier(),
    )
    assert result.verdict == CheckVerdict.INCONCLUSIVE
    assert result.verdict != CheckVerdict.FAIL and result.verdict != CheckVerdict.ERROR
    assert result.exit_code == 2
    assert "declared inconclusive by exit-code-v1" in result.inconclusive_reason
    assert "no signature block to compare" in result.inconclusive_reason
    assert result.binding_status == BindingStatus.BOUND


def test_a_target_mismatch_is_an_error_and_the_verifier_never_runs(tmp_path):
    runtime = make_runtime(tmp_path, state_resolver=lambda: "state-B")
    verifier = ForgingVerifier()
    result = runtime.run_check(
        request("k-moved", "print('ok')", target_hash="state-A"), verifier=verifier
    )
    assert verifier.calls == 0
    assert result.verdict == CheckVerdict.ERROR
    assert result.binding_status == BindingStatus.MISMATCH
    assert result.error == "target state mismatch: expected state-A, observed state-B"


@pytest.mark.parametrize("mode", ["no_resolver", "resolver_raises", "resolver_returns_none"])
def test_an_unavailable_target_is_an_error_and_the_verifier_never_runs(tmp_path, mode):
    def raises():
        raise OSError("unreadable")

    resolver = {"no_resolver": None, "resolver_raises": raises,
                "resolver_returns_none": lambda: None}[mode]
    runtime = make_runtime(tmp_path, state_resolver=resolver)
    verifier = ForgingVerifier()
    result = runtime.run_check(
        request("k-gone", "print('ok')", target_hash="state-A"), verifier=verifier
    )
    assert verifier.calls == 0
    assert result.verdict == CheckVerdict.ERROR
    assert result.binding_status == BindingStatus.UNAVAILABLE
    # Infrastructure uncertainty is not epistemic uncertainty.
    assert result.verdict != CheckVerdict.INCONCLUSIVE
    assert result.inconclusive_reason is None


def test_a_raising_verifier_is_an_error(tmp_path):
    class Boom:
        def run(self, request):
            raise RuntimeError("boom")

    runtime = make_runtime(tmp_path)
    result = runtime.run_check(request("k-boom", "print('ok')"), verifier=Boom())
    assert result.verdict == CheckVerdict.ERROR
    assert result.error == "verifier raised RuntimeError: boom"


@pytest.mark.parametrize(
    ("result_factory", "expected"),
    [
        (lambda r: "PASSED", "verifier returned str, not a CheckResult"),
        (
            lambda r: CheckResult(check_id=r.check_id, verdict="MOSTLY_FINE"),
            "verifier returned an unknown verdict: 'MOSTLY_FINE'",
        ),
        (
            lambda r: CheckResult(check_id="some-other-check", verdict=CheckVerdict.PASS),
            "verifier returned a result for check 'some-other-check'",
        ),
        (
            lambda r: CheckResult(check_id=r.check_id, verdict=CheckVerdict.INCONCLUSIVE),
            "inconclusive result carries no reason",
        ),
        (
            lambda r: CheckResult(
                check_id=r.check_id, verdict=CheckVerdict.PASS, verdict_policy_id="my-own-rules"
            ),
            "verifier applied policy 'my-own-rules', but 'exit-code-v1' was declared",
        ),
    ],
)
def test_an_unusable_result_is_an_error(tmp_path, result_factory, expected):
    class Odd:
        def run(self, request):
            return result_factory(request)

    runtime = make_runtime(tmp_path)
    result = runtime.run_check(request("k-odd", "print('ok')"), verifier=Odd())
    assert result.verdict == CheckVerdict.ERROR
    assert result.error == expected


def test_an_exit_code_the_declared_policy_does_not_map_is_an_error(tmp_path):
    runtime = make_runtime(tmp_path)
    result = runtime.run_check(
        request("k-unmapped", "import sys; sys.exit(7)"), verifier=declared_verifier()
    )
    assert result.verdict == CheckVerdict.ERROR
    assert result.exit_code == 7
    assert "exit code 7 is not mapped by exit-code-v1" in result.error


def test_the_runtime_observation_wins_over_the_verifier_account(tmp_path):
    runtime = make_runtime(tmp_path, state_resolver=lambda: "state-A")
    verifier = ForgingVerifier()
    result = runtime.run_check(
        CheckRequest("k-forged", "t1", target_state_hash="state-A"), verifier=verifier
    )
    assert verifier.calls == 1
    assert result.verdict == CheckVerdict.PASS
    # A verifier cannot say what state it checked.
    assert result.observed_target_state_hash == "state-A"
    completed = next(iter(runtime.ledger.events_by_kind(("check.completed",))))
    assert completed.payload["observed_target_state_hash"] == "state-A"
    assert completed.payload["binding"]["observed_state_hash"] == "state-A"


# ---------------- what a verdict does to a claim ----------------


def claim(runtime: Runtime, claim_id: str) -> None:
    runtime.record_claim(
        Claim(claim_id=claim_id, task_id="t1", statement="the marker is gone", source_call_id="m")
    )


def test_pass_and_fail_move_the_claims_the_check_named(tmp_path):
    runtime = make_runtime(tmp_path, state_resolver=lambda: "state-A")
    claim(runtime, "c-pass")
    claim(runtime, "c-fail")
    claim(runtime, "c-untouched")
    runtime.run_check(request("k1", "print('ok')", claims=("c-pass",)), verifier=declared_verifier())
    runtime.run_check(
        request("k2", "import sys; sys.exit(1)", claims=("c-fail",)), verifier=declared_verifier()
    )
    claims = runtime.claims_for_task("t1")
    assert claims["c-pass"].evidence_class == EvidenceClass.REPRODUCED
    assert claims["c-fail"].status == ClaimStatus.REFUTED
    assert claims["c-untouched"].evidence_class == EvidenceClass.ASSERTED


def test_inconclusive_is_durable_evidence_that_changes_no_status(tmp_path):
    runtime = make_runtime(tmp_path, state_resolver=lambda: "state-A")
    claim(runtime, "c-maybe")
    result = runtime.run_check(
        request("k-maybe", "import sys; sys.exit(2)", claims=("c-maybe",)),
        verifier=declared_verifier(),
    )
    assert result.verdict == CheckVerdict.INCONCLUSIVE

    recorded = runtime.inconclusive_checks_for_claim("c-maybe")
    assert len(recorded) == 1
    assert recorded[0]["check_id"] == "k-maybe"
    assert recorded[0]["inconclusive_reason"]
    # The attempt is recorded; the claim is exactly where it was.
    standing = runtime.claims_for_task("t1")["c-maybe"]
    assert standing.status == ClaimStatus.ASSERTED
    assert standing.evidence_class == EvidenceClass.ASSERTED


def test_an_errored_check_is_never_negative_evidence_against_a_claim(tmp_path):
    runtime = make_runtime(tmp_path, state_resolver=lambda: "state-B")
    claim(runtime, "c-safe")
    result = runtime.run_check(
        request("k-error", "import sys; sys.exit(1)", claims=("c-safe",), target_hash="state-A"),
        verifier=declared_verifier(),
    )
    assert result.verdict == CheckVerdict.ERROR

    # verification failed != claim disproved
    standing = runtime.claims_for_task("t1")["c-safe"]
    assert standing.status == ClaimStatus.ASSERTED
    assert runtime.inconclusive_checks_for_claim("c-safe") == ()
    kinds = [e.kind for e in runtime.ledger.read_all()]
    assert "claim.status" not in kinds and "claim.evidence" not in kinds


def test_the_deliberate_evidence_path_separates_errored_from_inconclusive(tmp_path):
    from codeai.evidence import CHECK, SUPPORTS, EvidenceRecord, EvidenceRefused

    runtime = make_runtime(tmp_path, state_resolver=lambda: "state-B")
    runtime.run_check(
        request("k-error", "print('ok')", claims=("c1",), target_hash="state-A"),
        verifier=declared_verifier(),
    )
    runtime.run_check(
        request("k-maybe", "import sys; sys.exit(2)", claims=("c1",)), verifier=declared_verifier()
    )
    with pytest.raises(EvidenceRefused) as errored:
        runtime.record_claim_evidence(
            EvidenceRecord("ev-1", "c1", CHECK, SUPPORTS, "reviewer", check_id="k-error")
        )
    with pytest.raises(EvidenceRefused) as inconclusive:
        runtime.record_claim_evidence(
            EvidenceRecord("ev-2", "c1", CHECK, SUPPORTS, "reviewer", check_id="k-maybe")
        )
    assert "check_errored" in errored.value.reasons
    assert "check_inconclusive" in inconclusive.value.reasons
    assert "check_errored" not in inconclusive.value.reasons


# ---------------- reconstruction ----------------


def test_the_record_reconstructs_target_policy_verifier_and_verdict(tmp_path):
    path = tmp_path / "ledger.sqlite"
    ledger = SQLiteLedger(path)
    runtime = Runtime(ledger, state_resolver=lambda: "state-A")
    runtime.run_check(
        request("k-full", "import sys; sys.exit(2)", claims=("c1",), target_hash="state-A"),
        verifier=declared_verifier(),
    )
    events = ledger.read_all()
    ledger.close()

    reopened = SQLiteLedger(path)
    assert reopened.read_all() == events
    requested = next(iter(reopened.events_by_kind(("check.requested",))))
    completed = next(iter(reopened.events_by_kind(("check.completed",))))
    reopened.close()

    # Declared before execution...
    assert requested.payload["verdict_policy"] == EXIT_POLICY
    assert requested.payload["target_state_hash"] == "state-A"
    # ...and recorded with what was actually established.
    assert completed.causation_id == requested.event_id
    assert completed.payload["verdict"] == "INCONCLUSIVE"
    assert completed.payload["inconclusive_reason"]
    assert completed.payload["declared_verdict_policy"] == EXIT_POLICY
    assert completed.payload["verdict_policy_id"] == "exit-code-v1"
    assert completed.payload["binding"]["status"] == "bound"
    assert completed.payload["binding"]["requested_state_hash"] == "state-A"
    assert completed.payload["verifier_identity"]["name"] == "LocalCommandVerifier"
    assert completed.payload["verifier_identity"]["version"] == "local-command-v2"
    assert completed.payload["verification_version"] == VERIFICATION_V1


def test_an_unbound_check_says_so_rather_than_implying_a_subject(tmp_path):
    runtime = make_runtime(tmp_path)
    result = runtime.run_check(request("k-unbound", "print('ok')"), verifier=declared_verifier())
    assert result.verdict == CheckVerdict.PASS
    assert result.binding_status == BindingStatus.UNBOUND
    assert result.observed_target_state_hash is None
    completed = next(iter(runtime.ledger.events_by_kind(("check.completed",))))
    assert "not pinned" in completed.payload["binding"]["reason"]


def test_binding_reads_the_state_once_and_appends_nothing(tmp_path):
    reads = []

    def observe():
        reads.append(len(reads))
        return "A" if not reads[:-1] else "B"

    runtime = make_runtime(tmp_path, state_resolver=observe)
    binding = bind_check_target(runtime, CheckRequest("k", "t1", target_state_hash="B"))
    assert reads == [0]
    assert binding.status == BindingStatus.MISMATCH
    assert binding.observed_state_hash == "A"
    assert binding.permits_verification is False
    assert runtime.ledger.read_all() == ()


def test_the_default_policy_keeps_the_ordinary_shell_reading(tmp_path):
    # No declared mapping: 0 passes, anything else fails, nothing is inconclusive.
    runtime = make_runtime(tmp_path)
    plain = LocalCommandVerifier()
    passed = runtime.run_check(
        CheckRequest("k-ok", "t1", command=command("print('ok')")), verifier=plain
    )
    failed = runtime.run_check(
        CheckRequest("k-no", "t1", command=command("import sys; sys.exit(3)")), verifier=plain
    )
    assert passed.verdict == CheckVerdict.PASS
    assert failed.verdict == CheckVerdict.FAIL


def test_the_policy_maps_exit_codes_it_declares_and_refuses_the_rest():
    policy = ExitCodePolicy(pass_codes=(0,), fail_codes=(1,), inconclusive_codes=(2,))
    assert policy.verdict_for(0) == CheckVerdict.PASS
    assert policy.verdict_for(1) == CheckVerdict.FAIL
    assert policy.verdict_for(2) == CheckVerdict.INCONCLUSIVE
    assert policy.verdict_for(9) is None
    assert ExitCodePolicy().verdict_for(9) == CheckVerdict.FAIL
    assert ExitCodePolicy.from_payload(EXIT_POLICY).inconclusive_codes == (2,)
    assert ExitCodePolicy.from_payload(None) is None


def test_a_request_policy_overrides_the_verifier_default(tmp_path):
    # The mapping a later reader can see is the one in the ledger.
    runtime = make_runtime(tmp_path)
    verifier = LocalCommandVerifier()  # default: nothing is inconclusive
    result = runtime.run_check(
        request("k-declared", "import sys; sys.exit(2)"), verifier=verifier
    )
    assert result.verdict == CheckVerdict.INCONCLUSIVE
    assert verifier.policy.inconclusive_codes == ()


def test_a_check_establishes_the_result_not_the_adequacy_of_the_test(tmp_path):
    """A passing check is a fact about a command, not a proof about the system."""
    runtime = make_runtime(tmp_path, state_resolver=lambda: sha256(b"repo").hexdigest())
    result = runtime.run_check(
        request("k-narrow", "print('ok')", target_hash=sha256(b"repo").hexdigest()),
        verifier=declared_verifier(),
    )
    completed = next(iter(runtime.ledger.events_by_kind(("check.completed",))))
    assert result.verdict == CheckVerdict.PASS
    # Everything the record claims is mechanical: this command, this exit code,
    # this target reading, under this declared policy. Nothing about coverage,
    # independence, hermeticity or whether the criterion was the right one.
    assert completed.payload["exit_code"] == 0
    assert set(completed.payload["binding"]) == {
        "requested_state_hash", "observed_state_hash", "status", "reason", "version"
    }


# ---------------- the artifact a check was given (W1-R3) ----------------


def stored(runtime, text="the cache is intended to make page loads faster"):
    return runtime.artifact_store.store_text(text, artifact_type="candidate_output")


def artifact_check(check_id, ref, *, script=None, consumes=True):
    """A check over stored bytes. ``consumes`` decides whether it is given them."""
    from codeai.acceptance import artifact_target

    command = ()
    if script is not None:
        subject = "{artifact}" if consumes else "/no/such/path"
        command = (sys.executable, "-c", script, subject)
    return CheckRequest(
        check_id=check_id,
        task_id="t1",
        command=command,
        cwd=".",
        target=artifact_target(ref.sha256),
        verdict_policy=EXIT_POLICY,
    )


READS_IT = (
    "import sys, pathlib; "
    "text = pathlib.Path(sys.argv[1]).read_text(); "
    "sys.exit(0 if 'cache' in text else 1)"
)


def test_a_command_given_the_artifact_reads_the_bytes_the_runtime_verified(tmp_path):
    runtime = make_runtime(tmp_path)
    ref = stored(runtime)
    result = runtime.run_check(
        artifact_check("k", ref, script=READS_IT), verifier=declared_verifier()
    )
    assert result.verdict == CheckVerdict.PASS
    assert result.artifact_binding_status == "bound"

    completed = next(iter(runtime.ledger.events_by_kind(("check.completed",))))
    binding = completed.payload["artifact_binding"]
    assert binding["status"] == "bound"
    assert binding["requested_artifact_sha256"] == ref.sha256
    assert binding["resolved_artifact_sha256"] == ref.sha256
    assert binding["materialized"] is True


def test_a_command_that_never_references_the_artifact_is_an_error(tmp_path):
    """Audit gap 3: a check that was never given the bytes cannot have read them."""
    runtime = make_runtime(tmp_path)
    ref = stored(runtime)
    result = runtime.run_check(
        artifact_check("k", ref, script="import sys; sys.exit(0)", consumes=False),
        verifier=declared_verifier(),
    )
    assert result.verdict == CheckVerdict.ERROR
    assert result.verdict != CheckVerdict.PASS
    assert result.artifact_binding_status == "unconsumed"
    assert "never given to it" in result.error


def test_an_artifact_that_is_not_stored_is_an_error_not_a_verdict(tmp_path):
    from codeai.acceptance import artifact_target, text_sha256

    runtime = make_runtime(tmp_path)
    result = runtime.run_check(
        CheckRequest("k", "t1", target=artifact_target(text_sha256("never stored"))),
        verifier=declared_verifier(),
    )
    assert result.verdict == CheckVerdict.ERROR
    assert result.artifact_binding_status == "missing"


def test_a_check_with_no_artifact_target_stays_unbound_and_runs(tmp_path):
    runtime = make_runtime(tmp_path)
    result = runtime.run_check(
        CheckRequest("k", "t1", command=command("print('ok')"), cwd=".",
                     target="file:some/path", verdict_policy=EXIT_POLICY),
        verifier=declared_verifier(),
    )
    assert result.verdict == CheckVerdict.PASS
    assert result.artifact_binding_status == "unbound"
    completed = next(iter(runtime.ledger.events_by_kind(("check.completed",))))
    assert "does not name an artifact" in completed.payload["artifact_binding"]["reason"]


def test_a_verifier_without_a_command_is_still_handed_the_bytes(tmp_path):
    """Static verifiers keep working, and the record says what they were given."""
    seen = {}

    class Static:
        def run(self, request):
            seen["path"] = request.materialized_artifact_path
            seen["content"] = Path(request.materialized_artifact_path).read_text(encoding="utf-8")
            return CheckResult(check_id=request.check_id, verdict=CheckVerdict.PASS)

    runtime = make_runtime(tmp_path)
    ref = stored(runtime)
    result = runtime.run_check(artifact_check("k", ref), verifier=Static())
    assert result.verdict == CheckVerdict.PASS
    assert result.artifact_binding_status == "bound"
    assert "cache" in seen["content"]


def test_the_caller_cannot_set_the_materialized_path(tmp_path):
    seen = {}

    class Static:
        def run(self, request):
            # Read it here: the materialized copy is per-check and transient.
            seen["path"] = request.materialized_artifact_path
            seen["content"] = Path(seen["path"]).read_text(encoding="utf-8")
            return CheckResult(check_id=request.check_id, verdict=CheckVerdict.PASS)

    runtime = make_runtime(tmp_path)
    ref = stored(runtime)
    forged = replace(artifact_check("k", ref), materialized_artifact_path="/tmp/whatever-I-like")
    runtime.run_check(forged, verifier=Static())
    assert seen["path"] != "/tmp/whatever-I-like"
    assert seen["content"].startswith("the cache")
    # The copy does not outlive the check; the artifact itself lives in the store.
    assert not Path(seen["path"]).exists()


def test_corrupted_stored_bytes_are_a_mismatch_not_a_verdict(tmp_path):
    runtime = make_runtime(tmp_path)
    ref = stored(runtime)
    # Overwrite the stored bytes behind the store's back.
    path = runtime.artifact_store._path_for(ref.sha256)
    path.write_text("something else entirely", encoding="utf-8")

    result = runtime.run_check(
        artifact_check("k", ref, script=READS_IT), verifier=declared_verifier()
    )
    assert result.verdict == CheckVerdict.ERROR
    assert result.artifact_binding_status == "mismatch"
    assert result.verdict != CheckVerdict.FAIL


def test_the_two_bindings_are_independent(tmp_path):
    """State binding and artifact binding answer different questions."""
    runtime = make_runtime(tmp_path, state_resolver=lambda: "state-A")
    ref = stored(runtime)
    request = replace(artifact_check("k", ref, script=READS_IT), target_state_hash="state-A")
    result = runtime.run_check(request, verifier=declared_verifier())
    assert (result.binding_status, result.artifact_binding_status) == ("bound", "bound")

    moved = replace(request, check_id="k2", target_state_hash="state-B")
    second = runtime.run_check(moved, verifier=declared_verifier())
    # The state moved; the artifact is still exactly what it was.
    assert second.verdict == CheckVerdict.ERROR
    assert (second.binding_status, second.artifact_binding_status) == ("mismatch", "bound")


def test_artifact_binding_is_not_artifact_adequacy(tmp_path):
    """BOUND says which bytes were supplied. It says nothing about the check.

    Both of these are legitimately BOUND: one examines the artifact, the other
    is handed it and ignores it. The runtime claims only the first thing.
    """
    runtime = make_runtime(tmp_path)
    ref = stored(runtime)
    meaningful = runtime.run_check(
        artifact_check("k-real", ref, script=READS_IT), verifier=declared_verifier()
    )
    ignores_it = runtime.run_check(
        artifact_check("k-lazy", ref, script="import sys; sys.exit(0)"),
        verifier=declared_verifier(),
    )
    assert meaningful.artifact_binding_status == "bound"
    assert ignores_it.artifact_binding_status == "bound"
    assert (meaningful.verdict, ignores_it.verdict) == (CheckVerdict.PASS, CheckVerdict.PASS)
    # The record establishes supply, not scrutiny. Whether the criterion was the
    # right one, and whether the command applied it, is verification adequacy,
    # which this seam deliberately does not address.
    completed = {e.stream_id: e.payload for e in runtime.ledger.events_by_kind(("check.completed",))}
    for check_id in ("k-real", "k-lazy"):
        assert completed[check_id]["artifact_binding"]["materialized"] is True
