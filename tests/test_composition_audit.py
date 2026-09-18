"""The Wave 1 composition audit, pinned.

These assertions record what the audit *observed*, including the gaps. The
frozen baseline is 11fedcd (result: experiments/W1-composition-results.json).
Where a repair has landed, the assertion moves and names the repair; the frozen
result never moves.
Several of them assert behaviour that is wrong and known to be wrong: that is the
point of a frozen baseline. When a repair lands, the failing assertion here is
the signal to write a new audit entry citing the old one -- never to edit the
frozen result.

Protocol:  experiments/W1-composition-prereg.md
Result:    experiments/W1-composition-results.md / .json
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

AUDIT = Path(__file__).resolve().parents[1] / "experiments" / "wave1_composition_audit.py"


def load():
    spec = importlib.util.spec_from_file_location("wave1_composition_audit", AUDIT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit(capsys_module=None):
    module = load()
    return module.main()


@pytest.fixture(scope="module")
def probes(audit):
    return {probe["probe_id"]: probe for probe in audit["probes"]}


def test_the_happy_path_crosses_every_seam_and_survives_a_reopen(audit):
    lifecycle = audit["happy_path"]
    operations = [
        step["operation"] for step in lifecycle["trace"] if step["step"].startswith("decision")
    ]
    # Post-W1-R2 the lifecycle chain grants ACCEPT, so nothing is owed at step 3.
    # The ASK_HUMAN branch is probe N, where the chain grants none.
    assert operations == ["CALL", "CHECK", "STOP", "STOP"]
    assert lifecycle["reopen_identical"] is True
    assert lifecycle["reprojected_status"] == "completed"
    assert lifecycle["reprojected_decision"] == "STOP"
    for kind in (
        "directive.opened", "task.created", "scheduler.decision_recorded", "action.requested",
        "action.authorized", "action.execution_started", "action.completed", "check.requested",
        "check.completed", "task.accepted", "task.completed",
    ):
        assert kind in lifecycle["kinds"], kind


@pytest.mark.parametrize(
    ("probe_id", "classification"),
    [
        ("A", "ENFORCED"),       # directive -> action authority
        ("B", "ENFORCED"),       # directive -> acceptance authority (gap 2, repaired by W1-R2)
        ("C", "ENFORCED"),       # decision -> execution      (gap 4, repaired by W1-R4)
        ("D", "DERIVED"),        # report -> effect state   (gap 1, repaired by W1-R1)
        ("E", "ENFORCED"),       # effect -> reconciliation
        ("F", "ENFORCED"),       # observed state -> binding
        ("G", "ENFORCED"),       # command -> verdict semantics
        ("H", "ENFORCED"),       # verification -> claim evidence (gap 5, repaired by W1-R5)
        ("I", "ENFORCED"),       # verification -> acceptance  (gap 3, repaired by W1-R3)
        ("J", "ENFORCED"),       # acceptance -> completion
        ("K", "ENFORCED"),       # replay -> current authority
        ("L", "DERIVED"),        # durable state -> decision
        ("M", "ENFORCED"),       # check -> artifact identity  (gap 3, repaired by W1-R3)
        ("N", "ABSENT"),         # human gate -> acceptance grant  (new, raised by W1-R2)
    ],
)
def test_the_frozen_classification_of_each_joint(probes, probe_id, classification):
    assert probes[probe_id]["classification"] == classification


# ---------------- the four gaps, as reproducers ----------------


def test_gap_a_repaired_a_caller_can_no_longer_supply_acceptance_authority(probes):
    """Audit gap 2, repaired by W1-R2 (experiments/W1-R2-authority-symmetry.md).

    Baseline at f0c730b: the same caller claim completed the task.
    """
    observed = " | ".join(probes["B"]["observed"])
    assert "allows accept: False" in observed
    assert "rejected: acceptance_not_granted:d-root" in observed
    assert "task.accepted events: 0; durable refusals: 1" in observed
    assert "refusal names the directive it resolved: d-root" in observed
    assert "what the caller claimed, recorded as a claim: ['accept']" in observed
    assert probes["B"]["classification"] == "ENFORCED"


def test_the_human_gate_now_names_a_grant_nobody_can_add(probes):
    """Raised by W1-R2, not repaired: recorded as a finding for the author."""
    observed = " | ".join(probes["N"]["observed"])
    assert "scheduler: ASK_HUMAN" in observed
    assert "a human answering that gate -> rejected: acceptance_not_granted" in observed
    assert "the task ends at: ASK_HUMAN" in observed


def test_gap_b_repaired_a_governed_operation_must_match_its_decision(probes):
    """Audit gap 4, repaired by W1-R4 (experiments/W1-R4-decision-execution-binding.md).

    Baseline at f0c730b: 'caller executed an action anyway -> succeeded'.
    """
    observed = " | ".join(probes["C"]["observed"])
    assert "recorded decision: CHECK" in observed
    assert "action claiming that CHECK decision -> failed, adapter calls 0" in observed
    assert "scheduler governance with no decision -> failed, adapter calls 0" in observed
    # The external path stays open, and says what it is.
    assert "action claiming nothing -> succeeded" in observed
    assert "recorded as ungoverned/external_request" in observed
    # Freshness: a decision the world has overtaken no longer permits anything.
    assert "check under that fresh CHECK decision -> PASS" in observed
    assert "the first decision, after the state moved -> ERROR" in observed
    assert probes["C"]["classification"] == "ENFORCED"


def test_gap_c_repaired_a_worker_report_alone_no_longer_observes_the_effect(probes):
    """Audit gap 1, repaired by W1-R1 (experiments/W1-R1-effect-observation.md).

    Baseline at f0c730b: 'projected effect state: observed' from the report alone.
    """
    observed = " | ".join(probes["D"]["observed"])
    assert "changed nothing: state hash moved = False" in observed
    assert "projected effect state: unknown" in observed
    assert "identical: True" in observed
    assert "effect_state trusts the report alone: False" in observed
    assert probes["D"]["classification"] == "DERIVED"


def test_gap_d_repaired_a_labelled_check_no_longer_stands_in_for_the_bytes(probes):
    """Audit gap 3, repaired by W1-R3 (experiments/W1-R3-artifact-binding.md).

    Baseline at f0c730b: 'acceptance citing that check -> completed'.
    """
    observed = " | ".join(probes["M"]["observed"])
    assert "it never opened the artifact" in observed
    assert "acceptance citing that check -> rejected" in observed
    assert "artifact binding: unconsumed" in observed
    assert "the check's own verdict: ERROR" in observed
    assert probes["M"]["classification"] == "ENFORCED"


# ---------------- the properties that did survive composition ----------------


def test_effect_unknown_never_becomes_retry_safe(probes):
    observed = " | ".join(probes["E"]["observed"])
    assert "effect state unknown; next operation reconcile_effect" in observed
    assert "same-key retry -> failed; adapter invocations total 1" in observed
    assert "ASK_HUMAN (unresolved_effect_requires_reconciliation)" in observed
    assert "after reconciliation: effect observed, reported status still failed" in observed


def test_binding_failure_stays_error_through_composition(probes):
    observed = " | ".join(probes["F"]["observed"])
    assert "MISMATCH: verdict ERROR" in observed
    assert "UNAVAILABLE: verdict ERROR" in observed
    assert "binding failure never produced INCONCLUSIVE: True" in observed
    assert "verifier claimed 'whatever-I-like'" in observed


def test_four_verdicts_survive_the_claim_and_acceptance_paths(probes):
    verdicts = " | ".join(probes["G"]["observed"])
    assert "k-pass: PASS" in verdicts and "k-fail: FAIL" in verdicts
    assert "k-maybe: INCONCLUSIVE" in verdicts and "k-unmapped: ERROR" in verdicts

    claims = " | ".join(probes["H"]["observed"])
    assert "PASS -> E3_REPRODUCED; FAIL -> refuted" in claims
    assert "INCONCLUSIVE -> status asserted, durable attempts 1" in claims
    assert "ERROR" in claims and "-> status asserted" in claims

    acceptance = " | ".join(probes["I"]["observed"])
    assert "check_not_passed:k-maybe:INCONCLUSIVE" in acceptance
    assert "check_not_passed:k-err:ERROR" in acceptance
    assert "check_wrong_artifact:k-other" in acceptance


def test_gap_e_repaired_an_errored_verification_is_discoverable_from_the_claim(probes):
    """Audit gap 5, repaired by W1-R5 (experiments/W1-R5-claim-attempt-trace.md).

    Baseline at f0c730b: the claim-side index was empty and the attempt could be
    found only by scanning every check.requested for the claim id.
    """
    observed = " | ".join(probes["H"]["observed"])
    assert "attempts discoverable from the errored claim: [('k-error', 'ERROR', False)]" in observed
    assert "'c-maybe': [('k-maybe', 'INCONCLUSIVE')]" in observed
    # Visible as an attempt, and still moving nothing.
    assert "'c-error': [False]" in observed and "'c-maybe': [False]" in observed
    assert "'c-pass': [True]" in observed and "'c-fail': [True]" in observed
    # A reference, not a second opinion.
    assert "['check_completed_event_id', 'check_id', 'claim_id', 'version']" in observed
    assert probes["H"]["classification"] == "ENFORCED"
    # ENFORCED is scoped: linkage and admissibility, never the truth of evidence.
    assert "linkage and its admissibility rules are ENFORCED" in observed
    assert "evidentiary state remains DERIVED" in observed


def test_replay_is_decided_against_the_record_as_it_stands(probes):
    observed = " | ".join(probes["K"]["observed"])
    assert "same-key replay under the same grant: succeeded, adapter calls 1" in observed
    assert "child directive id is already registered" in observed
    assert "asking for a capability the child lacks: denied" in observed
