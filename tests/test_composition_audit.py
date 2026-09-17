"""The Wave 1 composition audit, pinned.

These assertions record what the audit *observed* at 11fedcd, including the gaps.
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
    assert operations == ["CALL", "CHECK", "ASK_HUMAN", "STOP"]
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
        ("B", "RECORDED"),       # directive -> acceptance authority   (gap)
        ("C", "CONVENTIONAL"),   # decision -> execution               (gap)
        ("D", "RECORDED"),       # report -> effect state              (gap)
        ("E", "ENFORCED"),       # effect -> reconciliation
        ("F", "ENFORCED"),       # observed state -> binding
        ("G", "ENFORCED"),       # command -> verdict semantics
        ("H", "DERIVED"),        # verification -> claim evidence      (gap: claim-side trace)
        ("I", "RECORDED"),       # verification -> acceptance
        ("J", "ENFORCED"),       # acceptance -> completion
        ("K", "ENFORCED"),       # replay -> current authority
        ("L", "DERIVED"),        # durable state -> decision
        ("M", "CONVENTIONAL"),   # check -> artifact identity          (gap)
    ],
)
def test_the_frozen_classification_of_each_joint(probes, probe_id, classification):
    assert probes[probe_id]["classification"] == classification


# ---------------- the four gaps, as reproducers ----------------


def test_gap_a_caller_supplied_acceptance_authority_completes_the_task(probes):
    """A record that grants no ACCEPT does not stop an acceptance that claims it."""
    observed = " | ".join(probes["B"]["observed"])
    assert "allows accept: False" in observed
    assert "-> completed" in observed
    assert "task.accepted payload names a directive: False" in observed


def test_gap_b_the_recorded_decision_does_not_gate_the_effect(probes):
    observed = " | ".join(probes["C"]["observed"])
    assert "recorded decision: CHECK" in observed
    assert "caller executed an action anyway -> succeeded" in observed
    assert "with no decision recorded at all -> succeeded" in observed
    assert "references a decision: False" in observed


def test_gap_c_a_worker_report_alone_makes_the_effect_observed(probes):
    """Chapter 19's question, answered by the worker rather than by the world."""
    observed = " | ".join(probes["D"]["observed"])
    assert "changed nothing: state hash moved = False" in observed
    assert "projected effect state: observed" in observed
    assert "identical: True" in observed
    assert "a check of the intended post-condition: FAIL" in observed


def test_gap_d_acceptance_trusts_the_artifact_label_on_the_check(probes):
    observed = " | ".join(probes["M"]["observed"])
    assert "it never opened the artifact" in observed
    assert "acceptance citing that check -> completed" in observed
    assert "the completed check's own binding: unbound" in observed


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


def test_an_errored_verification_leaves_no_trace_on_the_claim(probes):
    """Registered in advance as a distinct property; confirmed as a gap."""
    observed = " | ".join(probes["H"]["observed"])
    assert "'claim-scoped inconclusive index': 0" in observed
    assert "by scanning check.requested for the claim id: ['k-error']" in observed


def test_replay_is_decided_against_the_record_as_it_stands(probes):
    observed = " | ".join(probes["K"]["observed"])
    assert "same-key replay under the same grant: succeeded, adapter calls 1" in observed
    assert "child directive id is already registered" in observed
    assert "asking for a capability the child lacks: denied" in observed
