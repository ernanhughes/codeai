"""The book's teaching examples must run, and must say what the chapter says.

Each example is small, executable and asserted here, so a chapter can print a
reduction of it without the reduction drifting away from working code.
"""

import importlib.util
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "applied_ai"


def load(name):
    path = EXAMPLES / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_ch22_action_recovery_example(capsys):
    summary = load("ch22_action_recovery.py").main()
    capsys.readouterr()

    assert summary["reported_status"] == "failed"
    assert summary["effect_before_reconciliation"] == "unknown"
    assert summary["retry_unsafe"] is True
    assert summary["lines_written"] == 1
    assert summary["adapter_calls"] == 1
    # Later evidence settles the effect without rewriting the reported status.
    assert summary["effect_after_reconciliation"] == "observed"
    assert summary["result_status_after_reconciliation"] == "failed"


def test_ch19_effect_observation_example(capsys):
    summary = load("ch19_effect_observation.py").main()
    capsys.readouterr()

    # Every worker said the same thing.
    assert summary["reports"] == ["succeeded"]
    # The record said four different things.
    assert summary["diligent"]["effect"] == "observed"
    assert summary["busy"]["effect"] == "observed"
    assert summary["idle"]["effect"] == "unknown"
    assert summary["unwatched"]["effect"] == "reported"
    # A changed scope is not the intended change: same effect state, opposite verdicts.
    assert summary["diligent"]["verdict"] == "PASS"
    assert summary["busy"]["verdict"] == "FAIL"
    # A success the readings contradict becomes a person's problem, not a retry.
    assert summary["idle"]["next"] == "reconcile_effect"
    assert summary["idle"]["duplicate_effect_risk"] is True
    assert summary["open_effects"] == ["a-idle"]


def test_ch21_verification_example(capsys):
    summary = load("ch21_verification.py").main()
    capsys.readouterr()

    assert summary["verdicts"] == ["PASS", "FAIL", "INCONCLUSIVE", "ERROR"]
    assert summary["bindings"] == ["bound", "bound", "bound", "mismatch"]
    # The inconclusive check ran and said why; the errored one measured nothing.
    assert "declared inconclusive by exit-code-v1" in summary["inconclusive_reason"]
    assert "cannot read it as prose" in summary["inconclusive_reason"]
    assert summary["errored_check_ran_nothing"] is True
    assert summary["error_reason"].startswith("target state mismatch")
    # The mapping was declared before the command ran, and recorded with it.
    assert summary["declared_policy"]["inconclusive_codes"] == [2]
    assert summary["verifier_identity"] == "LocalCommandVerifier"


def test_ch28_scheduler_example(capsys):
    summary = load("ch28_scheduler.py").main()
    capsys.readouterr()

    # Two tasks, identical but for one recorded capability, diverge at step 3.
    assert summary["gated"] == ["CALL", "CHECK", "ASK_HUMAN", "ASK_HUMAN"]
    assert summary["granted"] == ["CALL", "CHECK", "STOP", "STOP"]
    # And the acceptance is decided the same way the scheduler read it.
    assert summary["acceptance"]["t-gated"].startswith("refused (acceptance_not_granted")
    assert summary["acceptance"]["t-full"] == "completed"
    assert summary["final_state"]["process_complete"] is True
    assert summary["final_state"]["acceptance_authority_available"] is True
    # Decision 2 is history. The world moved on; the record did not.
    assert summary["recorded_second"] == "CHECK"
    assert summary["reprojected_now"] == "STOP"
    assert summary["replayed_second"] == "CHECK"
    assert summary["second_state_check_satisfied"] is False
    assert summary["basis_events"] >= 1


def test_ch20_authority_example(capsys):
    summary = load("ch20_authority.py").main()
    capsys.readouterr()

    assert summary["chain"] == ["review-child", "review-root"]
    assert summary["effective"] == ["write"]
    assert summary["write"] == "succeeded"
    assert summary["destructive"] == "denied"
    # The caller claimed DESTRUCTIVE; the record did not grant it.
    assert summary["forged"] == "denied"
    assert summary["worker_calls"] == 1
    assert summary["grant_source"] == "recorded_directive"
    assert summary["basis_events"] == 2
