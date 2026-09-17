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

    # Each decision came from facts the ledger established, in this order.
    assert summary["decisions"] == ["CALL", "CHECK", "ASK_HUMAN", "STOP"]
    # The task stopped because an acceptance completed it, not because a caller
    # said so: the directive never granted ACCEPT.
    assert summary["final_state"]["process_complete"] is True
    assert summary["final_state"]["acceptance_authority_available"] is False
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
