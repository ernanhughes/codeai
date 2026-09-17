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
