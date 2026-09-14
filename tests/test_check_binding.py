from hashlib import sha256

import pytest

from codeai.adapters import CheckRequest, CheckResult, CheckVerdict
from codeai.ledger import SQLiteLedger
from codeai.runtime import Runtime


class CountingVerifier:
    def __init__(self):
        self.calls = 0

    def run(self, request):
        self.calls += 1
        return CheckResult(request.check_id, CheckVerdict.PASS,
                           observed_target_state_hash='forged by verifier')


@pytest.mark.parametrize('mode', ['match', 'mismatch', 'missing', 'raises'])
def test_binding_records_one_reading_and_refuses_unestablished_state(tmp_path, mode):
    target = tmp_path / 'target.txt'
    target.write_bytes(b'actual target')
    actual = sha256(target.read_bytes()).hexdigest()
    reads = []

    def observe():
        reads.append('read')
        if mode == 'raises':
            raise OSError('unreadable fixture')
        return sha256(target.read_bytes()).hexdigest()

    ledger_path = tmp_path / 'ledger.sqlite'
    ledger = SQLiteLedger(ledger_path)
    runtime = Runtime(ledger, state_resolver=None if mode == 'missing' else observe)
    verifier = CountingVerifier()
    result = runtime.run_check(
        CheckRequest('check', 'task', claim_ids=('claim',),
                     target_state_hash='other' if mode == 'mismatch' else actual),
        verifier=verifier,
    )
    assert len(reads) == (0 if mode == 'missing' else 1)
    assert verifier.calls == (1 if mode == 'match' else 0)
    assert result.verdict == (CheckVerdict.PASS if mode == 'match' else CheckVerdict.ERROR)
    assert result.observed_target_state_hash == (actual if mode in {'match', 'mismatch'} else None)
    events = ledger.read_all()
    if mode != 'match':
        assert [e.kind for e in events] == ['check.requested', 'check.completed']
        assert result.error
    completed = next(iter(ledger.events_by_kind(('check.completed',))))
    assert completed.causation_id == events[0].event_id
    assert completed.payload['observed_target_state_hash'] == result.observed_target_state_hash
    ledger._conn.close()
    reopened = SQLiteLedger(ledger_path)
    assert reopened.read_all() == events
    reopened._conn.close()


def test_binding_does_not_reread_moving_state_to_decide_or_describe_mismatch():
    reads = []

    def observe():
        reads.append(len(reads))
        return 'A' if len(reads) == 1 else 'B'

    ledger = SQLiteLedger()
    verifier = CountingVerifier()
    result = Runtime(ledger, state_resolver=observe).run_check(
        CheckRequest('check', 'task', target_state_hash='B'), verifier=verifier)
    assert reads == [0]
    assert verifier.calls == 0
    assert result.observed_target_state_hash == 'A'
    assert result.error == 'target state mismatch: expected B, observed A'
    ledger._conn.close()


class RaisingVerifier:
    def __init__(self):
        self.calls = 0

    def run(self, request):
        self.calls += 1
        raise RuntimeError('boom')


def test_raising_verifier_becomes_durable_error_without_promotion(tmp_path):
    ledger_path = tmp_path / 'ledger.sqlite'
    ledger = SQLiteLedger(ledger_path)
    runtime = Runtime(ledger, state_resolver=lambda: 'state-A')
    verifier = RaisingVerifier()
    result = runtime.run_check(
        CheckRequest('check', 'task', claim_ids=('claim',), target_state_hash='state-A'),
        verifier=verifier,
    )
    assert verifier.calls == 1
    assert result.verdict == CheckVerdict.ERROR
    assert result.verdict != CheckVerdict.FAIL
    assert result.error == 'verifier raised RuntimeError: boom'
    assert result.observed_target_state_hash == 'state-A'
    events = ledger.read_all()
    assert [e.kind for e in events] == ['check.requested', 'check.completed']
    completed = next(iter(ledger.events_by_kind(('check.completed',))))
    assert completed.causation_id == events[0].event_id
    assert completed.payload['verdict'] == 'ERROR'
    assert completed.payload['error'] == 'verifier raised RuntimeError: boom'
    assert completed.payload['observed_target_state_hash'] == 'state-A'
    assert list(ledger.events_by_kind(('claim.evidence', 'claim.status'))) == []
    ledger._conn.close()
    reopened = SQLiteLedger(ledger_path)
    assert reopened.read_all() == events
    reopened._conn.close()


def test_unbound_raising_verifier_records_error_without_observation():
    ledger = SQLiteLedger()
    verifier = RaisingVerifier()
    result = Runtime(ledger).run_check(
        CheckRequest('check', 'task', claim_ids=('claim',)), verifier=verifier)
    assert verifier.calls == 1
    assert result.verdict == CheckVerdict.ERROR
    assert result.error == 'verifier raised RuntimeError: boom'
    assert result.observed_target_state_hash is None
    assert list(ledger.events_by_kind(('claim.evidence', 'claim.status'))) == []
    ledger._conn.close()


def test_unbound_check_remains_supported_without_inventing_observation():
    def observe():
        raise AssertionError('unbound check should not request a state reading')

    ledger = SQLiteLedger()
    verifier = CountingVerifier()
    result = Runtime(ledger, state_resolver=observe).run_check(
        CheckRequest('check', 'task'), verifier=verifier)
    assert result.verdict == CheckVerdict.PASS
    assert verifier.calls == 1
    assert result.observed_target_state_hash is None
    ledger._conn.close()
