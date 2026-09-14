from dataclasses import replace

import pytest

from codeai.domain import Authority, Budget, Capability, Directive
from codeai.ledger import SQLiteLedger
from codeai.runtime import Runtime


def directive(name, capabilities, *, parent=None, tokens=100):
    return Directive(name, 'review', (), Budget(max_tokens=tokens),
                     Authority(frozenset(capabilities)), parent)


def test_child_narrowing_is_checked_after_reopen(tmp_path):
    path = tmp_path / 'ledger.sqlite'
    ledger = SQLiteLedger(path)
    parent = Runtime(ledger).open_directive(
        directive('parent', {Capability.READ, Capability.WRITE}))
    ledger._conn.close()
    ledger = SQLiteLedger(path)
    runtime = Runtime(ledger)
    child = runtime.open_directive(directive('child', {Capability.READ}, parent='parent', tokens=50))
    assert child.causation_id == parent.event_id
    assert child.payload['authority']['capabilities'] == ['read']
    assert child.payload['budget']['max_tokens'] == 50
    ledger._conn.close()


@pytest.mark.parametrize('caps,tokens,message', [
    ({Capability.READ, Capability.WRITE}, 50, 'authority must narrow'),
    ({Capability.READ}, 101, 'budget must narrow'),
    ({Capability.READ}, None, 'budget must narrow'),
])
def test_child_cannot_widen_recorded_parent(caps, tokens, message):
    ledger = SQLiteLedger()
    runtime = Runtime(ledger)
    runtime.open_directive(directive('parent', {Capability.READ}))
    before = ledger.read_all()
    with pytest.raises(ValueError, match=message):
        runtime.open_directive(directive('child', caps, parent='parent', tokens=tokens))
    assert ledger.read_all() == before
    ledger._conn.close()


@pytest.mark.parametrize('mode', ['missing', 'ambiguous', 'self', 'duplicate-child'])
def test_invalid_parent_or_child_identity_appends_nothing(mode):
    ledger = SQLiteLedger()
    runtime = Runtime(ledger)
    parent = directive('parent', {Capability.READ})
    child = directive('child', {Capability.READ}, parent='parent')
    if mode in {'ambiguous', 'duplicate-child'}:
        runtime.open_directive(parent)
        runtime.open_directive(parent if mode == 'ambiguous' else child)
    if mode == 'self':
        child = replace(child, parent_directive_id='child')
    before = ledger.read_all()
    with pytest.raises(ValueError):
        runtime.open_directive(child)
    assert ledger.read_all() == before
    ledger._conn.close()


def test_failed_widening_uses_durable_parent_and_survives_reopen(tmp_path):
    path = tmp_path / 'ledger.sqlite'
    ledger = SQLiteLedger(path)
    runtime = Runtime(ledger)
    parent = directive('parent', {Capability.READ})
    parent_event = runtime.open_directive(parent)
    # A caller can manufacture a wider parent object, but cannot pass it to
    # registration in place of the durable parent selected by the child's ID.
    alleged_parent = replace(parent, authority=Authority(frozenset({Capability.READ, Capability.WRITE})))
    child = directive('child', {Capability.WRITE}, parent='parent')
    alleged_parent.validate_child(child)
    with pytest.raises(ValueError, match='authority must narrow'):
        runtime.open_directive(child)
    assert ledger.read_all() == (parent_event,)
    assert runtime.show_run('child') is None
    ledger._conn.close()
    reopened = SQLiteLedger(path)
    assert reopened.read_all() == (parent_event,)
    assert Runtime(reopened).show_run('child') is None
    reopened._conn.close()
