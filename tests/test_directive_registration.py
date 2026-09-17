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
    # No child is registered, and the attempt is now durable: a later reader can
    # tell "nobody tried this" from "someone tried this and was refused".
    after = ledger.read_all()
    assert [e.kind for e in after[len(before):]] == [
        'directive.registration_requested',
        'directive.registration_refused',
    ]
    refusal = after[-1]
    assert refusal.payload['directive_id'] == 'child'
    assert refusal.payload['parent_directive_id'] == 'parent'
    assert message.split()[0] in refusal.payload['reason']
    assert refusal.payload['basis_event_ids']
    assert not [e for e in after if e.kind == 'directive.opened' and e.stream_id == 'child']
    ledger._conn.close()


@pytest.mark.parametrize('mode', ['missing', 'ambiguous', 'self', 'duplicate-child'])
def test_invalid_parent_or_child_identity_is_refused_durably(mode):
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
    after = ledger.read_all()
    assert [e.kind for e in after[len(before):]] == [
        'directive.registration_requested',
        'directive.registration_refused',
    ]
    assert after[-1].payload['reason']
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
    recorded = ledger.read_all()
    assert [e.kind for e in recorded] == [
        'directive.registration_requested',
        'directive.opened',
        'directive.registration_requested',
        'directive.registration_refused',
    ]
    assert recorded[1] == parent_event
    # The refusal names the parent it actually read, not the one the caller held.
    assert recorded[-1].payload['parent_capabilities'] == ['read']
    assert runtime.show_run('child') is None
    ledger._conn.close()
    reopened = SQLiteLedger(path)
    reread = reopened.read_all()
    assert [e.kind for e in reread] == [
        'directive.registration_requested',
        'directive.opened',
        'directive.registration_requested',
        'directive.registration_refused',
    ]
    assert reread[1] == parent_event
    assert Runtime(reopened).show_run('child') is None
    reopened._conn.close()
