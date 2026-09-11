import pytest

from codeai.context import ContextCompiler, ContextSealViolation
from codeai.domain import ActorRef, Seal
from codeai.ledger import Event


def test_context_package_is_deterministic():
    event = Event.create(stream_id="d1", kind="claim.created", actor_id="m1", payload={"call_id": "c1"})
    actor = ActorRef(actor_id="m2", kind="model", provider="example", model="x", version="1")
    compiler = ContextCompiler()

    a = compiler.compile(task_id="t1", actor=actor, prompt="solve", events=[event])
    b = compiler.compile(task_id="t1", actor=actor, prompt="solve", events=[event])

    assert a.package_id == b.package_id


def test_seal_blocks_sibling_call_output():
    sibling = Event.create(
        stream_id="d1",
        kind="call.completed",
        actor_id="m1",
        payload={"call_id": "call-a"},
    )
    actor = ActorRef(actor_id="m2", kind="model")

    with pytest.raises(ContextSealViolation):
        ContextCompiler().compile(
            task_id="t1",
            actor=actor,
            prompt="independent answer",
            events=[sibling],
            seal=Seal(forbidden_call_ids=frozenset({"call-a"})),
        )
