import json

import pytest

from codeai.adapters import InvalidControlError, UnknownControlError
from codeai.artifacts import FileArtifactStore
from codeai.context import ContextCompiler
from codeai.domain import ActorRef, CallSpec
from codeai.ledger import SQLiteLedger
from codeai.modelconfig import ModelConfig, ModelMapping
from codeai.providers import HttpResponse, OpenCodeCognitionAdapter
from codeai.runtime import Runtime


def spec(parameters=None):
    actor = ActorRef('reviewer', 'model', provider='opencode', model='fixture')
    return CallSpec('call', 'task', actor,
                    ContextCompiler().compile(task_id='task', actor=actor, prompt='Review P'),
                    'unique', parameters=parameters or {})


def test_messages_semantics_and_config():
    adapter = ModelConfig({'review': ModelMapping('review', 'opencode', 'fixture',
                          protocol='messages', gateway_plan='go')}).build_adapter('review')
    prepared = adapter.prepare(spec({'max_tokens': 256, 'temperature': .2,
                                     'reasoning_effort': 'low'}))
    assert prepared.body['max_tokens'] == 256
    assert prepared.body['messages'][0]['content'] == 'Review P'
    assert prepared.effective_controls == {'max_tokens': 256, 'temperature': .2}
    assert prepared.omitted_unsupported == ('reasoning_effort',)
    assert prepared.recorded_effective()['gateway_plan'] == 'go'
    assert prepared.public_headers['anthropic-version'] == '2023-06-01'
    assert prepared.plan_version == 'opencode-messages-request-plan-v1'
    default = adapter.prepare(spec())
    assert default.defaulted_controls == {'max_tokens': 1024}
    with pytest.raises(InvalidControlError):
        adapter.prepare(spec({'max_tokens': 0}))
    with pytest.raises(UnknownControlError):
        adapter.prepare(spec({'future_control': True}))


@pytest.mark.parametrize('reason,text,state', [
    ('end_turn', 'Review.', 'complete'), ('max_tokens', 'Review.', 'truncated'),
    ('future_reason', 'Review.', 'unknown'), ('tool_use', '', 'empty'),
    ('end_turn', '', 'empty'),
])
def test_messages_recorded_boundary(tmp_path, reason, text, state):
    payload = {'content': [{'type': 'thinking', 'thinking': 'private'},
                           {'type': 'text', 'text': text},
                           {'type': 'tool_use', 'name': 'unexecuted'}],
               'stop_reason': reason, 'usage': {'input_tokens': 10, 'output_tokens': 5,
                                               'cache_read_input_tokens': 3}}
    body = json.dumps(payload).encode()
    requests = []

    def post(url, request, headers, timeout):
        requests.append(request)
        assert url.endswith('/v1/messages')
        assert headers['x-api-key'] == 'secret-decoy'
        assert headers['Authorization'] == 'Bearer secret-decoy'
        assert headers['anthropic-version'] == '2023-06-01'
        return HttpResponse(200, {'request-id': 'req'}, body, 'application/json')

    ledger = SQLiteLedger(tmp_path / 'ledger.sqlite')
    store = FileArtifactStore(tmp_path / 'artifacts', ledger)
    runtime = Runtime(ledger, artifact_store=store)
    adapter = OpenCodeCognitionAdapter(model='fixture', protocol='messages',
                                       api_key='secret-decoy', http_post=post)
    call = runtime.invoke_recorded_call(spec(), adapter=adapter)
    interpretation = runtime.interpretations_for_attempt(call.attempts[0].attempt_id)[0]
    assert interpretation.generation_state == state
    observation = runtime.get_attempt_observation(call.attempts[0].attempt_id)
    assert store.read_bytes(observation['response_body_artifact']['artifact_id']) == body
    assert 'secret-decoy' not in json.dumps([e.payload for e in ledger.read_all()])
    assert len(requests) == 1
    if state in ('truncated', 'empty'):
        assert call.status != 'succeeded'
