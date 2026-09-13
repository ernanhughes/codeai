"""Stage 12: separate route captures from offline boundary conformance.

No output directory is reused. Response bytes are preserved; outbound hashes
identify canonical semantic objects, not serialized HTTP bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from codeai.artifacts import FileArtifactStore
from codeai.context import ContextCompiler
from codeai.domain import ActorRef, Authority, Budget, CallSpec, Task
from codeai.experiments import ExperimentBudget, ExperimentUsage, _budget_allows, build_config
from codeai.ledger import Event, SQLiteLedger
from codeai.providers import HttpResponse, OpenCodeCognitionAdapter, TransportFailure
from codeai.runtime import Runtime

PROMPT = ('Review this code comment for an unsupported performance claim. '
          'Reply with one short sentence identifying the missing evidence. '
          'Comment: This function always completes in under one millisecond.')
ROUTES = {'responses': 'gpt-5.6-luna', 'chat_completions': 'mimo-v2.5',
          'messages': 'minimax-m2.7'}
SOURCE = 'https://opencode.ai/docs/go/'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n',
                    encoding='utf-8')


def make_runtime(path):
    path.mkdir(parents=True)
    ledger = SQLiteLedger(path / 'ledger.sqlite')
    return Runtime(ledger, artifact_store=FileArtifactStore(path / 'artifacts', ledger))


def invoke(runtime, protocol, model, post=None, controls=None):
    task_id, call_id = str(uuid.uuid4()), str(uuid.uuid4())
    actor = ActorRef('reviewer', 'model', provider='opencode', model=model)
    runtime.create_task(Task(task_id, 'protocol-conformance', PROMPT,
                             ('Record route outcome; no quality ranking',), Budget(), Authority()))
    context = ContextCompiler().compile(task_id=task_id, actor=actor, prompt=PROMPT,
                                        prompt_version='protocol-review-v1')
    spec = CallSpec(call_id, task_id, actor, context, str(uuid.uuid4()),
                    chamber='deep-review', parameters=controls or {'max_tokens': 1024})
    adapter = OpenCodeCognitionAdapter(model=model, protocol=protocol, gateway_plan='go',
                                       api_key='offline-decoy' if post else None,
                                       http_post=post, timeout=60)
    call = runtime.invoke_recorded_call(spec, adapter=adapter, max_attempts=1)
    attempt = call.attempts[0]
    observation = runtime.get_attempt_observation(attempt.attempt_id)
    interpretation = runtime.interpretations_for_attempt(attempt.attempt_id)[0]
    envelope = json.loads(runtime.artifact_store.read_bytes(attempt.raw_artifact.artifact_id))
    row = {'protocol': protocol, 'gateway': 'opencode', 'gateway_plan': 'go', 'model': model,
           'task_id': task_id, 'call_id': call_id, 'attempt_id': attempt.attempt_id,
           'manifest': asdict(call.manifest), 'observation': observation,
           'interpretation': asdict(interpretation), 'usage': asdict(attempt.usage),
           'call_status': call.status, 'task_status': 'not automatically completed',
           'canonical_text': envelope.get('raw_output', envelope.get('output_text', ''))}
    # Read the compatible result view, without branching on protocol.
    completed = [e.payload for e in runtime.ledger.events_by_kind(('call.completed',))][-1]
    row['canonical_text'] = completed.get('raw_output', row['canonical_text'])
    write(runtime.artifact_store.base_dir.parent / 'result.json', row)
    write(runtime.artifact_store.base_dir.parent / 'events.json', [asdict(e) for e in runtime.ledger.read_all()])
    return row


def live(output):
    reason = ('User-authorized Stage 12 route check: at most three single attempts, '
              '1024 output tokens each, 60 seconds each; Go subscription per-call cost '
              'is unknown. No quality comparison, automatic retry or overage setting change.')
    config = build_config(name='Stage 12 route execution', hypothesis='Configured routes execute',
                          primary_metric='route outcome', task_ids=(), arms=(),
                          budget=ExperimentBudget(max_calls=3, max_cost_usd=.5,
                             allow_unknown_cost=True, unknown_cost_reason=reason),
                          stopping_rule='One attempt per route; at most three effects')
    write(output / 'budget.json', asdict(config))
    rows = []
    for protocol, model in ROUTES.items():
        if not os.getenv('OPENCODE_ZEN_API_KEY'):
            rows.append({'protocol': protocol, 'status': 'blocked_missing_credentials'})
            continue
        # Cost is prospectively unknown: require the guard's recorded override even on first call.
        allowed, message, state = _budget_allows(config, ExperimentUsage(
            calls=len(rows), cost_complete=False, unknown_cost_calls=max(1, len(rows))))
        if not allowed:
            rows.append({'protocol': protocol, 'status': 'blocked_budget', 'reason': message})
            break
        runtime = make_runtime(output / protocol)
        runtime.ledger.append(Event.create(stream_id=config.experiment_id, kind='budget.override',
            actor_id='runtime', payload={'reason': reason, 'state': state, 'config': asdict(config)}))
        row = invoke(runtime, protocol, model)
        row['evidence_class'] = 'live_transport_observation'
        row['route_source'] = SOURCE
        obs = row['observation'] or {}
        ref = obs.get('response_body_artifact')
        if ref:
            body = runtime.artifact_store.read_bytes(ref['artifact_id'])
            (output / protocol / 'response.bin').write_bytes(body)
            write(output / protocol / 'fixture.json', {
                'protocol': protocol, 'model': model, 'evidence_class': 'captured_transport_bytes',
                'source_call_id': row['call_id'], 'source_attempt_id': row['attempt_id'],
                'source_observation': obs, 'body_sha256': hashlib.sha256(body).hexdigest(),
                'body_file': 'response.bin', 'route_source': SOURCE,
                'expected_text': row['canonical_text'],
                'expected_generation_state': row['interpretation']['generation_state'],
                'expectation_basis': 'captured adapter view; independently inspect against raw body'})
        rows.append(row)
        print(protocol, obs.get('http_status'), row['interpretation']['generation_state'], flush=True)
    write(output / 'route-execution-report.json', rows)


def synthetic(protocol, variant):
    text = '' if variant in ('empty', 'tool_only') else 'Measurement is missing.'
    reason = {'responses': 'completed', 'chat_completions': 'stop', 'messages': 'end_turn'}[protocol]
    if variant == 'truncated':
        reason = {'responses': 'max_output_tokens', 'chat_completions': 'length',
                  'messages': 'max_tokens'}[protocol]
    if variant == 'unknown_reason':
        reason = 'unrecognized_reason'
    if protocol == 'responses':
        payload = {'status': reason, 'output': [{'type': 'message', 'content': [
            {'type': 'output_text', 'text': text}]}]}
        if variant == 'truncated':
            payload.update(status='incomplete', incomplete_details={'reason': reason})
        if variant == 'tool_only':
            payload['output'] = [{'type': 'function_call', 'name': 'not_executed'}]
    elif protocol == 'chat_completions':
        payload = {'choices': [{'finish_reason': reason, 'message': {'content': text}}]}
        if variant == 'tool_only':
            payload['choices'][0]['message'] = {'content': '', 'tool_calls': [{'name': 'not_executed'}]}
    else:
        payload = {'stop_reason': reason, 'content': [{'type': 'text', 'text': text}]}
        if variant == 'tool_only':
            payload['content'] = [{'type': 'tool_use', 'name': 'not_executed'}]
    if variant == 'http_error':
        return HttpResponse(400, {}, b'{"error":"invalid request"}', 'application/json')
    if variant == 'malformed':
        return HttpResponse(200, {}, b'{broken', 'application/json')
    return HttpResponse(200, {'request-id': 'synthetic'}, json.dumps(payload).encode(), 'application/json')


def offline(output, captures):
    rows = []
    with patch.object(socket.socket, 'connect', side_effect=AssertionError('Offline network denied')):
        for protocol, model in ROUTES.items():
            cases = [(v, synthetic(protocol, v), 'synthetic', None) for v in
                     ('complete', 'truncated', 'unknown_reason', 'empty', 'tool_only',
                      'http_error', 'malformed', 'no_response')]
            if captures:
                source = captures / protocol / 'fixture.json'
                if source.exists():
                    fixture = json.loads(source.read_text())
                    body = (source.parent / fixture['body_file']).read_bytes()
                    assert hashlib.sha256(body).hexdigest() == fixture['body_sha256']
                    obs = fixture['source_observation']
                    cases.append(('captured', HttpResponse(obs['http_status'],
                        obs.get('response_headers', {}), body, obs.get('content_type')),
                        'captured_transport_bytes', fixture))
            for name, reply, evidence_class, fixture in cases:
                sent = []

                def post(url, body, headers, timeout, sent=sent, name=name, reply=reply):
                    sent.append(body)
                    if name == 'no_response':
                        raise TransportFailure('TimeoutError', 'test timeout')
                    return reply

                runtime = make_runtime(output / protocol / name)
                controls = {'max_tokens': 256, 'temperature': .2, 'reasoning_effort': 'low'}
                row = invoke(runtime, protocol, model, post, controls)
                row.update(case=name, evidence_class=evidence_class, source_fixture=fixture)
                assert len(sent) == 1
                canonical_hash = hashlib.sha256(json.dumps(sent[0], sort_keys=True,
                                              separators=(',', ':')).encode()).hexdigest()
                assert row['manifest']['request_body_sha256'] == canonical_hash
                effective = row['manifest']['effective_parameters']
                for key in ('temperature', 'max_tokens', 'max_output_tokens', 'reasoning'):
                    if key in effective:
                        assert sent[0][key] == effective[key]
                state = row['interpretation']['generation_state']
                if name == 'complete':
                    assert state == 'complete' and row['canonical_text'] == 'Measurement is missing.'
                elif name == 'truncated':
                    assert state == 'truncated' and row['call_status'] != 'succeeded'
                elif name == 'unknown_reason':
                    assert state == 'unknown'
                elif name in ('empty', 'tool_only'):
                    assert state != 'complete' and not row['canonical_text']
                elif name in ('http_error', 'malformed', 'no_response'):
                    assert row['call_status'] != 'succeeded'
                elif fixture:
                    assert row['canonical_text'] == fixture['expected_text']
                    assert state == fixture['expected_generation_state']
                if name != 'no_response':
                    ref = row['observation']['response_body_artifact']
                    assert runtime.artifact_store.read_bytes(ref['artifact_id']) == reply.body
                rows.append(row)
    write(output / 'conformance-report.json', {'network_calls': 0, 'passed': len(rows),
          'captured_cases': sum(r['evidence_class'] == 'captured_transport_bytes' for r in rows),
          'quality_comparison': False, 'rows': rows})


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--offline', action='store_true')
    mode.add_argument('--live', action='store_true')
    parser.add_argument('--captures', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    write(args.output / 'manifest.json', {'question': 'Does the protocol boundary conform?',
        'mode': 'offline' if args.offline else 'live', 'command': sys.argv,
        'codeai_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
        'source_hashes': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [*sorted((root / 'src/codeai').glob('*.py')), Path(__file__)]},
        'prompt': PROMPT, 'route_source': SOURCE,
        'request_hash_semantics': 'canonical semantic JSON; not outbound HTTP bytes'})
    if args.offline:
        offline(args.output, args.captures)
    else:
        live(args.output)
    write(args.output / 'hashes.json', {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(args.output.rglob('*')) if p.is_file() and p.suffix != '.sqlite'})


if __name__ == '__main__':
    main()

