"""Read-only verification of Stage 12 bundle hashes and causal links."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    inventory = json.loads((root / 'hashes.json').read_text(encoding='utf-8'))
    for name, expected in inventory.items():
        path = (root / name).resolve()
        assert path.is_relative_to(root.resolve()), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    report_path = root / 'conformance-report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding='utf-8'))
        assert report['network_calls'] == 0
        assert report['passed'] == len(report['rows'])
        assert len({row['protocol'] for row in report['rows']}) == 3
        for row in report['rows']:
            assert row['manifest']['provider'] == 'opencode'
            assert row['manifest']['call_id'] == row['call_id']
            assert row['interpretation']['attempt_id'] == row['attempt_id']
            assert row['observation']['attempt_id'] == row['attempt_id']
            assert row['task_status'] == 'not automatically completed'
            fixture = row.get('source_fixture')
            if fixture:
                assert row['observation']['response_body_artifact']['sha256'] == fixture['body_sha256']
    for events_path in root.rglob('events.json'):
        events = json.loads(events_path.read_text(encoding='utf-8'))
        kinds = [e['kind'] for e in events]
        for a, b in zip(('call.manifest', 'attempt.started', 'attempt.observed',
                         'attempt.interpreted', 'call.status_decided'),
                        ('attempt.started', 'attempt.observed', 'attempt.interpreted',
                         'call.status_decided', 'call.completed')):
            assert kinds.index(a) < kinds.index(b)
        interpretations = {e['payload']['interpretation_id'] for e in events
                           if e['kind'] == 'attempt.interpreted'}
        assert interpretations
        for event in events:
            if event['kind'] == 'attempt.retry_decided':
                assert event['payload']['interpretation_id'] in interpretations
            if event['kind'] == 'call.status_decided':
                assert set(event['payload']['interpretation_ids']) <= interpretations
        assert 'task.completed' not in kinds
    print(f'PASS: {len(inventory)} file hashes and recorded event chains in {root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path)
    verify(parser.parse_args().bundle)
