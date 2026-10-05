#!/usr/bin/env python3
"""Repeat the September 20 three-code retrieval screens with serial decoding."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8731')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    def call(path, payload=None):
        req = urllib.request.Request(args.base_url + path, data=None if payload is None else json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=1200) as response:
            return json.load(response)

    health = call('/health')
    if health['prompt_cache']['mode'] != 0 or health['prompt_lookup'] != 'off' or health['context'] != 262144:
        parser.error('requires a 262144-context diagnostic profile with cache and prompt lookup off')
    record = {'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'health': health, 'method': 'Same deterministic documents and serial, greedy, thinking-off request policy as September 20. One run per length; no general accuracy qualification.', 'checks': []}
    for count in [7910, 959]:
        indices = [count // 10, count // 2, 9 * count // 10]
        lines = [f'Archive record {i:06d}: owner=team-{i%97:02d}; state=closed; retrieval_code=delta-{(i*7919+104729)%100000:05d}.' for i in range(count)]
        messages = [{'role': 'system', 'content': 'Retrieve the requested records exactly. Reply with only the three retrieval codes in order, separated by commas and no spaces.'}, {'role': 'user', 'content': '\n'.join(lines) + f'\nReturn the retrieval_code values for archive records {indices[0]:06d}, {indices[1]:06d}, and {indices[2]:06d}, in that order.'}]
        expected = ','.join(f'delta-{(i*7919+104729)%100000:05d}' for i in indices)
        options = {'model': health['model'], 'temperature': 0, 'seed': 1, 'reasoning_effort': 'none', 'max_tokens': 1024, 'stream': False, 'drafter': 'serial'}
        print(f'Starting {count} records', flush=True)
        start = time.monotonic()
        response = call('/v1/chat/completions', {**options, 'messages': messages})
        output = response['choices'][0]['message']['content'].strip()
        record['checks'].append({'records': count, 'messages_sha256': hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(), 'expected': expected, 'output': output, 'passed': output == expected, 'correct_codes': sum(a == b for a, b in zip(expected.split(','), output.split(','))), 'wall_seconds': time.monotonic() - start, 'request_options': options, 'response': response})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + '\n')
        assert response['timings']['draft_n'] == 0 and response['timings']['cache_n'] == 0, response
        print(f'{count}: {output}; {record["checks"][-1]["correct_codes"]}/3', flush=True)
    return int(any(not c['passed'] for c in record['checks']))


if __name__ == '__main__':
    raise SystemExit(main())
