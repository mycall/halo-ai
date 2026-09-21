#!/usr/bin/env python3
"""Check live Halogen MTP against serial greedy decoding on the fixed suite."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib/halo_ai'))
import rocmfpx_quality


def request(base_url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base_url + path, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8731')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    health = request(args.base_url, '/health')
    if not (health.get('engine', {}).get('responds') and health.get('drafter_weights_loaded')):
        parser.error('a healthy engine with loaded MTP weights is required')
    if health.get('drafter_default') != 'mtp' or health.get('prompt_cache', {}).get('mode') != 0:
        parser.error('start the 32k-mtp profile: MTP default and cache mode 0 are required')
    if health.get('prompt_lookup') != 'off':
        parser.error('disable prompt lookup to isolate the MTP head')
    suite, digest = rocmfpx_quality.load_suite(ROOT / 'config/benchmarks/rocmfpx-quality-v1.json')
    document = {
        'schema_version': 1, 'kind': 'halogen-mtp-integration-check',
        'recorded_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'health': health, 'suite_sha256': digest,
        'method': 'Sequential serial versus MTP requests in one process; temperature 0, seed 1, thinking off, cache off, prompt lookup off. Equality compares returned content bytes, not hidden engine token IDs. This is a bounded integration check, not general qualification.',
        'cases': [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(document, indent=2) + '\n')

    def generate(messages, limit, drafter):
        payload = {'model': health['model'], 'messages': messages, 'max_tokens': limit,
                   'temperature': 0, 'seed': 1, 'reasoning_effort': 'none', 'stream': False}
        if drafter is not None:
            payload['drafter'] = drafter
        start = time.monotonic()
        response = request(args.base_url, '/v1/chat/completions', payload)
        content = response['choices'][0]['message']['content']
        return {'output': content, 'sha256': hashlib.sha256(content.encode()).hexdigest(),
                'wall_seconds': round(time.monotonic() - start, 6),
                'finish_reason': response['choices'][0]['finish_reason'],
                'usage': response.get('usage'), 'timings': response.get('timings', {})}

    # Omit drafter on the candidate request to check the integration's default.
    probe = [{'role': 'user', 'content': 'Write a Python LinkedList class with push, pop, reverse, __len__, and __iter__. Include docstrings and type hints.'}]
    document['generation_probe'] = {}
    for name, drafter in [('serial', 'serial'), ('default_mtp', None)]:
        document['generation_probe'][name] = generate(probe, 512, drafter)
        save()
        print(f'probe {name}: {document["generation_probe"][name]["timings"]}', flush=True)
    for index, case in enumerate(suite['cases']):
        result = {'case_id': case['id'], 'expected': case['validator']['expected']}
        # Alternate order so the MTP arm does not always inherit warm disk pages.
        order = ['serial', 'mtp'] if index % 2 == 0 else ['mtp', 'serial']
        for drafter in order:
            result[drafter] = generate(rocmfpx_quality.case_messages(suite, case), case['max_tokens'], drafter)
            result[drafter]['passed'], _ = rocmfpx_quality.score_case(case, result[drafter]['output'])
        result['identical_text'] = result['serial']['output'] == result['mtp']['output']
        document['cases'].append(result)
        save()
        print(f'{index+1}/{len(suite["cases"])} {case["id"]}: identical={result["identical_text"]} serial={result["serial"]["passed"]} mtp={result["mtp"]["passed"]}', flush=True)
    totals = {}
    for arm in ['serial', 'mtp']:
        entries = [r[arm] for r in document['cases']]
        entries.append(document['generation_probe']['serial' if arm == 'serial' else 'default_mtp'])
        totals[arm] = {key: sum(e['timings'].get(key, 0) for e in entries) for key in ['draft_n', 'draft_n_accepted']}
    probe_equal = document['generation_probe']['serial']['output'] == document['generation_probe']['default_mtp']['output']
    document['summary'] = {
        'cases': len(document['cases']),
        'serial_passed': sum(c['serial']['passed'] for c in document['cases']),
        'mtp_passed': sum(c['mtp']['passed'] for c in document['cases']),
        'identical_cases': sum(c['identical_text'] for c in document['cases']),
        'probe_identical_text': probe_equal, 'speculative_totals': totals,
        'mtp_working': totals['mtp']['draft_n'] >= totals['mtp']['draft_n_accepted'] > 0 and totals['serial']['draft_n'] == 0,
    }
    save()
    print(json.dumps(document['summary'], indent=2))
    return 0 if document['summary']['mtp_working'] and probe_equal and all(c['identical_text'] for c in document['cases']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
