#!/usr/bin/env python3
"""Measure installed Halogen serving: short/long code prompts and cache reuse.

Makes HTTP requests only; does not start services or acquire any artifacts.
Run against otherwise idle, freshly started runtimes for an upgrade comparison.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8731')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--records', type=int, nargs='+', default=[0, 959])
    parser.add_argument('--max-tokens', type=int, default=512)
    args = parser.parse_args()
    if not 1 <= args.rounds <= 10 or not 128 <= args.max_tokens <= 8192:
        parser.error('rounds must be 1..10 and max-tokens 128..8192')
    if any(not 0 <= n <= 7900 for n in args.records) or len(set(args.records)) != len(args.records):
        parser.error('records must be distinct counts in 0..7900')

    def call(path, payload=None):
        req = urllib.request.Request(args.base_url + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=1200) as response:
            return json.load(response)

    health = call('/health')
    if health.get('prompt_cache', {}).get('mode') != 2 or health.get('drafter_default') != 'mtp':
        parser.error('requires a serving profile with cache mode 2 and MTP')
    record = {
        'recorded_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'health_before': health, 'completed': False, 'runs': [],
        'method': 'Sequential requests; NPU loaded if selected but idle. Each distinct prompt is requested twice, recording actual cached tokens. First use need not be fully cold: shared prefixes may hit. Synthetic archive plus code generation; greedy and sampled with thinking off to isolate decoding under a fixed output cap. Not a task-quality score or a test of the full serving reasoning policy. Sampled text is not expected to match across releases. Runtime order and disk page cache are not controlled by this script.',
        'options': {'rounds': args.rounds, 'records': args.records, 'max_tokens': args.max_tokens},
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + '\n')

    save()
    for count in args.records:
        archive = '\n'.join(f'Archive record {i:06d}: owner=team-{i%97:02d}; state=closed; retrieval_code=delta-{(i*7919+104729)%100000:05d}.' for i in range(count))
        for iteration in range(args.rounds):
            modes = ['greedy', 'sampled'] if iteration % 2 == 0 else ['sampled', 'greedy']
            for mode in modes:
                messages = [{'role': 'user', 'content': f'Decode benchmark v1 case {count}/{iteration}/{mode}.\n' + archive + '\nWrite a complete Python archive index class with parsing, lookup by record number, lookup by owner, and update operations. Include type hints, docstrings, validation, and usage examples.'}]
                options = {'model': health['model'], 'seed': iteration + 1,
                    'temperature': 0 if mode == 'greedy' else 1.0,
                    'reasoning_effort': 'none',
                    'max_tokens': args.max_tokens, 'stream': False, 'drafter': 'mtp'}
                if mode == 'sampled':
                    options.update(top_p=0.95, top_k=20)
                for cache_pass in ['first', 'repeat']:
                    start = time.monotonic()
                    response = call('/v1/chat/completions', {**options, 'messages': messages})
                    wall = time.monotonic() - start
                    entry = {'records': count, 'round': iteration + 1, 'mode': mode, 'cache_pass': cache_pass,
                        'messages_sha256': hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
                        'request_options': options, 'wall_seconds': wall, 'response': response}
                    record['runs'].append(entry)
                    save()
                    print(count, iteration + 1, mode, cache_pass, response.get('timings', {}), flush=True)
    record['summary'] = []
    for count in args.records:
        for mode in ['greedy', 'sampled']:
            for cache_pass in ['first', 'repeat']:
                runs = [r for r in record['runs'] if (r['records'], r['mode'], r['cache_pass']) == (count, mode, cache_pass)]
                record['summary'].append({'records': count, 'mode': mode, 'cache_pass': cache_pass, 'runs': len(runs),
                    'median_decode_tps': statistics.median(r['response']['timings']['predicted_per_second'] for r in runs),
                    'median_wall_seconds': statistics.median(r['wall_seconds'] for r in runs),
                    'prompt_tokens': [r['response']['usage']['prompt_tokens'] for r in runs],
                    'cached_tokens': [r['response']['timings']['cache_n'] for r in runs],
                    'completion_tokens': [r['response']['usage']['completion_tokens'] for r in runs]})
    record['health_after'] = call('/health')
    record['completed'] = True
    save()


if __name__ == '__main__':
    main()
