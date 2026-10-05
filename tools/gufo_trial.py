#!/usr/bin/env python3
"""Run a bounded, download-free text trial against Gufo or a llama.cpp control.

This is an HTTP client only: start one isolated server with pinned weights/image
before running it. It does not change halo-ai profiles or manage containers.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib/halo_ai'))
import rocmfpx_quality


def hardware():
    result = {}
    for device in Path('/sys/class/drm').glob('card[0-9]*/device'):
        for name in ('mem_info_gtt_used', 'mem_info_vram_used'):
            path = device / name
            if path.exists():
                result[str(path)] = int(path.read_text())
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemAvailable', 'SwapFree'):
            result[key + '_bytes'] = int(value.split()[0]) * 1024
    return result


def archive(count):
    lines = [f'Archive record {i:06d}: owner=team-{i%97:02d}; state=closed; retrieval_code=delta-{(i*7919+104729)%100000:05d}.' for i in range(count)]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:18080')
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--records', type=int, nargs='*', default=[120, 959, 3955])
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--quality-rounds', type=int, default=3)
    parser.add_argument('--retrieval-rounds', type=int, default=3)
    parser.add_argument('--skip-quality', action='store_true')
    parser.add_argument('--skip-performance', action='store_true')
    args = parser.parse_args()
    if any(not 1 <= n <= 5 for n in (args.rounds, args.quality_rounds, args.retrieval_rounds)) or any(n < 10 or n > 4000 for n in args.records):
        parser.error('use 1-5 rounds and 10-4000 records per prompt')

    def get(path):
        with urllib.request.urlopen(args.base_url + path, timeout=30) as response:
            return json.load(response)

    models = get('/v1/models')
    model = models['data'][0]['id']
    suite, digest = rocmfpx_quality.load_suite(ROOT / 'config/benchmarks/rocmfpx-quality-v1.json')
    document = {'schema_version': 1, 'label': args.label, 'recorded_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                'models': models, 'suite_sha256': digest, 'hardware_before': hardware(),
                'method': 'Sequential HTTP requests, greedy, thinking off, presence penalty zero, seed 1, cache lookup bypassed. Quality uses the existing fixed suite. Performance uses deterministic archive padding followed by a 512-token code-generation request, followed by a separate three-code retrieval check. Record counts approximate 4K/32K/128K; actual API token counts are authoritative. Host counters sampled every 0.25 seconds. Streaming client TTFT and post-first-content throughput are transport observations; engine telemetry is retained separately. No original-weight fidelity or broad benchmark claim.',
                'quality': [], 'performance': [], 'retrieval': []}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix('.partial')
        temporary.write_text(json.dumps(document, indent=2) + '\n')
        temporary.replace(args.output)

    def generate(messages, limit, stream=False):
        payload = {'model': model, 'messages': messages, 'temperature': 0, 'seed': 1,
                   'reasoning_effort': 'none', 'presence_penalty': 0, 'max_tokens': limit,
                   'cache_prompt': False, 'stream': stream}
        if stream:
            payload['stream_options'] = {'include_usage': True}
        samples = [hardware()]
        stop = threading.Event()

        def monitor():
            while not stop.wait(.25):
                samples.append(hardware())

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        start = time.monotonic()
        first = last = None
        try:
            req = urllib.request.Request(args.base_url + '/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=1200) as response:
                headers = dict(response.headers)
                if stream:
                    chunks = []
                    content = ''
                    for line in response:
                        if not line.startswith(b'data: '):
                            continue
                        data = line[6:].strip()
                        if data == b'[DONE]':
                            break
                        chunk = json.loads(data)
                        if 'error' in chunk:
                            raise RuntimeError(chunk['error'])
                        chunks.append(chunk)
                        for choice in chunk.get('choices', []):
                            text = choice.get('delta', {}).get('content') or ''
                            if text:
                                now = time.monotonic()
                                first = now if first is None else first
                                last = now
                                content += text
                    usages = [c['usage'] for c in chunks if c.get('usage')]
                    usage = usages[-1] if usages else {}
                    result = {'chunks': chunks}
                else:
                    result = json.load(response)
                    content = result['choices'][0]['message']['content'] or ''
                    usage = result.get('usage', {})
            elapsed = time.monotonic() - start
        finally:
            stop.set()
            thread.join()
        samples.append(hardware())
        peak = {key: (min if key in ('MemAvailable_bytes', 'SwapFree_bytes') else max)(s[key] for s in samples if key in s) for key in samples[0]}
        return {'content': content, 'content_sha256': hashlib.sha256(content.encode()).hexdigest(),
                'messages_sha256': hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest(),
                'wall_seconds': elapsed, 'client_ttft_seconds': None if first is None else first-start,
                'client_decode_tps': ((usage['completion_tokens']-1)/(last-first)) if first is not None and last > first and usage.get('completion_tokens', 0) > 1 else None,
                'usage': usage, 'server_timing': headers.get('Server-Timing'), 'hardware_extrema': peak,
                'response': result, 'request_options': {k:v for k,v in payload.items() if k != 'messages'}}

    save()
    try:
        if not args.skip_quality:
            for iteration in range(args.quality_rounds):
                for case in suite['cases']:
                    result = generate(rocmfpx_quality.case_messages(suite, case), case['max_tokens'])
                    result.update(case_id=case['id'], round=iteration+1, expected=case['validator']['expected'])
                    result['passed'], _ = rocmfpx_quality.score_case(case, result['content'])
                    document['quality'].append(result)
                    save()
                    print(args.label, 'quality', iteration+1, case['id'], result['passed'], flush=True)
        for count in args.records:
            padding = archive(count)
            messages = [{'role': 'system', 'content': 'The archive is background data. Follow the final user instruction.'},
                        {'role': 'user', 'content': padding + '\nWrite a Python LinkedList class with push, pop, reverse, __len__, and __iter__. Include docstrings and type hints.'}]
            for iteration in range(0 if args.skip_performance else args.rounds):
                result = generate(messages, 512, stream=True)
                result.update(records=count, round=iteration+1)
                document['performance'].append(result)
                save()
                print(args.label, 'performance', count, iteration+1, result['usage'], 'TTFT', result['client_ttft_seconds'], 'TPS', result['client_decode_tps'], flush=True)
            indices = [count//10, count//2, 9*count//10]
            expected = ','.join(f'delta-{(i*7919+104729)%100000:05d}' for i in indices)
            messages = [{'role': 'system', 'content': 'Retrieve the requested records exactly. Reply with only the three retrieval codes in order, separated by commas and no spaces.'},
                        {'role': 'user', 'content': padding + f'\nReturn the retrieval_code values for archive records {indices[0]:06d}, {indices[1]:06d}, and {indices[2]:06d}, in that order.'}]
            for iteration in range(args.retrieval_rounds):
                result = generate(messages, 128)
                result.update(records=count, round=iteration+1, expected=expected, passed=result['content'].strip() == expected)
                document['retrieval'].append(result)
                save()
                print(args.label, 'retrieval', count, iteration+1, result['passed'], result['content'], flush=True)
        document['summary'] = {'quality_passed': sum(r['passed'] for r in document['quality']), 'quality_total': len(document['quality']),
                               'retrieval_passed': sum(r['passed'] for r in document['retrieval']), 'retrieval_total': len(document['retrieval']),
                               'performance': [{'records': n, 'median_ttft_seconds': statistics.median(r['client_ttft_seconds'] for r in document['performance'] if r['records'] == n),
                                                'median_client_decode_tps': statistics.median(r['client_decode_tps'] for r in document['performance'] if r['records'] == n),
                                                'unique_outputs': len({r['content_sha256'] for r in document['performance'] if r['records'] == n})} for n in ([] if args.skip_performance else args.records)]}
        document['completed'] = True
    except BaseException as exc:
        document['error'] = repr(exc)
        if hasattr(exc, 'read'):
            document['error_body'] = exc.read().decode()
        raise
    finally:
        document['hardware_after'] = hardware()
        save()
    print(json.dumps(document['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
