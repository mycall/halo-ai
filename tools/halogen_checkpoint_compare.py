#!/usr/bin/env python3
"""Compare installed W4B and v2 checkpoints, then restore the serving profile."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib/halo_ai'))
import cli
from halogen_validate import request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--restore-profile', default='qwen3.8-halogen-v2-npu')
    args = parser.parse_args()
    if not 2 <= args.rounds <= 10:
        parser.error('--rounds must be between 2 and 10')
    config = cli.load_config(None)
    catalog = cli.load_catalog(config)
    if not cli.resolve_profile(catalog, args.restore_profile):
        parser.error('unknown restore profile')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = f'http://127.0.0.1:{cli.port_for(config, "halogen")}'
    record = {
        'recorded_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'method': 'Fresh W4B then v2, same pinned engine and host. NPU/vision disabled during comparison. Cache and prompt lookup off; temperature 0, seed 1, thinking off. Existing quality suite warms each runtime before repeated 512-token probes; serial/MTP order alternates. Retrieval has one run per length/checkpoint. Model order is not randomized; timings are a bounded local comparison, not broad quality or performance qualification.',
        'rounds': args.rounds, 'image': cli.image_for(config, 'halogen'),
        'restore_profile': args.restore_profile, 'models': {},
    }

    def save():
        (output / 'comparison.json').write_text(json.dumps(record, indent=2) + '\n')

    def start(profile):
        print(f'START {profile}', flush=True)
        result = cli.command_start(config, catalog, argparse.Namespace(profile_id=profile, switch=True))
        if result:
            raise RuntimeError(f'start failed: {profile}: {result}')

    def run_screen(script, filename):
        path = output / filename
        result = subprocess.run([sys.executable, str(ROOT / 'tools' / script), '--base-url', base, '--output', str(path)])
        if result.returncode not in (0, 1) or not path.exists():
            raise RuntimeError(f'{script} did not finish: {result.returncode}')
        data = json.loads(path.read_text())
        if script == 'halogen_validate.py' and 'summary' not in data:
            raise RuntimeError('incomplete quality screen')
        if script == 'halogen_retrieval_validate.py' and len(data.get('checks', [])) != 2:
            raise RuntimeError('incomplete retrieval screen')
        return data

    try:
        for arm, profile_id in [('w4b', 'qwen3.8-flash-next-halogen-32k-mtp'), ('v2', 'qwen3.8-flash-next-halogen-v2-32k-mtp')]:
            start(profile_id)
            entry = {'profile': copy.deepcopy(catalog.profiles[profile_id]), 'health': request(base, '/health')}
            record['models'][arm] = entry
            save()
            quality = run_screen('halogen_validate.py', f'{arm}-quality.json')
            entry['quality_summary'] = quality['summary']
            entry['hardware_after_quality'] = cli.hardware_snapshot(config)
            entry['decode_runs'] = []
            for iteration in range(args.rounds):
                order = ['serial', 'mtp'] if iteration % 2 == 0 else ['mtp', 'serial']
                for drafter in order:
                    payload = {'model': entry['health']['model'], 'messages': [{'role': 'user', 'content': 'Write a Python LinkedList class with push, pop, reverse, __len__, and __iter__. Include docstrings and type hints.'}], 'max_tokens': 512, 'temperature': 0, 'seed': 1, 'reasoning_effort': 'none', 'stream': False, 'drafter': drafter}
                    begin = time.monotonic()
                    response = request(base, '/v1/chat/completions', payload)
                    text = response['choices'][0]['message']['content']
                    timings = response['timings']
                    assert response['usage']['completion_tokens'] == 512, response
                    assert timings['cache_n'] == 0, response
                    assert (timings['draft_n'] == 0) == (drafter == 'serial'), response
                    run = {'round': iteration + 1, 'drafter': drafter, 'wall_seconds': time.monotonic() - begin, 'sha256': hashlib.sha256(text.encode()).hexdigest(), 'response': response}
                    entry['decode_runs'].append(run)
                    save()
                    print(f'{arm} {iteration + 1}/{args.rounds} {drafter}: {timings["predicted_per_second"]} tok/s', flush=True)
            entry['decode_median_tps'] = {d: statistics.median(r['response']['timings']['predicted_per_second'] for r in entry['decode_runs'] if r['drafter'] == d) for d in ['serial', 'mtp']}
            entry['decode_identical_within_checkpoint'] = len({r['sha256'] for r in entry['decode_runs']}) == 1
            entry['hardware_after_decode'] = cli.hardware_snapshot(config)
            (output / f'{arm}-32k-container.log').write_text(subprocess.check_output(['podman', 'logs', 'halo-halogen'], stderr=subprocess.STDOUT, text=True))
            diagnostic = copy.deepcopy(catalog.profiles[profile_id])
            diagnostic['id'] = f'halogen-{arm}-comparison-262k'
            diagnostic['context'] = 262144
            diagnostic['settings']['kv_pool_positions'] = 262144
            catalog.profiles[diagnostic['id']] = diagnostic
            entry['retrieval_profile'] = diagnostic
            save()
            start(diagnostic['id'])
            retrieval = run_screen('halogen_retrieval_validate.py', f'{arm}-retrieval.json')
            entry['retrieval'] = [{k: c[k] for k in ['records', 'correct_codes', 'passed', 'output', 'expected', 'wall_seconds', 'messages_sha256']} | {'timings': c['response']['timings']} for c in retrieval['checks']]
            entry['hardware_after_retrieval'] = cli.hardware_snapshot(config)
            save()
        record['decode_change_percent_v2_vs_w4b'] = {d: 100 * (record['models']['v2']['decode_median_tps'][d] / record['models']['w4b']['decode_median_tps'][d] - 1) for d in ['serial', 'mtp']}
        record['completed'] = True
        save()
    except BaseException as exc:
        record['error'] = repr(exc)
        save()
        raise
    finally:
        start(args.restore_profile)
        record['restored_health'] = request(base, '/health')
        record['restored_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()
    print(json.dumps({'completed': True, 'decode_change_percent_v2_vs_w4b': record['decode_change_percent_v2_vs_w4b'], 'restored': args.restore_profile}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
