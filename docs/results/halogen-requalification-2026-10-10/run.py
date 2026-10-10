#!/usr/bin/env python3
"""Rerun Halogen qualification serially using the checkout config; leave it stopped.

Run from an idle host with the pinned images, model artifacts, and SciFact dataset
already present. No host tuning, model acquisition, or system redeployment.
"""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = Path(__file__).resolve().parent
CONFIG = ROOT / '.halo-ai-workspace.env'
sys.path.insert(0, str(ROOT / 'lib/halo_ai'))
import cli
import engine_halogen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=SCRIPTS)
    parser.add_argument('--cache-dir', type=Path, help='Optional SciFact vector cache; defaults to a separate cache for this output directory')
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    cache_dir = args.cache_dir or (Path('/var/cache/halo-ai/benchmarks/scifact') /
                                  ('qualification-' + hashlib.sha256(str(OUT).encode()).hexdigest()[:16]))
    OUT.mkdir(parents=True, exist_ok=True)
    # The test API is loopback; do not route it through inherited HTTP proxies.
    for key in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(key, "")
        os.environ[key] = ",".join(filter(None, [existing, "127.0.0.1", "localhost"]))
    config = cli.load_config(str(CONFIG))
    active = subprocess.check_output(['podman', 'ps', '--format', '{{.Names}}'], text=True).strip()
    if active:
        raise RuntimeError(f'Requires idle container host, found: {active}')
    if (OUT / 'run-state.json').exists():
        raise RuntimeError('Use a fresh output directory; preserve earlier evidence')
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    state = {'started_at': started, 'completed': False, 'initial_running_containers': [],
             'runtime_release': engine_halogen.RELEASE, 'image': cli.image_for(config, 'halogen'),
             'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             'config': str(CONFIG.relative_to(ROOT)), 'config_sha256': hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
             'method': 'Sequential tests, native v2 only, same host and pinned engine. No reset of disk page cache or changes to host tuning. Three alternating serial/MTP timing rounds on v2; retrieval once per prompt length. Serving performance three rounds per prompt/mode/cache pass. Five alternating GPU idle/NPU load pairs. SciFact full corpus; cache reuse is reported in the result. Restore initial stopped state.',
             'source_files_sha256': {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'lib/halo_ai/engine_halogen.py', ROOT/'lib/halo_ai/halogen_npu_models.json', ROOT/'config/halo-ai.env.example']},
             'tools_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'tools').glob('halogen_*.py')},
             'stages': []}
    def save():
        temp = OUT/'run-state.json.partial'
        temp.write_text(json.dumps(state, indent=2)+'\n')
        temp.replace(OUT/'run-state.json')
    def capture(name, command):
        r = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (OUT/name).write_text(r.stdout)
        return r.returncode
    def stage(name, command, allowed=(0,)):
        entry = {'name': name, 'command': command, 'started_at': dt.datetime.now(dt.timezone.utc).isoformat()}
        state['stages'].append(entry); save()
        print(f'STAGE {name}', flush=True)
        begin = time.monotonic()
        with (OUT/f'{name}.log').open('w') as log:
            r = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        entry.update(returncode=r.returncode, elapsed_seconds=time.monotonic()-begin, finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        save(); print(f'FINISHED {name}: rc={r.returncode}, {entry["elapsed_seconds"]:.1f}s', flush=True)
        if r.returncode not in allowed:
            raise RuntimeError(f'{name} failed: {r.returncode}; inspect its log')
    def screen(name, script, *options):
        stage(name, [sys.executable, f'tools/{script}', *options, '--output', str(OUT/f'{name}.json')])
    save()
    capture('doctor.txt', ['bin/halo-ai', '--config', str(CONFIG), 'doctor'])
    capture('host-profile.json', ['bin/halo-ai', '--config', str(CONFIG), 'host-profile', 'status'])
    capture('power-profile.txt', ['powerprofilesctl', 'get'])
    capture('fabric-clock.txt', ['halogen-fabric-clock', 'status'])
    capture('kernel.txt', ['uname', '-a'])
    capture('image.json', ['podman', 'image', 'inspect', state['image']])
    state['hardware_before'] = cli.hardware_snapshot(config)
    state['boot_id'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    save()
    stop_sample = threading.Event()
    def sample():
        sensor_root = Path('/sys/class/drm/card1/device')
        with (OUT/'hardware.jsonl').open('w') as stream:
            while not stop_sample.is_set():
                try:
                    row = {'time': dt.datetime.now(dt.timezone.utc).isoformat(), 'stage': state['stages'][-1]['name'] if state['stages'] else 'prepare', 'hardware': cli.hardware_snapshot(config)}
                    row['sensors'] = {str(p.relative_to(sensor_root)): p.read_text().strip() for pat in ['hwmon/hwmon*/power1_average', 'hwmon/hwmon*/temp1_input', 'gpu_busy_percent'] for p in sensor_root.glob(pat)}
                    stream.write(json.dumps(row)+'\n'); stream.flush()
                except Exception as exc:
                    stream.write(json.dumps({'sampling_error': repr(exc)})+'\n'); stream.flush()
                stop_sample.wait(2)
    thread = threading.Thread(target=sample, daemon=True); thread.start()
    try:
        stage('v2-diagnostics', [sys.executable, str(SCRIPTS/'diagnostics.py'), '--output-dir', str(OUT/'v2-diagnostics')])
        stage('start-serving', ['bin/halo-ai', '--config', str(CONFIG), 'start', 'qwen3.8-halogen-v2-npu', '--switch'])
        stage('serving-smoke', ['bin/halo-ai', '--config', str(CONFIG), 'test', 'qwen3.8-halogen-v2-npu'])
        screen('serving', 'halogen_serving_validate.py')
        screen('performance', 'halogen_performance_validate.py', '--rounds', '3')
        screen('npu-contention', 'halogen_npu_validate.py', '--rounds', '5')
        screen('ingestion', 'halogen_npu_evaluate.py', 'burst')
        screen('scifact', 'halogen_npu_evaluate.py', 'quality', '--cache-dir', str(cache_dir.resolve()))
        stage('post-load-smoke', ['bin/halo-ai', '--config', str(CONFIG), 'test', 'qwen3.8-halogen-v2-npu'])
        screen('post-load-npu', 'halogen_npu_validate.py', '--rounds', '0')
        state['completed'] = True
    except BaseException as exc:
        state['error'] = repr(exc)
        raise
    finally:
        capture('serving-container.log', ['podman', 'logs', 'halo-halogen'])
        capture('container-before-stop.json', ['podman', 'inspect', 'halo-halogen'])
        capture('kernel-journal.txt', ['journalctl', '-k', '--since', started, '--no-pager'])
        state['stop_returncode'] = capture('stop.log', ['bin/halo-ai', '--config', str(CONFIG), 'stop', 'qwen3.8-halogen-v2-npu'])
        stop_sample.set(); thread.join(timeout=10)
        state['hardware_after'] = cli.hardware_snapshot(config)
        state['final_running_containers'] = subprocess.check_output(['podman', 'ps', '--format', '{{.Names}}'], text=True).splitlines()
        state['finished_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
        save()
        print(json.dumps({k:state[k] for k in ['completed','stop_returncode','final_running_containers','finished_at']}), flush=True)


if __name__ == '__main__':
    main()
