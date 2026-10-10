#!/usr/bin/env python3
"""Summarize completed records without issuing inference requests."""
from pathlib import Path
import argparse
import json
import re
import statistics

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input-dir',type=Path,default=Path(__file__).resolve().parent)
OUT=parser.parse_args().input_dir.resolve()

def read(name):
    return json.loads((OUT/name).read_text())

state = read('run-state.json')
if not state['completed'] or state['stop_returncode'] or state['final_running_containers']:
    raise RuntimeError('Run incomplete or initial stopped state not restored')
diagnostic = read('v2-diagnostics/summary.json')
assert diagnostic['completed']
summary = {'runtime_release': state['runtime_release'], 'image': state['image'],
           'started_at': state['started_at'], 'finished_at': state['finished_at'],
           'method': state['method']}
log = (OUT/'v2-diagnostics/32k-container.log').read_text()
memory = re.search(r'memory: ([\d.]+) GiB of weights locked in RAM, ([\d.]+) GiB of KV pool, ([\d.]+) GiB of working memory, ([\d.]+) GiB in all', log)
summary['v2'] = {'quality': diagnostic['quality_summary'],
    'decode_median_tps': diagnostic['decode_median_tps'],
    'decode_identical': diagnostic['decode_identical'],
    'retrieval': diagnostic['retrieval'],
    'engine_memory_gib_at_32k': dict(zip(['locked_weights','kv_pool','working','total'], map(float,memory.groups()))) if memory else None}
serving = read('serving.json')
summary['serving'] = {'passed': sum(c['passed'] for c in serving['checks']), 'total':len(serving['checks']), 'failures':[c['name'] for c in serving['checks'] if not c['passed']]}
performance = read('performance.json')
assert performance['completed']
summary['performance'] = performance['summary']
contention = read('npu-contention.json')
summary['npu_contention'] = contention['summary'] | {'checks_passed':sum(c['passed'] for c in contention['checks']), 'checks_total':len(contention['checks']), 'probes':len(contention['probes']), 'all_gpu_outputs_identical':len({p['gpu']['sha256'] for p in contention['probes']})==1}
summary['npu_contention']['decode_reduction_percent'] = 100*(1-contention['summary']['busy_median_gpu_decode_tps']/contention['summary']['idle_median_gpu_decode_tps'])
ingestion=read('ingestion.json'); assert ingestion['completed']
summary['ingestion'] = [{k:v for k,v in s.items() if k!='results'} for s in ingestion['scenarios']]
scifact=read('scifact.json'); assert scifact['completed']
summary['scifact'] = {'method':scifact['method'], 'cache':scifact['cache'], 'metrics':scifact['summary'], 'paired_ndcg10_change':scifact['paired_ndcg10_change'], 'corpus_embedding_seconds':scifact['corpus_embedding_seconds_this_run'], 'mean_rerank_seconds':statistics.mean(q['rerank_seconds'] for q in scifact['queries'])}
post=read('post-load-npu.json')
summary['post_load_npu'] = post['summary']
inspect=read('container-before-stop.json')[0]
summary['container_oom_killed'] = inspect['State']['OOMKilled']
summary['final_running_containers'] = state['final_running_containers']
samples=[json.loads(line) for line in (OUT/'hardware.jsonl').read_text().splitlines()]
summary['hardware'] = {'max_sampled_gtt_gib':max(s['hardware']['HALO_AI_GTT_USED_BYTES']/2**30 for s in samples if 'hardware' in s), 'min_sampled_mem_available_gib':min(s['hardware']['HALO_AI_CPU_MEM_AVAILABLE_BYTES']/2**30 for s in samples if 'hardware' in s), 'note':'MemAvailable includes pinned file-cache weights that cannot actually be reclaimed; see engine memory logs. Hardware samples span all phases and are not isolated model allocations.'}
summary['hardware']['power_by_stage'] = {}
for stage in dict.fromkeys(s['stage'] for s in samples if 'stage' in s):
    watts = [float(value)/1e6 for s in samples if s.get('stage') == stage
             for key, value in s.get('sensors', {}).items() if key.endswith('power1_average')]
    if watts:
        summary['hardware']['power_by_stage'][stage] = {
            'samples': len(watts), 'minimum_watts': min(watts),
            'median_watts': statistics.median(watts), 'maximum_watts': max(watts)}
summary['hardware']['power_note'] = 'AMDGPU reported SoC power, sampled every two seconds; not a firmware TDP readback. The paired contention screen does not isolate power sharing from memory contention.'
if (OUT/'power-observation.json').exists():
    summary['hardware']['power_observation'] = read('power-observation.json')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
