#!/usr/bin/env python3
"""Measure native v2 MTP, memory, decode, and retrieval on the pinned runtime.

Invoked by run.py, which owns stopped-state restoration. Uses the explicit
checkout config and normal lifecycle gates; acquires no model artifacts.
"""
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

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'lib/halo_ai'), str(ROOT/'tools')]
import cli
from halogen_validate import request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    OUT=args.output_dir.resolve()
    config = cli.load_config(str(ROOT/'.halo-ai-workspace.env'))
    catalog = cli.load_catalog(config)
    OUT.mkdir(exist_ok=True)
    base = f'http://127.0.0.1:{cli.port_for(config,"halogen")}'
    record = {'recorded_at':dt.datetime.now(dt.timezone.utc).isoformat(), 'completed':False,
              'image':cli.image_for(config,'halogen'), 'rounds':3,
              'method':'V2 only. Cache and prompt lookup off, NPU and vision off. Temperature zero, seed 1, thinking off. Fixed quality suite warms runtime before three alternating serial/MTP 512-token pairs. One retrieval request at each depth; no general quality claim.'}
    def save():
        (OUT/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
    def start(profile):
        print(f'START {profile}',flush=True)
        rc=cli.command_start(config,catalog,argparse.Namespace(profile_id=profile,switch=True))
        if rc: raise RuntimeError(f'start {profile}: {rc}')
    def screen(script,filename):
        result=subprocess.run([sys.executable,str(ROOT/'tools'/script),'--base-url',base,'--output',str(OUT/filename)])
        if result.returncode not in (0,1): raise RuntimeError(f'{script}: {result.returncode}')
        return json.loads((OUT/filename).read_text())
    try:
        profile='qwen3.8-flash-next-halogen-v2-32k-mtp'
        start(profile)
        record['profile']=copy.deepcopy(catalog.profiles[profile])
        record['health']=request(base,'/health');save()
        quality=screen('halogen_validate.py','quality.json')
        record['quality_summary']=quality['summary'];record['decode_runs']=[];save()
        for iteration in range(3):
            for drafter in (['serial','mtp'] if iteration%2==0 else ['mtp','serial']):
                payload={'model':record['health']['model'],'messages':[{'role':'user','content':'Write a Python LinkedList class with push, pop, reverse, __len__, and __iter__. Include docstrings and type hints.'}],'max_tokens':512,'temperature':0,'seed':1,'reasoning_effort':'none','stream':False,'drafter':drafter}
                begin=time.monotonic(); response=request(base,'/v1/chat/completions',payload)
                assert response['usage']['completion_tokens']==512,response
                assert response['timings']['cache_n']==0,response
                assert (response['timings']['draft_n']==0)==(drafter=='serial'),response
                record['decode_runs'].append({'round':iteration+1,'drafter':drafter,'wall_seconds':time.monotonic()-begin,'sha256':hashlib.sha256(response['choices'][0]['message']['content'].encode()).hexdigest(),'request':payload,'response':response})
                save();print(f'{iteration+1}/3 {drafter}: {response["timings"]["predicted_per_second"]} tok/s',flush=True)
        record['decode_median_tps']={d:statistics.median(r['response']['timings']['predicted_per_second'] for r in record['decode_runs'] if r['drafter']==d) for d in ['serial','mtp']}
        record['decode_identical']=len({r['sha256'] for r in record['decode_runs']})==1
        (OUT/'32k-container.log').write_text(subprocess.check_output(['podman','logs','halo-halogen'],stderr=subprocess.STDOUT,text=True))
        diagnostic=copy.deepcopy(catalog.profiles[profile]);diagnostic['id']='halogen-v2-requalification-262k';diagnostic['context']=262144;diagnostic['settings']['kv_pool_positions']=262144
        catalog.profiles[diagnostic['id']]=diagnostic
        record['retrieval_profile']=diagnostic;save()
        start(diagnostic['id'])
        retrieval=screen('halogen_retrieval_validate.py','retrieval.json')
        assert len(retrieval['checks'])==2
        record['retrieval']=[{k:c[k] for k in ['records','correct_codes','passed','output','expected','wall_seconds','messages_sha256']}|{'timings':c['response']['timings'],'usage':c['response']['usage']} for c in retrieval['checks']]
        (OUT/'262k-container.log').write_text(subprocess.check_output(['podman','logs','halo-halogen'],stderr=subprocess.STDOUT,text=True))
        record['completed']=True
    except BaseException as exc:
        record['error']=repr(exc);raise
    finally: save()


if __name__=='__main__': main()
