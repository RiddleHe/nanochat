#!/usr/bin/env python3
"""Validate the frozen run; on explicit launch, finish GPU checks before training."""
import argparse
from datetime import datetime,timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--dry-run',action='store_true');mode.add_argument('--launch',action='store_true')
    parser.add_argument('--gpus');args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('frozen_launcher_core',HERE/'launcher_core.py')
    core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    m=json.loads((HERE/'manifest.json').read_text());core.verify(m)
    if not args.launch:
        print(json.dumps({k:m[k] for k in ('research_name','expected_steps','validation')},indent=2))
        print('Standby: no queue, GPU work, or training started.');return
    if not args.gpus:parser.error('--launch requires --gpus')
    parts=args.gpus.split(',')
    if len(parts)!=2 or not all(x.isdigit() for x in parts) or len(set(map(int,parts)))!=2:
        parser.error('Exactly two distinct physical GPU indices required')
    inventory={}
    for row in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).splitlines():
        i,uid=row.split(',');inventory[int(i)]=uid.strip()
    if any(int(i) not in inventory for i in parts):parser.error('Unknown GPU index')
    m['selected_gpus']=[{'index':int(i),'uuid':inventory[int(i)]} for i in parts]
    with (HERE/'preflight.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        core.verify(m);core.verify_gpus(m)
        if m['validation']['status']=='cpu_passed_pending_gpu':
            if m['validation']['cpu']['status']!='passed':raise RuntimeError('CPU validation required')
            env=dict(os.environ,**m['env']);env['CUDA_VISIBLE_DEVICES']=m['selected_gpus'][0]['uuid']
            for key in ('RANK','LOCAL_RANK','WORLD_SIZE'):env.pop(key,None)
            stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            log=HERE/f'gpu_preflight_{stamp}.log'
            print(f'Running synthetic single-GPU validation: {log}',flush=True)
            with log.open('x') as output:
                subprocess.run([m['command'][0],str(HERE/'gpu_smoke.py')],cwd=m['repo'],env=env,
                               stdout=output,stderr=subprocess.STDOUT,check=True)
            subprocess.run([m['command'][0],str(HERE/'finalize_validation.py')],cwd=m['repo'],env=env,check=True)
        # Core re-verifies dependencies, checks both GPUs, then runs the two-rank
        # synthetic preflight before creating any dataset-training process.
        core.main()

if __name__=='__main__':main()
