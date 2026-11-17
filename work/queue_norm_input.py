#!/usr/bin/env python3
"""Wait for both selected GPUs to be idle, then invoke one frozen launcher."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def idle_gpus(selected):
    inventory={}
    for row in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
                                         '--format=csv,noheader,nounits'],text=True,timeout=20).splitlines():
        idx,uid,mem,util=[x.strip() for x in row.split(',')]
        inventory[int(idx)]=(uid,int(mem),int(util))
    occupied={r.split(',')[0].strip() for r in subprocess.check_output(
        ['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20).splitlines() if r.strip()}
    for gpu in selected:
        uid,mem,util=inventory[gpu['index']]
        if uid!=gpu['uuid']:raise RuntimeError('GPU identity changed')
        if uid in occupied or mem>100 or util>5:return False
    return True


def verify_target(config):
    root=Path(config['target_run'])
    for name,digest in config['target_hashes'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Frozen target changed: {name}')
    m=json.loads((root/'manifest.json').read_text())
    if m['model_type']!='gpt_base_single_qk_embed_no_embed_norm' or not m['model_config']['norm_value_input']:
        raise RuntimeError('Queue target must be the normalized single-head model')
    if m['validation']['status'] not in ('passed','cpu_passed_pending_gpu'):raise RuntimeError('CPU validation is incomplete')
    for p in (root/'launch_request.json',Path(m['training_log']),Path(m['checkpoint_dir'])):
        if p.exists():raise RuntimeError(f'Target already launched: {p}')


def wait_and_launch(config,publish,sleep=time.sleep):
    observations=0
    while time.time()<config['deadline_epoch']:
        ready=idle_gpus(config['gpus'])
        observations=observations+1 if ready else 0
        publish('waiting_for_idle_gpus',idle_observations=observations)
        if observations>=2:
            verify_target(config)
            publish('launching')
            result=subprocess.run(['bash',str(Path(config['target_run'])/'launch.sh'),'--launch','--gpus',
                                   ','.join(str(g['index']) for g in config['gpus'])],
                                  cwd=config['target_run'],stdin=subprocess.DEVNULL)
            if result.returncode:raise RuntimeError(f'Launcher exited {result.returncode}; no automatic retry')
            publish('finished',launcher_returncode=0)
            return
        sleep(config['poll_seconds'])
    raise RuntimeError('Queue deadline expired')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args();config=json.loads(args.config.read_text());verify_target(config)
    assert len(config['gpus'])==2 and len({g['index'] for g in config['gpus']})==2
    if args.check:
        print(json.dumps({'target_verified':True,'both_gpus_idle':idle_gpus(config['gpus'])}));return
    root=args.config.parent;previous=None
    def publish(status,**extra):
        nonlocal previous
        state=dict(status=status,pid=os.getpid(),target_run=config['target_run'],
                   gpu_indices=[g['index'] for g in config['gpus']],updated_at=datetime.now(timezone.utc).isoformat(),**extra)
        tmp=root/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(root/'status.json')
        target=Path(config['target_run'])/'status.json';s=json.loads(target.read_text())
        if not s.get('training_started'):
            s.update(status='queued' if status.startswith('waiting_') else 'queue_'+status,
                     queue_dir=str(root),queue_pid=os.getpid(),selected_gpu_indices=state['gpu_indices'])
            tmp=target.with_name('status.queue.tmp');tmp.write_text(json.dumps(s,indent=2)+'\n');tmp.replace(target)
        if status!=previous:print(json.dumps(state),flush=True);previous=status
    def stop(signum,frame):
        publish('cancelled',signal=signum);raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:wait_and_launch(config,publish)
        except Exception as exc:publish('failed',error=str(exc));raise

if __name__=='__main__':main()
