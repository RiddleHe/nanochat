"""Queue the 18 local adapter runs (+ scaling, baseline_eval) on explicitly given idle GPUs.

python launch15.py --gpus 0,1,2
No automatic retry. A failed job is marked failed; its worker stops; others continue.
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path
from shared15 import ROOT, now, write_json, verify, frozen_selection, jobs

PYTHON = sys.executable


def gpu_idle(index):
    out = subprocess.check_output(['nvidia-smi', f'--id={index}', '--query-gpu=uuid,memory.used',
                                   '--format=csv,noheader,nounits'], text=True).strip()
    uuid, mem = [s.strip() for s in out.split(',')]
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    return float(mem) < 100 and uuid not in apps


def edit_queue(fn):
    with open(ROOT / 'queue.lock', 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        queue = json.loads((ROOT / 'queue.json').read_text())
        result = fn(queue)
        write_json(ROOT / 'queue.json', queue)
        return result


def claim(gpu):
    def fn(queue):
        for job in queue:
            if job['phase'] == 'pending':
                job.update(phase='running', gpu=gpu, pid=os.getpid(), started_utc=now())
                return job['job']
        return None
    return edit_queue(fn)


def mark(name, phase, code):
    def fn(queue):
        for job in queue:
            if job['job'] == name:
                job.update(phase=phase, returncode=code, finished_utc=now())
    edit_queue(fn)


def worker(gpu):
    guard = open(f'/tmp/nanochat-looping-gpu{gpu}.lock', 'a+')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), NANOCHAT_DTYPE='bfloat16', PYTHONUNBUFFERED='1',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    while True:
        verify()
        if not gpu_idle(gpu):
            raise RuntimeError(f'GPU {gpu} has a foreign process; stopping this worker')
        job = claim(gpu)
        if job is None:
            return 0
        with open(ROOT / 'logs' / f'{job}.log', 'x') as log:
            code = subprocess.call([PYTHON, '-u', str(ROOT / 'run15.py'), job], cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        mark(job, 'complete' if code == 0 else 'failed', code)
        if code:
            return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus')
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    if args.worker is not None:
        sys.exit(worker(args.worker))
    gpus = [int(g) for g in args.gpus.split(',')]
    supervisor = open(ROOT / 'supervisor.lock', 'a+')
    fcntl.flock(supervisor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert verify() is not None, 'Not frozen'
    chosen, _ = frozen_selection()
    assert not (ROOT / 'queue.json').exists(), 'Refusing duplicate launch'
    for g in gpus:
        assert gpu_idle(g), f'GPU {g} not idle'
    (ROOT / 'logs').mkdir(exist_ok=True)
    names = jobs(chosen['locations']) + ['scaling', 'baseline_eval']
    write_json(ROOT / 'queue.json', [dict(job=j, phase='pending') for j in names])
    children = [subprocess.Popen([PYTHON, '-u', str(ROOT / 'launch15.py'), '--worker', str(g)]) for g in gpus]
    write_json(ROOT / 'status.json', dict(phase='running', time=now(), gpus=gpus, jobs=len(names),
                                          workers=[c.pid for c in children]))
    codes = [c.wait() for c in children]
    queue = json.loads((ROOT / 'queue.json').read_text())
    done = all(j['phase'] == 'complete' for j in queue)
    write_json(ROOT / 'status.json', dict(phase='complete' if done else 'needs_attention', time=now(),
                                          worker_returncodes=codes,
                                          failed=[j['job'] for j in queue if j['phase'] == 'failed'],
                                          pending=[j['job'] for j in queue if j['phase'] == 'pending']))


if __name__ == '__main__':
    main()
