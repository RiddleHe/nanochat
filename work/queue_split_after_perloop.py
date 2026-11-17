#!/usr/bin/env python3
"""Wait for the named per-loop runs, then invoke the frozen split-Emb launcher once."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def process_start(pid):
    try:
        # Linux stat's command can contain spaces or parentheses.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    except FileNotFoundError:
        return None


def dependency_state(dep):
    alive = process_start(dep['pid']) == dep['start_ticks']
    root = Path(dep['run_dir'])
    done = root / 'done.json'
    if not done.exists():
        return ('waiting', 'still training') if alive else ('failed', 'process exited without final completion')
    try:
        result = json.loads(done.read_text())
    except json.JSONDecodeError:
        return ('waiting', 'completion file being written') if alive else ('failed', 'invalid completion file')
    if result.get('final_step') != dep['final_step'] or not math.isfinite(result.get('val_bpb', float('nan'))):
        return 'failed', 'completion does not match planned final step and finite validation'
    for prefix, suffix in [('model', 'pt'), ('meta', 'json'), ('state', 'pt')]:
        p = root / 'checkpoints' / f"{prefix}_{dep['final_step']:06d}.{suffix}"
        if not p.is_file() or p.stat().st_size == 0:
            return 'failed', f'missing final checkpoint: {p.name}'
    if alive:
        return 'waiting', 'final files saved; waiting for process exit'
    return 'complete', 'final result and checkpoints verified'


def idle_gpus(selected):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True, timeout=20)
    inventory = {}
    for row in rows.splitlines():
        idx, uid, mem, util = (x.strip() for x in row.split(','))
        inventory[int(idx)] = (uid, int(mem), int(util))
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader,nounits'], text=True, timeout=20)
    occupied = {r.split(',')[0].strip() for r in apps.splitlines() if r.strip()}
    for gpu in selected:
        uid, mem, util = inventory[gpu['index']]
        if uid != gpu['uuid']:
            raise RuntimeError(f"GPU {gpu['index']} identity changed")
        if uid in occupied or mem > 100 or util > 5:
            return False
    return True


def verify_target(config):
    root = Path(config['target_run'])
    for name, expected in config['target_hashes'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen target changed: {name}')
    manifest = json.loads((root / 'manifest.json').read_text())
    for path in [root / 'launch_request.json', Path(manifest['training_log']), Path(manifest['checkpoint_dir'])]:
        if path.exists():
            raise RuntimeError(f'Target already launched or has output: {path}')


def wait_and_launch(config, publish, sleep=time.sleep):
    deadline = config['deadline_epoch']
    idle_observations = 0
    while time.time() < deadline:
        states = {d['name']: dependency_state(d) for d in config['dependencies']}
        if any(s[0] == 'failed' for s in states.values()):
            raise RuntimeError(f'Dependency failed: {states}')
        if all(s[0] == 'complete' for s in states.values()):
            idle_observations = idle_observations + 1 if idle_gpus(config['gpus']) else 0
            publish('waiting_for_idle_gpus', dependencies=states, idle_observations=idle_observations)
            if idle_observations >= 2:
                verify_target(config)
                publish('launching', dependencies=states)
                command = ['bash', str(Path(config['target_run']) / 'launch.sh'), '--launch', '--gpus', '4,7']
                result = subprocess.run(command, cwd=config['target_run'], stdin=subprocess.DEVNULL)
                if result.returncode:
                    raise RuntimeError(f'Frozen launcher exited {result.returncode}; inspect queue.log and target status')
                publish('finished', launcher_returncode=0)
                return
        else:
            idle_observations = 0
            publish('waiting_for_dependencies', dependencies=states)
        sleep(config['poll_seconds'])
    raise RuntimeError('Queue deadline expired without launch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--check', action='store_true', help='Readiness check only; do not wait or launch')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    verify_target(config)
    if args.check:
        print(json.dumps({d['name']: dependency_state(d) for d in config['dependencies']}, indent=2))
        return
    root = args.config.parent
    previous = None
    def publish(status, **extra):
        nonlocal previous
        state = dict(status=status, pid=os.getpid(), target_run=config['target_run'],
                     gpu_indices=[4,7], updated_at=datetime.now(timezone.utc).isoformat(), **extra)
        tmp = root / 'status.json.tmp'
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(root / 'status.json')
        # Make the queue discoverable from the prepared run's normal status.
        # Once training starts, its frozen launcher exclusively owns that status.
        target_status = Path(config['target_run']) / 'status.json'
        target = json.loads(target_status.read_text())
        if not target.get('training_started'):
            target.update(status=('queued' if status.startswith('waiting_') else 'queue_' + status),
                          queue_dir=str(root), queue_pid=os.getpid(),
                          selected_gpu_indices=[4, 7])
            temporary = target_status.with_name('status.queue.tmp')
            temporary.write_text(json.dumps(target, indent=2) + '\n')
            temporary.replace(target_status)
        if status != previous:
            print(json.dumps(state), flush=True)
            previous = status
    def stop(signum, frame):
        publish('cancelled', signal=signum)
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            wait_and_launch(config, publish)
        except Exception as exc:
            publish('failed', error=str(exc))
            raise


if __name__ == '__main__':
    main()
