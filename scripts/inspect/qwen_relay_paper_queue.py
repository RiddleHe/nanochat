"""Wait for one idle GPU; never launch unrelated experiment scans or kill jobs."""
import csv
import datetime as dt
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import time


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def snapshot():
    raw = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, timeout=20)
    processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True, timeout=20)
    busy = {line.split(',')[0].strip() for line in processes.splitlines() if ',' in line}
    result = []
    for index, uuid, memory, util in csv.reader(io.StringIO(raw)):
        result.append({'index': int(index), 'uuid': uuid.strip(),
                       'memory_mib': int(memory), 'utilization': int(util),
                       'has_process': uuid.strip() in busy})
    return result


def available(g):
    return not g['has_process'] and g['memory_mib'] < 1000 and g['utilization'] < 5


def wait_gpu(status, samples, poll):
    stable = {}
    while True:
        try:
            state = snapshot()
        except Exception as exc:
            stable = {}
            status('waiting', query_error=repr(exc))
            time.sleep(poll)
            continue
        for g in state:
            uid = g['uuid']
            stable[uid] = stable.get(uid, 0) + 1 if available(g) else 0
        candidates = sorted((g for g in state if stable[g['uuid']] >= samples),
                            key=lambda g: g['index'])
        for g in candidates:
            lock = Path(f'/tmp/relay-x-gpu-{g["uuid"]}.lock').open('a')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fresh = next(x for x in snapshot() if x['uuid'] == g['uuid'])
                if available(fresh):
                    return fresh, lock
            except BlockingIOError:
                pass
            lock.close()
        status('waiting', stable_samples=stable, gpus=state)
        time.sleep(poll)
