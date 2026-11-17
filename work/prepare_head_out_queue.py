"""Freeze and start the per-head-output queue after the running Single QK run."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', action='store_true', help='Explicitly start the detached queue')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    python = '/hdd/mh3897/nanochat/.venv/bin/python'
    target = Path((repo/'work/split-embed-head-out-prepared-run.txt').read_text().strip())
    source = Path((repo/'work/single-qk-prepared-run.txt').read_text().strip())
    pointer = repo/'work/split-embed-head-out-queue.txt'
    if pointer.exists():
        raise RuntimeError(f'Queue already recorded at {pointer}; inspect it before any retry')
    target_state = json.loads((target/'status.json').read_text())
    assert target_state['status'] == 'ready' and not target_state['training_started']
    subprocess.run(['bash', str(target/'launch.sh'), '--dry-run'], check=True, stdout=subprocess.DEVNULL)
    state = json.loads((source/'status.json').read_text())
    manifest = json.loads((source/'manifest.json').read_text())
    assert state['status'] == 'running'
    pid = state['training_pid']
    launcher_pid = int(next(line.split()[1] for line in Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith('PPid:')))
    assert str(source/'launch.py') in Path(f'/proc/{launcher_pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    spec = importlib.util.spec_from_file_location('queue_helper', repo/'work/queue_split_after_perloop.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    dep = dict(name='single_qk_variant_3',run_dir=str(source),pid=pid,start_ticks=helper.process_start(pid),
               launcher_pid=launcher_pid,launcher_start_ticks=helper.process_start(launcher_pid),
               final_step=manifest['expected_steps'],checkpoint_dir=manifest['checkpoint_dir'],gpu_count=manifest['gpu_count'])
    assert dep['start_ticks'] is not None and dep['launcher_start_ticks'] is not None
    inventory = {}
    for row in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True).splitlines():
        index, uuid = row.split(','); inventory[int(index)] = uuid.strip()
    config = dict(target_run=str(target),dependencies=[dep],gpus=[dict(index=i,uuid=inventory[i]) for i in (4,7)],
                  poll_seconds=30,deadline_epoch=time.time()+48*3600,
                  target_hashes={name:hashlib.sha256((target/name).read_bytes()).hexdigest() for name in ('manifest.json','launch.py','launch.sh')})
    helper.verify_target(config)
    if not args.start:
        print(json.dumps(config,indent=2))
        return
    root = Path('/local-ssd/mh3897/run_queues')/f'head_out_after_single_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}'
    root.mkdir()
    for name in ('queue_head_out_after_single.py','queue_split_after_perloop.py'):
        shutil.copy2(repo/'work'/name,root/name)
    shutil.copy2(Path(__file__),root/'prepare_queue.py')
    (root/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    (root/'README.md').write_text('Wait for Single QK successful exit, finite final validation and all final checkpoint shards. Then require two idle observations on GPUs 4/7 and launch the frozen target once. The target runs its two-rank synthetic preflight before dataset training. Dependency or preflight failure stops the queue. Waiting expires after 48 hours. See config.json, status.json and queue.log.\n')
    command = [python,'-u',str(root/'queue_head_out_after_single.py'),'--config',str(root/'config.json')]
    subprocess.run(command+['--check'],check=True)
    with (root/'queue.log').open('x') as output:
        proc = subprocess.Popen(command,cwd=root,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    (root/'queue.pid').write_text(str(proc.pid)+'\n')
    pointer.write_text(str(root)+'\n')
    print(json.dumps(dict(queue=str(root),pid=proc.pid)),flush=True)


if __name__ == '__main__':
    main()
