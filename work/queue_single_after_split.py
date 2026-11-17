#!/usr/bin/env python3
"""Queue scalar-enabled Single QK on GPUs 4/7 after the frozen split-Emb run."""
import importlib.util
import json
import math
from pathlib import Path

helper = Path(__file__).with_name('queue_split_after_perloop.py')
spec = importlib.util.spec_from_file_location('queue_runner', helper)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def dependency_state(dep):
    root = Path(dep['run_dir'])
    state = json.loads((root / 'status.json').read_text())
    alive = runner.process_start(dep['pid']) == dep['start_ticks']
    launcher_alive = runner.process_start(dep['launcher_pid']) == dep['launcher_start_ticks']
    if state['status'] == 'failed' or state.get('returncode', 0) != 0:
        return 'failed', 'chunked training failed'
    if state['status'] != 'exited':
        return ('waiting', 'chunked training still running') if alive or launcher_alive else ('failed', 'chunked processes exited without success status')
    if state.get('returncode') != 0 or not state.get('training_started'):
        return 'failed', 'missing successful training exit'
    if alive or launcher_alive:
        return 'waiting', 'waiting for chunked processes to exit'
    step = dep['final_step']
    ckpt = Path(dep['checkpoint_dir'])
    names = [f'model_{step:06d}.pt', f'meta_{step:06d}.json']
    names += [f'optim_{step:06d}_rank{rank}.pt' for rank in range(dep['gpu_count'])]
    for name in names:
        p = ckpt / name
        if not p.is_file() or not p.stat().st_size:
            return 'failed', f'missing final checkpoint: {name}'
    meta = json.loads((ckpt / f'meta_{step:06d}.json').read_text())
    if meta.get('step') != step or not math.isfinite(meta.get('val_bpb', float('nan'))):
        return 'failed', 'final step or validation is invalid'
    return 'complete', 'successful exit, final validation and all checkpoint shards verified'


runner.dependency_state = dependency_state
if __name__ == '__main__':
    runner.main()
