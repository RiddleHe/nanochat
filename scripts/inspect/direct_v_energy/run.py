"""Execute collection sequentially from each model's frozen code and environment."""
import argparse
import os
from pathlib import Path
import subprocess
from .common import read, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    a = p.parse_args()
    root = a.run_dir.resolve()
    parent = read(root/'manifest.json')
    for name in parent['models']:
        run = root/name
        assert sha(run/'manifest.json') == parent['child_manifest_hashes'][name]
        m = read(run/'manifest.json')
        env = dict(os.environ, **m['environment_overrides'])
        for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'PYTHONPATH'):
            env.pop(key, None)
        with (run/'logs/collect.log').open('a') as log:
            proc = subprocess.Popen([m['python'], '-m', 'scripts.inspect.direct_v_energy.collect', '--run-dir', str(run)],
                cwd=run/'code', env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                print(name+': '+line, end='', flush=True); log.write(line); log.flush()
            if proc.wait():
                raise RuntimeError(f'{name} collection failed; see {run}/logs/collect.log')


if __name__ == '__main__':
    main()
