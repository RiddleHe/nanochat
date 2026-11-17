"""Execute a stage using the run's frozen code and recorded environment."""
import argparse
import os
from pathlib import Path
import subprocess
from .common import read_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=['collect','summarize'], required=True)
    args = parser.parse_args(); run = args.run_dir.resolve()
    manifest = read_json(run/'manifest.json')
    for path, digest in manifest['code_hashes'].items():
        assert sha256(run/'code'/path) == digest, path
    env = dict(os.environ, **manifest['environment_overrides'])
    for key in ('RANK','LOCAL_RANK','WORLD_SIZE','PYTHONPATH'):
        env.pop(key, None)
    if args.stage == 'summarize': env['CUDA_VISIBLE_DEVICES'] = ''
    command = [manifest['python_executable'], '-m', 'scripts.inspect.raw_activation_norms.'+args.stage, '--run-dir', str(run)]
    with (run/'logs'/f'{args.stage}.log').open('a', buffering=1) as log:
        process = subprocess.Popen(command, cwd=run/'code', env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='', flush=True); log.write(line)
        code = process.wait()
    if code: raise SystemExit(code)


if __name__ == '__main__': main()
