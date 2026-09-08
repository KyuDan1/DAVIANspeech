#!/usr/bin/env python3
"""Launch the two frozen v67 low-band Music variants with durable logs/status."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/lowband_music_v67.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', default='2,4')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    variants = list(config['variants'])
    gpus = args.gpus.split(',')
    if len(gpus) != len(variants) or len(set(gpus)) != len(gpus):
        raise ValueError('one distinct GPU per declared variant required')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    active, results = [], []
    manifest = dict(config=str(args.config.resolve()),
                    config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                    python=sys.executable, runs=[])
    try:
        for name, gpu in zip(variants, gpus):
            command = [sys.executable, str(ROOT / 'scripts/train_lowband_music_v67.py'),
                       '--config', str(args.config.resolve()), '--variant', name,
                       '--output', str((args.output / name).resolve())]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='6', OPENBLAS_NUM_THREADS='1')
            env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
            handle = (args.output / f'{name}.log').open('x')
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
            active.append((name, process, handle))
            manifest['runs'].append(dict(name=name, gpu=gpu, pid=process.pid, command=command))
            print(f'launched {name}: GPU {gpu}, PID {process.pid}', flush=True)
        (args.output / 'launch.json').write_text(json.dumps(manifest, indent=2) + '\n')
        while active:
            remaining = []
            for name, process, handle in active:
                code = process.poll()
                if code is None:
                    remaining.append((name, process, handle))
                    continue
                handle.close()
                results.append(dict(name=name, returncode=code))
                (args.output / 'status.json').write_text(json.dumps(results, indent=2) + '\n')
                print(f'finished {name}: {code}', flush=True)
            active = remaining
            if active:
                time.sleep(1)
    finally:
        for _, process, handle in active:
            if process.poll() is None:
                process.terminate()
            handle.close()
    if any(r['returncode'] for r in results):
        raise RuntimeError(f'training failed: {results}')


if __name__ == '__main__':
    main()
