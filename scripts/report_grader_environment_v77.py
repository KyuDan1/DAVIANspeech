#!/usr/bin/env python3
"""Check a separate grader-version core stack; not a full server/L4 replica."""
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    expected = {}
    for line in (ROOT / 'configs/requirements-grader-v77.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            name, version = line.split('==')
            expected[name] = version
    actual = {name: importlib.metadata.version(name) for name in expected}
    mismatch = {name: dict(expected=version, actual=actual[name])
                for name, version in expected.items() if actual[name] != version}
    if mismatch:
        raise RuntimeError(f'core package mismatch: {mismatch}')
    import numpy as np
    import pandas as pd
    import librosa
    import torch
    import torchaudio
    torch.set_num_threads(2)
    # Exercises native torch/torchaudio libraries and NumPy/pandas binary imports.
    audio = torch.linspace(-1, 1, 1600)
    downsampled = torchaudio.functional.resample(audio, 16000, 8000)
    if not torch.isfinite(downsampled).all() or len(downsampled) != 800:
        raise ValueError('torchaudio native functional smoke failed')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        matrix = torch.eye(32, device='cuda')
        result = matrix @ matrix
    if not torch.isfinite(result).all() or not torch.equal(result.float(), matrix):
        raise ValueError('CUDA bf16 smoke failed')
    report = dict(status='complete_core_environment_smoke', executable=sys.executable,
        python=platform.python_version(), platform=platform.platform(), packages=actual,
        torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(), native_torchaudio_import_and_resample=True,
        pandas_numpy_abi_import=True, cuda_bf16_matmul=True,
        limitations=['Python 3.11.16 locally vs supplied grader 3.11.15',
            'B200 research hardware, not L4; CPU/RAM limits not enforced',
            'Only relevant pinned core packages replicated; unspecified transitive dependencies may differ',
            'No model probability equivalence, submission installation time, or official score implied'],
        automatic_submission_allowed=False)
    output = ROOT / 'reports/grader_environment_v77'
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
