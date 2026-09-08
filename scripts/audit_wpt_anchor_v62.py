#!/usr/bin/env python3
"""Verify packaged WPT assets against the submitted ZIP and re-export dev predictions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def digest_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    package = ROOT / 'v50_v57m_challenger_v2'
    assets = [package / 'script.py', *sorted((package / 'model/src').glob('*.py')),
              *sorted((package / 'model/spectra-aasist').glob('*'))]
    assets = [p for p in assets if p.is_file()]
    assets.extend(sorted((package / 'model/spectra-aasist/xlsr_config').glob('*.json')))
    checks = {}
    with zipfile.ZipFile(package.with_suffix('.zip')) as archive:
        for path in assets:
            relative = str(path.relative_to(package))
            with path.open('rb') as source:
                current = digest_stream(source)
            with archive.open(relative) as source:
                zipped = digest_stream(source)
            if current != zipped:
                raise ValueError(f'package directory differs from submitted ZIP: {relative}')
            checks[relative] = current
    print(json.dumps(dict(verified_zip_assets=len(checks))), flush=True)
    # Use the actual packaged inference code, not the dirty working-tree copy.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    sys.path.insert(0, str(package / 'model/src'))
    from wpt_spectra_inference import predict_wpt_tasks
    from pipeline import find_audio_files
    sample = pd.read_csv(ROOT / 'data/eval/codec_mixed_dev_v4/truth.csv', usecols=['ID'], dtype=str)
    paths = {p.stem: p for p in find_audio_files(ROOT / 'data/eval/codec_mixed_dev_v4/audio')}
    if sample.ID.duplicated().any() or set(sample.ID) != set(paths):
        raise ValueError('authorized development audio keys differ')
    started = time.monotonic()
    model_dir = package / 'model/spectra-aasist'
    probabilities = predict_wpt_tasks([paths[i] for i in sample.ID], model_dir,
        model_dir / 'wpt_spectra_multitask.pt', device='cuda', file_batch_size=6,
        file_views=5, file_temperature=2.)
    elapsed = time.monotonic() - started
    cache = ROOT / 'reports/wpt_spectra_v1/codec_mixed_dev_v4_seed06_5view_t2.npz'
    with np.load(cache, allow_pickle=False) as archive:
        ids, values = archive['ids'].astype(str), archive['probabilities']
    if len(set(ids)) != len(ids) or set(ids) != set(sample.ID):
        raise ValueError('historical dev cache ID mismatch')
    positions = {item: i for i, item in enumerate(ids)}
    old = values[[positions[item] for item in sample.ID]]
    differences = np.abs(probabilities - old)
    args.output.mkdir(parents=True)
    result = sample.copy()
    result['DATASET'] = 'codec_mixed_dev_v4'
    for i, task in enumerate(['VOICE', 'MUSIC', 'FILE']):
        result[f'{task}_FAKE_PROB'] = probabilities[:, i]
    result.to_csv(args.output / 'predictions.csv', index=False)
    report = dict(scope='Exact submitted WPT module on authorized development, not new detector score',
        zip_assets_sha256=checks, dev_rows=len(result), seconds=elapsed,
        historical_cache_max_abs_difference=float(differences.max()),
        historical_cache_file_max_abs_difference=float(differences[:, 2].max()),
        historical_cache_close=bool(np.allclose(probabilities, old, atol=1e-5, rtol=1e-5)),
        file_views=5, file_temperature=2., wpt_weight=.4)
    (args.output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'zip_assets_sha256'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
