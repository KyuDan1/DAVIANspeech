#!/usr/bin/env python3
"""Completed common-probe inference equivalence on authorized development."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import pandas as pd
import torch

from src.common_encoder_probe_inference import CommonProbePredictor
from src.evaluate_diagnostic import PREDICTION_COLUMNS
from src.pipeline import load_audio
from scripts.verify_paired_checkpoint_inference_v68 import resolve_development, compare_arrays
from scripts.train_paired_wpt_file_v60 import sha256


def select_rows(frame, full=False):
    if full:
        return frame.copy()
    # Deterministic infrastructure sample: two rows per domain and presence /
    # component-label cell. This is NOT an accuracy evaluation subset.
    keys = ['DATASET', 'VOICE_PRESENT', 'MUSIC_PRESENT', 'VOICE_FAKE', 'MUSIC_FAKE']
    return frame.sort_values(['DATASET', 'ID']).groupby(keys, dropna=False).head(2).copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--pooling', '--variant', choices=['mean', 'attention', 'control', 'adapted', 'file_only', 'dense_supervised', 'adapted_dense'], required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full-development', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not (args.run / 'completed.json').is_file():
        raise ValueError('run is not complete')
    checkpoint = args.run / args.pooling / 'head.pt'
    predictions = args.run / args.pooling / 'development_predictions.csv'
    hashes = {str(p): sha256(p) for p in [checkpoint, predictions]}
    frame = select_rows(pd.read_csv(predictions, dtype={'ID': str, 'DATASET': str}), args.full_development)
    paths = resolve_development(frame, ROOT / 'configs/data_partitions.yaml')
    torch.set_num_threads(4)
    predictor = CommonProbePredictor(checkpoint, ROOT, allow_smoke=args.allow_smoke)
    actual = np.stack([predictor(load_audio(path)) for path in paths])
    reverse = np.stack([predictor(load_audio(path)) for path in paths[::-1]])[::-1]
    expected = frame[PREDICTION_COLUMNS].to_numpy(float)
    report = compare_arrays(expected, actual, reverse, tolerance=5e-4)
    report.update(scope='inference equivalence, NOT accuracy or generalization',
                  full_development=args.full_development, artifacts_sha256=hashes,
                  per_column_max_abs_difference=dict(zip(PREDICTION_COLUMNS, np.abs(actual - expected).max(0).tolist())),
                  code_sha256={str(p): sha256(p) for p in [Path(__file__), ROOT / 'src/common_encoder_probe_inference.py', ROOT / 'src/common_encoder_probe.py']})
    if any(sha256(Path(path)) != digest for path, digest in hashes.items()):
        raise RuntimeError('checkpoint/predictions changed during verification')
    table = frame[['DATASET', 'ID']].copy()
    for index, column in enumerate(PREDICTION_COLUMNS):
        table[f'saved_{column}'] = expected[:, index]
        table[f'independent_{column}'] = actual[:, index]
        table[f'reverse_{column}'] = reverse[:, index]
    args.output.mkdir(parents=True)
    table.to_csv(args.output / 'predictions.csv', index=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if not report['pass_check']:
        raise RuntimeError('file-local inference differs; investigate, do not submit')


if __name__ == '__main__':
    main()
