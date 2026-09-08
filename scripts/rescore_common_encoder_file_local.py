#!/usr/bin/env python3
"""One fixed completed readout, full authorized-development file-local scoring.

This corrects evaluation execution, not checkpoint/weight selection or tuning.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
import torch

from src.common_encoder_probe_inference import CommonProbePredictor
from src.evaluate_diagnostic import PREDICTION_COLUMNS, LABEL_COLUMNS
from src.pipeline import load_audio
from train_common_encoder_probe import metrics
from train_paired_wpt_file_v60 import sha256
from train_three_stream_anchor_residual import authorized_partitions
from verify_paired_checkpoint_inference_v68 import resolve_development


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pooling', '--variant', choices=['mean', 'attention', 'control', 'adapted', 'file_only', 'dense_supervised', 'adapted_dense'], default='mean')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not (args.run / 'completed.json').is_file():
        raise RuntimeError('only completed frozen readouts may be rescored')
    checkpoint = args.run / args.pooling / 'head.pt'
    saved = args.run / args.pooling / 'development_predictions.csv'
    hashes = {str(p): sha256(p) for p in [checkpoint, saved]}
    frame = pd.read_csv(saved, dtype={'DATASET': str, 'ID': str})
    # Labels for scoring must still agree with registered development truth.
    for partition in authorized_partitions(ROOT / 'configs/data_partitions.yaml', 'development', list(frame.DATASET.unique())):
        truth = pd.read_csv(partition.truth_path, dtype={'ID': str}).set_index('ID')
        rows = frame.loc[frame.DATASET.eq(partition.name)].set_index('ID')
        np.testing.assert_allclose(rows[LABEL_COLUMNS].to_numpy(float),
                                   truth.loc[rows.index, LABEL_COLUMNS].to_numpy(float), equal_nan=True)
    paths = resolve_development(frame, ROOT / 'configs/data_partitions.yaml')
    expected = frame[PREDICTION_COLUMNS].to_numpy(float).copy()
    torch.set_num_threads(4)
    predictor = CommonProbePredictor(checkpoint, ROOT)
    start = time.monotonic()
    actual = []
    for index, path in enumerate(paths):
        actual.append(predictor(load_audio(path)))
        if index % 250 == 0:
            print(json.dumps(dict(files=index + 1, total=len(paths), seconds=time.monotonic() - start)), flush=True)
    actual = np.stack(actual)
    # Limited deterministic repeat check; never describe it as a full reverse audit.
    repeats = sorted(set([0, len(paths) // 2, len(paths) - 1]))
    for index in repeats[::-1]:
        np.testing.assert_array_equal(actual[index], predictor(load_audio(paths[index])))
    if any(sha256(Path(p)) != digest for p, digest in hashes.items()):
        raise RuntimeError('checkpoint/predictions changed while evaluating')
    frame[PREDICTION_COLUMNS] = actual
    summary, slices = metrics(frame)
    report = dict(scope='fixed checkpoint full file-local development, NOT official score',
                  rows=len(frame), pooling=args.pooling, encoder=predictor.encoder.name,
                  elapsed_seconds=time.monotonic() - start, repeated_file_indices=repeats,
                  repeated_files_bit_exact=True, checkpoint_reselected=False,
                  max_batch_vs_file_difference=float(np.abs(actual - expected).max()),
                  summary=summary, artifacts_sha256=hashes, automatic_submission_allowed=False,
                  code_sha256={str(p): sha256(p) for p in [Path(__file__), ROOT / 'src/common_encoder_probe.py',
                               ROOT / 'src/common_encoder_probe_inference.py']})
    frame.to_csv(args.output / 'development_predictions.csv', index=False)
    report['file_local_prediction_sha256'] = sha256(args.output / 'development_predictions.csv')
    slices.to_csv(args.output / 'development_slices.csv', index=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
