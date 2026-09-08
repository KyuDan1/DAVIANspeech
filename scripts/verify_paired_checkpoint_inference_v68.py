#!/usr/bin/env python3
"""Compare completed/smoke paired-model dev predictions with file-local inference.

Only checks the already-authorized development predictions. No locked data,
new thresholds, checkpoint selection, or leaderboard score estimates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from src.pipeline import load_audio, find_audio_files
from train_three_stream_anchor_residual import authorized_partitions


def resolve_development(prediction, partition_config):
    keys = ['DATASET', 'ID']
    if prediction[keys].duplicated().any():
        raise ValueError('duplicate prediction keys')
    paths = {}
    for partition in authorized_partitions(partition_config, 'development', list(prediction.DATASET.unique())):
        truth = pd.read_csv(partition.truth_path, dtype={'ID': str})
        wanted = prediction[prediction.DATASET.eq(partition.name)].ID
        if not set(wanted).issubset(set(truth.ID)):
            raise ValueError('prediction contains IDs outside authorized development')
        audio = {}
        for p in find_audio_files(partition.truth_path.parent / 'audio'):
            if p.stem in audio:
                raise ValueError('ambiguous audio stem')
            audio[p.stem] = p
        for identity in wanted:
            if identity not in audio:
                raise FileNotFoundError(identity)
            paths[(partition.name, identity)] = audio[identity]
    return [paths[key] for key in map(tuple, prediction[keys].to_numpy())]


def compare_arrays(expected, independent, reversed_scores, tolerance=5e-4):
    arrays = [np.asarray(x, dtype=float) for x in [expected, independent, reversed_scores]]
    if not len(arrays[0]) or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError('same nonempty shapes required')
    if any(not np.isfinite(a).all() or (a < 0).any() or (a > 1).any() for a in arrays):
        raise ValueError('finite probabilities required')
    difference = np.abs(arrays[0] - arrays[1])
    order_exact = np.array_equal(arrays[1], arrays[2])
    return dict(rows=len(arrays[0]), maximum_absolute_difference=float(difference.max()),
        mean_absolute_difference=float(difference.mean()), tolerance=tolerance,
        saved_evaluation_close=bool(difference.max() <= tolerance),
        file_order_bit_exact=bool(order_exact),
        pass_check=bool(difference.max() <= tolerance and order_exact))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not (args.run / 'complete.json').is_file():
        raise ValueError('run is incomplete; do not verify changing checkpoints')
    checkpoint_path = args.run / 'best.pt'
    prediction_path = args.run / 'dev_predictions.csv'
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [checkpoint_path, prediction_path]}
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    smoke = checkpoint.get('smoke', True)
    if smoke and not args.allow_smoke:
        raise ValueError('smoke checkpoint requires explicit --allow-smoke')
    prediction = pd.read_csv(prediction_path, dtype={'DATASET': str, 'ID': str})
    if len(prediction) != checkpoint['provenance']['development_rows']:
        raise ValueError('prediction count differs from training manifest')
    paths = resolve_development(prediction, ROOT / checkpoint['config']['partition_config'])
    device = torch.device('cuda')
    torch.set_num_threads(6)
    if checkpoint['model_type'] == 'lowband_music_v67':
        from src.lowband_music_inference_v67 import load_model, predict_one_audio
        model, config = load_model(checkpoint_path, device, allow_smoke=args.allow_smoke)
        task = 'MUSIC'
        def infer(path):
            return predict_one_audio(model, load_audio(path), config, device)
    elif checkpoint['model_type'] == 'full_coverage_wpt_file_v60':
        from src.full_coverage_wpt_inference import load_full_coverage_model, predict_one_audio
        model, config = load_full_coverage_model(ROOT / checkpoint['config']['model_dir'], checkpoint_path,
                                                device, allow_smoke=args.allow_smoke)
        task = 'FILE'
        def infer(path):
            return predict_one_audio(model, load_audio(path), config, device)[0]
    else:
        raise ValueError('unsupported checkpoint architecture')
    actual = [infer(path) for path in paths]
    reversed_scores = [infer(path) for path in reversed(paths)][::-1]
    expected = prediction[f'{task}_FAKE_PROB'].to_numpy(dtype=float)
    report = compare_arrays(expected, actual, reversed_scores)
    for p, h in hashes.items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h:
            raise ValueError('checkpoint or saved predictions changed during verification')
    report.update(scope='inference equivalence only, not generalization or official score',
        smoke=smoke, task=task, artifacts=hashes,
        verifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inference_source_sha256=hashlib.sha256(Path(sys.modules[predict_one_audio.__module__].__file__).read_bytes()).hexdigest(),
        gpu=torch.cuda.get_device_name())
    # Record per-ID values; no input audio or evaluation labels are changed.
    result = prediction[['DATASET', 'ID']].copy()
    result['SAVED_PROB'] = expected
    result['INDEPENDENT_PROB'] = actual
    result['REVERSED_PROB'] = reversed_scores
    args.output.mkdir(parents=True)
    result.to_csv(args.output / 'predictions.csv', index=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if not report['pass_check']:
        raise RuntimeError('independent inference differs; investigate before deployment')


if __name__ == '__main__':
    main()
