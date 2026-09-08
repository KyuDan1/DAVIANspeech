#!/usr/bin/env python3
"""Choose completed v67 variant by full dev, then audit a frozen 10% Music blend."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from finalize_paired_wpt_v60 import wait_for_training
from build_prospective_mixed_phone_v3 import sha256_file

VARIANTS = ('magnitude', 'magnitude_phase')


def completed(directory):
    required = ['complete.json', 'manifest.json', 'best.pt', 'history.csv', 'dev_predictions.csv']
    if any(not (directory / p).is_file() for p in required):
        raise ValueError('lowband run is incomplete')
    checkpoint = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
    if checkpoint.get('smoke', True) or checkpoint.get('model_type') != 'lowband_music_v67':
        raise ValueError('not an eligible full lowband checkpoint')
    if checkpoint['phase'] is not checkpoint['config']['variants'][checkpoint['variant']]['phase']:
        raise ValueError('phase/config mismatch')
    history = pd.read_csv(directory / 'history.csv')
    selected = history[history.epoch.eq(checkpoint['best_epoch'])]
    completion = json.loads((directory / 'complete.json').read_text())
    if len(selected) != 1 or not np.isfinite(checkpoint['selection']):
        raise ValueError('invalid selected epoch')
    if any(not np.isclose(checkpoint['selection'], x, atol=1e-12, rtol=0) for x in
           [float(selected.iloc[0].selection), completion['best_selection']]):
        raise ValueError('selection differs between artifacts')
    if len(pd.read_csv(directory / 'dev_predictions.csv')) != checkpoint['provenance']['development_rows']:
        raise ValueError('development row count differs')
    return dict(best_epoch=int(checkpoint['best_epoch']), selection=checkpoint['selection'],
        pooled_music_eer=float(selected.iloc[0].pooled_music_eer),
        files={p: sha256_file(directory / p) for p in required})


def blend_music(anchor, candidate, weight):
    if not 0 <= weight <= 1:
        raise ValueError('weight outside [0,1]')
    keys = ['DATASET', 'ID']
    for frame in [anchor, candidate]:
        if frame[keys].duplicated().any():
            raise ValueError('duplicate keys')
    ids = list(map(tuple, anchor[keys].to_numpy()))
    if set(ids) != set(map(tuple, candidate[keys].to_numpy())):
        raise ValueError('exact matching keys required')
    aligned = candidate.set_index(keys).loc[ids]
    values = [anchor.MUSIC_FAKE_PROB.to_numpy(dtype=float), aligned.MUSIC_FAKE_PROB.to_numpy(dtype=float)]
    for p in values:
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError('invalid probabilities')
    result = anchor.copy()
    if weight == 0:
        return result
    old, new = [np.log(np.clip(p, 1e-5, 1-1e-5)) - np.log1p(-np.clip(p, 1e-5, 1-1e-5)) for p in values]
    probability = np.exp(-np.logaddexp(0, -((1-weight)*old + weight*new)))
    result['MUSIC_FAKE_PROB'] = [f'{x:.10f}' for x in probability]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-for-training', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config_path = ROOT / 'configs/lowband_music_v67.yaml'
    config = yaml.safe_load(config_path.read_text())
    runs = ROOT / 'reports/lowband_music_v67'
    if args.wait_for_training:
        wait_for_training(runs, variants=VARIANTS)
    info = {name: completed(runs / name) for name in VARIANTS}
    # Tie goes to simpler magnitude; no choice using 600-row blend results.
    chosen = max(VARIANTS, key=lambda name: info[name]['selection'])
    for name in VARIANTS:
        manifest = json.loads((runs / name / 'manifest.json').read_text())
        if manifest['config'] != config or manifest['config_sha256'] != sha256_file(config_path):
            raise ValueError('training no longer matches frozen config')
    args.output.mkdir(parents=True)
    anchor_path = ROOT / 'reports/three_stream_all_type_v57_strict/v50_authorized_v2/v1_strict__music_only_predictions.csv'
    candidate_path = runs / chosen / 'dev_predictions.csv'
    anchor, prediction = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in [anchor_path, candidate_path]]
    prediction = prediction[prediction.DATASET.eq('codec_mixed_dev_v4')]
    blended = blend_music(anchor, prediction, config['decision']['music_logit_weight'])
    unchanged = [c for c in anchor if c != 'MUSIC_FAKE_PROB']
    if not blended[unchanged].equals(anchor[unchanged]):
        raise ValueError('other outputs changed')
    blended_path = args.output / 'blended_predictions.csv'
    blended.to_csv(blended_path, index=False)
    evaluator = [sys.executable, str(ROOT / 'scripts/evaluate_music_specialist_v58.py'),
                 '--matrix', str(config_path), '--task', 'music', '--datasets', 'codec_mixed_dev_v4',
                 '--incumbent', str(anchor_path)]
    for candidate, filename in [(candidate_path, 'standalone_diagnostic.json'), (blended_path, 'blend_primary.json')]:
        subprocess.run(evaluator + ['--candidate', str(candidate), '--output', str(args.output / filename)], cwd=ROOT, check=True)
    primary = json.loads((args.output / 'blend_primary.json').read_text())['decision']
    for name in VARIANTS:
        if any(sha256_file(runs / name / p) != h for p, h in info[name]['files'].items()):
            raise ValueError('completed artifacts changed during evaluation')
    report = dict(stage='development_only', chosen=chosen, variants=info, primary_decision=primary,
        music_logit_weight=config['decision']['music_logit_weight'], other_four_outputs_text_exact=True,
        official_improvement_verified=False, automatic_submission_allowed=False,
        remaining_if_pass=['source_and_generator_prospective', 'independent_file_inference', 'L4_runtime', 'offline_package'],
        files={str(p): sha256_file(p) for p in [config_path, anchor_path, candidate_path, blended_path]})
    (args.output / 'decision.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(chosen=chosen, primary=primary)), flush=True)


if __name__ == '__main__':
    main()
