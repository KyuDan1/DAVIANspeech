#!/usr/bin/env python3
"""Audit completed v60 variants against each other and the exact official File anchor.

Never scores an incomplete checkpoint, locked bank, or a weight sweep. Passing
development only freezes a research candidate, not permission to submit.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_prospective_mixed_phone_v3 import sha256_file

VARIANTS = ('paired_bce', 'paired_consistency')
ANCHOR = ROOT / 'reports/three_stream_all_type_v57_strict/v50_authorized_v2/v1_strict__music_only_predictions.csv'


def completed_variant(directory):
    required = ['complete.json', 'best.pt', 'dev_predictions.csv', 'history.csv', 'manifest.json']
    for name in required:
        if not (directory / name).is_file():
            raise ValueError(f'run is not complete: {directory / name}')
    completion = json.loads((directory / 'complete.json').read_text())
    checkpoint = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
    if checkpoint.get('smoke', True) or checkpoint.get('model_type') != 'full_coverage_wpt_file_v60':
        raise ValueError('not an eligible full v60 checkpoint')
    history = pd.read_csv(directory / 'history.csv')
    selected = history.loc[history.epoch.eq(checkpoint['best_epoch'])]
    if len(selected) != 1:
        raise ValueError('selected epoch not uniquely present in history')
    score = float(checkpoint['selection'])
    if not np.isfinite(score) or not np.isclose(score, completion['best_selection'], rtol=0, atol=1e-12):
        raise ValueError('checkpoint/completion selection differs')
    if not np.isclose(score, float(selected.iloc[0].selection), rtol=0, atol=1e-12):
        raise ValueError('checkpoint/history selection differs')
    prediction = pd.read_csv(directory / 'dev_predictions.csv')
    if len(prediction) != checkpoint['provenance']['development_rows']:
        raise ValueError('development prediction row count differs')
    return dict(best_epoch=int(checkpoint['best_epoch']), selection=score,
                pooled_file_eer=float(selected.iloc[0].pooled_file_eer),
                files={name: sha256_file(directory / name) for name in required})


def choose_candidate(variants, gates):
    eligible = [name for name in VARIANTS if gates[name]['pass_gate']]
    # Predeclared full-development selection; no use of locked scores.
    return max(eligible, key=lambda name: variants[name]['selection']) if eligible else None


def wait_for_training(runs, variants=VARIANTS):
    launch = json.loads((runs / 'launch.json').read_text())
    if {r['name'] for r in launch['runs']} != set(variants):
        raise ValueError('unexpected training launch manifest')
    previous_live = None
    while True:
        live = []
        for run in launch['runs']:
            process = Path('/proc') / str(run['pid']) / 'cmdline'
            try:
                command = process.read_bytes().split(b'\x00')
            except FileNotFoundError:
                continue
            actual = [item.decode() for item in command if item]
            if not actual:  # Exited zombie; completion files still must validate.
                continue
            if actual != run['command']:
                raise ValueError('PID was reused or training command changed; manual inspection required')
            live.append(run['pid'])
        if not live:
            return
        if live != previous_live:
            print(json.dumps(dict(waiting_for_verified_training_pids=live)), flush=True)
            previous_live = live
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'reports/paired_wpt_file_v60')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-for-training', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.wait_for_training:
        wait_for_training(args.runs)
    variants = {name: completed_variant(args.runs / name) for name in VARIANTS}
    args.output.mkdir(parents=True)
    common = [sys.executable, str(ROOT / 'scripts/evaluate_music_specialist_v58.py'),
              '--task', 'file', '--matrix', str(ROOT / 'configs/paired_wpt_file_v60.yaml')]
    subprocess.run(common + ['--incumbent', str(args.runs / 'paired_bce/dev_predictions.csv'),
                   '--candidate', str(args.runs / 'paired_consistency/dev_predictions.csv'),
                   '--output', str(args.output / 'consistency_ablation.json')], cwd=ROOT, check=True)
    gates = {}
    for name in VARIANTS:
        report_path = args.output / f'{name}_vs_anchor.json'
        subprocess.run(common + ['--datasets', 'codec_mixed_dev_v4', '--incumbent', str(ANCHOR),
                       '--candidate', str(args.runs / name / 'dev_predictions.csv'),
                       '--output', str(report_path)], cwd=ROOT, check=True)
        gates[name] = json.loads(report_path.read_text())['decision']
    # Ensure training/output files did not change while comparisons were run.
    for name, metadata in variants.items():
        for filename, digest in metadata['files'].items():
            if sha256_file(args.runs / name / filename) != digest:
                raise ValueError('completed artifact changed during comparison')
    result = dict(stage='development_decision_only', variants=variants, anchor_gates=gates,
        frozen_candidate=choose_candidate(variants, gates), anchor_predictions_sha256=sha256_file(ANCHOR),
        automatic_submission_allowed=False, official_score_improvement_verified=False,
        remaining_if_pass=['independent_file_inference_equivalence', 'prospective_all_type_confirmation',
                           'long_voice_confirmation', 'full_pipeline_L4_runtime', 'offline_package_checks'])
    (args.output / 'decision.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
