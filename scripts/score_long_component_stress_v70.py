#!/usr/bin/env python3
"""Freeze candidates first, then score v70 only as a mechanism diagnostic."""
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
import yaml

from src.common_encoder_probe_inference import CommonProbePredictor
from src.evaluate_diagnostic import PREDICTION_COLUMNS, score_frame
from src.pipeline import load_audio
from train_paired_wpt_file_v60 import sha256


def validate_bank(bank):
    roles = yaml.safe_load((ROOT / 'configs/data_partitions.yaml').read_text())
    truth = bank / 'truth.csv'
    if str(truth.resolve().relative_to(ROOT)) not in roles['stress_eval']:
        raise ValueError('bank must be registered as stress_eval, never train/development')
    validation = json.loads((bank / 'validation.json').read_text())
    if not validation['passed'] or validation['selection_allowed'] or sha256(truth) != validation['truth_sha256']:
        raise ValueError('invalid/changed stress bank')
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='mode', required=True)
    freeze = subs.add_parser('freeze')
    freeze.add_argument('--bank', type=Path, default=ROOT / 'data/eval/long_component_stress_v70')
    freeze.add_argument('--output', type=Path, required=True)
    freeze.add_argument('--candidate', action='append', required=True, help='name=checkpoint')
    score = subs.add_parser('score')
    score.add_argument('--frozen', type=Path, required=True)
    score.add_argument('--candidate', required=True)
    score.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode == 'freeze':
        validation = validate_bank(args.bank)
        candidates = {}
        for item in args.candidate:
            name, checkpoint = item.split('=', 1)
            path = (ROOT / checkpoint).resolve()
            if name in candidates or not (path.parent.parent / 'completed.json').is_file():
                raise ValueError('duplicate candidate or incomplete training')
            state = torch.load(path, map_location='cpu', weights_only=False)
            if state.get('smoke', False):
                raise ValueError('smoke is not an eligible comparison model')
            candidates[name] = dict(path=str(path), sha256=sha256(path), epoch=state['epoch'])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(bank=str(args.bank.resolve()), candidates=candidates,
            truth_sha256=validation['truth_sha256'], purpose='fixed-candidate mechanism stress, not model selection',
            selection_allowed=False, automatic_submission_allowed=False), indent=2) + '\n')
        print(json.dumps(candidates), flush=True)
        return
    frozen = json.loads(args.frozen.read_text())
    frozen_hash = sha256(args.frozen)
    if frozen['selection_allowed']:
        raise ValueError('not a stress-only freeze')
    bank = Path(frozen['bank'])
    validation = validate_bank(bank)
    if validation['truth_sha256'] != frozen['truth_sha256']:
        raise ValueError('bank changed after candidates were fixed')
    record = frozen['candidates'][args.candidate]
    path = Path(record['path'])
    if sha256(path) != record['sha256']:
        raise ValueError('checkpoint changed after freeze')
    torch.set_num_threads(4)
    predictor = CommonProbePredictor(path, ROOT)
    truth = pd.read_csv(bank / 'truth.csv', dtype={'ID': str})
    args.output.mkdir(parents=True)
    predictions = []
    started = time.monotonic()
    for index, identity in enumerate(truth.ID):
        predictions.append(predictor(load_audio(bank / 'audio' / (identity + '.flac'))))
        if index % 100 == 0:
            print(json.dumps(dict(candidate=args.candidate, files=index + 1, seconds=time.monotonic() - started)), flush=True)
    values = np.stack(predictions)
    for index in sorted(set([0, len(truth) // 2, len(truth) - 1]), reverse=True):
        again = predictor(load_audio(bank / 'audio' / (truth.ID.iloc[index] + '.flac')))
        np.testing.assert_array_equal(again, values[index])
    if sha256(path) != record['sha256'] or sha256(args.frozen) != frozen_hash:
        raise ValueError('candidate or frozen protocol changed while scoring')
    truth[PREDICTION_COLUMNS] = values
    truth.to_csv(args.output / 'predictions.csv', index=False)
    rows = []
    for axes in [['DURATION'], ['MIX_MODE'], ['CHANNEL'], ['DURATION', 'MIX_MODE'], ['DURATION', 'MIX_MODE', 'CHANNEL']]:
        for keys, selected in truth.groupby(axes):
            if not isinstance(keys, tuple):
                keys = (keys,)
            rows.append(dict(axis='|'.join(axes), group='|'.join(map(str, keys)), **score_frame(selected)))
    pd.DataFrame(rows).to_csv(args.output / 'by_condition.csv', index=False)
    calibration = []
    for keys, selected in truth.groupby(['DURATION', 'MIX_MODE', 'COMPONENT_CASE', 'CHANNEL']):
        row = dict(zip(['DURATION', 'MIX_MODE', 'COMPONENT_CASE', 'CHANNEL'], keys))
        for task in ['FILE', 'VOICE', 'MUSIC']:
            active = selected if task == 'FILE' else selected.loc[selected[task + '_PRESENT'].eq(1)]
            if len(active):
                probabilities = active[task + '_FAKE_PROB']
                row[task + '_MEDIAN_PROB'] = float(probabilities.median())
                row[task + '_ERROR_AT_FIXED_HALF'] = float(((probabilities >= .5).astype(int) != active[task + '_FAKE']).mean())
        calibration.append(row)
    pd.DataFrame(calibration).to_csv(args.output / 'fixed_half_diagnostics.csv', index=False)
    report = dict(candidate=args.candidate, rows=len(truth), overall=score_frame(truth),
        elapsed_seconds=time.monotonic() - started, selection_allowed=False, automatic_submission_allowed=False,
        checkpoint_sha256=record['sha256'], frozen_sha256=frozen_hash, truth_sha256=frozen['truth_sha256'],
        repeated_files=3, repeated_files_bit_exact=True,
        scope='development-derived mechanism stress, not natural-call or source-OOD generalization')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
