#!/usr/bin/env python3
"""Paired descriptive summaries of the pre-frozen v70 comparison; no tuning."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd

from src.evaluate_diagnostic import score_frame, PREDICTION_COLUMNS, LABEL_COLUMNS
from train_paired_wpt_file_v60 import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'reports/long_component_stress_v70_comparison')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frozen = json.loads((args.runs / 'frozen.json').read_text())
    truth = pd.read_csv(Path(frozen['bank']) / 'truth.csv', dtype={'ID': str})
    if sha256(Path(frozen['bank']) / 'truth.csv') != frozen['truth_sha256']:
        raise ValueError('changed frozen truth')
    frames, hashes = {}, {}
    for name, record in frozen['candidates'].items():
        path = args.runs / name
        if not (path / 'report.json').is_file():
            raise RuntimeError('all frozen candidates must finish before comparison')
        report = json.loads((path / 'report.json').read_text())
        if report['checkpoint_sha256'] != record['sha256'] or report['frozen_sha256'] != sha256(args.runs / 'frozen.json'):
            raise ValueError('comparison candidates differ from frozen manifest')
        prediction = pd.read_csv(path / 'predictions.csv', dtype={'ID': str})
        if prediction.ID.duplicated().any() or set(prediction.ID) != set(truth.ID):
            raise ValueError('prediction identities do not match truth')
        prediction = prediction.set_index('ID').loc[truth.ID].reset_index()
        np.testing.assert_allclose(prediction[LABEL_COLUMNS].to_numpy(float), truth[LABEL_COLUMNS].to_numpy(float), equal_nan=True)
        values = prediction[PREDICTION_COLUMNS].to_numpy(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError('invalid probabilities')
        frames[name] = prediction
        hashes[str(path / 'predictions.csv')] = sha256(path / 'predictions.csv')
    results = []
    for name, frame in frames.items():
        for axes in [['AUDIO_TYPE'], ['DURATION', 'AUDIO_TYPE'], ['DURATION', 'AUDIO_TYPE', 'CHANNEL'],
                     ['DURATION', 'MIX_MODE', 'CHANNEL']]:
            for keys, selected in frame.groupby(axes):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                results.append(dict(candidate=name, axis='|'.join(axes), group='|'.join(map(str, keys)), **score_frame(selected)))
    table = pd.DataFrame(results)
    args.output.mkdir(parents=True)
    table.to_csv(args.output / 'by_type_length_layout.csv', index=False)
    control = table.loc[table.candidate.eq('eat_control')].set_index(['axis', 'group'])
    adapted = table.loc[table.candidate.eq('eat_adapted')].set_index(['axis', 'group'])
    paired = control[['N', 'FILE_EER', 'VOICE_EER', 'MUSIC_EER', 'ADS']].join(
        adapted[['FILE_EER', 'VOICE_EER', 'MUSIC_EER', 'ADS']], lsuffix='_control', rsuffix='_adapted')
    for task in ['FILE_EER', 'VOICE_EER', 'MUSIC_EER', 'ADS']:
        paired[task + '_delta'] = paired[task + '_adapted'] - paired[task + '_control']
    paired.to_csv(args.output / 'paired_differences.csv')
    report = dict(scope='paired synthetic mechanism stress only', independent_source_groups=int(truth.GROUP_ID.nunique()),
                  rows=len(truth), candidates=list(frames), selection_allowed=False,
                  generalization_verified=False, automatic_submission_allowed=False,
                  artifacts_sha256=hashes, frozen_sha256=sha256(args.runs / 'frozen.json'))
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(paired.loc[paired.index.get_level_values('axis') == 'DURATION|AUDIO_TYPE'].to_string(), flush=True)


if __name__ == '__main__':
    main()
