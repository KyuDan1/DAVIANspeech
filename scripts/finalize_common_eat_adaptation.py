#!/usr/bin/env python3
"""Close paired EAT development evidence; no fitting or submission."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
import torch

from src.evaluate_diagnostic import PREDICTION_COLUMNS, score_frame
from train_common_encoder_probe import metrics
from train_paired_wpt_file_v60 import sha256
from compare_common_encoder_probe import align_truth, paired_bootstrap, typed_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = ROOT / 'reports/common_eat_adaptation/full'
    original = ROOT / 'reports/common_encoder_probe/eat_large'
    local = ROOT / 'reports/common_eat_adaptation/full_file_local'
    for path in [run / 'completed.json', original / 'completed.json', local / 'report.json']:
        if not path.is_file():
            raise RuntimeError(f'completed training and full independent evaluation required: {path}')
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest['smoke']:
        raise ValueError('smoke cannot be finalized as a full model')
    for source, digest in manifest['code_sha256'].items():
        if sha256(run / 'source_snapshot' / Path(source).name) != digest:
            raise ValueError('training source snapshot corrupted')
    for filename in ['train_ids.csv', 'development_ids.csv']:
        if sha256(run / filename) != sha256(original / filename):
            raise ValueError('ordered experiment identities differ')
    for epoch in range(1, 7):
        if not np.array_equal(np.load(run / f'epoch_{epoch}_train_draws.npy'), np.load(original / f'epoch_{epoch}_train_draws.npy')):
            raise ValueError('matched sampler draws differ')
    old = pd.read_csv(original / 'mean/development_predictions.csv', dtype={'ID': str, 'DATASET': str})
    control = pd.read_csv(run / 'control/development_predictions.csv', dtype={'ID': str, 'DATASET': str})
    if not np.array_equal(old[PREDICTION_COLUMNS].to_numpy(), control[PREDICTION_COLUMNS].to_numpy()):
        raise ValueError('paired control does not reproduce the previous frozen baseline')
    verification = json.loads((local / 'report.json').read_text())
    for path, digest in verification['artifacts_sha256'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('independent evaluation checkpoint/predictions changed')
    adapted_path = local / 'development_predictions.csv'
    adapted = pd.read_csv(adapted_path, dtype={'ID': str, 'DATASET': str})
    aligned = align_truth(control.drop(columns=PREDICTION_COLUMNS), adapted)
    if verification['rows'] != len(control) or not verification['repeated_files_bit_exact']:
        raise ValueError('incomplete independent evaluation')
    checkpoint_path = run / 'adapted/head.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    history = json.loads((run / 'history.json').read_text())
    expected = max((row for row in history if row['variant'] == 'adapted'), key=lambda row: row['selection'])
    if expected['epoch'] != checkpoint['epoch']:
        raise ValueError('checkpoint selection does not follow the declared criterion')
    args.output.mkdir(parents=True)
    rows, types = [], []
    for name, frame in [('control', control), ('adapted', aligned)]:
        summary, slices = metrics(frame)
        rows.append(dict(variant=name, **summary['overall'], macro_ads=summary['macro_ads'], selection=summary['selection']))
        slices.to_csv(args.output / f'{name}_slices.csv', index=False)
        types.extend(dict(variant=name, **item) for item in typed_metrics(frame))
    pd.DataFrame(rows).to_csv(args.output / 'overall.csv', index=False)
    pd.DataFrame(types).to_csv(args.output / 'by_type.csv', index=False)
    # A direct replacement diagnostic, without calibrating or sweeping weights.
    anchor_path = ROOT / 'reports/three_stream_all_type_v57_strict/v50_authorized_v2/v1_strict__music_only_predictions.csv'
    codec_truth = control.loc[control.DATASET.eq('codec_mixed_dev_v4')].drop(columns=PREDICTION_COLUMNS)
    anchor = align_truth(codec_truth, pd.read_csv(anchor_path, dtype={'ID': str, 'DATASET': str}))
    candidate = align_truth(codec_truth, adapted.loc[adapted.DATASET.eq('codec_mixed_dev_v4')])
    music_only = anchor.copy()
    music_only['MUSIC_FAKE_PROB'] = candidate['MUSIC_FAKE_PROB'].to_numpy()
    unchanged = [c for c in PREDICTION_COLUMNS if c != 'MUSIC_FAKE_PROB']
    if not np.array_equal(anchor[unchanged].to_numpy(), music_only[unchanged].to_numpy()):
        raise ValueError('Music-only diagnostic changed another output')
    codec_rows = []
    for name, frame in [('official_anchor', anchor), ('adapted_all_outputs', candidate), ('adapted_music_only', music_only)]:
        codec_rows.append(dict(variant=name, channel='all', **score_frame(frame)))
        codec_rows.extend(dict(variant=name, channel=channel, **score_frame(block)) for channel, block in frame.groupby('CHANNEL'))
    pd.DataFrame(codec_rows).to_csv(args.output / 'codec_comparison.csv', index=False)
    pd.DataFrame(paired_bootstrap(anchor, music_only)).to_csv(args.output / 'music_replacement_paired_intervals.csv', index=False)
    music_only.to_csv(args.output / 'music_only_replacement_diagnostic.csv', index=False)
    report = dict(stage='completed paired adaptation development audit',
        matched_control_probabilities_exact=True, all_six_sampler_epochs_exact=True,
        adapted_best_epoch=checkpoint['epoch'], overall=rows,
        codec_anchor=score_frame(anchor), codec_direct_music_replacement=score_frame(music_only),
        other_four_music_replacement_outputs_exact=True, source_ood_verified=False,
        official_improvement_verified=False, automatic_submission_allowed=False,
        limitations=['fixed checkpoint re-evaluated, not reselected using independent inference',
                     'bootstrap is descriptive development evidence, not corrected for model selection',
                     'development has no 30-60 second files; generator families overlap train'],
        artifacts_sha256={str(p): sha256(p) for p in [checkpoint_path, adapted_path, anchor_path, run / 'completed.json', local / 'report.json']})
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['codec_direct_music_replacement']), flush=True)


if __name__ == '__main__':
    main()
