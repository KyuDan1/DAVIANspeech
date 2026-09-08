#!/usr/bin/env python3
"""Descriptive, paired-channel diagnosis of the existing best submission on dev.

No training, threshold deployment, candidate selection or locked-data inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

CONTRASTS = [('FILE', 'RR', 'FR'), ('FILE', 'RR', 'RF'), ('FILE', 'RR', 'FF'),
             ('VOICE', 'RR', 'FR'), ('VOICE', 'RF', 'FF'),
             ('MUSIC', 'RR', 'RF'), ('MUSIC', 'FR', 'FF')]
PROBS = [f'{task}_FAKE_PROB' for task in ['FILE', 'VOICE', 'MUSIC']]


def eer_point(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if not set(labels).issubset({0, 1}) or not np.isfinite(scores).all():
        raise ValueError('binary labels and finite scores required')
    if (scores < 0).any() or (scores > 1).any():
        raise ValueError('probabilities must be in [0,1]')
    if len(set(labels)) < 2:
        return None, None
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    index = int(np.argmin(np.abs(fpr - (1 - tpr))))
    return float((fpr[index] + 1 - tpr[index]) / 2), float(thresholds[index])


def align(truth, prediction):
    keys = ['DATASET', 'ID']
    for frame in [truth, prediction]:
        if frame[keys].duplicated().any():
            raise ValueError('duplicate DATASET/ID')
    if set(map(tuple, truth[keys].to_numpy())) != set(map(tuple, prediction[keys].to_numpy())):
        raise ValueError('prediction keys do not exactly match truth')
    merged = truth.merge(prediction[keys + PROBS], on=keys, validate='one_to_one')
    if not (merged.VOICE_PRESENT.eq(1) & merged.MUSIC_PRESENT.eq(1)).all():
        raise ValueError('this diagnostic requires mixed audio with both components present')
    expected = np.where(merged.VOICE_FAKE.eq(1), 'F', 'R').astype(object)
    expected += np.where(merged.MUSIC_FAKE.eq(1), 'F', 'R')
    if not np.array_equal(expected, merged.COMPONENT_CASE.to_numpy()):
        raise ValueError('component case and labels disagree')
    if not np.array_equal(np.maximum(merged.VOICE_FAKE, merged.MUSIC_FAKE), merged.FILE_FAKE):
        raise ValueError('File fake OR rule violated')
    if merged[['BASE_ID', 'CHANNEL']].duplicated().any():
        raise ValueError('duplicate base/channel')
    expected_channels = set(merged.CHANNEL)
    for _, block in merged.groupby('BASE_ID'):
        if set(block.CHANNEL) != expected_channels or 'clean' not in expected_channels:
            raise ValueError('incomplete paired channels')
        for col in ['FILE_FAKE', 'VOICE_FAKE', 'MUSIC_FAKE', 'COMPONENT_CASE', 'MIX_MODE', 'SNR_DB']:
            if block[col].nunique(dropna=False) != 1:
                raise ValueError(f'paired metadata changes: {col}')
    for task in ['FILE', 'VOICE', 'MUSIC']:
        eer_point(merged[f'{task}_FAKE'], merged[f'{task}_FAKE_PROB'])
    return merged


def diagnose(frame):
    slices = [('pooled', 'all', frame)]
    for axis in ['CHANNEL', 'MIX_MODE', 'SNR_DB']:
        slices.extend((axis, str(key), block) for key, block in frame.groupby(axis))
    slices.extend(('CHANNEL+MIX_MODE', f'{channel}|{mode}', block)
                  for (channel, mode), block in frame.groupby(['CHANNEL', 'MIX_MODE']))
    eers, contrasts = [], []
    for axis, group, block in slices:
        for task in ['FILE', 'VOICE', 'MUSIC']:
            value, _ = eer_point(block[f'{task}_FAKE'], block[f'{task}_FAKE_PROB'])
            eers.append(dict(axis=axis, group=group, task=task, n=len(block),
                             base_count=block.BASE_ID.nunique(), eer=value))
        for task, negative, positive in CONTRASTS:
            selected = block[block.COMPONENT_CASE.isin([negative, positive])]
            value, _ = eer_point(selected[f'{task}_FAKE'], selected[f'{task}_FAKE_PROB'])
            contrasts.append(dict(axis=axis, group=group, task=task,
                contrast=f'{negative}_vs_{positive}', n=len(selected),
                base_count=selected.BASE_ID.nunique(), eer=value))
    clean = frame[frame.CHANNEL.eq('clean')].set_index('BASE_ID')
    operating, paired = [], []
    for task in ['FILE', 'VOICE', 'MUSIC']:
        _, threshold = eer_point(clean[f'{task}_FAKE'], clean[f'{task}_FAKE_PROB'])
        if threshold is None or not np.isfinite(threshold):
            raise ValueError('defined finite clean EER threshold required for fixed-threshold diagnosis')
        for (channel, cell), block in frame.groupby(['CHANNEL', 'COMPONENT_CASE']):
            scores = block[f'{task}_FAKE_PROB'].to_numpy()
            labels = block[f'{task}_FAKE'].to_numpy()
            error = ((scores >= threshold) != labels).mean()
            operating.append(dict(task=task, channel=channel, cell=cell, n=len(block),
                clean_dev_threshold=threshold, error_rate=float(error),
                error_type='FNR' if labels[0] else 'FPR'))
            if channel == 'clean':
                continue
            original = clean.loc[block.BASE_ID, f'{task}_FAKE_PROB'].to_numpy()
            signed = (2 * labels - 1) * (scores - original)
            paired.append(dict(task=task, channel=channel, cell=cell, n=len(block),
                mean_signed_confidence_change=float(signed.mean()),
                median_signed_confidence_change=float(np.median(signed)),
                fraction_confidence_worsened=float((signed < -1e-12).mean())))
    return dict(eers=pd.DataFrame(eers), contrasts=pd.DataFrame(contrasts),
                fixed_clean_threshold=pd.DataFrame(operating), paired_changes=pd.DataFrame(paired))


def main():
    from train_three_stream_anchor_residual import authorized_partitions
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    partitions = ROOT / 'configs/data_partitions.yaml'
    selected = authorized_partitions(partitions, 'development', ['codec_mixed_dev_v4'])
    if len(selected) != 1:
        raise ValueError('expected one authorized development partition')
    truth_path = selected[0].truth_path
    truth = pd.read_csv(truth_path, dtype={'ID': str, 'BASE_ID': str})
    truth['DATASET'] = selected[0].name
    prediction_path = ROOT / 'reports/three_stream_all_type_v57_strict/v50_authorized_v2/v1_strict__music_only_predictions.csv'
    prediction = pd.read_csv(prediction_path, dtype={'ID': str, 'DATASET': str})
    frame = align(truth, prediction)
    tables = diagnose(frame)
    args.output.mkdir(parents=True)
    for name, table in tables.items():
        table.to_csv(args.output / f'{name}.csv', index=False)
    report = dict(scope='authorized development only; descriptive diagnosis, no candidate selection',
        rows=len(frame), distinct_base_mixtures=frame.BASE_ID.nunique(),
        voice_sources=frame.VOICE_SOURCE_ID.nunique(), music_sources=frame.MUSIC_SOURCE_ID.nunique(),
        channels=sorted(frame.CHANNEL.unique()),
        limitations=['same source audio repeats across channels; rows are not independent',
            'component contrasts have both classes; individual cells do not have EER',
            'clean EER thresholds are diagnostic only, not deployment thresholds',
            'all files are mixed; no music-only, voice-only or CPS AUC coverage',
            'does not estimate official test distribution, EER or total score'],
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [truth_path, prediction_path, partitions, Path(__file__)]})
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(tables['eers'].query("axis == 'CHANNEL'").to_string(index=False))
    print(tables['contrasts'].query("axis == 'CHANNEL' and task == 'FILE'").to_string(index=False))


if __name__ == '__main__':
    main()
