#!/usr/bin/env python3
"""Music-only replacement audit on the predeclared development rows, never test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_curve

from evaluate_three_stream_v57 import load_truths, load_candidate_predictions


def component_metrics(frame: pd.DataFrame, column: str, task: str) -> dict:
    if task not in ('music', 'voice', 'file'):
        raise ValueError('unknown component task')
    if task != 'file':
        frame = frame.loc[frame[f'{task.upper()}_PRESENT'].eq(1)]
    labels = frame[f'{task.upper()}_FAKE'].to_numpy(dtype=int)
    scores = frame[column].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Predictions must be finite probabilities")
    positives, negatives = int((labels == 1).sum()), int((labels == 0).sum())
    eer = None
    if positives and negatives:
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
        fnr = 1 - tpr
        index = np.argmin(np.abs(fpr - fnr))
        eer = float((fpr[index] + fnr[index]) / 2)
    return dict(n=len(labels), real=negatives, fake=positives, eer=eer,
                fpr_at_05=float((scores[labels == 0] >= .5).mean()) if negatives else None,
                fnr_at_05=float((scores[labels == 1] < .5).mean()) if positives else None)


def music_metrics(frame: pd.DataFrame, column: str) -> dict:
    return component_metrics(frame, column, 'music')


def decision(rows: list[dict], gate: dict, task: str = 'music') -> dict:
    pooled = next(row for row in rows if row['axis'] == 'pooled')
    domains = [r for r in rows if r['axis'] == 'DATASET' and r['gain'] is not None]
    channels = [r for r in rows if r['axis'] == 'CHANNEL_V57' and r['gain'] is not None]
    if pooled['gain'] is None or not domains or not channels:
        return dict(pass_gate=False, reason='insufficient defined EER groups')
    mean_gain = float(np.mean([r['gain'] for r in domains]))
    worst_gain = (max(r['incumbent']['eer'] for r in domains)
                  - max(r['candidate']['eer'] for r in domains))
    checks = {
        'pooled_improves': pooled['gain'] >= gate[f'minimum_pooled_{task}_eer_gain'] - 1e-12,
        'domains_nonregressing': min(r['gain'] for r in domains) >= -gate[f'maximum_domain_{task}_eer_regression'] - 1e-12,
        'channels_nonregressing': min(r['gain'] for r in channels) >= -gate[f'maximum_channel_{task}_eer_regression'] - 1e-12,
        'mean_domain_improves': mean_gain > 1e-12 if gate['require_mean_domain_improvement'] else True,
        'worst_domain_nonworse': worst_gain >= -1e-12 if gate['require_worst_domain_nonworse'] else True,
    }
    return dict(pass_gate=all(checks.values()), checks=checks,
                pooled_gain=pooled['gain'], mean_domain_gain=mean_gain,
                worst_domain_gain=worst_gain,
                max_domain_regression=max(0., -min(r['gain'] for r in domains)),
                max_channel_regression=max(0., -min(r['gain'] for r in channels)))


def compare(truth: pd.DataFrame, incumbent: pd.DataFrame, candidate: pd.DataFrame, task: str = 'music') -> list[dict]:
    keys = ['DATASET', 'ID']
    expected = set(map(tuple, truth[keys].to_numpy()))
    frame = truth.copy()
    probability = f'{task.upper()}_FAKE_PROB'
    for name, predictions in [('incumbent', incumbent), ('candidate', candidate)]:
        if predictions[keys].duplicated().any() or set(map(tuple, predictions[keys].to_numpy())) != expected:
            raise ValueError(f'{name} does not exactly match declared development rows')
        frame = frame.merge(predictions[keys + [probability]].rename(
            columns={probability: name}), on=keys, validate='one_to_one')
    groups = [('pooled', 'all', frame)]
    for axis in ['DATASET', 'CHANNEL_V57', 'LAYOUT_V57', 'CELL_V57', 'MUSIC_GENERATOR_V57']:
        groups.extend((axis, str(name), group) for name, group in frame.groupby(axis, dropna=False))
    rows = []
    for axis, name, group in groups:
        old, new = component_metrics(group, 'incumbent', task), component_metrics(group, 'candidate', task)
        gain = None if old['eer'] is None else old['eer'] - new['eer']
        rows.append(dict(axis=axis, group=name, incumbent=old, candidate=new, gain=gain))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=Path('configs/music_specialist_v58.yaml'))
    parser.add_argument('--partitions', type=Path, default=Path('configs/data_partitions.yaml'))
    parser.add_argument('--incumbent', type=Path, default=Path('reports/three_stream_all_type_v57_strict/v1_robust_l2/dev_predictions.csv'))
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=['music', 'voice', 'file'], default='music')
    parser.add_argument('--datasets', nargs='+', help='Explicit subset of the declared development datasets only')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    matrix = yaml.safe_load(args.matrix.read_text())
    data_matrix = Path(matrix.get('data_matrix', args.matrix))
    truth = load_truths(data_matrix, args.partitions)
    incumbent = pd.read_csv(args.incumbent, dtype={'DATASET': str, 'ID': str})
    candidate = pd.read_csv(args.candidate, dtype={'DATASET': str, 'ID': str})
    if args.datasets:
        if not set(args.datasets).issubset(set(truth.DATASET)):
            raise ValueError('requested dataset is not declared development')
        truth, incumbent, candidate = [f[f.DATASET.isin(args.datasets)] for f in [truth, incumbent, candidate]]
    rows = compare(truth, incumbent, candidate, args.task)
    result = {'decision': decision(rows, matrix['decision'], args.task), 'rows': rows,
              'scope': f'{args.task} only; not official ADS or total score estimate',
              'files': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in [args.matrix, data_matrix, args.partitions, args.incumbent, args.candidate]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['decision'], indent=2))


if __name__ == '__main__':
    main()
