#!/usr/bin/env python3
"""Compare COMPLETED matched encoder/readout runs on authorized development.

No model fitting, ensembling, protected-set inference or submission. Bootstrap
intervals are descriptive development intervals, not selection-corrected tests.
"""
from __future__ import annotations

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

from src.evaluate_diagnostic import LABEL_COLUMNS, PREDICTION_COLUMNS, score_frame, official_eer
from train_common_encoder_probe import metrics
from train_paired_wpt_file_v60 import sha256
from train_three_stream_anchor_residual import authorized_partitions

ENCODERS = ('eat_large', 'spear', 'xlsr')
KEYS = ['DATASET', 'ID']


def completed_matched_runs(directory, encoders=ENCODERS):
    """Fail before reading any predictions when even one run is incomplete."""
    if not encoders or len(set(encoders)) != len(encoders):
        raise ValueError('unique nonempty encoder list required')
    paths = {name: directory / name for name in encoders}
    for path in paths.values():
        if not (path / 'completed.json').is_file():
            raise RuntimeError(f'comparison requires all completed runs: {path}')
    manifests = {name: json.loads((path / 'manifest.json').read_text()) for name, path in paths.items()}
    first = manifests[encoders[0]]['config']
    # Later encoder additions may extend only the checkpoint registry. Existing
    # runs retain their original config/snapshot; all experiment settings match.
    settings = lambda config: {k: v for k, v in config.items() if k != 'encoders'}
    if any(settings(m['config']) != settings(first) for m in manifests.values()):
        raise ValueError('encoder experiment configs differ')
    registry = {}
    for manifest in manifests.values():
        for name, path in manifest['config'].get('encoders', {}).items():
            if name in registry and registry[name] != path:
                raise ValueError('encoder checkpoint registry changed')
            registry[name] = path
    if any(m['audit']['manifests'] != manifests[encoders[0]]['audit']['manifests'] for m in manifests.values()):
        raise ValueError('source manifest hashes differ between encoders')
    if first['schema'] != 'common_encoder_probe_v1':
        raise ValueError('unexpected experiment schema')
    for filename in ['train_ids.csv', 'development_ids.csv']:
        if len({sha256(path / filename) for path in paths.values()}) != 1:
            raise ValueError(f'ordered {filename} mismatch')
    evidence = []
    for epoch in range(1, first['epochs'] + 1):
        files = [path / f'epoch_{epoch}_train_draws.npy' for path in paths.values()]
        arrays = [np.load(path, allow_pickle=False) for path in files]
        if not all(np.array_equal(arrays[0], array) for array in arrays[1:]):
            raise ValueError(f'epoch {epoch}: sampler draws differ')
        if len(arrays[0]) != first['samples_per_epoch']:
            raise ValueError('wrong epoch sample count')
        evidence.append(dict(epoch=epoch, count=len(arrays[0]), sha256=sha256(files[0])))
    for name, manifest in manifests.items():
        if manifest['encoder'] != name:
            raise ValueError('encoder name mismatch')
        for original, digest in manifest['code_sha256'].items():
            snapshot = paths[name] / 'source_snapshot' / Path(original).name
            if sha256(snapshot) != digest:
                raise ValueError(f'code snapshot mismatch: {snapshot}')
    return paths, manifests, evidence


def align_truth(truth, prediction):
    for frame in [truth, prediction]:
        if frame[KEYS].isna().any().any() or frame[KEYS].duplicated().any():
            raise ValueError('invalid or duplicate prediction/truth keys')
    if set(map(tuple, truth[KEYS].to_numpy())) != set(map(tuple, prediction[KEYS].to_numpy())):
        raise ValueError('prediction/truth keys differ')
    result = truth.merge(prediction[KEYS + PREDICTION_COLUMNS], on=KEYS, validate='one_to_one')
    values = result[PREDICTION_COLUMNS].to_numpy(float)
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError('invalid probabilities')
    return result


def typed_metrics(frame):
    labels = np.where(frame.VOICE_PRESENT.eq(1),
                      np.where(frame.MUSIC_PRESENT.eq(1), 'mixed', 'voice_only'),
                      np.where(frame.MUSIC_PRESENT.eq(1), 'music_only', 'neither'))
    rows = []
    for kind in sorted(set(labels)):
        rows.append(dict(type=kind, **score_frame(frame.loc[labels == kind])))
    return rows


def paired_bootstrap(anchor, candidate, repetitions=500, seed=20260905):
    """Stratified by RR/RF/FR/FF, resample BASE_ID including all its channels."""
    if not np.array_equal(anchor[KEYS].to_numpy(), candidate[KEYS].to_numpy()):
        raise ValueError('bootstrap requires identically ordered pairs')
    if repetitions <= 0 or 'BASE_ID' not in candidate or 'COMPONENT_CASE' not in candidate:
        raise ValueError('positive repetitions and mixed base metadata required')
    metadata = candidate[['BASE_ID', 'COMPONENT_CASE']].drop_duplicates()
    if metadata.BASE_ID.duplicated().any():
        raise ValueError('component labels change within a base')
    groups = {base: np.flatnonzero(candidate.BASE_ID.eq(base)) for base in metadata.BASE_ID}
    strata = [block.BASE_ID.to_numpy() for _, block in metadata.groupby('COMPONENT_CASE')]
    rng = np.random.default_rng(seed)
    values = {task: [] for task in ['FILE', 'VOICE', 'MUSIC']}
    labels = {task: candidate[f'{task}_FAKE'].to_numpy(int) for task in values}
    for _ in range(repetitions):
        selected = np.concatenate([np.concatenate([groups[base] for base in rng.choice(bases, len(bases), replace=True)])
                                   for bases in strata])
        for task in values:
            column = f'{task}_FAKE_PROB'
            delta = official_eer(labels[task][selected], candidate[column].to_numpy()[selected]) - official_eer(
                labels[task][selected], anchor[column].to_numpy()[selected])
            values[task].append(delta)
    result = []
    for task, deltas in values.items():
        finite = np.asarray(deltas)[np.isfinite(deltas)]
        if not len(finite):
            raise ValueError('bootstrap has no defined EER samples')
        result.append(dict(task=task, base_count=len(groups), rows=len(candidate), repetitions=repetitions,
                           delta_eer_ci_low=float(np.quantile(finite, .025)),
                           delta_eer_ci_high=float(np.quantile(finite, .975)),
                           sign='candidate minus anchor; negative is better'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'reports/common_encoder_probe')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-wavlm', action='store_true', help='Require completed matched WavLM extension too')
    parser.add_argument('--include-spear-independent', action='store_true', help='Include corrected window-local SPEAR run')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    encoders = ENCODERS + ('wavlm',) if args.include_wavlm else ENCODERS
    if args.include_spear_independent:
        encoders += ('spear_independent',)
    paths, manifests, draws = completed_matched_runs(args.runs, encoders)
    reference = manifests['eat_large']
    names = [Path(item['path']).parent.name for item in reference['audit']['manifests'] if item['role'] == 'development']
    partitions = authorized_partitions(ROOT / reference['config']['partition_config'], 'development', names)
    truths = []
    for partition in partitions:
        recorded = next(item for item in reference['audit']['manifests']
                        if item['role'] == 'development' and Path(item['path']).resolve() == partition.truth_path.resolve())
        if sha256(partition.truth_path) != recorded['sha256']:
            raise ValueError(f'development truth changed: {partition.truth_path}')
        frame = pd.read_csv(partition.truth_path, dtype={'ID': str, 'BASE_ID': str})
        frame['DATASET'] = partition.name
        truths.append(frame)
    truth = pd.concat(truths, ignore_index=True)
    ids = pd.read_csv(paths['eat_large'] / 'development_ids.csv', dtype=str)
    if not np.array_equal(truth[KEYS].to_numpy(), ids[KEYS].to_numpy()):
        raise ValueError('ordered development identities changed')
    from evaluate_three_stream_v57 import add_diagnostic_axes
    truth = add_diagnostic_axes(truth)
    anchor_path = ROOT / 'reports/three_stream_all_type_v57_strict/v50_authorized_v2/v1_strict__music_only_predictions.csv'
    anchor_scores = pd.read_csv(anchor_path, dtype={'ID': str, 'DATASET': str})
    codec_truth = truth[truth.DATASET.eq('codec_mixed_dev_v4')].copy()
    anchor = align_truth(codec_truth, anchor_scores)
    args.output.mkdir(parents=True)
    summaries, types, channels, intervals = [], [], [], []
    artifacts = {str(anchor_path): sha256(anchor_path)}
    for encoder, path in paths.items():
        history = json.loads((path / 'history.json').read_text())
        for pooling in reference['config']['pooling_conditions']:
            current = path / pooling
            prediction_path, checkpoint_path = current / 'development_predictions.csv', current / 'head.pt'
            prediction = pd.read_csv(prediction_path, dtype={'ID': str, 'DATASET': str})
            frame = align_truth(truth, prediction)
            summary, slices = metrics(frame)
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            expected = max((row for row in history if row['pooling'] == pooling), key=lambda row: row['selection'])
            if checkpoint['epoch'] != expected['epoch'] or abs(summary['selection'] - expected['selection']) > 1e-10:
                raise ValueError('selected checkpoint or metric mismatch')
            identity = dict(encoder=encoder, pooling=pooling, epoch=checkpoint['epoch'])
            summaries.append(dict(**identity, **summary['overall'], macro_ads=summary['macro_ads'], selection=summary['selection']))
            types.extend(dict(**identity, **row) for row in typed_metrics(frame))
            candidate = align_truth(codec_truth, prediction[prediction.DATASET.eq('codec_mixed_dev_v4')])
            for channel, block in candidate.groupby('CHANNEL'):
                channels.append(dict(**identity, channel=channel, **score_frame(block)))
            for row in paired_bootstrap(anchor, candidate):
                task = row['task']
                row['anchor_eer'] = score_frame(anchor)[f'{task}_EER']
                row['candidate_eer'] = score_frame(candidate)[f'{task}_EER']
                intervals.append(dict(**identity, **row))
            slices.to_csv(args.output / f'{encoder}_{pooling}_slices.csv', index=False)
            artifacts.update({str(p): sha256(p) for p in [prediction_path, checkpoint_path, path / 'completed.json']})
    table = pd.DataFrame(summaries).sort_values('selection', ascending=False)
    table.to_csv(args.output / 'overall.csv', index=False)
    pd.DataFrame(types).to_csv(args.output / 'by_type.csv', index=False)
    pd.DataFrame(channels).to_csv(args.output / 'codec_channels.csv', index=False)
    pd.DataFrame(intervals).to_csv(args.output / 'paired_codec_intervals.csv', index=False)
    report = dict(stage='completed development benchmark; not submission approval',
                  development_rank_leader=table.iloc[0][['encoder', 'pooling', 'epoch']].to_dict(),
                  next_adaptation_candidate=None,
                  promotion_requires='file-local encoder equivalence for each candidate; scores alone cannot promote',
                  sampler_equality=draws, artifacts_sha256=artifacts,
                  unique_sampled_train_rows=int(len(np.unique(np.concatenate([
                      np.load(paths['eat_large'] / f'epoch_{epoch}_train_draws.npy', allow_pickle=False)
                      for epoch in range(1, reference['config']['epochs'] + 1)])))),
                  eligible_train_rows=reference['train_rows'],
                  limitations=['development selection is not unseen-generator evidence',
                    'bootstrap is conditional on the balanced local codec bank, not official test composition',
                    'intervals do not correct for checkpoint or model selection',
                    'native-width projections have different parameter counts',
                    'no probabilities were ensemble-fitted or calibrated',
                    'SPEAR mean independent verification failed; saved batch metrics are provisional'],
                  official_improvement_verified=False, automatic_submission_allowed=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(table[['encoder', 'pooling', 'FILE_EER', 'VOICE_EER', 'MUSIC_EER', 'ADS', 'selection']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
