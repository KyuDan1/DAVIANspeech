#!/usr/bin/env python3
"""Close paired dense-supervision evidence, without fitting or submission.

Only completed runs and full file-local development predictions are accepted.
Direct component replacement is diagnostic, never an ensemble weight sweep.
"""
import argparse
from collections import Counter
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

VARIANTS = ['file_only', 'dense_supervised']


def require_declared_rows(frame, declared):
    keys = ['DATASET', 'ID']
    if (frame[keys].isna().any().any() or frame[keys].duplicated().any()
            or len(frame) != len(declared)
            or set(map(tuple, frame[keys].to_numpy())) != set(map(tuple, declared[keys].to_numpy()))):
        raise ValueError('predictions do not cover the exact declared development rows')


def audit_draws(run, config):
    """Every recorded source must come from the frozen authorized train list."""
    train = pd.read_csv(run / 'train_ids.csv', dtype=str)
    catalog = pd.read_csv(run / 'source_catalog.csv', dtype={'ID': str, 'GROUP_ID': str}).set_index('ID')
    if catalog.index.duplicated().any():
        raise ValueError('duplicate catalog IDs')
    payloads, counts, epochs = {}, Counter(), []
    for epoch in range(1, config['epochs'] + 1):
        path = run / f'epoch_{epoch}_draws.json'
        draws = json.loads(path.read_text())
        if len(draws) != config['samples_per_epoch']:
            raise ValueError('incomplete epoch draws')
        for index, draw in enumerate(draws):
            if draw['draw'] != index or draw['epoch'] != epoch:
                raise ValueError('unexpected epoch/draw order')
            counts[draw['kind']] += 1
            if draw['kind'] == 'original_train':
                if not 0 <= draw['row'] < len(train):
                    raise ValueError('original row outside strict train')
                row = train.iloc[draw['row']]
                if (row.ID, row.DATASET) != (draw['id'], draw['dataset']):
                    raise ValueError('original draw identity differs from strict train')
            elif draw['kind'] == 'synthetic_train':
                if (draw['channel'] not in config['synthetic_channels']
                        or draw['layout'] not in config['synthetic_layouts']
                        or draw['seconds'] not in config['synthetic_durations']):
                    raise ValueError('undeclared synthesis condition')
                for source in draw['sources'].values():
                    if source['id'] not in catalog.index or catalog.loc[source['id'], 'GROUP_ID'] != source['group']:
                        raise ValueError('synthetic source not in authorized catalog')
                    digest = source['sha256']
                    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                        raise ValueError('invalid consumed payload checksum')
                    if source['id'] in payloads and payloads[source['id']] != digest:
                        raise ValueError('source payload changed between training draws')
                    payloads[source['id']] = digest
                if draw['layout'] in ['sparse_voice', 'sparse_music']:
                    key = 'VR' if draw['layout'] == 'sparse_voice' else 'MR'
                    if draw['sources'][key]['group'] == draw['sources']['REAL_INSERT']['group']:
                        raise ValueError('real insertion control did not change source group')
                counts['layout:' + draw['layout']] += 1
                counts['channel:' + draw['channel']] += 1
                counts['duration:' + str(draw['seconds'])] += 1
            else:
                raise ValueError('unrecognized training draw')
        epochs.append(dict(epoch=epoch, draws=len(draws), sha256=sha256(path)))
    return dict(epochs=epochs, counts=dict(counts), unique_consumed_raw_sources=len(payloads),
                consumed_source_sha256=payloads)


def audit_run(run):
    completion = json.loads((run / 'completed.json').read_text())
    manifest = json.loads((run / 'manifest.json').read_text())
    if completion['smoke'] or manifest['smoke']:
        raise ValueError('smoke cannot be finalized as accuracy evidence')
    config = manifest['config']
    if config['schema'] != 'dense_component_v71' or not config['no_automatic_submission']:
        raise ValueError('unexpected experiment schema or submission policy')
    for filename, count in [('train_ids.csv', 'train_rows'), ('development_ids.csv', 'development_rows')]:
        identities = pd.read_csv(run / filename, dtype=str)
        if len(identities) != manifest[count] or identities[['DATASET', 'ID']].duplicated().any():
            raise ValueError('declared train/development identity count changed')
    for path, digest in manifest['code_sha256'].items():
        if sha256(run / 'source_snapshot' / Path(path).name) != digest:
            raise ValueError('training source snapshot changed')
    if sha256(run / 'source_catalog.csv') != manifest['source_catalog_sha256']:
        raise ValueError('catalog snapshot changed')
    parent_path = ROOT / config['parent_checkpoint']
    if sha256(parent_path) != manifest['parent_checkpoint_sha256']:
        raise ValueError('frozen parent checkpoint changed')
    history = json.loads((run / 'history.json').read_text())
    selected = {}
    for variant in VARIANTS:
        rows = [row for row in history if row['variant'] == variant]
        if [row['epoch'] for row in rows] != list(range(2, config['epochs'] + 1, 2)):
            raise ValueError('missing scheduled development evaluations')
        best = max(rows, key=lambda row: row['selection'])
        checkpoint = torch.load(run / variant / 'head.pt', map_location='cpu', weights_only=False)
        if (checkpoint['smoke'] or checkpoint['model_type'] != 'dense_component_v71'
                or checkpoint['variant'] != variant or checkpoint['epoch'] != best['epoch']
                or checkpoint['config'] != config
                or checkpoint['selection']['selection'] != best['selection']
                or completion['best_selection'][variant] != best['selection']
                or checkpoint['parent_checkpoint_sha256'] != manifest['parent_checkpoint_sha256']):
            raise ValueError('checkpoint does not match declared development selection')
        selected[variant] = best
    return dict(config=config, selected=selected, training_draws=audit_draws(run, config))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'reports/dense_component_v71/full')
    parser.add_argument('--local-root', type=Path, default=ROOT / 'reports/dense_component_v71/full_file_local')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    audit = audit_run(args.run)
    frames = {}
    artifacts = [args.run / 'completed.json', args.run / 'manifest.json']
    declared = pd.read_csv(args.run / 'development_ids.csv', dtype=str)
    for variant in VARIANTS:
        directory = args.local_root / variant
        verification = json.loads((directory / 'report.json').read_text())
        saved = pd.read_csv(args.run / variant / 'development_predictions.csv', dtype={'ID': str, 'DATASET': str})
        local = pd.read_csv(directory / 'development_predictions.csv', dtype={'ID': str, 'DATASET': str})
        require_declared_rows(saved, declared)
        require_declared_rows(local, declared)
        if sha256(directory / 'development_predictions.csv') != verification['file_local_prediction_sha256']:
            raise ValueError('file-local prediction payload changed')
        if verification['rows'] != len(saved) or not verification['repeated_files_bit_exact'] or verification['checkpoint_reselected']:
            raise ValueError('incomplete or reselected file-local evaluation')
        for path, digest in verification['artifacts_sha256'].items():
            if sha256(Path(path)) != digest:
                raise ValueError('verified inference artifacts changed')
        checkpoint = args.run / variant / 'head.pt'
        if sha256(checkpoint) not in verification['artifacts_sha256'].values():
            raise ValueError('file-local report does not verify the selected checkpoint')
        frames[variant] = align_truth(saved.drop(columns=PREDICTION_COLUMNS), local)
        artifacts.extend([checkpoint, directory / 'report.json', directory / 'development_predictions.csv'])
    parent_path = ROOT / 'reports/common_eat_adaptation/full_file_local/development_predictions.csv'
    frames['parent_global'] = align_truth(frames['file_only'].drop(columns=PREDICTION_COLUMNS),
        pd.read_csv(parent_path, dtype={'ID': str, 'DATASET': str}))
    artifacts.append(parent_path)
    summaries, types, slices = [], [], []
    for variant, frame in frames.items():
        summary, grouped = metrics(frame)
        summaries.append(dict(variant=variant, **summary['overall'], macro_ads=summary['macro_ads'], selection=summary['selection']))
        types.extend(dict(variant=variant, **row) for row in typed_metrics(frame))
        grouped['variant'] = variant
        slices.append(grouped)
    anchor_path = ROOT / 'reports/three_stream_all_type_v57_strict/v50_authorized_v2/v1_strict__music_only_predictions.csv'
    truth = frames['file_only'].loc[frames['file_only'].DATASET.eq('codec_mixed_dev_v4')].drop(columns=PREDICTION_COLUMNS)
    anchor = align_truth(truth, pd.read_csv(anchor_path, dtype={'ID': str}))
    replacements, intervals = [], []
    for variant, frame in frames.items():
        candidate = align_truth(truth, frame.loc[frame.DATASET.eq('codec_mixed_dev_v4')])
        for component in ['FILE', 'VOICE', 'MUSIC']:
            column = component + '_FAKE_PROB'
            replaced = anchor.copy()
            replaced[column] = candidate[column].to_numpy()
            untouched = [key for key in PREDICTION_COLUMNS if key != column]
            if not np.array_equal(anchor[untouched].to_numpy(), replaced[untouched].to_numpy()):
                raise ValueError('one-column diagnostic changed another output')
            name = variant + ':' + component
            replacements.append(dict(variant=name, channel='all', **score_frame(replaced)))
            replacements.extend(dict(variant=name, channel=channel, **score_frame(block)) for channel, block in replaced.groupby('CHANNEL'))
            intervals.extend(dict(variant=name, **row) for row in paired_bootstrap(anchor, replaced))
    replacements.append(dict(variant='official_anchor', channel='all', **score_frame(anchor)))
    artifacts.append(anchor_path)
    args.output.mkdir(parents=True)
    pd.DataFrame(summaries).to_csv(args.output / 'overall.csv', index=False)
    pd.DataFrame(types).to_csv(args.output / 'by_type.csv', index=False)
    pd.concat(slices).to_csv(args.output / 'slices.csv', index=False)
    pd.DataFrame(replacements).to_csv(args.output / 'codec_single_column_diagnostics.csv', index=False)
    pd.DataFrame(intervals).to_csv(args.output / 'paired_intervals.csv', index=False)
    report = dict(stage='paired dense supervision development audit', audit=audit, overall=summaries,
        official_improvement_verified=False, source_ood_verified=False, automatic_submission_allowed=False,
        artifacts_sha256={str(path): sha256(path) for path in artifacts},
        limitations=['TRAIN-only synthetic data and frozen parent shared by both heads',
            'parent_global differs in both training inputs and temporal aggregation; only paired heads isolate dense loss',
            'existing development has no 30-60 second files or within-component brief fake insertions',
            'direct output replacement/bootstrap is descriptive development evidence, not new checkpoint or weight selection',
            'no v70/locked/Suno data used in this finalizer'])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
