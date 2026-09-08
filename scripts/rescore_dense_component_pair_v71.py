#!/usr/bin/env python3
"""Full file-local scoring of two selected heads, sharing ONE frozen encoder.

No cross-file batching: all windows in a forward belong to the current file.
No checkpoint reselection, blending, calibration, protected-audio access or API.
"""
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

from src.common_encoder_probe import complete_windows
from src.dense_component_inference_v71 import DenseComponentPredictor
from src.dense_component_v71 import DenseComponentHead, paired_dense_logits
from src.evaluate_diagnostic import PREDICTION_COLUMNS, LABEL_COLUMNS
from src.pipeline import load_audio
from train_common_encoder_probe import metrics
from train_paired_wpt_file_v60 import sha256
from train_three_stream_anchor_residual import authorized_partitions
from verify_paired_checkpoint_inference_v68 import resolve_development
from compare_common_encoder_probe import align_truth
from finalize_dense_component_v71 import VARIANTS, audit_run, require_declared_rows


@torch.no_grad()
def predict_one(encoder, heads, audio, config):
    windows, lengths, _ = complete_windows(audio)
    outputs, _, _ = paired_dense_logits(encoder, heads, torch.from_numpy(windows), torch.from_numpy(lengths),
        [len(windows)], config['encoder_chunk'], config['temperature'])
    result = {name: values.sigmoid()[0].cpu().numpy().astype(np.float64) for name, values in outputs.items()}
    if not all(np.isfinite(values).all() for values in result.values()):
        raise ValueError('nonfinite file-local probabilities')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'reports/dense_component_v71/full')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    audit = audit_run(args.run)
    config = audit['config']
    frames, artifacts = {}, {}
    declared = pd.read_csv(args.run / 'development_ids.csv', dtype=str)
    for variant in VARIANTS:
        checkpoint = args.run / variant / 'head.pt'
        saved = args.run / variant / 'development_predictions.csv'
        artifacts[variant] = {str(path): sha256(path) for path in [checkpoint, saved]}
        frames[variant] = pd.read_csv(saved, dtype={'ID': str, 'DATASET': str})
        require_declared_rows(frames[variant], declared)
    reference = frames['file_only'].drop(columns=PREDICTION_COLUMNS)
    frames['dense_supervised'] = align_truth(reference, frames['dense_supervised'])
    for partition in authorized_partitions(ROOT / 'configs/data_partitions.yaml', 'development', list(reference.DATASET.unique())):
        truth = pd.read_csv(partition.truth_path, dtype={'ID': str}).set_index('ID')
        rows = reference.loc[reference.DATASET.eq(partition.name)].set_index('ID')
        np.testing.assert_allclose(rows[LABEL_COLUMNS].to_numpy(float), truth.loc[rows.index, LABEL_COLUMNS].to_numpy(float), equal_nan=True)
    paths = resolve_development(reference, ROOT / 'configs/data_partitions.yaml')
    torch.set_num_threads(4)
    first = DenseComponentPredictor(args.run / 'file_only/head.pt', ROOT)
    second = torch.load(args.run / 'dense_supervised/head.pt', map_location='cpu', weights_only=False)
    # audit_run already verified the same frozen parent and configuration.
    heads = {'file_only': first.head, 'dense_supervised': DenseComponentHead(
        width=config['head_width'], pooling=config['pooling']).cuda().eval()}
    heads['dense_supervised'].load_state_dict(second['state_dict'], strict=True)
    values = {name: [] for name in heads}
    started = time.monotonic()
    for index, path in enumerate(paths):
        predictions = predict_one(first.encoder, heads, load_audio(path), config)
        for name in heads:
            values[name].append(predictions[name])
        if index % 250 == 0:
            print(json.dumps(dict(files=index + 1, total=len(paths), paired_heads=2,
                seconds=time.monotonic() - started)), flush=True)
    actual = {name: np.stack(items) for name, items in values.items()}
    repeats = sorted(set([0, len(paths) // 2, len(paths) - 1]))
    for index in repeats[::-1]:
        repeated = predict_one(first.encoder, heads, load_audio(paths[index]), config)
        for name in heads:
            np.testing.assert_array_equal(actual[name][index], repeated[name])
    for records in artifacts.values():
        if any(sha256(Path(path)) != digest for path, digest in records.items()):
            raise ValueError('selected model or saved predictions changed during scoring')
    args.output.mkdir(parents=True)
    summaries = {}
    for name, frame in frames.items():
        expected = frame[PREDICTION_COLUMNS].to_numpy(float).copy()
        frame[PREDICTION_COLUMNS] = actual[name]
        summary, slices = metrics(frame)
        directory = args.output / name
        directory.mkdir()
        frame.to_csv(directory / 'development_predictions.csv', index=False)
        slices.to_csv(directory / 'development_slices.csv', index=False)
        report = dict(scope='full authorized development, file-local paired encoder reuse; NOT official score',
            rows=len(frame), pooling=name, encoder=first.encoder.name, elapsed_seconds=time.monotonic() - started,
            shared_encoder=True, repeated_file_indices=repeats, repeated_files_bit_exact=True,
            checkpoint_reselected=False, max_batch_vs_file_difference=float(np.abs(actual[name] - expected).max()),
            file_local_prediction_sha256=sha256(directory / 'development_predictions.csv'),
            summary=summary, artifacts_sha256=artifacts[name], automatic_submission_allowed=False)
        (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        summaries[name] = summary
    (args.output / 'completed.json').write_text(json.dumps(dict(rows=len(paths), paired_heads=2,
        summaries=summaries, elapsed_seconds=time.monotonic() - started), indent=2) + '\n')
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    main()
