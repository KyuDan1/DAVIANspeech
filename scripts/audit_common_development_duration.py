#!/usr/bin/env python3
"""Read authorized-development audio headers and report duration-specific EER.

No train/holdout audio and no checkpoint/mixture-weight selection. Separately
shows the competition's 4--60 second range instead of silently pooling it.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
import soundfile as sf

from src.evaluate_diagnostic import score_frame
from verify_paired_checkpoint_inference_v68 import resolve_development
from train_paired_wpt_file_v60 import sha256


def duration_bin(seconds):
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('finite positive audio duration required')
    if seconds < 4:
        return 'below_4s'
    if seconds <= 10.24:
        return '4_to_10.24s'
    if seconds <= 30:
        return '10.24_to_30s'
    if seconds <= 60:
        return '30_to_60s'
    return 'above_60s'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    runs = ROOT / 'reports/common_encoder_probe'
    reference_path = runs / 'eat_large/mean/development_predictions.csv'
    reference = pd.read_csv(reference_path, dtype={'DATASET': str, 'ID': str})
    paths = resolve_development(reference, ROOT / 'configs/data_partitions.yaml')
    def header(path):
        info = sf.info(path)
        return dict(duration_actual=info.frames / info.samplerate, sample_rate=info.samplerate, channels=info.channels)
    with ThreadPoolExecutor(max_workers=8) as pool:
        metadata = pd.DataFrame(list(pool.map(header, paths)))
    frame = pd.concat([reference[['DATASET', 'ID']].reset_index(drop=True), metadata], axis=1)
    frame['duration_bin'] = frame.duration_actual.map(duration_bin)
    frame.to_csv(args.output / 'durations.csv', index=False)
    counts = frame.groupby(['DATASET', 'duration_bin']).size().rename('rows').reset_index()
    counts.to_csv(args.output / 'counts.csv', index=False)
    results, hashes = [], {str(reference_path): sha256(reference_path)}
    for encoder in ['eat_large', 'xlsr', 'spear', 'wavlm', 'spear_independent']:
        if not (runs / encoder / 'completed.json').is_file():
            continue
        for pooling in ['mean', 'attention']:
            path = runs / encoder / pooling / 'development_predictions.csv'
            prediction = pd.read_csv(path, dtype={'DATASET': str, 'ID': str})
            joined = prediction.merge(frame, on=['DATASET', 'ID'], validate='one_to_one')
            if len(joined) != len(frame) or len(prediction) != len(frame):
                raise ValueError('duration/prediction identities do not match')
            for name, selected in joined.groupby('duration_bin'):
                results.append(dict(encoder=encoder, pooling=pooling, duration_bin=name, **score_frame(selected)))
            eligible = joined.duration_actual.between(4, 60)
            results.append(dict(encoder=encoder, pooling=pooling, duration_bin='competition_range_4_to_60s',
                                **score_frame(joined.loc[eligible])))
            hashes[str(path)] = sha256(path)
    pd.DataFrame(results).to_csv(args.output / 'duration_scores.csv', index=False)
    report = dict(rows=len(frame), duration_counts=frame.duration_bin.value_counts().to_dict(),
        competition_length_rows=int(frame.duration_actual.between(4, 60).sum()),
        minimum_seconds=float(frame.duration_actual.min()), maximum_seconds=float(frame.duration_actual.max()),
        artifacts_sha256=hashes, scope='development length audit only; no official generalization claim',
        limitations=['saved batch scores for SPEAR/XLSR remain provisional pending independent rescoring',
                     'duration slice was added after observing short-file infrastructure failures; diagnostic only'],
        automatic_submission_allowed=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
