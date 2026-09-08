#!/usr/bin/env python3
"""Descriptive all-fake Suno audit of ONE frozen v73 model, without tuning."""
import json
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
import torch
import yaml
from src.component_composition_inference_v73 import ComponentCompositionPredictor, checksum
from src.evaluate_diagnostic import PREDICTION_COLUMNS, LABEL_COLUMNS
from src.pipeline import load_audio


def main():
    output = ROOT / 'reports/component_composition_v73/suno_fixed_audit'
    if output.exists():
        raise FileExistsError('frozen audit already reserved; no repeated model search')
    bank = ROOT / 'data/eval/suno_vocals_v1'
    truth_path = bank / 'truth.csv'
    roles = yaml.safe_load((ROOT / 'configs/data_partitions.yaml').read_text())
    memberships = {role for role, paths in roles.items() if str(truth_path.relative_to(ROOT)) in paths}
    if memberships != {'ood_holdout'}:
        raise ValueError('Suno must remain protected OOD')
    truth = pd.read_csv(truth_path, dtype={'ID': str})
    if truth.ID.duplicated().any() or len(truth) != 13 or not truth[LABEL_COLUMNS].eq(1).all().all():
        raise ValueError('expected fixed 13 all-fake mixed clips')
    candidate = ROOT / 'reports/component_composition_v73/frozen.json'
    hashes = {str(path): checksum(path) for path in [candidate, truth_path, Path(__file__)]}
    paths = [bank / 'audio' / (identity + '.flac') for identity in truth.ID]
    hashes.update({str(path): checksum(path) for path in paths})
    output.mkdir(parents=True)
    frozen = dict(variant='equal_voice_pair_eat_music', artifacts_sha256=hashes,
        threshold=.5, scope='13 user Suno all-fake clips; not EER/AUC or full original tracks',
        selection_allowed=False, automatic_submission_allowed=False)
    (output / 'frozen.json').write_text(json.dumps(frozen, indent=2) + '\n')
    warnings.filterwarnings('ignore', category=FutureWarning)
    torch.set_num_threads(2)
    model = ComponentCompositionPredictor(candidate, ROOT)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    values = np.stack([model(load_audio(path)) for path in paths])
    reverse = np.stack([model(load_audio(path)) for path in paths[::-1]])[::-1]
    np.testing.assert_array_equal(values, reverse)
    if any(checksum(path) != expected for path, expected in hashes.items()):
        raise ValueError('frozen audit input changed')
    result = truth.copy()
    result[PREDICTION_COLUMNS] = values
    result.to_csv(output / 'predictions.csv', index=False)
    summary = {column: dict(minimum=float(values[:, i].min()), median=float(np.median(values[:, i])),
                           maximum=float(values[:, i].max()), count_at_least_half=int((values[:, i] >= .5).sum()))
               for i, column in enumerate(PREDICTION_COLUMNS)}
    report = dict(**frozen, status='complete', rows=len(truth), summary=summary,
        seconds=time.monotonic() - started, peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
        reverse_bit_exact=True, eer=None, auc=None,
        limitation='No matched real songs; cannot measure false positives, EER or generator-wide generalization.')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
