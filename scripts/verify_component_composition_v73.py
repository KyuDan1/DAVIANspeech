#!/usr/bin/env python3
"""Three-model integrated waveform inference versus cached-score composition."""
import argparse
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

from src.component_composition_inference_v73 import ComponentCompositionPredictor, checksum
from src.evaluate_diagnostic import PREDICTION_COLUMNS
from src.pipeline import load_audio
from verify_common_encoder_probe_inference import select_rows
from verify_paired_checkpoint_inference_v68 import resolve_development, compare_arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'reports/component_composition_v73')
    parser.add_argument('--variant', choices=['xlsr_voice_eat_music', 'equal_voice_pair_eat_music'], default='equal_voice_pair_eat_music')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frozen = args.run / 'frozen.json'
    predictions = args.run / (args.variant + '_predictions.csv')
    hashes = {str(path): checksum(path) for path in [frozen, predictions]}
    frame = select_rows(pd.read_csv(predictions, dtype={'DATASET': str, 'ID': str}))
    paths = resolve_development(frame, ROOT / 'configs/data_partitions.yaml')
    torch.set_num_threads(4)
    warnings.filterwarnings('ignore', category=FutureWarning)
    model = ComponentCompositionPredictor(frozen, ROOT, args.variant)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    actual = np.stack([model(load_audio(path)) for path in paths])
    reverse = np.stack([model(load_audio(path)) for path in paths[::-1]])[::-1]
    expected = frame[PREDICTION_COLUMNS].to_numpy(float)
    report = compare_arrays(expected, actual, reverse, tolerance=5e-4)
    report.update(scope='integrated file-local implementation equivalence; NOT accuracy or L4 time guarantee',
        variant=args.variant, checkpoints=model.checkpoints, elapsed_seconds=time.monotonic() - started,
        cuda_peak_mb=torch.cuda.max_memory_allocated() / 2**20, gpu=torch.cuda.get_device_name(),
        artifacts_sha256=hashes, automatic_submission_allowed=False)
    if any(checksum(path) != digest for path, digest in hashes.items()):
        raise ValueError('frozen comparison changed during execution')
    args.output.mkdir(parents=True)
    table = frame[['DATASET', 'ID']].copy()
    for index, column in enumerate(PREDICTION_COLUMNS):
        table['saved_' + column] = expected[:, index]
        table['integrated_' + column] = actual[:, index]
        table['reverse_' + column] = reverse[:, index]
    table.to_csv(args.output / 'predictions.csv', index=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    if not report['pass_check']:
        raise ValueError('integrated composition does not reproduce cached evaluation')


if __name__ == '__main__':
    main()
