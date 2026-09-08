#!/usr/bin/env python3
"""Reproduce SPEAR padding/batch dependence, using authorized dev only.

No training or score-based tuning. The existing checkpoint remains unchanged.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
import torch

from src.common_encoder_probe import FrozenEncoderTokens, CommonTokenHead, complete_windows
from src.pipeline import load_audio
from src.evaluate_diagnostic import PREDICTION_COLUMNS
from train_common_encoder_probe import file_logits
from verify_paired_checkpoint_inference_v68 import resolve_development
from train_paired_wpt_file_v60 import sha256


@torch.no_grad()
def compare_pair(encoder, head, windows, lengths):
    together, mask = encoder(windows, lengths)
    alone, alone_mask = encoder(windows[:1], lengths[:1])
    count = int(alone_mask[0].sum())
    batched_score = head(together, mask).sigmoid()[0]
    alone_score = head(alone, alone_mask).sigmoid()[0]
    return dict(lengths=lengths.tolist(), valid_tokens=count,
                token_max_abs_difference=float((together[0, :, :count] - alone[0, :, :count]).abs().max()),
                probability_max_abs_difference=float((batched_score - alone_score).abs().max()),
                batched_probability=batched_score.tolist(), alone_probability=alone_score.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--encoder', choices=['spear', 'spear_independent'], default='spear')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    run = ROOT / 'reports/common_encoder_probe/spear'
    checkpoint_path = run / 'mean/head.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    encoder = FrozenEncoderTokens(args.encoder, ROOT / checkpoint['config']['encoders']['spear'])
    head = CommonTokenHead(checkpoint['input_dimension'], pooling='mean').cuda().eval()
    head.load_state_dict(checkpoint['state_dict'])
    prediction = pd.read_csv(run / 'mean/development_predictions.csv', dtype={'ID': str, 'DATASET': str})
    suspect = pd.read_csv(ROOT / 'reports/common_encoder_probe/spear_independent_mean/predictions.csv', dtype=str)
    differences = np.stack([(suspect['saved_' + c].astype(float) - suspect['independent_' + c].astype(float)).abs()
                            for c in PREDICTION_COLUMNS], axis=1).max(1)
    worst = suspect.iloc[np.argsort(-differences)[:4]]
    records = []
    for row in worst.itertuples():
        index = np.flatnonzero(prediction.DATASET.eq(row.DATASET) & prediction.ID.eq(row.ID)).item()
        start = index // checkpoint['config']['batch_size'] * checkpoint['config']['batch_size']
        frame = prediction.iloc[start:start + checkpoint['config']['batch_size']]
        paths = resolve_development(frame, ROOT / 'configs/data_partitions.yaml')
        arrays = [complete_windows(load_audio(path))[:2] for path in paths]
        windows = torch.from_numpy(np.concatenate([x[0] for x in arrays]))
        lengths = torch.from_numpy(np.concatenate([x[1] for x in arrays]))
        counts = [len(x[0]) for x in arrays]
        with torch.no_grad():
            actual = file_logits(encoder, {'mean': head}, windows, lengths, counts, 2, 2.)['mean'].sigmoid()
        expected = frame[PREDICTION_COLUMNS].to_numpy(float)
        offsets = np.cumsum([0] + counts)
        target_window = int(offsets[index - start])
        first = windows[target_window:target_window + 1]
        first_length = lengths[target_window:target_window + 1]
        pairs = []
        for neighbor in sorted(set([0, int(lengths.argmax()), int(lengths.argmin())])):
            pairs.append(compare_pair(encoder, head,
                torch.cat([first, windows[neighbor:neighbor + 1]]),
                torch.cat([first_length, lengths[neighbor:neighbor + 1]])))
        # Same raw target paired with an identical copy removes differing lengths.
        pairs.append(compare_pair(encoder, head, first.repeat(2, 1), first_length.repeat(2)))
        records.append(dict(dataset=row.DATASET, id=row.ID,
            replay_saved_batch_max_abs_difference=float(np.abs(actual.cpu().numpy() - expected).max()),
            target_length=int(first_length[0]), pairs=pairs))
        print(json.dumps(records[-1]), flush=True)
    report = dict(scope='batch dependence diagnosis; no model promotion', encoder=args.encoder, records=records,
                  checkpoint_sha256=sha256(checkpoint_path), code_sha256=sha256(Path(__file__)))
    report['all_pairs_independent'] = all(p['probability_max_abs_difference'] <= 5e-4
                                         for row in records for p in row['pairs'])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.encoder == 'spear_independent' and not report['all_pairs_independent']:
        raise RuntimeError('corrected encoder is not file independent; do not train')


if __name__ == '__main__':
    main()
