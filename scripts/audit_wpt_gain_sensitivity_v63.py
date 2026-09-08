#!/usr/bin/env python3
"""Gain-only diagnostic on declared development, without checkpoint/weight selection."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def peak_normalize_windows(waveforms):
    return waveforms / (waveforms.abs().amax(dim=-1, keepdim=True) + 1e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    package = ROOT / 'v50_v57m_challenger_v2'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(package / 'model/src'))
    from wpt_spectra_inference import load_wpt_model, fixed_windows, _preemphasis, aggregate_view_logits
    from pipeline import load_audio, find_audio_files
    truth = pd.read_csv(ROOT / 'data/eval/codec_mixed_dev_v4/truth.csv')
    pieces = [g.sample(n=3, random_state=20260905) for _, g in truth.groupby(['CHANNEL', 'FILE_FAKE'])]
    sample = pd.concat(pieces).sort_values('ID')
    paths = {p.stem: p for p in find_audio_files(ROOT / 'data/eval/codec_mixed_dev_v4/audio')}
    model_dir = package / 'model/spectra-aasist'
    model, config = load_wpt_model(model_dir, model_dir / 'wpt_spectra_multitask.pt', torch.device('cuda'))
    model.eval()
    records = []
    with torch.inference_mode():
        for row in sample.itertuples():
            windows = fixed_windows(load_audio(paths[row.ID]), config['window'], 5)
            for normalized in (False, True):
                for gain in (.1, 1., 10.):
                    values = _preemphasis(torch.from_numpy(windows[None] * gain).to('cuda'))
                    if normalized:
                        values = peak_normalize_windows(values)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits = aggregate_view_logits(model.forward_windows(values), 2.)
                    records.append(dict(ID=row.ID, CHANNEL=row.CHANNEL, PEAK_NORMALIZED=normalized,
                                        GAIN=gain, FILE_PROB=float(logits[0, 2].sigmoid())))
    result = pd.DataFrame(records)
    summaries = []
    for (normalized, channel), group in result.groupby(['PEAK_NORMALIZED', 'CHANNEL']):
        table = group.pivot(index='ID', columns='GAIN', values='FILE_PROB')
        changes = np.abs(table[[.1, 10.]].to_numpy() - table[[1.]].to_numpy())
        summaries.append(dict(peak_normalized=bool(normalized), channel=channel, files=len(table),
            mean_absolute_probability_change=float(changes.mean()),
            maximum_absolute_probability_change=float(changes.max()),
            files_with_probability_range_gt_01=int((table.max(axis=1) - table.min(axis=1) > .1).sum())))
    args.output.mkdir(parents=True)
    result.to_csv(args.output / 'probabilities.csv', index=False)
    sample[['ID', 'CHANNEL', 'FILE_FAKE']].to_csv(args.output / 'development_samples.csv', index=False)
    (args.output / 'report.json').write_text(json.dumps(dict(
        scope='30 declared-development files; gain invariance only, not authenticity accuracy or a submission candidate',
        no_waveform_clipping_applied=True, comparison='same old WPT weights, peak normalization after preemphasis',
        rows=summaries), indent=2) + '\n')
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
