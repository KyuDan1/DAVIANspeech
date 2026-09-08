#!/usr/bin/env python3
"""Actual local-weight, authorized TRAIN-only common encoder/head smoke.

Not a benchmark: no dev/holdout detector scoring, EER or model selection.
All complete file windows are pooled before labels supervise the head.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import pandas as pd
import torch
import yaml

from src.common_encoder_probe import CommonTokenHead, FrozenEncoderTokens, complete_windows, component_loss
from src.pipeline import load_audio
from src.full_coverage_wpt import masked_lme
from train_paired_wpt_file_v60 import load_data, sha256

MODELS = {'eat_base': 'models/eat-base-as2m', 'eat_large': 'models/eat-large-as2m-v56',
          'spear': 'models/spear-xlarge-speech-audio-v2', 'xlsr': 'models/xls-r-2b-anti-deepfake',
          'wavlm': 'models/wavlm-large', 'spear_independent': 'models/spear-xlarge-speech-audio-v2'}


def pad_token_batches(token_list, mask_list):
    # SPEAR removes trailing batch padding, so frame counts may differ between
    # chunks even when the raw input arrays have equal length.
    maximum = max(tokens.shape[2] for tokens in token_list)
    return (torch.cat([torch.nn.functional.pad(tokens, (0, 0, 0, maximum - tokens.shape[2]))
                       for tokens in token_list]),
            torch.cat([torch.nn.functional.pad(mask, (0, maximum - mask.shape[1]), value=False)
                       for mask in mask_list]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', choices=list(MODELS), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(20260905)
    np.random.seed(20260905)
    config = yaml.safe_load((ROOT / 'configs/paired_wpt_file_v60.yaml').read_text())
    train, _, audit = load_data(config)
    # One complete train file from each of eight presence/authenticity cells.
    keys = ['VOICE_PRESENT', 'MUSIC_PRESENT', 'VOICE_FAKE', 'MUSIC_FAKE']
    frame = train.sort_values(['DATASET', 'ID']).groupby(keys, dropna=False, sort=True).head(1)
    frame = frame.drop_duplicates(keys).reset_index(drop=True)
    columns = ['FILE_FAKE', 'VOICE_FAKE', 'MUSIC_FAKE', 'VOICE_PRESENT', 'MUSIC_PRESENT']
    targets = torch.tensor(frame[columns].to_numpy(np.float32), device='cuda')
    arrays, lengths, starts, owners = [], [], [], []
    for i, row in frame.iterrows():
        windows, valid, offsets = complete_windows(load_audio(Path(row.PATH)))
        arrays.extend(windows)
        lengths.extend(valid)
        starts.append(offsets.tolist())
        owners.extend([i] * len(windows))
    waveform = torch.tensor(np.stack(arrays))
    lengths = torch.tensor(lengths)
    device = torch.device('cuda')
    began = time.monotonic()
    encoder = FrozenEncoderTokens(args.encoder, ROOT / MODELS[args.encoder], device)
    load_seconds = time.monotonic() - began
    token_list, mask_list = [], []
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    for offset in range(0, len(waveform), 2):
        tokens, mask = encoder(waveform[offset:offset + 2], lengths[offset:offset + 2])
        token_list.append(tokens)
        mask_list.append(mask)
    tokens, mask = pad_token_batches(token_list, mask_list)
    torch.cuda.synchronize()
    extraction_seconds = time.monotonic() - began
    if tokens.is_inference():
        raise RuntimeError('inference tensors cannot feed trainable readout')
    maximum = max(owners.count(i) for i in range(len(frame)))
    bag_mask = torch.zeros(len(frame), maximum, dtype=torch.bool, device=device)
    for i in range(len(frame)):
        bag_mask[i, :owners.count(i)] = True
    owner_index = [torch.tensor([j for j, owner in enumerate(owners) if owner == i], device=device) for i in range(len(frame))]
    head = CommonTokenHead(tokens.shape[-1]).to(device)
    initial = copy.deepcopy(head.state_dict())
    variants = {}
    for pooling in ['mean', 'attention']:
        head = CommonTokenHead(tokens.shape[-1], pooling=pooling).to(device)
        head.load_state_dict(initial)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001)
        losses = []
        for step in range(6):
            logits = head(tokens, mask)
            bags = logits.new_zeros((len(frame), maximum, 5))
            for i, indices in enumerate(owner_index):
                bags[i, :len(indices)] = logits[indices]
            loss = component_loss(masked_lme(bags, bag_mask, 2.), targets)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite smoke loss')
            optimizer.zero_grad()
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters()):
                raise RuntimeError('nonfinite smoke gradient')
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            one = head(tokens[:1], mask[:1])
            together = head(tokens, mask)[:1]
        difference = float((one - together).abs().max())
        if difference > 1e-5:
            raise RuntimeError(f'head file independence mismatch {difference}')
        variants[pooling] = dict(losses=losses, parameters=sum(p.numel() for p in head.parameters()),
                                 head_batch_max_abs_difference=difference)
    model_files = sorted((ROOT / MODELS[args.encoder]).glob('*.safetensors'))
    if not model_files:
        model_files = sorted((ROOT / MODELS[args.encoder]).glob('*.bin'))
    report = dict(status='passed', purpose='train-only execution/gradient smoke; NOT accuracy evidence',
                  encoder=args.encoder, layers=encoder.indices, input_shape=list(waveform.shape),
                  input_lengths=lengths.tolist(), window_starts=starts,
                  input_sha256=__import__('hashlib').sha256(waveform.numpy().tobytes()).hexdigest(),
                  token_shape=list(tokens.shape), valid_token_counts=mask.sum(1).tolist(),
                  load_seconds=load_seconds, extraction_seconds=extraction_seconds,
                  gpu=torch.cuda.get_device_name(), peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                  variants=variants, source_audit=audit,
                  code_sha256=sha256(ROOT / 'src/common_encoder_probe.py'),
                  runner_sha256=sha256(Path(__file__)),
                  checkpoint_sha256={str(p): sha256(p) for p in model_files})
    frame[['DATASET', 'ID', 'PATH'] + columns].to_csv(args.output / 'train_rows.csv', index=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['status', 'encoder', 'token_shape', 'extraction_seconds', 'variants']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
