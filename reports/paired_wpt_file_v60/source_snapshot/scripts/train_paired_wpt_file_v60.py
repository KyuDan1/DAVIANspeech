#!/usr/bin/env python3
"""Strict-source raw WPT File training with full-coverage matched codec bags."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from full_coverage_wpt import coverage_windows, codec_pair, forward_bags, resolve_ffmpeg
from wpt_spectra import WPTSpectraMultitask
from pipeline import load_audio, find_audio_files
from train_wpt_spectra_multitask import load_spectra, finite_eer, worker_seed
from train_three_stream_anchor_residual import (
    authorized_partitions, load_train_identity_exclusions, filter_train_pair,
    assert_component_split_contract, assert_strict_identity_split,
)
from evaluate_three_stream_v57 import add_diagnostic_axes


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_data(config):
    matrix_path = ROOT / config['data_matrix']
    matrix = yaml.safe_load(matrix_path.read_text())
    frames = {}
    manifests = []
    removed = []
    exclusions = load_train_identity_exclusions(ROOT / matrix['train_identity_exclusions'])
    for role, key in [('train', 'train_set'), ('development', 'development_set')]:
        pieces = []
        for partition in authorized_partitions(ROOT / config['partition_config'], role, matrix[matrix[key]]):
            frame = pd.read_csv(partition.truth_path, dtype={'ID': str})
            frame['DATASET'] = partition.name
            if role == 'train':
                (frame, _, _), rejected = filter_train_pair((frame, {}, {}), exclusions)
                removed.extend(rejected)
            paths = {path.stem: str(path) for path in find_audio_files(partition.truth_path.parent / 'audio')}
            missing = set(frame.ID).difference(paths)
            if missing:
                raise FileNotFoundError(f'{partition.name}: missing {len(missing)} audio IDs')
            frame['PATH'] = [paths[item] for item in frame.ID]
            pieces.append(frame)
            manifests.append(dict(role=role, path=str(partition.truth_path), sha256=sha256(partition.truth_path)))
        frames[role] = pd.concat(pieces, ignore_index=True)
    audit = dict(component=assert_component_split_contract(frames['train'], frames['development']),
                 strict=assert_strict_identity_split(frames['train'], frames['development']),
                 excluded_train_ids=removed, manifests=manifests,
                 matrix_sha256=sha256(matrix_path))
    return frames['train'], add_diagnostic_axes(frames['development']), audit


class Bags(Dataset):
    def __init__(self, frame, config, paired):
        self.frame = frame.reset_index(drop=True)
        self.config = config
        self.paired = paired
        self.ffmpeg = resolve_ffmpeg() if paired else None

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        audio = load_audio(Path(row.PATH))
        # Whole-file gain augmentation preserves temporal labels and coverage.
        if self.paired:
            audio = audio * np.float32(np.random.uniform(.7, 1.3))
        windows, starts = coverage_windows(audio, self.config['window'])
        if self.paired:
            variant = self.config['channels'][np.random.randint(len(self.config['channels']))]
            changed = codec_pair(windows, variant, self.ffmpeg, int(np.random.randint(2**30)))
        else:
            changed = None
        return windows, changed, float(row.FILE_FAKE), index


def collate_bags(items):
    # Original bags first, corresponding channel bags second.
    arrays = [item[0] for item in items]
    paired = items[0][1] is not None
    if paired:
        arrays += [item[1] for item in items]
    maximum = max(len(array) for array in arrays)
    audio = np.zeros((len(arrays), maximum, arrays[0].shape[1]), dtype=np.float32)
    mask = np.zeros(audio.shape[:2], dtype=bool)
    for i, array in enumerate(arrays):
        audio[i, :len(array)] = array
        mask[i, :len(array)] = True
    targets = torch.tensor([item[2] for item in items], dtype=torch.float32)
    indices = torch.tensor([item[3] for item in items])
    return torch.from_numpy(audio), torch.from_numpy(mask), targets, indices


def balanced_weights(frame):
    # Equal corpus -> File class -> component cell mass. No evaluation priors.
    cells = [(str(r.DATASET), int(r.FILE_FAKE), int(r.VOICE_PRESENT), int(r.MUSIC_PRESENT),
              int(0 if pd.isna(r.VOICE_FAKE) else r.VOICE_FAKE),
              int(0 if pd.isna(r.MUSIC_FAKE) else r.MUSIC_FAKE)) for r in frame.itertuples()]
    counts = Counter(cells)
    strata = Counter(key[:2] for key in counts)
    classes = Counter(set(key[:2] for key in counts))
    class_count = Counter(key[0] for key in classes)
    return torch.tensor([1 / (counts[key] * strata[key[:2]] * class_count[key[0]]) for key in cells], dtype=torch.double)


def loader(dataset, config, training=False, generator=None):
    sampler = WeightedRandomSampler(balanced_weights(dataset.frame), config['samples_per_epoch'],
                                    replacement=True, generator=generator) if training else None
    return DataLoader(dataset, batch_size=config['batch_size' if training else 'eval_batch_size'],
                      sampler=sampler, shuffle=False, num_workers=config['workers'],
                      collate_fn=collate_bags, pin_memory=True, worker_init_fn=worker_seed,
                      persistent_workers=config['workers'] > 0, generator=generator)


@torch.inference_mode()
def evaluate(model, batches, device, config):
    model.eval()
    scores, indices = [], []
    for windows, mask, _, index in batches:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            output = forward_bags(model, windows.to(device), mask.to(device),
                                  config['temperature'], config['window_chunk_size'])
        scores.extend(output[:, 2].float().sigmoid().cpu().tolist())
        indices.extend(index.tolist())
    frame = batches.dataset.frame.iloc[indices].copy()
    frame['FILE_FAKE_PROB'] = scores
    metrics = []
    for axis in ['DATASET', 'CHANNEL_V57', 'LAYOUT_V57', 'CELL_V57']:
        for name, group in frame.groupby(axis):
            metrics.append(dict(axis=axis, group=str(name), n=len(group),
                                real=int(group.FILE_FAKE.eq(0).sum()), fake=int(group.FILE_FAKE.eq(1).sum()),
                                FILE_EER=finite_eer(group.FILE_FAKE, group.FILE_FAKE_PROB)))
    metrics = pd.DataFrame(metrics)
    domains = metrics.loc[metrics.axis.eq('DATASET'), 'FILE_EER'].dropna().to_numpy()
    pooled = finite_eer(frame.FILE_FAKE, frame.FILE_FAKE_PROB)
    if not len(domains) or not np.isfinite(pooled):
        raise ValueError('no defined File EER for selection')
    selection = float(1 - .5 * pooled - .25 * domains.mean() - .25 * domains.max())
    return frame[['DATASET', 'ID', 'FILE_FAKE_PROB']], metrics, selection, pooled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/paired_wpt_file_v60.yaml')
    parser.add_argument('--variant', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    variant = config['variants'][args.variant]
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    torch.set_num_threads(6)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    train, development, audit = load_data(config)
    if args.smoke:
        train = train.iloc[:8].copy()
        # Ensure a defined File EER in a small diagnostic-only CUDA run.
        development = pd.concat([development[development.FILE_FAKE.eq(label)].head(4) for label in [0, 1]])
        config.update(epochs=1, eval_every=1, samples_per_epoch=8, workers=0, batch_size=2)
    args.output.mkdir(parents=True)
    model_dir = ROOT / config['model_dir']
    provenance = dict(config=config, variant=args.variant, smoke=args.smoke, split_audit=audit,
                      config_sha256=sha256(args.config), script_sha256=sha256(__file__),
                      model_sha256=sha256(model_dir / 'model.safetensors'),
                      source_sha256=sha256(ROOT / 'src/full_coverage_wpt.py'),
                      train_rows=len(train), development_rows=len(development))
    (args.output / 'manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'validated': True, 'train': len(train), 'development': len(development),
                      'identity_overlap': audit['strict']['overlap'], 'smoke': args.smoke}), flush=True)
    device = torch.device(args.device)
    model = WPTSpectraMultitask(load_spectra(model_dir, device), temperature=config['temperature']).to(device)
    prompts = list(model.prompt_encoder.prompts.parameters())
    heads = list(model.task_head.parameters())
    excluded = {id(p) for p in prompts + heads}
    backend = [p for p in model.parameters() if p.requires_grad and id(p) not in excluded]
    optimizer = torch.optim.AdamW([dict(params=prompts, lr=config['prompt_lr']),
                                   dict(params=heads, lr=config['head_lr']),
                                   dict(params=backend, lr=config['backend_lr'])], weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(config['seed'])
    train_loader = loader(Bags(train, config, True), config, True, generator)
    dev_loader = loader(Bags(development, config, False), config)
    best, stale, history = -float('inf'), 0, []
    start_time = time.monotonic()
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        losses = []
        for step, (windows, mask, targets, _) in enumerate(train_loader):
            targets = targets.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                logits = forward_bags(model, windows.to(device), mask.to(device), config['temperature'], config['window_chunk_size'])[:, 2]
            clean, channel = logits.chunk(2)
            bce = .5 * (F.binary_cross_entropy_with_logits(clean, targets) + F.binary_cross_entropy_with_logits(channel, targets))
            retention = F.smooth_l1_loss(channel, clean.detach())
            repeated = targets.repeat(2)
            pos, neg = logits[repeated.eq(1)], logits[repeated.eq(0)]
            rank = F.softplus(-(pos[:, None] - neg[None])).mean() if len(pos) and len(neg) else logits.sum() * 0
            loss = bce + variant['consistency_weight'] * retention + config['ranking_weight'] * rank
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 3., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % 100 == 0:
                print(json.dumps(dict(epoch=epoch, step=step, loss=float(np.mean(losses[-100:])))), flush=True)
        if epoch % config['eval_every'] and epoch != config['epochs']:
            continue
        predictions, metrics, selection, pooled = evaluate(model, dev_loader, device, config)
        record = dict(epoch=epoch, loss=float(np.mean(losses)), selection=selection,
                      pooled_file_eer=pooled, elapsed_seconds=time.monotonic() - start_time)
        history.append(record)
        pd.DataFrame(history).to_csv(args.output / 'history.csv', index=False)
        print(json.dumps(record), flush=True)
        if selection > best + 1e-5:
            best, stale = selection, 0
            torch.save(dict(model_type='full_coverage_wpt_file_v60', state=model.trainable_state_dict(),
                            config=config, variant=args.variant, smoke=args.smoke, best_epoch=epoch,
                            selection=selection, provenance=provenance), args.output / 'best.pt')
            predictions.to_csv(args.output / 'dev_predictions.csv', index=False)
            metrics.to_csv(args.output / 'dev_metrics.csv', index=False)
        else:
            stale += 1
            if stale >= config['patience']:
                break
    (args.output / 'complete.json').write_text(json.dumps(dict(best_selection=best, elapsed_seconds=time.monotonic() - start_time)) + '\n')


if __name__ == '__main__':
    main()
