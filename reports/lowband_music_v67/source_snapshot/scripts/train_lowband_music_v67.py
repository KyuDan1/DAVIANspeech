#!/usr/bin/env python3
"""Train Music-only low-band CNNs on strict raw paired codec mixtures."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from src.lowband_music_v67 import LowbandMusic
from train_paired_wpt_file_v60 import load_data, Bags, collate_bags, balanced_weights, sha256, worker_seed
from evaluate_music_specialist_v58 import component_metrics


class MusicBags(Bags):
    def __getitem__(self, index):
        original, channel, _, idx = super().__getitem__(index)
        row = self.frame.iloc[index]
        # Absent Music is never a negative training label. Eval labels are unused.
        target = float(row.MUSIC_FAKE) if row.MUSIC_PRESENT == 1 else 0.
        if self.paired and row.MUSIC_PRESENT != 1:
            raise ValueError('training contains music-absent row')
        return original, channel, target, idx


def music_weights(frame):
    if not frame.MUSIC_PRESENT.eq(1).all() or not frame.MUSIC_FAKE.isin([0, 1]).all():
        raise ValueError('binary Music-present training rows required')
    # Reuse corpus -> target class -> component-cell balancing, with Music as target.
    sampler_frame = frame.copy()
    sampler_frame['FILE_FAKE'] = sampler_frame.MUSIC_FAKE
    return balanced_weights(sampler_frame)


def make_loader(frame, config, training, generator=None):
    sampler = WeightedRandomSampler(music_weights(frame), config['samples_per_epoch'], True,
                                    generator=generator) if training else None
    return DataLoader(MusicBags(frame, config, training),
        batch_size=config['batch_size' if training else 'eval_batch_size'],
        sampler=sampler, shuffle=False, num_workers=config['workers'],
        collate_fn=collate_bags, pin_memory=True, worker_init_fn=worker_seed,
        persistent_workers=config['workers'] > 0, generator=generator)


@torch.inference_mode()
def evaluate(model, loader, config, device):
    model.eval()
    scores, indices = [], []
    for windows, mask, _, idx in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = model(windows.to(device), mask.to(device), config['temperature'], config['window_chunk_size'])
        scores.extend(logits.float().sigmoid().cpu().tolist())
        indices.extend(idx.tolist())
    frame = loader.dataset.frame.iloc[indices].copy()
    frame['MUSIC_FAKE_PROB'] = scores
    rows = []
    for axis in ['DATASET', 'CHANNEL_V57', 'LAYOUT_V57', 'CELL_V57']:
        for group, block in frame.groupby(axis):
            rows.append(dict(axis=axis, group=str(group),
                **component_metrics(block, 'MUSIC_FAKE_PROB', 'music')))
    pooled = component_metrics(frame, 'MUSIC_FAKE_PROB', 'music')['eer']
    domains = [r['eer'] for r in rows if r['axis'] == 'DATASET' and r['eer'] is not None]
    if pooled is None or not domains:
        raise ValueError('Music EER selection is undefined')
    selection = 1 - .5 * pooled - .25 * np.mean(domains) - .25 * max(domains)
    return frame[['DATASET', 'ID', 'MUSIC_FAKE_PROB']], rows, float(selection), pooled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/lowband_music_v67.yaml')
    parser.add_argument('--variant', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = yaml.safe_load(args.config.read_text())
    phase = config['variants'][args.variant]['phase']
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    torch.set_num_threads(6)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    train, development, audit = load_data(config)
    all_train_rows = len(train)
    train = train[train.MUSIC_PRESENT.eq(1)].copy()
    if args.smoke:
        train = pd.concat([train[train.MUSIC_FAKE.eq(y)].head(4) for y in [0, 1]])
        development = pd.concat([development[development.MUSIC_PRESENT.eq(1) & development.MUSIC_FAKE.eq(y)].head(4) for y in [0, 1]])
        config.update(epochs=1, eval_every=1, samples_per_epoch=8, batch_size=2, workers=0)
    args.output.mkdir(parents=True)
    provenance = dict(config=config, variant=args.variant, smoke=args.smoke, split_audit=audit,
        train_rows=len(train), all_authorized_train_rows=all_train_rows, development_rows=len(development),
        config_sha256=sha256(args.config), script_sha256=sha256(__file__),
        source_sha256=sha256(ROOT / 'src/lowband_music_v67.py'),
        data_loader_sha256=sha256(ROOT / 'scripts/train_paired_wpt_file_v60.py'))
    (args.output / 'manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(dict(validated=True, train=len(train), development=len(development),
                         identity_overlap=audit['strict']['overlap'], smoke=args.smoke)), flush=True)
    device = torch.device(args.device)
    model = LowbandMusic(phase=phase).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(config['seed'])
    train_loader = make_loader(train, config, True, generator)
    dev_loader = make_loader(development, config, False)
    history, best, stale = [], -float('inf'), 0
    start = time.monotonic()
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        losses = []
        for step, (windows, mask, target, _) in enumerate(train_loader):
            target = target.to(device).repeat(2)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                logits = model(windows.to(device), mask.to(device), config['temperature'], config['window_chunk_size'])
            pos, neg = logits[target.eq(1)], logits[target.eq(0)]
            ranking = F.softplus(-(pos[:, None] - neg[None])).mean() if len(pos) and len(neg) else logits.sum() * 0
            loss = F.binary_cross_entropy_with_logits(logits, target) + config['ranking_weight'] * ranking
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite Music loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % 100 == 0:
                print(json.dumps(dict(epoch=epoch, step=step, loss=float(np.mean(losses[-100:])))), flush=True)
        if epoch % config['eval_every'] and epoch != config['epochs']:
            continue
        prediction, metrics, selection, eer = evaluate(model, dev_loader, config, device)
        record = dict(epoch=epoch, loss=float(np.mean(losses)), selection=selection,
                      pooled_music_eer=eer, elapsed_seconds=time.monotonic() - start)
        history.append(record)
        pd.DataFrame(history).to_csv(args.output / 'history.csv', index=False)
        print(json.dumps(record), flush=True)
        if selection > best + 1e-5:
            best, stale = selection, 0
            torch.save(dict(model_type='lowband_music_v67', state=model.state_dict(), config=config,
                variant=args.variant, phase=phase, smoke=args.smoke, best_epoch=epoch,
                selection=selection, provenance=provenance), args.output / 'best.pt')
            prediction.to_csv(args.output / 'dev_predictions.csv', index=False)
            pd.DataFrame(metrics).to_csv(args.output / 'dev_metrics.csv', index=False)
        else:
            stale += 1
            if stale >= config['patience']:
                break
    (args.output / 'complete.json').write_text(json.dumps(dict(best_selection=best,
        elapsed_seconds=time.monotonic() - start)) + '\n')


if __name__ == '__main__':
    main()
