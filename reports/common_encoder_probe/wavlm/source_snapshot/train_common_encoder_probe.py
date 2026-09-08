#!/usr/bin/env python3
"""Train paired mean/attention readouts on identical frozen raw encoder tokens."""
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import yaml

from src.common_encoder_probe import CommonTokenHead, FrozenEncoderTokens, complete_windows, component_loss
from src.pipeline import load_audio
from src.full_coverage_wpt import masked_lme
from src.evaluate_diagnostic import score_frame, PREDICTION_COLUMNS, LABEL_COLUMNS
from train_paired_wpt_file_v60 import load_data, balanced_weights, sha256
from src.data_guard import (identity_tokens, PROTECTED_FROM_TRAIN_ROLES,
                            ROW_ONLY_PROTECTED_ROLES, assert_development_eval_separation)


def audit_protected_sources(train, partition_config):
    """Identity metadata only; no protected audio/features/predictions read."""
    assert_development_eval_separation(partition_config)
    roles = yaml.safe_load(Path(partition_config).read_text())
    root = Path(partition_config).resolve().parent.parent
    tokens = identity_tokens(train)
    ids = set(train.ID.astype(str))
    records = []
    for role in (*PROTECTED_FROM_TRAIN_ROLES, *ROW_ONLY_PROTECTED_ROLES):
        for relative in roles.get(role, []):
            path = root / relative
            protected = pd.read_csv(path, dtype=str)
            overlap = (ids & set(protected.ID)) if role in ROW_ONLY_PROTECTED_ROLES else (tokens & identity_tokens(protected))
            if overlap:
                raise ValueError(f'protected {role} identity overlap: {path}: {sorted(overlap)[:5]}')
            records.append(dict(role=role, path=str(path), sha256=sha256(path), overlap=0))
    return records


class FileBags(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        windows, lengths, _ = complete_windows(load_audio(Path(row.PATH)))
        target = row[LABEL_COLUMNS].to_numpy(np.float32)
        return windows, lengths, target, index


def collate(items):
    counts = [len(item[0]) for item in items]
    return (torch.from_numpy(np.concatenate([item[0] for item in items])),
            torch.from_numpy(np.concatenate([item[1] for item in items])), counts,
            torch.tensor(np.stack([item[2] for item in items])), [item[3] for item in items])


def file_logits(encoder, heads, windows, lengths, counts, chunk_size, temperature):
    scores = {name: [] for name in heads}
    for start in range(0, len(windows), chunk_size):
        tokens, mask = encoder(windows[start:start + chunk_size], lengths[start:start + chunk_size])
        for name, head in heads.items():
            scores[name].append(head(tokens, mask))
    result = {}
    for name, pieces in scores.items():
        values = torch.cat(pieces)
        mask = torch.zeros(len(counts), max(counts), dtype=torch.bool, device=values.device)
        padded = values.new_zeros((len(counts), max(counts), 5))
        offset = 0
        for i, count in enumerate(counts):
            padded[i, :count] = values[offset:offset + count]
            mask[i, :count] = True
            offset += count
        result[name] = masked_lme(padded, mask, temperature)
    return result


def metrics(frame):
    overall = score_frame(frame)
    records = []
    for axis in ['DATASET', 'CHANNEL_V57', 'LAYOUT_V57', 'VOICE_GENERATOR', 'MUSIC_GENERATOR']:
        if axis in frame:
            for group, selected in frame.groupby(axis, dropna=False):
                records.append(dict(axis=axis, group=str(group), **score_frame(selected)))
    domains = [r for r in records if r['axis'] == 'DATASET']
    # Pure-only domains have undefined component metrics; macro EACH task over
    # its valid domains, then apply official task weights. Do not discard pure.
    macro = 0.
    for key, weight in [('FILE_EER', .5), ('VOICE_EER', .2), ('MUSIC_EER', .3)]:
        values = [r[key] for r in domains if np.isfinite(r[key])]
        if not values:
            raise ValueError(f'no valid development domains for {key}')
        macro += weight * (1 - float(np.mean(values)))
    selection = .5 * overall['ADS'] + .5 * macro
    return dict(overall=overall, macro_ads=macro, selection=selection), pd.DataFrame(records)


@torch.no_grad()
def evaluate(encoder, heads, batches, config):
    for head in heads.values():
        head.eval()
    predictions, indices = {name: [] for name in heads}, []
    for windows, lengths, counts, _, index in batches:
        outputs = file_logits(encoder, heads, windows, lengths, counts,
                              config['encoder_chunk'], config['temperature'])
        for name, values in outputs.items():
            predictions[name].extend(values.sigmoid().cpu().numpy())
        indices.extend(index)
    results = {}
    for name, values in predictions.items():
        frame = batches.dataset.frame.iloc[indices].copy()
        frame[PREDICTION_COLUMNS] = np.asarray(values)
        summary, slices = metrics(frame)
        results[name] = (frame, summary, slices)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', choices=['eat_large', 'spear', 'xlsr', 'wavlm'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/common_encoder_probe.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    started = time.monotonic()
    train, dev, audit = load_data(config)
    audit['all_protected_roles'] = audit_protected_sources(train, ROOT / config['partition_config'])
    train[['DATASET', 'ID']].to_csv(args.output / 'train_ids.csv', index=False)
    dev[['DATASET', 'ID']].to_csv(args.output / 'development_ids.csv', index=False)
    source_files = [Path(__file__), ROOT / 'src/common_encoder_probe.py', args.config]
    snapshot = args.output / 'source_snapshot'
    snapshot.mkdir()
    for path in source_files:
        (snapshot / path.name).write_bytes(path.read_bytes())
    manifest = dict(encoder=args.encoder, config=config, audit=audit, train_rows=len(train),
                    development_rows=len(dev), purpose='frozen readout benchmark, not a submission',
                    code_sha256={str(p): sha256(p) for p in source_files})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    generator = torch.Generator().manual_seed(config['seed'])
    sampler = WeightedRandomSampler(balanced_weights(train), config['samples_per_epoch'],
                                    replacement=True, generator=generator)
    kwargs = dict(batch_size=config['batch_size'], num_workers=config['workers'],
                  collate_fn=collate, persistent_workers=config['workers'] > 0)
    training = DataLoader(FileBags(train), sampler=sampler, **kwargs)
    development = DataLoader(FileBags(dev), shuffle=False, **kwargs)
    encoder = FrozenEncoderTokens(args.encoder, ROOT / config['encoders'][args.encoder])
    first = FileBags(train)[0]
    tokens, _ = encoder(torch.tensor(first[0][:1]), torch.tensor(first[1][:1]))
    width = tokens.shape[-1]
    template = CommonTokenHead(width, width=config['head_width']).cuda()
    initial = copy.deepcopy(template.state_dict())
    heads = {name: CommonTokenHead(width, width=config['head_width'], pooling=name).cuda()
             for name in config['pooling_conditions']}
    for head in heads.values():
        head.load_state_dict(initial)
    optimizers = {name: torch.optim.AdamW(head.parameters(), lr=config['learning_rate'],
                                         weight_decay=config['weight_decay']) for name, head in heads.items()}
    best = {name: -float('inf') for name in heads}
    history = []
    for epoch in range(1, config['epochs'] + 1):
        for head in heads.values():
            head.train()
        losses = {name: [] for name in heads}
        draws = []
        for step, (windows, lengths, counts, target, index) in enumerate(training):
            outputs = file_logits(encoder, heads, windows, lengths, counts,
                                  config['encoder_chunk'], config['temperature'])
            target = target.cuda()
            for name, output in outputs.items():
                loss = component_loss(output, target)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'{name}: nonfinite loss')
                optimizers[name].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(heads[name].parameters(), 5., error_if_nonfinite=True)
                optimizers[name].step()
                losses[name].append(float(loss.detach()))
            draws.extend(index)
            if step % 100 == 0:
                print(json.dumps(dict(epoch=epoch, step=step, losses={k: float(np.mean(v)) for k, v in losses.items()})), flush=True)
        np.save(args.output / f'epoch_{epoch}_train_draws.npy', np.asarray(draws))
        if epoch % config['eval_every'] == 0:
            result = evaluate(encoder, heads, development, config)
            for name, (frame, summary, slices) in result.items():
                directory = args.output / name
                directory.mkdir(exist_ok=True)
                row = dict(epoch=epoch, pooling=name, loss=float(np.mean(losses[name])), **summary)
                history.append(row)
                print(json.dumps(row), flush=True)
                if summary['selection'] > best[name]:
                    best[name] = summary['selection']
                    frame.to_csv(directory / 'development_predictions.csv', index=False)
                    slices.to_csv(directory / 'development_slices.csv', index=False)
                    torch.save(dict(state_dict=heads[name].state_dict(), input_dimension=width,
                                    pooling=name, epoch=epoch, config=config, encoder=args.encoder,
                                    selection=summary, purpose='frozen-readout benchmark'), directory / 'head.pt')
            (args.output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    (args.output / 'completed.json').write_text(json.dumps(dict(best_selection=best, elapsed_seconds=time.monotonic() - started), indent=2) + '\n')


if __name__ == '__main__':
    main()
