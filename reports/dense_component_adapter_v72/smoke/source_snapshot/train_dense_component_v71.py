#!/usr/bin/env python3
"""Paired file-only versus dense-interval heads sharing one frozen EAT."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from src.common_encoder_probe import component_loss
from src.common_eat_adaptation_inference import CommonEatAdaptationPredictor
from src.dense_component_v71 import DenseComponentHead, paired_dense_logits, dense_component_loss
from src.dense_component_data_v71 import DenseTrainingBags, dense_collate
from src.data_guard import identity_tokens, PROTECTED_FROM_TRAIN_ROLES
from src.evaluate_diagnostic import PREDICTION_COLUMNS
from src.full_coverage_wpt import resolve_ffmpeg
from train_common_encoder_probe import FileBags, collate, metrics, audit_protected_sources
from train_paired_wpt_file_v60 import load_data, balanced_weights, sha256


def audited_catalog(config, train):
    path = ROOT / config['source_catalog']
    report = json.loads((path.parent / 'report.json').read_text())
    if report['source_catalog_sha256'] != sha256(path):
        raise ValueError('source catalog checksum changed')
    catalog = pd.read_csv(path, dtype={'ID': str, 'SOURCE_ID': str, 'GROUP_ID': str}).fillna({'GENERATOR': 'unknown'})
    if catalog.ID.duplicated().any():
        raise ValueError('duplicate catalog identities')
    # Raw SOURCE_ID is not a generic data_guard column: explicitly map to ID.
    raw_ids = catalog[['SOURCE_ID', 'GROUP_ID']].rename(columns={'SOURCE_ID': 'ID'})
    tokens = identity_tokens(raw_ids)
    train_tokens = identity_tokens(train)
    if not set(catalog.SOURCE_ID).issubset(train_tokens):
        raise ValueError('raw catalog contains a source not represented in strict training')
    roles = yaml.safe_load((ROOT / config['partition_config']).read_text())
    for role in PROTECTED_FROM_TRAIN_ROLES:
        for relative in roles.get(role, []):
            if tokens & identity_tokens(pd.read_csv(ROOT / relative, dtype=str)):
                raise ValueError(f'raw training catalog overlaps protected role {role}: {relative}')
    for bank, frame in catalog.groupby('SOURCE_BANK'):
        raw = ROOT / 'data/eval' / bank / 'truth.csv'
        if set(frame.RAW_TRUTH_SHA256) != {sha256(raw)}:
            raise ValueError('source truth changed since catalog construction')
    return catalog


@torch.no_grad()
def evaluate(encoder, heads, loader, config):
    for head in heads.values():
        head.eval()
    predictions, indices = {name: [] for name in heads}, []
    for windows, lengths, counts, _, index in loader:
        outputs, _, _ = paired_dense_logits(encoder, heads, windows, lengths, counts,
                                             config['encoder_chunk'], config['temperature'])
        for name, logits in outputs.items():
            predictions[name].append(logits.sigmoid().cpu().numpy())
        indices.extend(index)
    result = {}
    for name in heads:
        frame = loader.dataset.frame.iloc[indices].copy()
        frame[PREDICTION_COLUMNS] = np.concatenate(predictions[name])
        result[name] = frame
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/dense_component_v71.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    train, dev, audit = load_data(config)
    audit['all_protected_roles'] = audit_protected_sources(train, ROOT / config['partition_config'])
    catalog = audited_catalog(config, train)
    if args.smoke:
        dev = dev.head(8).copy()
    for name, frame in [('train', train), ('development', dev)]:
        frame[['DATASET', 'ID']].to_csv(args.output / f'{name}_ids.csv', index=False)
    catalog.to_csv(args.output / 'source_catalog.csv', index=False)
    sources = [Path(__file__), args.config, ROOT / 'src/dense_component_v71.py',
        ROOT / 'src/dense_component_data_v71.py', ROOT / 'src/long_component_stress.py',
        ROOT / 'src/common_encoder_probe.py', ROOT / 'src/common_eat_adaptation.py',
        ROOT / 'src/common_eat_adaptation_inference.py', ROOT / 'src/telephone_channel.py',
        ROOT / 'scripts/train_common_encoder_probe.py', ROOT / 'scripts/train_paired_wpt_file_v60.py',
        ROOT / 'scripts/reserve_dense_component_sources_v71.py']
    snapshot = args.output / 'source_snapshot'
    snapshot.mkdir()
    for path in sources:
        (snapshot / path.name).write_bytes(path.read_bytes())
    manifest = dict(config=config, audit=audit, train_rows=len(train), development_rows=len(dev),
        smoke=args.smoke, parent_checkpoint_sha256=sha256(ROOT / config['parent_checkpoint']),
        source_catalog_sha256=sha256(ROOT / config['source_catalog']),
        code_sha256={str(p): sha256(p) for p in sources},
        purpose='TRAIN-only dense labels; paired shared encoder; no automatic submission')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(stage='source_audit_pass', train=len(train), dev=len(dev), raw_sources=len(catalog))), flush=True)
    parent = CommonEatAdaptationPredictor(ROOT / config['parent_checkpoint'], ROOT)
    encoder = parent.encoder.eval().requires_grad_(False)
    heads = {name: DenseComponentHead(width=config['head_width'], pooling=config['pooling']).cuda()
             for name in ['file_only', 'dense_supervised']}
    for head in heads.values():
        head.readout.load_state_dict(copy.deepcopy(parent.head.state_dict()), strict=True)
    for key in heads['file_only'].state_dict():
        torch.testing.assert_close(heads['file_only'].state_dict()[key], heads['dense_supervised'].state_dict()[key], rtol=0, atol=0)
    optimizers = {name: torch.optim.AdamW(head.parameters(), lr=config['learning_rate'],
                    weight_decay=config['weight_decay']) for name, head in heads.items()}
    kwargs = dict(batch_size=config['batch_size'], num_workers=0 if args.smoke else config['workers'])
    development = DataLoader(FileBags(dev), shuffle=False, collate_fn=collate, **kwargs)
    best, history = {name: -float('inf') for name in heads}, []
    probabilities = balanced_weights(train)
    ffmpeg = resolve_ffmpeg()
    for epoch in range(1, (1 if args.smoke else config['epochs']) + 1):
        dataset = DenseTrainingBags(train, catalog, config, probabilities, ffmpeg,
                                   draws=8 if args.smoke else config['samples_per_epoch'], epoch=epoch)
        training = DataLoader(dataset, shuffle=False, collate_fn=dense_collate, **kwargs)
        for head in heads.values():
            head.train()
        losses, traces, dense_count = {name: [] for name in heads}, [], 0
        for step, (windows, lengths, counts, target, dense, valid, trace) in enumerate(training):
            outputs, dense_logits, temporal_mask = paired_dense_logits(encoder, heads, windows, lengths, counts,
                                                config['encoder_chunk'], config['temperature'])
            active = valid.cuda() & temporal_mask[..., None]
            dense_count += int(active.sum())
            for name, output in outputs.items():
                loss = component_loss(output, target.cuda())
                if name == 'dense_supervised':
                    loss = loss + config['dense_loss_weight'] * dense_component_loss(dense_logits[name], dense.cuda(), active)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite paired loss')
                optimizers[name].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(heads[name].parameters(), 5., error_if_nonfinite=True)
                optimizers[name].step()
                losses[name].append(float(loss.detach()))
            if any(p.grad is not None for p in encoder.parameters()):
                raise ValueError('frozen parent accumulated gradients')
            traces.extend(trace)
            if step % 100 == 0:
                print(json.dumps(dict(epoch=epoch, step=step, losses={k: float(np.mean(v)) for k, v in losses.items()},
                    dense_task_bins=dense_count, cuda_peak_mb=torch.cuda.max_memory_allocated() / 2**20)), flush=True)
        if dense_count == 0:
            raise ValueError('paired experiment received no active dense supervision')
        (args.output / f'epoch_{epoch}_draws.json').write_text(json.dumps(traces) + '\n')
        if args.smoke or epoch % 2 == 0:
            results = evaluate(encoder, heads, development, config)
            for name, frame in results.items():
                summary, slices = ({'selection': 0., 'smoke_only': True}, None) if args.smoke else metrics(frame)
                row = dict(epoch=epoch, variant=name, loss=float(np.mean(losses[name])), **summary)
                history.append(row)
                print(json.dumps(row), flush=True)
                if summary['selection'] > best[name]:
                    best[name] = summary['selection']
                    directory = args.output / name
                    directory.mkdir(exist_ok=True)
                    frame.to_csv(directory / 'development_predictions.csv', index=False)
                    if slices is not None:
                        slices.to_csv(directory / 'development_slices.csv', index=False)
                    torch.save(dict(model_type='dense_component_v71', state_dict=heads[name].state_dict(),
                        config=config, variant=name, epoch=epoch, selection=summary, smoke=args.smoke,
                        parent_checkpoint_sha256=manifest['parent_checkpoint_sha256']), directory / 'head.pt')
            (args.output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    (args.output / 'completed.json').write_text(json.dumps(dict(best_selection=best, smoke=args.smoke,
        elapsed_seconds=time.monotonic() - started, cuda_peak_mb=torch.cuda.max_memory_allocated() / 2**20), indent=2) + '\n')


if __name__ == '__main__':
    main()
