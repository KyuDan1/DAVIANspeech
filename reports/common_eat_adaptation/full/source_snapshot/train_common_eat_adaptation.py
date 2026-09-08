#!/usr/bin/env python3
"""Paired EAT frozen-control versus representation adaptation on all audio types."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml

from src.common_encoder_probe import FrozenEncoderTokens, CommonTokenHead, component_loss
from src.common_eat_adaptation import PairedEatRepresentation, paired_file_logits
from src.evaluate_diagnostic import PREDICTION_COLUMNS
from train_common_encoder_probe import FileBags, collate, metrics, audit_protected_sources
from train_paired_wpt_file_v60 import load_data, balanced_weights, sha256


@torch.no_grad()
def evaluate(encoder, heads, loader, config):
    encoder.eval()
    for head in heads.values():
        head.eval()
    predictions = {name: [] for name in heads}
    indices = []
    for windows, lengths, counts, _, index in loader:
        outputs = paired_file_logits(encoder, heads, windows, lengths, counts,
                                     config['encoder_chunk'], config['temperature'])
        for name, output in outputs.items():
            predictions[name].append(output.sigmoid().cpu().numpy())
        indices.extend(index)
    result = {}
    for name in heads:
        frame = loader.dataset.frame.iloc[indices].copy()
        frame[PREDICTION_COLUMNS] = np.concatenate(predictions[name])
        result[name] = frame
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/common_eat_adaptation.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    started = time.monotonic()
    train, dev, audit = load_data(config)
    audit['all_protected_roles'] = audit_protected_sources(train, ROOT / config['partition_config'])
    if args.smoke:
        # Infrastructure only; never create development accuracy claims from smoke.
        train, dev = train.head(8).copy(), dev.head(8).copy()
    for name, frame in [('train', train), ('development', dev)]:
        frame[['DATASET', 'ID']].to_csv(args.output / f'{name}_ids.csv', index=False)
    source = [Path(__file__), args.config, ROOT / 'src/common_eat_adaptation.py',
              ROOT / 'src/common_encoder_probe.py', ROOT / 'src/eat_music_adapter.py',
              ROOT / 'scripts/train_common_encoder_probe.py']
    snapshot = args.output / 'source_snapshot'
    snapshot.mkdir()
    for path in source:
        (snapshot / path.name).write_bytes(path.read_bytes())
    manifest = dict(config=config, audit=audit, train_rows=len(train), development_rows=len(dev),
                    smoke=args.smoke, purpose='paired representation experiment, no automatic submission',
                    code_sha256={str(p): sha256(p) for p in source})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    base = FrozenEncoderTokens('eat_large', ROOT / config['encoder_path'])
    paired = PairedEatRepresentation(base.model, config['adapter_blocks'], config['adapter_bottleneck']).cuda()
    first = FileBags(train)[0]
    # Verify the native control implementation and zero-initialized adapters
    # before any training. Also checks frontend/captured-layer equivalence.
    with torch.no_grad():
        windows, lengths = torch.tensor(first[0][:1]), torch.tensor(first[1][:1])
        reference_tokens, reference_mask = base(windows, lengths)
        pair, mask = paired(windows, lengths)
        torch.testing.assert_close(mask, reference_mask)
        for tokens in pair.values():
            torch.testing.assert_close(tokens, reference_tokens, rtol=0, atol=0)
    width = reference_tokens.shape[-1]
    template = CommonTokenHead(width, width=config['head_width'], pooling=config['pooling']).cuda()
    initial = copy.deepcopy(template.state_dict())
    heads = {name: CommonTokenHead(width, width=config['head_width'], pooling=config['pooling']).cuda()
             for name in ['control', 'adapted']}
    for head in heads.values():
        head.load_state_dict(initial)
    optimizers = {
        'control': torch.optim.AdamW(heads['control'].parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay']),
        'adapted': torch.optim.AdamW([
            {'params': heads['adapted'].parameters(), 'lr': config['learning_rate']},
            {'params': paired.adapters.parameters(), 'lr': config['adapter_learning_rate']}], weight_decay=config['weight_decay'])}
    draws_count = 8 if args.smoke else config['samples_per_epoch']
    sampler = WeightedRandomSampler(balanced_weights(train), draws_count, replacement=True,
                                    generator=torch.Generator().manual_seed(config['seed']))
    kwargs = dict(batch_size=config['batch_size'], num_workers=0 if args.smoke else config['workers'],
                  collate_fn=collate)
    training = DataLoader(FileBags(train), sampler=sampler, **kwargs)
    development = DataLoader(FileBags(dev), shuffle=False, **kwargs)
    manifest['trainable_adapter_parameters'] = sum(p.numel() for p in paired.adapters.parameters())
    manifest['initial_control_and_adapted_tokens_exact'] = True
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    best = {name: -float('inf') for name in heads}
    history = []
    for epoch in range(1, (1 if args.smoke else config['epochs']) + 1):
        paired.train()
        for head in heads.values():
            head.train()
        losses, draws = {name: [] for name in heads}, []
        for step, (windows, lengths, counts, target, index) in enumerate(training):
            outputs = paired_file_logits(paired, heads, windows, lengths, counts,
                                         config['encoder_chunk'], config['temperature'])
            for name, output in outputs.items():
                loss = component_loss(output, target.cuda())
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite paired loss')
                optimizers[name].zero_grad()
                loss.backward()
                parameters = list(heads[name].parameters()) + (list(paired.adapters.parameters()) if name == 'adapted' else [])
                torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
                optimizers[name].step()
                losses[name].append(float(loss.detach()))
            if any(p.grad is not None for p in paired.base.parameters()):
                raise RuntimeError('frozen backbone unexpectedly accumulated gradients')
            draws.extend(index)
            if step % 100 == 0:
                print(json.dumps(dict(epoch=epoch, step=step, losses={k: float(np.mean(v)) for k, v in losses.items()})), flush=True)
        np.save(args.output / f'epoch_{epoch}_train_draws.npy', np.asarray(draws))
        if args.smoke or epoch % config['eval_every'] == 0:
            results = evaluate(paired, heads, development, config)
            for name, frame in results.items():
                if args.smoke:
                    summary, slices = {'selection': 0., 'smoke_only': True}, None
                else:
                    summary, slices = metrics(frame)
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
                    torch.save(dict(model_type='common_eat_adaptation_v1', state_dict=heads[name].state_dict(),
                        adapters=paired.adapters.state_dict() if name == 'adapted' else None,
                        input_dimension=width, config=config, variant=name, epoch=epoch,
                        selection=summary, smoke=args.smoke), directory / 'head.pt')
            (args.output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    (args.output / 'completed.json').write_text(json.dumps(dict(best_selection=best, smoke=args.smoke,
        elapsed_seconds=time.monotonic() - started), indent=2) + '\n')


if __name__ == '__main__':
    main()
