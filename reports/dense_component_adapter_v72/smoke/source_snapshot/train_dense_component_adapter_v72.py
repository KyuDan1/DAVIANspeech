#!/usr/bin/env python3
"""v71 matched data, but allow the parent EAT adapters to learn dense labels."""
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
from torch.utils.data import DataLoader
import yaml

from src.common_encoder_probe import component_loss
from src.common_eat_adaptation_inference import CommonEatAdaptationPredictor
from src.dense_component_v71 import DenseComponentHead, dense_component_loss, paired_dense_logits
from src.dense_component_adapter_v72 import adapted_dense_logits
from src.dense_component_data_v71 import DenseTrainingBags, dense_collate
from src.full_coverage_wpt import resolve_ffmpeg
from train_dense_component_v71 import audited_catalog, evaluate
from train_common_encoder_probe import FileBags, collate, metrics, audit_protected_sources
from train_paired_wpt_file_v60 import load_data, balanced_weights, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/dense_component_adapter_v72.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    overrides = yaml.safe_load(args.config.read_text())
    config = yaml.safe_load((ROOT / overrides['base_config']).read_text())
    config.update(overrides)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    started = time.monotonic()
    train, dev, audit = load_data(config)
    audit['all_protected_roles'] = audit_protected_sources(train, ROOT / config['partition_config'])
    catalog = audited_catalog(config, train)
    if args.smoke:
        dev = dev.head(8).copy()
    for name, frame in [('train', train), ('development', dev)]:
        frame[['DATASET', 'ID']].to_csv(args.output / f'{name}_ids.csv', index=False)
    catalog.to_csv(args.output / 'source_catalog.csv', index=False)
    sources = [Path(__file__), args.config, ROOT / overrides['base_config'], ROOT / 'src/dense_component_adapter_v72.py',
        ROOT / 'src/dense_component_v71.py', ROOT / 'src/dense_component_data_v71.py',
        ROOT / 'src/common_encoder_probe.py', ROOT / 'src/common_eat_adaptation.py', ROOT / 'src/long_component_stress.py',
        ROOT / 'src/common_eat_adaptation_inference.py', ROOT / 'src/telephone_channel.py',
        ROOT / 'scripts/train_dense_component_v71.py', ROOT / 'scripts/train_common_encoder_probe.py',
        ROOT / 'scripts/train_paired_wpt_file_v60.py']
    snapshot = args.output / 'source_snapshot'
    snapshot.mkdir()
    for path in sources:
        (snapshot / path.name).write_bytes(path.read_bytes())
    manifest = dict(config=config, audit=audit, train_rows=len(train), development_rows=len(dev), smoke=args.smoke,
        source_catalog_sha256=sha256(ROOT / config['source_catalog']),
        parent_checkpoint_sha256=sha256(ROOT / config['parent_checkpoint']),
        code_sha256={str(path): sha256(path) for path in sources})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(stage='source_audit_pass', train=len(train), dev=len(dev), raw_sources=len(catalog))), flush=True)
    parent = CommonEatAdaptationPredictor(ROOT / config['parent_checkpoint'], ROOT)
    encoder = parent.encoder.eval().requires_grad_(False)
    encoder.adapters.requires_grad_(True)
    head = DenseComponentHead(width=config['head_width'], pooling=config['pooling']).cuda()
    head.readout.load_state_dict(copy.deepcopy(parent.head.state_dict()), strict=True)
    optimizer = torch.optim.AdamW([
        {'params': head.parameters(), 'lr': config['learning_rate']},
        {'params': encoder.adapters.parameters(), 'lr': config['adapter_learning_rate']}], weight_decay=config['weight_decay'])
    parameters = list(head.parameters()) + list(encoder.adapters.parameters())
    kwargs = dict(batch_size=config['batch_size'], num_workers=0 if args.smoke else config['workers'])
    development = DataLoader(FileBags(dev), shuffle=False, collate_fn=collate, **kwargs)
    probabilities, ffmpeg = balanced_weights(train), resolve_ffmpeg()
    # Match initial logits to v71 before any optimization, including differentiable
    # versus no-grad frozen-tail implementations on a whole synthetic TRAIN file.
    first_dataset = DenseTrainingBags(train, catalog, config, probabilities, ffmpeg, draws=1)
    windows, lengths, counts, _, _, _, _ = dense_collate([first_dataset[0]])
    with torch.no_grad():
        reference, _, _ = paired_dense_logits(encoder, {'dense_supervised': head}, windows, lengths, counts,
            config['encoder_chunk'], config['temperature'])
    initial, dense, valid = adapted_dense_logits(encoder, head, windows, lengths, counts, config['encoder_chunk'], config['temperature'])
    initial_difference = float((initial.sigmoid() - reference['dense_supervised'].sigmoid()).abs().max().detach())
    if initial_difference > 5e-4:
        raise ValueError('initial adapter-gradient forward does not reproduce v71')
    del initial, dense, valid, reference
    manifest.update(initial_v71_probability_difference=initial_difference,
        trainable_adapter_parameters=sum(p.numel() for p in encoder.adapters.parameters()),
        trainable_head_parameters=sum(p.numel() for p in head.parameters()))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    history, best = [], -float('inf')
    for epoch in range(1, (1 if args.smoke else config['epochs']) + 1):
        dataset = DenseTrainingBags(train, catalog, config, probabilities, ffmpeg,
                                   draws=8 if args.smoke else config['samples_per_epoch'], epoch=epoch)
        training = DataLoader(dataset, shuffle=False, collate_fn=dense_collate, **kwargs)
        encoder.train()
        head.train()
        losses, traces = [], []
        for step, (windows, lengths, counts, target, dense, valid, trace) in enumerate(training):
            output, dense_logits, temporal_mask = adapted_dense_logits(encoder, head, windows, lengths, counts,
                config['encoder_chunk'], config['temperature'])
            loss = component_loss(output, target.cuda()) + config['dense_loss_weight'] * dense_component_loss(
                dense_logits, dense.cuda(), valid.cuda() & temporal_mask[..., None])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite adapted dense loss')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
            if any(p.grad is not None for p in encoder.base.parameters()):
                raise ValueError('original EAT weights unexpectedly received gradients')
            if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.adapters.parameters()):
                raise ValueError('dense adaptation produced no adapter gradients')
            optimizer.step()
            losses.append(float(loss.detach()))
            traces.extend(trace)
            if step % 100 == 0:
                print(json.dumps(dict(epoch=epoch, step=step, loss=float(np.mean(losses)),
                    cuda_peak_mb=torch.cuda.max_memory_allocated() / 2**20)), flush=True)
        (args.output / f'epoch_{epoch}_draws.json').write_text(json.dumps(traces) + '\n')
        if args.smoke or epoch % 2 == 0:
            encoder.eval()
            frame = evaluate(encoder, {'adapted_dense': head}, development, config)['adapted_dense']
            summary, slices = ({'selection': 0., 'smoke_only': True}, None) if args.smoke else metrics(frame)
            row = dict(epoch=epoch, variant='adapted_dense', loss=float(np.mean(losses)), **summary)
            history.append(row)
            print(json.dumps(row), flush=True)
            if summary['selection'] > best:
                best = summary['selection']
                directory = args.output / 'adapted_dense'
                directory.mkdir(exist_ok=True)
                frame.to_csv(directory / 'development_predictions.csv', index=False)
                if slices is not None:
                    slices.to_csv(directory / 'development_slices.csv', index=False)
                torch.save(dict(model_type='dense_component_adapter_v72', state_dict=head.state_dict(),
                    adapters=encoder.adapters.state_dict(), config=config, variant='adapted_dense',
                    epoch=epoch, selection=summary, smoke=args.smoke,
                    parent_checkpoint_sha256=manifest['parent_checkpoint_sha256']), directory / 'head.pt')
            (args.output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    (args.output / 'completed.json').write_text(json.dumps(dict(smoke=args.smoke, best_selection=best,
        elapsed_seconds=time.monotonic() - started, cuda_peak_mb=torch.cuda.max_memory_allocated() / 2**20), indent=2) + '\n')


if __name__ == '__main__':
    main()
