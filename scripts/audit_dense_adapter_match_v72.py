#!/usr/bin/env python3
"""Verify completed v72 really changes adapter training, not data/exposure.

No inference, fitting, model selection, or protected audio is performed here.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import torch

from train_paired_wpt_file_v60 import sha256
from finalize_dense_component_v71 import audit_draws, audit_run


def compare_configs(reference, adapted):
    for key, value in reference.items():
        if key != 'schema' and adapted.get(key) != value:
            raise ValueError(f'v72 changed matched v71 configuration: {key}')
    if adapted.get('schema') != 'dense_component_adapter_v72':
        raise ValueError('not a v72 configuration')
    if adapted.get('adapter_learning_rate') != .0001:
        raise ValueError('adapter learning rate differs from preregistered value')


def audit_match(reference_run, adapted_run):
    reference_audit = audit_run(reference_run)
    completion = json.loads((adapted_run / 'completed.json').read_text())
    manifest = json.loads((adapted_run / 'manifest.json').read_text())
    if completion['smoke'] or manifest['smoke']:
        raise ValueError('smoke is not a completed training comparison')
    config = manifest['config']
    compare_configs(reference_audit['config'], config)
    old_manifest = json.loads((reference_run / 'manifest.json').read_text())
    for key in ['train_rows', 'development_rows', 'parent_checkpoint_sha256', 'source_catalog_sha256']:
        if manifest[key] != old_manifest[key]:
            raise ValueError('matched training metadata differs: ' + key)
    for filename in ['train_ids.csv', 'development_ids.csv', 'source_catalog.csv']:
        if sha256(reference_run / filename) != sha256(adapted_run / filename):
            raise ValueError('matched ordered data identities differ: ' + filename)
    for source, digest in manifest['code_sha256'].items():
        if sha256(adapted_run / 'source_snapshot' / Path(source).name) != digest:
            raise ValueError('v72 training source snapshot changed')
        if source in old_manifest['code_sha256'] and old_manifest['code_sha256'][source] != digest:
            raise ValueError('shared source implementation differs: ' + source)
    if manifest['initial_v71_probability_difference'] > 5e-4:
        raise ValueError('initial forward is not numerically matched')
    draws = audit_draws(adapted_run, config)
    trace_hashes = []
    for epoch in range(1, config['epochs'] + 1):
        filename = f'epoch_{epoch}_draws.json'
        digest = sha256(reference_run / filename)
        if sha256(adapted_run / filename) != digest:
            raise ValueError(f'actual data draws differ in epoch {epoch}')
        trace_hashes.append(dict(epoch=epoch, sha256=digest))
    history = json.loads((adapted_run / 'history.json').read_text())
    if [row['epoch'] for row in history] != list(range(2, config['epochs'] + 1, 2)):
        raise ValueError('missing scheduled development evaluation')
    best = max(history, key=lambda row: row['selection'])
    checkpoint_path = adapted_run / 'adapted_dense/head.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if (checkpoint['smoke'] or checkpoint['model_type'] != 'dense_component_adapter_v72'
            or checkpoint['variant'] != 'adapted_dense' or checkpoint['config'] != config
            or checkpoint['epoch'] != best['epoch'] or checkpoint['selection']['selection'] != best['selection']
            or completion['best_selection'] != best['selection']
            or checkpoint['parent_checkpoint_sha256'] != manifest['parent_checkpoint_sha256']):
        raise ValueError('v72 selected checkpoint violates the declared rule')
    parent_path = ROOT / config['parent_checkpoint']
    if sha256(parent_path) != manifest['parent_checkpoint_sha256']:
        raise ValueError('frozen initial parent checkpoint changed')
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    if set(parent['adapters']) != set(checkpoint['adapters']):
        raise ValueError('adapter state schema changed')
    changed = sum(not torch.equal(value, parent['adapters'][key]) for key, value in checkpoint['adapters'].items())
    if changed == 0:
        raise ValueError('adapter weights never changed')
    return dict(stage='completed v71/v72 matched training audit; NOT accuracy or official improvement',
        actual_epoch_draws_bit_exact=True, trace_hashes=trace_hashes,
        data_counts=draws['counts'], unique_consumed_raw_sources=draws['unique_consumed_raw_sources'],
        selected_epoch=best['epoch'], selected_development=best, adapter_tensors_changed=changed,
        initial_probability_difference=manifest['initial_v71_probability_difference'],
        head_parameters=manifest['trainable_head_parameters'], adapter_parameters=manifest['trainable_adapter_parameters'],
        checkpoint_sha256=sha256(checkpoint_path), automatic_submission_allowed=False,
        limitations=['matching is against v71 dense_supervised, not the earlier global EAT model',
                     'additional adapter parameters/compute are intentional and not a fixed-capacity comparison',
                     'full independent inference and transfer diagnostics are still required'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, default=ROOT / 'reports/dense_component_v71/full')
    parser.add_argument('--adapted-run', type=Path, default=ROOT / 'reports/dense_component_adapter_v72/full')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit_match(args.reference_run, args.adapted_run)
    args.output.mkdir(parents=True)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'trace_hashes'}), flush=True)


if __name__ == '__main__':
    main()
