#!/usr/bin/env python3
"""Evaluate a predeclared WPT-only slot replacement on authorized development."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from finalize_paired_wpt_v60 import completed_variant, wait_for_training, VARIANTS
from wpt_slot_replacement import replace_wpt_slot
from build_prospective_mixed_phone_v3 import sha256_file


def compose(anchor, old, new, weight):
    keys = ['DATASET', 'ID']
    expected = set(map(tuple, anchor[keys].to_numpy()))
    aligned = []
    for name, frame in [('anchor', anchor), ('old', old), ('new', new)]:
        if frame[keys].duplicated().any() or set(map(tuple, frame[keys].to_numpy())) != expected:
            raise ValueError(f'{name} ID/data keys differ')
        aligned.append(frame.set_index(keys).loc[list(map(tuple, anchor[keys].to_numpy()))])
    probabilities = [f.FILE_FAKE_PROB.to_numpy(dtype=float) for f in aligned]
    replaced = replace_wpt_slot(*probabilities, weight=weight)
    result = anchor.copy()
    result['FILE_FAKE_PROB'] = [f'{value:.10f}' for value in replaced]
    unchanged = [c for c in anchor.columns if c != 'FILE_FAKE_PROB']
    if not result[unchanged].equals(anchor[unchanged]):
        raise ValueError('non-File output changed')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/wpt_slot_v62.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-for-training', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = yaml.safe_load(args.config.read_text())
    runs = ROOT / config['training_runs']
    if args.wait_for_training:
        wait_for_training(runs)
    variants = {name: completed_variant(runs / name) for name in VARIANTS}
    chosen = max(VARIANTS, key=lambda name: variants[name]['selection'])
    verified_root = ROOT / 'reports/wpt_slot_v62/anchor_verification'
    verified = json.loads((verified_root / 'verification.json').read_text())
    if not verified['historical_cache_close']:
        raise ValueError('packaged WPT did not reproduce cached anchor expert; rebase needs investigation')
    if verified['file_views'] != config['old_wpt_views'] or verified['file_temperature'] != config['old_wpt_temperature']:
        raise ValueError('anchor view/temperature differs')
    anchor_path = ROOT / config['anchor_csv']
    new_path = runs / chosen / 'dev_predictions.csv'
    old_path = verified_root / 'predictions.csv'
    # Preserve every non-File CSV value as text, not another float round-trip.
    anchor, old, new = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in [anchor_path, old_path, new_path]]
    datasets = config['development_subset']
    anchor, old, new = [f[f.DATASET.isin(datasets)] for f in [anchor, old, new]]
    result = compose(anchor, old, new, config['slot_weight'])
    args.output.mkdir(parents=True)
    candidate = args.output / 'predictions.csv'
    result.to_csv(candidate, index=False)
    comparison = args.output / 'comparison.json'
    subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_music_specialist_v58.py'),
        '--task', 'file', '--matrix', str(args.config), '--datasets', *datasets,
        '--incumbent', str(anchor_path), '--candidate', str(candidate), '--output', str(comparison)],
        cwd=ROOT, check=True)
    gate = json.loads(comparison.read_text())['decision']
    for name, metadata in variants.items():
        for filename, digest in metadata['files'].items():
            if sha256_file(runs / name / filename) != digest:
                raise ValueError('completed checkpoint artifacts changed')
    report = dict(stage='development_only', experiment='v62_fixed_wpt_slot_replacement',
        chosen_training_variant=chosen, variant_choice=variants, slot_weight=config['slot_weight'],
        decision=gate, other_four_outputs_text_exact=True, official_improvement_verified=False,
        automatic_submission_allowed=False, remaining_gates=config['remaining_gates'],
        files={str(p): sha256_file(p) for p in [args.config, anchor_path, old_path, new_path, candidate,
                                             verified_root / 'verification.json']})
    (args.output / 'decision.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(chosen=chosen, decision=gate), indent=2), flush=True)


if __name__ == '__main__':
    main()
