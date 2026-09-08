#!/usr/bin/env python3
"""Finalize v64: matched training ablation plus primary fixed-WPT-slot gate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from finalize_paired_wpt_v60 import completed_variant, wait_for_training
from evaluate_wpt_slot_v62 import compose
from build_prospective_mixed_phone_v3 import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wait-for-training', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config_path = ROOT / 'configs/paired_peak_wpt_v64.yaml'
    config = yaml.safe_load(config_path.read_text())
    runs = ROOT / 'reports/paired_peak_wpt_v64'
    if args.wait_for_training:
        wait_for_training(runs, variants=('paired_peak',))
    candidate_dir = runs / 'paired_peak'
    control_dir = ROOT / config['decision']['baseline_training_run']
    candidate_info, control_info = completed_variant(candidate_dir), completed_variant(control_dir)
    checkpoint = torch.load(candidate_dir / 'best.pt', map_location='cpu', weights_only=False)
    if checkpoint['config'].get('peak_normalize') is not True:
        raise ValueError('candidate checkpoint normalization is not enabled')
    # Metadata contains numerical/data conditions, not just a claimed experiment name.
    ignored = {'schema_version', 'variants', 'decision', 'peak_normalize'}
    control_config = json.loads((control_dir / 'manifest.json').read_text())['config']
    if {k: v for k, v in control_config.items() if k not in ignored} != {
            k: v for k, v in checkpoint['config'].items() if k not in ignored}:
        raise ValueError('candidate training conditions do not match BCE control')
    args.output.mkdir(parents=True)
    evaluator = [sys.executable, str(ROOT / 'scripts/evaluate_music_specialist_v58.py'),
                 '--task', 'file', '--matrix', str(config_path)]
    def compare(incumbent, candidate, output, subset=False):
        command = evaluator + ['--incumbent', str(incumbent), '--candidate', str(candidate), '--output', str(output)]
        if subset:
            command += ['--datasets', 'codec_mixed_dev_v4']
        subprocess.run(command, cwd=ROOT, check=True)
        return json.loads(output.read_text())['decision']
    training_gate = compare(control_dir / 'dev_predictions.csv', candidate_dir / 'dev_predictions.csv',
                            args.output / 'matched_training_ablation.json')
    anchor_path = ROOT / 'reports/three_stream_all_type_v57_strict/v50_authorized_v2/v1_strict__music_only_predictions.csv'
    direct_gate = compare(anchor_path, candidate_dir / 'dev_predictions.csv', args.output / 'direct_vs_anchor.json', True)
    old_path = ROOT / 'reports/wpt_slot_v62/anchor_verification/predictions.csv'
    verification_path = old_path.parent / 'verification.json'
    if not json.loads(verification_path.read_text())['historical_cache_close']:
        raise ValueError('old expert cache was not reproduced')
    anchor, old, new = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in
                        [anchor_path, old_path, candidate_dir / 'dev_predictions.csv']]
    new = new[new.DATASET.eq('codec_mixed_dev_v4')]
    slot = compose(anchor, old, new, config['decision']['slot_weight'])
    slot_path = args.output / 'slot_predictions.csv'
    slot.to_csv(slot_path, index=False)
    slot_gate = compare(anchor_path, slot_path, args.output / 'slot_vs_anchor.json', True)
    for directory, info in [(candidate_dir, candidate_info), (control_dir, control_info)]:
        for name, digest in info['files'].items():
            if sha256_file(directory / name) != digest:
                raise ValueError('completed run artifacts changed during comparison')
    report = dict(stage='development_only', primary_decision=slot_gate,
        matched_training_ablation=training_gate, direct_replacement_diagnostic=direct_gate,
        candidate=candidate_info, control=control_info, slot_weight=config['decision']['slot_weight'],
        normalization_enabled=True, other_four_slot_outputs_text_exact=True,
        official_improvement_verified=False, automatic_submission_allowed=False,
        remaining_if_pass=['independent_file_equivalence', 'prospective_all_type', 'long_voice', 'L4_runtime', 'offline_package'],
        files={str(p): sha256_file(p) for p in [config_path, anchor_path, old_path, slot_path, verification_path]})
    (args.output / 'decision.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(primary_decision=slot_gate), indent=2), flush=True)


if __name__ == '__main__':
    main()
