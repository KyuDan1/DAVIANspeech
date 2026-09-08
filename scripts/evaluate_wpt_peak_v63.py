#!/usr/bin/env python3
"""One fixed normalization change to the submitted WPT slot, development only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_wpt_gain_sensitivity_v63 import peak_normalize_windows
from evaluate_wpt_slot_v62 import compose
from build_prospective_mixed_phone_v3 import sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config_path = ROOT / 'configs/wpt_peak_v63.yaml'
    config = yaml.safe_load(config_path.read_text())
    package = ROOT / 'v50_v57m_challenger_v2'
    verification_path = ROOT / 'reports/wpt_slot_v62/anchor_verification/verification.json'
    verification = json.loads(verification_path.read_text())
    if not verification['historical_cache_close']:
        raise ValueError('anchor cache verification failed')
    for relative, digest in verification['zip_assets_sha256'].items():
        if sha256_file(package / relative) != digest:
            raise ValueError('verified submitted asset changed')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(package / 'model/src'))
    from wpt_spectra_inference import load_wpt_model, fixed_windows, _preemphasis, aggregate_view_logits
    from pipeline import load_audio, find_audio_files
    anchor_path = ROOT / 'reports/three_stream_all_type_v57_strict/v50_authorized_v2/v1_strict__music_only_predictions.csv'
    anchor = pd.read_csv(anchor_path, dtype=str, keep_default_na=False)
    if set(anchor.DATASET) != set(config['development_subset']):
        raise ValueError('unexpected development dataset')
    paths = {p.stem: p for p in find_audio_files(ROOT / 'data/eval/codec_mixed_dev_v4/audio')}
    if set(paths) != set(anchor.ID):
        raise ValueError('development audio keys differ')
    model_dir = package / 'model/spectra-aasist'
    model, model_config = load_wpt_model(model_dir, model_dir / 'wpt_spectra_multitask.pt', torch.device('cuda'))
    values = []
    with torch.inference_mode():
        for offset in range(0, len(anchor), 6):
            ids = anchor.ID.iloc[offset:offset + 6]
            windows = np.stack([fixed_windows(load_audio(paths[i]), model_config['window'], config['file_views']) for i in ids])
            tensor = peak_normalize_windows(_preemphasis(torch.from_numpy(windows).to('cuda')))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = aggregate_view_logits(model.forward_windows(tensor), config['temperature'])
            values.extend(logits[:, 2].sigmoid().cpu().tolist())
    old_path = ROOT / 'reports/wpt_slot_v62/anchor_verification/predictions.csv'
    old = pd.read_csv(old_path, dtype=str, keep_default_na=False)
    new = anchor[['DATASET', 'ID']].copy()
    new['FILE_FAKE_PROB'] = values
    candidate = compose(anchor, old, new, config['slot_weight'])
    args.output.mkdir(parents=True)
    new.to_csv(args.output / 'normalized_wpt.csv', index=False)
    candidate_path = args.output / 'predictions.csv'
    candidate.to_csv(candidate_path, index=False)
    comparison = args.output / 'comparison.json'
    subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_music_specialist_v58.py'),
        '--task', 'file', '--matrix', str(config_path), '--datasets', *config['development_subset'],
        '--incumbent', str(anchor_path), '--candidate', str(candidate_path), '--output', str(comparison)], cwd=ROOT, check=True)
    decision = json.loads(comparison.read_text())['decision']
    report = dict(scope='One normalization change to old WPT slot; development only',
        decision=decision, other_four_outputs_text_exact=True, official_improvement_verified=False,
        automatic_submission_allowed=False,
        files={str(p): sha256_file(p) for p in [config_path, anchor_path, old_path, candidate_path, verification_path]})
    (args.output / 'decision.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == '__main__':
    main()
