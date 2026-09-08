#!/usr/bin/env python3
"""Build a metadata-only source-audited cache using fast directory enumeration.

No audio decoding or training. Does not modify any active run or core loader.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]


def main():
    import pandas as pd
    import yaml
    import train_paired_wpt_file_v60 as loader
    from train_common_encoder_probe import audit_protected_sources
    from src.audio_inventory_v78 import find_audio_files
    from src.diagnostic_axes_compat_v78 import add_diagnostic_axes
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/training_inventory_v78')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config_path = ROOT / 'configs/common_eat_adaptation.yaml'
    config = yaml.safe_load(config_path.read_text())
    reference = ROOT / 'reports/common_eat_adaptation/full'
    previous = json.loads((reference / 'manifest.json').read_text())
    calls = []

    def measured(path):
        began = time.monotonic()
        files = find_audio_files(path)
        calls.append(dict(directory=str(path), files=len(files), seconds=time.monotonic() - began))
        return files

    # Injection is local to this new process. The loader's source and all other
    # Python processes, including the running v74/v76 experiments, are untouched.
    original = loader.find_audio_files
    original_axes = loader.add_diagnostic_axes
    started = time.monotonic()
    loader.find_audio_files = measured
    loader.add_diagnostic_axes = add_diagnostic_axes
    try:
        train, dev, audit = loader.load_data(config)
    finally:
        loader.find_audio_files = original
        loader.add_diagnostic_axes = original_axes
    load_seconds = time.monotonic() - started
    if (len(train), len(dev)) != (18738, 4577):
        raise ValueError('unexpected authorized population')
    for name, frame in [('train', train), ('development', dev)]:
        expected = pd.read_csv(reference / (name + '_ids.csv'), dtype=str)
        pd.testing.assert_frame_equal(frame[['DATASET', 'ID']].astype(str).reset_index(drop=True),
            expected[['DATASET', 'ID']].reset_index(drop=True))
    if audit['manifests'] != previous['audit']['manifests']:
        raise ValueError('source manifests differ from the actual completed parent run')
    audit['all_protected_roles'] = audit_protected_sources(train, ROOT / config['partition_config'])
    sources = [Path(__file__), ROOT / 'src/audio_inventory_v78.py',
        ROOT / 'src/diagnostic_axes_compat_v78.py', config_path,
        ROOT / config['partition_config'], ROOT / config['data_matrix'],
        ROOT / 'scripts/train_paired_wpt_file_v60.py', ROOT / 'scripts/train_common_encoder_probe.py',
        reference / 'manifest.json', reference / 'train_ids.csv', reference / 'development_ids.csv']
    hashes = {str(path.resolve()): loader.sha256(path) for path in sources}
    hashes.update({record['path']: record['sha256'] for record in audit['manifests']})
    hashes.update({record['path']: record['sha256'] for record in audit['all_protected_roles']})
    args.output.mkdir(parents=True)
    tables = {}
    for name, frame in [('train', train), ('development', dev)]:
        path = args.output / (name + '.csv')
        frame.to_csv(path, index=False)
        tables[name] = dict(path=str(path.resolve()), sha256=loader.sha256(path), rows=len(frame))
    report = dict(status='complete_metadata_cache', config=config, audit=audit,
        artifacts_sha256=hashes, tables=tables, discovery_calls=calls,
        load_and_split_validation_seconds=load_seconds,
        directory_discovery_seconds=sum(r['seconds'] for r in calls),
        total_seconds=time.monotonic() - started, train_development_order_exact=True,
        source_manifests_equal_completed_parent=True, waveform_decoding=False,
        training_performed=False, automatic_submission_allowed=False)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['status', 'load_and_split_validation_seconds',
        'directory_discovery_seconds', 'total_seconds', 'train_development_order_exact']}), flush=True)


if __name__ == '__main__':
    main()
