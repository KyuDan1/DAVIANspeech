#!/usr/bin/env python3
"""Relocate the submitted entrypoint unchanged; verify before locked evaluation."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ['FILE_FAKE_PROB', 'VOICE_FAKE_PROB', 'MUSIC_FAKE_PROB',
           'VOICE_PRESENT_PROB', 'MUSIC_PRESENT_PROB']
ARCHIVE_SHA = 'a5730fde4e0c49100ee9aca5cc373eb550bf02948517072f23afe02fd62b91d6'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def attest_package(package, archive, expected_sha=ARCHIVE_SHA):
    if sha256(archive) != expected_sha:
        raise ValueError('not the recorded submitted archive')
    inventory = {}
    with zipfile.ZipFile(archive) as source:
        names = [i.filename for i in source.infolist() if not i.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError('duplicate archive paths')
        for item in source.infolist():
            relative = Path(item.filename)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('unsafe archive path')
            if item.is_dir():
                continue
            path = package / relative
            if path.stat().st_size != item.file_size:
                raise ValueError(f'package size changed: {relative}')
            digest, crc = hashlib.sha256(), 0
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
                    crc = zlib.crc32(block, crc)
            if crc != item.CRC:
                raise ValueError(f'package differs from submitted ZIP: {relative}')
            inventory[str(relative)] = digest.hexdigest()
        actual = {str(p.relative_to(package)) for p in (package / 'model').rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
        if actual != {p for p in inventory if p.startswith('model/')}:
            raise ValueError('extra/missing model files outside submitted archive')
    return dict(package=str(package), archive=str(archive), archive_sha256=expected_sha,
                files_sha256=inventory, archive_crc_matches_package=True)


def stage_inputs(output, package, rows):
    output.mkdir(parents=True, exist_ok=False)
    audio = output / 'data/test'
    audio.mkdir(parents=True)
    (output / 'model').symlink_to(package / 'model', target_is_directory=True)
    ids = [row['ID'] for row in rows]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('nonempty unique IDs required')
    for row in rows:
        identity, source = row['ID'], Path(row['PATH']).resolve(strict=True)
        if Path(identity).name != identity or identity in {'.', '..'}:
            raise ValueError('unsafe audio ID')
        (audio / (identity + source.suffix)).symlink_to(source)
    with (output / 'data/sample_submission.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['ID', *COLUMNS])
        writer.writeheader()
        writer.writerows(dict(ID=i, **{c: .5 for c in COLUMNS}) for i in ids)


def verify_inventory(attestation):
    package = Path(attestation['package'])
    for relative, expected in attestation['files_sha256'].items():
        if sha256(package / relative) != expected:
            raise ValueError(f'frozen package changed: {relative}')


def run_worker(stage):
    frozen_path = stage / 'frozen.json'
    frozen = json.loads(frozen_path.read_text())
    frozen_hash = sha256(frozen_path)
    verify_inventory(frozen['package_attestation'])
    package = Path(frozen['package_attestation']['package'])
    if (stage / 'output').exists():
        raise FileExistsError('refusing an already started output')
    for row in frozen['inputs']:
        staged = stage / 'data/test' / (row['ID'] + Path(row['PATH']).suffix)
        if staged.resolve() != Path(row['PATH']).resolve() or sha256(staged) != row['SHA256']:
            raise ValueError('staged audio changed')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    sys.argv = [str(package / 'script.py')]
    spec = importlib.util.spec_from_file_location('frozen_submitted_entrypoint', package / 'script.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Path relocation ONLY. All models, operations, arguments and weights remain
    # those of the exact submitted script. Imports resolve to package/model/src.
    module.BASE_DIR = stage
    os.chdir(stage)
    started = time.monotonic()
    module.main()
    if sha256(frozen_path) != frozen_hash:
        raise ValueError('frozen run changed')
    verify_inventory(frozen['package_attestation'])
    report = dict(status='complete', frozen_sha256=frozen_hash,
                  predictions_sha256=sha256(stage / 'output/submission.csv'),
                  seconds=time.monotonic() - started, automatic_submission_allowed=False)
    (stage / 'completed.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='mode', required=True)
    prepare = subs.add_parser('prepare-smoke')
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--limit', type=int, default=8)
    worker = subs.add_parser('worker')
    worker.add_argument('--stage', type=Path, required=True)
    check = subs.add_parser('check-smoke')
    check.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    if args.mode == 'worker':
        run_worker(args.stage.resolve())
        return
    sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
    import numpy as np
    import pandas as pd
    if args.mode == 'prepare-smoke':
        from verify_paired_checkpoint_inference_v68 import resolve_development
        if args.output.exists() or args.limit < 1:
            raise ValueError('new output and positive limit required')
        source = ROOT / 'reports/three_stream_all_type_v57_strict/v50_authorized_v2/v1_strict__music_only_predictions.csv'
        frame = pd.read_csv(source, dtype={'ID': str, 'DATASET': str}).head(args.limit)
        paths = resolve_development(frame, ROOT / 'configs/data_partitions.yaml')
        package = ROOT / 'v50_v57m_challenger_v2'
        attestation = attest_package(package, ROOT / 'v50_v57m_challenger_v2.zip')
        rows = [dict(ID=r.ID, DATASET=r.DATASET, PATH=str(p.resolve()), SHA256=sha256(p))
                for (_, r), p in zip(frame.iterrows(), paths)]
        stage_inputs(args.output, package, rows)
        frame.to_csv(args.output / 'expected.csv', index=False)
        frozen = dict(scope='authorized development entrypoint correspondence only',
            package_attestation=attestation, inputs=rows,
            source_predictions_sha256=sha256(source), expected_sha256=sha256(args.output / 'expected.csv'),
            runner_sha256=sha256(Path(__file__)), automatic_submission_allowed=False)
        (args.output / 'frozen.json').write_text(json.dumps(frozen, indent=2) + '\n')
        print(json.dumps(dict(status='prepared', files=len(rows), package_files=len(attestation['files_sha256']))), flush=True)
    else:
        frozen = json.loads((args.stage / 'frozen.json').read_text())
        completed = json.loads((args.stage / 'completed.json').read_text())
        if (completed['frozen_sha256'] != sha256(args.stage / 'frozen.json')
                or completed['predictions_sha256'] != sha256(args.stage / 'output/submission.csv')
                or frozen['expected_sha256'] != sha256(args.stage / 'expected.csv')):
            raise ValueError('changed comparison artifacts')
        expected = pd.read_csv(args.stage / 'expected.csv', dtype={'ID': str})
        actual = pd.read_csv(args.stage / 'output/submission.csv', dtype={'ID': str})
        if actual.ID.duplicated().any() or set(actual.ID) != set(expected.ID):
            raise ValueError('missing or duplicate output IDs')
        actual = actual.set_index('ID').loc[expected.ID]
        values = actual[COLUMNS].to_numpy(float)
        difference = np.abs(values - expected[COLUMNS].to_numpy(float))
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError('invalid output probabilities')
        report = dict(rows=len(actual), maximum_absolute_difference=float(difference.max()),
            by_column=dict(zip(COLUMNS, difference.max(0).tolist())),
            pass_check=bool(difference.max() <= 5e-4), tolerance=5e-4,
            scope='small development runtime correspondence, NOT full equivalence or accuracy')
        (args.stage / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
        if not report['pass_check']:
            raise ValueError('cached anchor differs from exact submitted entrypoint')


if __name__ == '__main__':
    main()
