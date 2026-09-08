#!/usr/bin/env python3
"""Prepare and fully restore-verify exact research weights; does NOT build/submit ZIP."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.lossless_weights_v77 import pack_file, restore_file, sha256

SOURCES = {
    'xlsr': 'models/xls-r-2b-anti-deepfake/model.safetensors',
    'spear': 'models/spear-xlarge-speech-audio-v2/model.safetensors',
    'eat': 'models/eat-large-as2m-v56/model.safetensors',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    codes = {str(path): sha256(path) for path in [Path(__file__), ROOT / 'src/lossless_weights_v77.py']}
    frozen = dict(sources=SOURCES, code_sha256=codes, workers=args.workers,
        purpose='lossless weights only; not a promoted candidate or submission ZIP',
        automatic_submission_allowed=False)
    (args.output / 'frozen.json').write_text(json.dumps(frozen, indent=2) + '\n')
    rows, started = [], time.monotonic()
    for name, relative in SOURCES.items():
        source = ROOT / relative
        begin = time.monotonic()
        packed = pack_file(source, args.output / name, workers=args.workers)
        pack_seconds = time.monotonic() - begin
        restored_directory = args.output / 'verified_restore' / name
        restored_directory.mkdir(parents=True)
        begin = time.monotonic()
        restored = restore_file(args.output / name, restored_directory / 'model.safetensors')
        row = dict(model=name, source=str(source), raw_bytes=packed['raw_bytes'],
            packed_bytes=packed['packed_bytes'], raw_sha256=packed['raw_sha256'],
            restored_sha256=restored['restored_sha256'],
            manifest_sha256=sha256(args.output / name / 'manifest.json'),
            pack_seconds=pack_seconds, restore_seconds=time.monotonic() - begin)
        if row['raw_sha256'] != row['restored_sha256']:
            raise ValueError('full reconstruction mismatch')
        rows.append(row)
        print(json.dumps(row), flush=True)
    if any(sha256(path) != digest for path, digest in codes.items()):
        raise ValueError('packer code changed during run')
    packed_bytes = sum(row['packed_bytes'] for row in rows)
    report = dict(**frozen, status='complete', results=rows,
        total_raw_bytes=sum(row['raw_bytes'] for row in rows), total_packed_bytes=packed_bytes,
        remaining_decimal_10gb_bytes=10_000_000_000 - packed_bytes,
        weights_only_under_decimal_10gb=packed_bytes < 10_000_000_000,
        all_models_restored_sha256_exact=True, elapsed_seconds=time.monotonic() - started,
        limitations=['Actual ZIP size includes code, heads, manifests and ZIP metadata',
            'No submission ZIP was created', 'No L4 end-to-end execution or package installation verified',
            'Full-size verified_restore files are research artifacts; do not include them in ZIP'])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
