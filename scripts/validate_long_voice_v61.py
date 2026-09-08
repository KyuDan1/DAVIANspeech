#!/usr/bin/env python3
"""Validate the long-voice bank's construction and provenance, never predictions."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_codec_mixed_blind_v7 as v7
from build_long_voice_v61 import read_reservation, uniform_start, UNIFORM_POLICY, SR
from build_prospective_mixed_phone_v3 import sha256_file
from reserve_long_voice_v61 import IDENTITIES


def validate_recipe(frame, sources, plan):
    positions = plan.get('positions', ['early', 'middle', 'late'])
    expected = plan['groups'] * 3 * len(positions) * 2 * 2 * 3
    if len(frame) != expected or frame.ID.nunique() != expected:
        raise ValueError('wrong row/ID count')
    axes = ['GROUP_ID', 'DURATION', 'POSITION', 'RECORDING', 'FILE_FAKE', 'CHANNEL']
    if len(frame.drop_duplicates(axes)) != expected:
        raise ValueError('factorial cells duplicated')
    for column, values in [('GROUP_ID', set(sources.SOURCE_GROUP)),
                            ('DURATION', {30, 45, 60}), ('POSITION', set(positions)),
                            ('RECORDING', {'direct', 'replay'}), ('FILE_FAKE', {0, 1}),
                            ('CHANNEL', set(plan['channels']))]:
        if set(frame[column]) != values:
            raise ValueError(f'unexpected {column}')
    lookup = sources.set_index('ID').to_dict('index')
    for row in frame.itertuples():
        first, second = lookup[row.FIRST_SOURCE_ID], lookup[row.SECOND_SOURCE_ID]
        if int(first['FILE_FAKE']) or int(second['FILE_FAKE']) != row.FILE_FAKE:
            raise ValueError('source/mixture label mismatch')
        if first['SOURCE_GROUP'] != row.GROUP_ID or second['SOURCE_GROUP'] != row.GROUP_ID:
            raise ValueError('source group mismatch')
        expected_start = (uniform_start(plan['seed'], row.GROUP_ID, row.DURATION) if row.POSITION == 'uniform' else
                          {'early': 1., 'middle': (row.DURATION - 2.) / 2., 'late': row.DURATION - 3.}[row.POSITION])
        expected_samples = [round(expected_start * SR), round((expected_start + 2) * SR)]
        if [round(row.INSERTION_START * SR), round(row.INSERTION_END * SR)] != expected_samples:
            raise ValueError('wrong insertion bounds')
        ranges = [[round(float(value) * SR) for value in bounds] for bounds in json.loads(row.FAKE_RANGES)]
        if ranges != ([expected_samples] if row.FILE_FAKE else []):
            raise ValueError('fake ranges mismatch')
        if row.RECORDING != second['RECORDING'] or row.GENERATOR != second['VOICE_GENERATOR']:
            raise ValueError('recording/generator mismatch')
        if row.FIRST_GROUP != first['VOICE_SPEAKER'] or row.SECOND_GROUP != second['VOICE_SPEAKER']:
            raise ValueError('speaker provenance mismatch')
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reservation', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources, plan = read_reservation(args.reservation)
    provenance = json.loads((args.bank / 'provenance.json').read_text())
    if provenance['source_manifest_sha256'] != plan['source_manifest_sha256']:
        raise ValueError('wrong source reservation')
    if provenance.get('temporal_geometry_policy') == UNIFORM_POLICY:
        plan = {**plan, 'positions': ['uniform']}
    frame = pd.read_csv(args.bank / 'truth.csv')
    count = validate_recipe(frame, sources, plan)
    snapshot = pd.read_csv(args.reservation / 'protected_inputs.csv')
    protection = v7.build_protection([ROOT / p for p in snapshot.loc[snapshot.KIND.eq('truth'), 'PATH']],
                                   [ROOT / p for p in snapshot.loc[snapshot.KIND.eq('source_hashes'), 'PATH']])
    source_root = Path(provenance['source_root'])
    if not source_root.is_absolute():
        source_root = ROOT / source_root
    if sha256_file(source_root / 'source_hashes.csv') != provenance['source_hashes_sha256']:
        raise ValueError('source hash manifest changed')
    source_hashes = pd.read_csv(source_root / 'source_hashes.csv')
    if set(source_hashes.ID) != set(sources.VOICE_ARCHIVE_ID) or source_hashes.PCM_SHA256.duplicated().any():
        raise ValueError('source hashes missing/duplicated')
    for row in source_hashes.itertuples():
        if {row.SHA256, row.SOURCE_PAYLOAD_SHA256} & protection.audio_hashes:
            raise ValueError('known protected source hash overlap')
        if sha256_file(source_root / 'audio' / f'{row.ID}.flac') != row.SHA256:
            raise ValueError('materialized source changed')
    seen = set()
    for row in sources.to_dict('records'):
        tokens = set().union(*(v7.identity_variants(row[c]) for c in IDENTITIES))
        if tokens & seen or tokens & protection.exact:
            raise ValueError('source identity overlap')
        seen.update(tokens)
    hashes = pd.read_csv(args.bank / 'audio_hashes.csv')
    if hashes.ID.nunique() != count or set(hashes.ID) != set(frame.ID):
        raise ValueError('audio hash coverage differs')
    by_id = hashes.set_index('ID').SHA256.to_dict()
    paths = list((args.bank / 'audio').glob('*.flac'))
    if len(paths) != count or {p.stem for p in paths} != set(frame.ID):
        raise ValueError('unexpected/missing audio files')
    def inspect(row):
        path = args.bank / 'audio' / f'{row.ID}.flac'
        if sha256_file(path) != by_id[row.ID]:
            raise ValueError('rendered audio hash mismatch')
        if by_id[row.ID] in protection.audio_hashes:
            raise ValueError('known protected rendered audio hash overlap')
        audio, rate = sf.read(path, dtype='float32', always_2d=True)
        if rate != SR or audio.shape != (row.DURATION * SR, 1) or not np.isfinite(audio).all():
            raise ValueError('audio format/duration/finite check failed')
        if not np.max(np.abs(audio)) > 1e-5:
            raise ValueError('silent file')
        return float(np.abs(audio).max())
    with ThreadPoolExecutor(max_workers=4) as executor:
        peaks = list(executor.map(inspect, frame.itertuples()))
    report = dict(status='PASS_CONSTRUCTION_NOT_DETECTION', files=count,
        source_groups=plan['groups'], independent_files=False, source_identity_overlap=0,
        known_protected_source_hash_overlap=0,
        known_protected_rendered_hash_overlap=0, maximum_peak=max(peaks),
        truth_sha256=sha256_file(args.bank / 'truth.csv'),
        audio_hashes_sha256=sha256_file(args.bank / 'audio_hashes.csv'),
        authenticity_scores_used=False, requires_partition_registration=True,
        scope='Synthetic long-voice File/Voice detection; not music/CPS or natural-call validation.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
