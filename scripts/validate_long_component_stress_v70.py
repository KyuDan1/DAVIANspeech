#!/usr/bin/env python3
"""Independent full header/hash/label/source checks for the v70 stress bank."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from src.data_guard import identity_tokens
from train_paired_wpt_file_v60 import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, default=ROOT / 'data/eval/long_component_stress_v70')
    args = parser.parse_args()
    output = args.bank / 'validation.json'
    if output.exists():
        raise FileExistsError(output)
    provenance = json.loads((args.bank / 'provenance.json').read_text())
    config = provenance['config']
    if provenance['stage'] != 'built_unscored' or provenance['selection_allowed']:
        raise ValueError('not an unscored stress-only bank')
    if sha256(args.bank / 'truth.csv') != provenance['truth_sha256']:
        raise ValueError('truth changed')
    if sha256(args.bank / 'audio_hashes.csv') != provenance['audio_hashes_sha256']:
        raise ValueError('audio hash manifest changed')
    frame = pd.read_csv(args.bank / 'truth.csv', dtype={'ID': str})
    hashes = pd.read_csv(args.bank / 'audio_hashes.csv', dtype=str).set_index('ID')
    parent = pd.read_csv(args.bank / 'parent_identity_manifest.csv', dtype=str)
    if len(frame) != config['expected_rows'] or frame.ID.duplicated().any() or set(frame.ID) != set(hashes.index):
        raise ValueError('sample count or IDs invalid')
    if frame.GROUP_ID.nunique() != config['groups'] or not frame.DURATION.between(4, 60).all():
        raise ValueError('invalid group count/duration')
    if not np.array_equal(frame.FILE_FAKE, np.maximum(frame.VOICE_FAKE, frame.MUSIC_FAKE)):
        raise ValueError('File OR labels disagree')
    for row in frame.itertuples():
        for component in ['VOICE', 'MUSIC']:
            present, fake = getattr(row, component + '_PRESENT'), getattr(row, component + '_FAKE')
            intervals = json.loads(getattr(row, component + '_INTERVALS'))
            fake_intervals = json.loads(getattr(row, component + '_FAKE_INTERVALS'))
            if bool(intervals) != bool(present) or bool(fake_intervals) != bool(fake):
                raise ValueError('presence/fake interval label mismatch')
            for start, end in intervals + fake_intervals:
                if not 0 <= start < end <= row.DURATION:
                    raise ValueError('interval outside waveform')
            for start, end in fake_intervals:
                if not any(a <= start < end <= b for a, b in intervals):
                    raise ValueError('fake interval outside its component')
            if row.MIX_MODE == 'sparse_' + component.lower() and fake:
                if len(fake_intervals) != 1 or not np.isclose(fake_intervals[0][1] - fake_intervals[0][0], 2):
                    raise ValueError('sparse fake is not exactly two seconds')
    expected_channels = set(config['channels'])
    for _, rows in frame.groupby('BASE_ID'):
        if set(rows.CHANNEL) != expected_channels or len(rows) != len(expected_channels):
            raise ValueError('missing/duplicated paired channels')
        if rows[['FILE_FAKE', 'VOICE_FAKE', 'MUSIC_FAKE', 'DURATION']].drop_duplicates().shape[0] != 1:
            raise ValueError('channel changed component labels or duration')
    def check(row):
        path = args.bank / 'audio' / (row.ID + '.flac')
        info = sf.info(path)
        if info.samplerate != 16000 or info.channels != 1 or info.frames != row.DURATION * 16000:
            raise ValueError('bad audio header')
        if sha256(path) != hashes.loc[row.ID, 'SHA256']:
            raise ValueError('audio content changed')
        return info.frames
    with ThreadPoolExecutor(max_workers=8) as pool:
        frames = list(pool.map(check, frame.itertuples()))
    roles = yaml.safe_load((ROOT / config['partition_config']).read_text())
    identities = identity_tokens(parent)
    for role in ['train', 'router_train', 'training_validation']:
        for relative in roles.get(role, []):
            if identities & identity_tokens(pd.read_csv(ROOT / relative, dtype=str)):
                raise ValueError('parent identity now overlaps training')
    report = dict(passed=True, files=len(frame), full_audio_sha_checks=len(frames), groups=frame.GROUP_ID.nunique(),
        duration_counts={str(k): int(v) for k, v in frame.DURATION.value_counts().items()},
        total_audio_seconds=sum(frames) / 16000, training_parent_identity_overlap=0,
        truth_sha256=sha256(args.bank / 'truth.csv'), provenance_sha256=sha256(args.bank / 'provenance.json'),
        selection_allowed=False, scope='full integrity, not detector accuracy or new-source generalization')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
