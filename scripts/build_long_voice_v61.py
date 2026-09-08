#!/usr/bin/env python3
"""Materialize a reserved, held-out long-voice temporal mechanism bank.

Synthetic looped backgrounds are explicit. Both real and fake insertions use
the same duration, position, gain policy and crossfade; codecs process the whole
rendered file. This is not a natural-call corpus or a music evaluation.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
import build_codec_mixed_blind_v7 as v7
from build_prospective_mixed_phone_v3 import sha256_file
from full_coverage_wpt import resolve_ffmpeg
from telephone_channel import apply_channel

SR = 16000
UNIFORM_POLICY = 'uniform_v61_sha256_group_duration_v1'


def uniform_start(seed, group, seconds):
    digest = hashlib.sha256(f'{seed}|{group}|{seconds}|uniform-v61'.encode()).hexdigest()
    fraction = int(digest[:16], 16) / 2**64
    return round((.25 + (seconds - 2.5) * fraction) * SR) / SR


def read_reservation(path):
    plan = json.loads((path / 'reservation.json').read_text())
    if sha256_file(path / 'truth_sources.csv') != plan['source_manifest_sha256']:
        raise ValueError('reserved source manifest changed')
    if sha256_file(path / 'protected_inputs.csv') != plan['protection_sha256']:
        raise ValueError('protection snapshot changed')
    v7.verify_snapshot(pd.read_csv(path / 'protected_inputs.csv'))
    return pd.read_csv(path / 'truth_sources.csv', dtype=str).fillna(''), plan


def normalize(audio):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or len(audio) < 2 * SR or not np.isfinite(audio).all():
        raise ValueError('source must be finite mono with at least 2 seconds')
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < 1e-5:
        raise ValueError('silent source')
    return audio * min(.08 / rms, .85 / float(np.abs(audio).max()))


def render_insertion(background, inserted, seconds, position, *, start_seconds=None):
    """Return waveform and pre-codec insertion bounds, with no fake-label input."""
    if seconds not in (30, 45, 60) or position not in ('early', 'middle', 'late', 'uniform'):
        raise ValueError('unsupported fixed temporal condition')
    background, inserted = normalize(background), normalize(inserted)
    length, segment, fade = int(seconds * SR), 2 * SR, 160
    # Loop joins are smoothed for every class. Repetition remains a limitation.
    base = background.copy()
    while len(base) < length:
        ramp = np.linspace(0., 1., fade, dtype=np.float32)
        join = base[-fade:] * (1 - ramp) + background[:fade] * ramp
        base = np.concatenate((base[:-fade], join, background[fade:]))
    base = base[:length].copy()
    # Use an energy-only 2 s crop so a label never points at an all-silent crop.
    starts = range(0, len(inserted) - segment + 1, SR // 4)
    crop_start = max(starts, key=lambda start: float(np.square(inserted[start:start + segment]).mean()))
    crop = inserted[crop_start:crop_start + segment]
    if position == 'uniform':
        if start_seconds is None or not np.isfinite(start_seconds) or not 0 <= start_seconds <= seconds - 2:
            raise ValueError('uniform insertion requires an explicit in-range start')
        start = round(start_seconds * SR)
    else:
        if start_seconds is not None:
            raise ValueError('fixed insertion must not override its start')
        start = {'early': SR, 'middle': (length - segment) // 2, 'late': length - segment - SR}[position]
    alpha = np.ones(segment, dtype=np.float32)
    alpha[:fade] = np.linspace(0., 1., fade)
    alpha[-fade:] = np.linspace(1., 0., fade)
    base[start:start + segment] = (1 - alpha) * base[start:start + segment] + alpha * crop
    return base, (start / SR, (start + segment) / SR), crop_start / SR


def materialize(args):
    frame, plan = read_reservation(args.reservation)
    if args.output.exists():
        raise FileExistsError(args.output)
    staging, publish = v7.atomic_output_dir(args.output)
    (staging / 'audio').mkdir()
    records, seen_pcm, found = [], set(), set()
    for shard, group in frame.groupby('VOICE_PARQUET_SHARD'):
        wanted = set(group.VOICE_ARCHIVE_ID)
        parquet_path = ROOT / 'data/external/echofake' / shard
        for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=64, columns=['utt_id', 'path']):
            for row in batch.to_pylist():
                key = str(row['utt_id'])
                if key not in wanted:
                    continue
                if key in found:
                    raise ValueError('duplicate archive ID')
                audio, rate = sf.read(BytesIO(row['path']['bytes']), dtype='float32', always_2d=True)
                audio = audio.mean(axis=1)
                if rate != SR:
                    divisor = np.gcd(rate, SR)
                    audio = resample_poly(audio, SR // divisor, rate // divisor).astype(np.float32)
                normalize(audio)  # Validate; preserve unnormalized source for provenance.
                digest = hashlib.sha256(np.asarray(audio, dtype='<f4').tobytes()).hexdigest()
                if digest in seen_pcm:
                    raise ValueError('duplicate decoded source audio; reservation requires revision before rendering')
                seen_pcm.add(digest)
                destination = staging / 'audio' / f'{key}.flac'
                sf.write(destination, audio, SR, format='FLAC', subtype='PCM_24')
                records.append(dict(ID=key, SHA256=sha256_file(destination), PCM_SHA256=digest,
                                    SOURCE_PAYLOAD_SHA256=hashlib.sha256(row['path']['bytes']).hexdigest(),
                                    DURATION=len(audio) / SR))
                found.add(key)
        print(json.dumps(dict(shard=shard, found=len(found))), flush=True)
    if found != set(frame.VOICE_ARCHIVE_ID):
        raise ValueError('reserved sources missing')
    pd.DataFrame(records).to_csv(staging / 'source_hashes.csv', index=False)
    (staging / 'complete.json').write_text(json.dumps(dict(
        source_manifest_sha256=plan['source_manifest_sha256'], sources=len(found),
        decoded_source_duplicates=0, full_bank_complete=False)) + '\n')
    publish()


def render(args):
    frame, plan = read_reservation(args.reservation)
    if args.uniform_position:
        plan = {**plan, 'positions': ['uniform'], 'temporal_geometry_policy': UNIFORM_POLICY}
    source_report = json.loads((args.sources / 'complete.json').read_text())
    if source_report['source_manifest_sha256'] != plan['source_manifest_sha256']:
        raise ValueError('sources are from another reservation')
    hashes = pd.read_csv(args.sources / 'source_hashes.csv', dtype=str)
    audio = {}
    for row in hashes.itertuples():
        path = args.sources / 'audio' / f'{row.ID}.flac'
        if sha256_file(path) != row.SHA256:
            raise ValueError('materialized source changed')
        audio[row.ID], rate = sf.read(path, dtype='float32')
        if rate != SR:
            raise ValueError('wrong source sample rate')
    staging, publish = v7.atomic_output_dir(args.output)
    (staging / 'audio').mkdir()
    ffmpeg, records, file_hashes = resolve_ffmpeg(), [], []
    for group_id, group in frame.groupby('SOURCE_GROUP'):
        roles = {r.ROLE: r for r in group.itertuples()}
        background = roles['background']
        for seconds in plan['durations']:
            for position in plan['positions']:
                for recording in ('direct', 'replay'):
                    for label, name in ((0, 'real_insert'), (1, 'fake_insert')):
                        source = roles[('replay_' if recording == 'replay' else '') + name]
                        insertion_start = uniform_start(plan['seed'], group_id, seconds) if position == 'uniform' else None
                        base, bounds, crop_start = render_insertion(
                            audio[background.VOICE_ARCHIVE_ID], audio[source.VOICE_ARCHIVE_ID], seconds, position,
                            start_seconds=insertion_start)
                        base_id = f'{group_id}_{seconds}_{position}_{recording}_{label}'
                        for channel in plan['channels']:
                            changed = apply_channel(base, channel, ffmpeg=ffmpeg, key=plan['seed'])
                            changed = np.pad(changed, (0, max(0, len(base) - len(changed))))[:len(base)]
                            if not np.isfinite(changed).all():
                                raise ValueError('non-finite codec audio')
                            sample_id = f'{base_id}__{channel}'
                            path = staging / 'audio' / f'{sample_id}.flac'
                            sf.write(path, changed, SR, format='FLAC', subtype='PCM_16')
                            records.append(dict(ID=sample_id, PARENT_ID=base_id, GROUP_ID=group_id,
                                FILE_FAKE=label, VOICE_FAKE=label, MUSIC_FAKE='', VOICE_PRESENT=1,
                                MUSIC_PRESENT=0, AUDIO_TYPE='voice', DURATION=seconds, POSITION=position,
                                RECORDING=recording, CHANNEL=channel, GENERATOR=source.VOICE_GENERATOR,
                                FIRST_SOURCE_ID=background.VOICE_SOURCE_ID, FIRST_GROUP=background.VOICE_SPEAKER,
                                SECOND_SOURCE_ID=source.VOICE_SOURCE_ID, SECOND_GROUP=source.VOICE_SPEAKER,
                                VOICE_REFERENCE_ID=source.VOICE_REFERENCE_ID,
                                VOICE_REFERENCE_SPEAKER=source.VOICE_REFERENCE_SPEAKER,
                                INSERTION_START=bounds[0], INSERTION_END=bounds[1], SOURCE_CROP_START=crop_start,
                                FAKE_RANGES=json.dumps([bounds] if label else []),
                                TEMPORAL_CONSTRUCTION='looped_real_background_same_edit_real_fake_insert'))
                            file_hashes.append(dict(ID=sample_id, SHA256=sha256_file(path)))
        print(json.dumps(dict(group=group_id, rendered=len(records))), flush=True)
    expected = plan['groups'] * len(plan['durations']) * len(plan['positions']) * 2 * 2 * len(plan['channels'])
    if len(records) != expected:
        raise ValueError('factorial count mismatch')
    pd.DataFrame(records).to_csv(staging / 'truth.csv', index=False)
    pd.DataFrame(file_hashes).to_csv(staging / 'audio_hashes.csv', index=False)
    (staging / 'provenance.json').write_text(json.dumps(
        render_provenance(plan, args.sources, len(records)), indent=2) + '\n')
    publish()


def render_provenance(plan, sources, count):
    return {**plan, 'stage': 'rendered_unscored_requires_integrity_validation',
            'rendered_rows': count, 'full_bank_complete': False, 'source_root': str(sources),
            'source_hashes_sha256': sha256_file(sources / 'source_hashes.csv'),
            'builder_sha256': sha256_file(Path(__file__))}


def publish_render(args):
    """Recover metadata publication only; verify every previously rendered file."""
    from concurrent.futures import ThreadPoolExecutor
    from validate_long_voice_v61 import validate_recipe
    sources, plan = read_reservation(args.reservation)
    if args.output.exists():
        raise FileExistsError(args.output)
    staging = args.output.with_name(args.output.name + '.partial')
    if (staging / 'provenance.json').exists():
        raise ValueError('publication repair only accepts the known missing-provenance state')
    frame = pd.read_csv(staging / 'truth.csv')
    count = validate_recipe(frame, sources, plan)
    hashes = pd.read_csv(staging / 'audio_hashes.csv')
    if len(hashes) != count or hashes.ID.nunique() != count or set(hashes.ID) != set(frame.ID):
        raise ValueError('incomplete render hash manifest')
    if {p.stem for p in (staging / 'audio').glob('*.flac')} != set(frame.ID):
        raise ValueError('unexpected/missing rendered files')
    def verify(row):
        if sha256_file(staging / 'audio' / f'{row.ID}.flac') != row.SHA256:
            raise ValueError('rendered file hash changed')
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(verify, hashes.itertuples()))
    payload = render_provenance(plan, args.sources, count)
    payload['publication_only_repair'] = True
    payload['repair_reason'] = 'duplicate stage keyword in final metadata; no waveform regeneration'
    (staging / 'provenance.json').write_text(json.dumps(payload, indent=2) + '\n')
    staging.rename(args.output)
    print(json.dumps(dict(published=str(args.output), files=count, waveforms_regenerated=False)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['materialize', 'render', 'publish-render'])
    parser.add_argument('--reservation', type=Path, required=True)
    parser.add_argument('--sources', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--uniform-position', action='store_true', help='Separate deterministic uniform-time companion bank')
    args = parser.parse_args()
    if args.uniform_position and args.stage != 'render':
        parser.error('--uniform-position only applies to a fresh render')
    if args.stage != 'materialize' and args.sources is None:
        parser.error('--sources is required for rendering')
    {'materialize': materialize, 'render': render, 'publish-render': publish_render}[args.stage](args)


if __name__ == '__main__':
    main()
