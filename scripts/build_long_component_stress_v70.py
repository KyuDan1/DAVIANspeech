#!/usr/bin/env python3
"""Build length/layout/partial-fake stress from already-authorized dev sources.

Not an unseen-generator bank. No detector inference, train writes, or selection.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from src.long_component_stress import SR, make_stream, geometry_start, render_case
from src.pipeline import load_audio
from src.data_guard import identity_tokens
from src.telephone_channel import apply_channel
from src.full_coverage_wpt import resolve_ffmpeg
from build_codec_mixed_blind_v7 import atomic_output_dir
from train_paired_wpt_file_v60 import sha256


def reserve(config):
    roles = yaml.safe_load((ROOT / config['partition_config']).read_text())
    if config['parent_truth'] not in roles['development']:
        raise ValueError('only registered development parents allowed')
    parent = pd.read_csv(ROOT / config['parent_truth'], dtype=str)
    parent = parent.loc[parent.CHANNEL.eq('clean')].copy()
    if parent.BASE_ID.duplicated().any():
        raise ValueError('ambiguous clean parents')
    parent_tokens = identity_tokens(parent)
    audits = []
    for role in ['train', 'router_train', 'training_validation']:
        for relative in roles.get(role, []):
            path = ROOT / relative
            other = pd.read_csv(path, dtype=str)
            overlap = parent_tokens & identity_tokens(other)
            if overlap:
                raise ValueError(f'stress sources overlap {role}: {path}: {sorted(overlap)[:5]}')
            audits.append(dict(role=role, path=relative, sha256=sha256(path), overlap=0))
    hashes = pd.read_csv(ROOT / config['parent_sources'], dtype=str)
    if hashes[['KIND', 'SOURCE_ID']].duplicated().any():
        raise ValueError('duplicate parent source hashes')
    records = []
    for kind, prefix in [('voice', 'VOICE'), ('music', 'MUSIC')]:
        sources = parent.drop_duplicates(prefix + '_SOURCE_ID')
        for label in [0, 1]:
            selected = sources.loc[sources[prefix + '_FAKE'].astype(int).eq(label)].copy()
            selected['_rank'] = selected[prefix + '_SOURCE_ID'].map(
                lambda identity: hashlib.sha256(f'{config["seed"]}|{kind}|{identity}'.encode()).hexdigest())
            selected = selected.sort_values('_rank')
            needed = config['groups'] * config['sources_per_pool_per_group']
            if len(selected) < needed:
                raise ValueError(f'not enough development-only {kind}/{label} sources')
            for index, row in enumerate(selected.head(needed).to_dict('records')):
                source_id = row[prefix + '_SOURCE_ID']
                source = hashes.loc[hashes.KIND.eq(kind) & hashes.SOURCE_ID.eq(source_id)]
                if len(source) != 1:
                    raise ValueError('missing source path/hash')
                record = source.iloc[0].to_dict()
                record.update(GROUP=f'lcs70_{index // config["sources_per_pool_per_group"]:02d}',
                              POOL=prefix[0] + ('F' if label else 'R'), LABEL=label,
                              PARENT_ID=row['BASE_ID'], GENERATOR=row[prefix + '_GENERATOR'])
                records.append(record)
    return parent, pd.DataFrame(records), audits


def build_group(group, selected, config, audio_directory, ffmpeg):
    streams, source_meta = {}, []
    maximum = max(config['durations'])
    fade = round(SR * config['crossfade_seconds'])
    for pool, rows in selected.groupby('POOL', sort=True):
        audios = []
        for row in rows.itertuples():
            path = Path(row.LOCAL_PATH)
            if sha256(path) != row.SHA256:
                raise ValueError(f'original development source changed: {path}')
            audio = load_audio(path)
            audios.append(audio)
            piece_seconds = config['voice_piece_seconds'] if pool[0] == 'V' else config['music_piece_seconds']
            source_meta.append(dict(GROUP=group, POOL=pool, SOURCE_ID=row.SOURCE_ID,
                DURATION=len(audio) / SR, WITHIN_PIECE_REPEATED=len(audio) < piece_seconds * SR,
                PCM_SHA256=hashlib.sha256(np.asarray(audio, dtype='<f4').tobytes()).hexdigest()))
        streams[pool] = make_stream(audios, maximum, piece_seconds, fade)
    records, output_hashes = [], []
    snr = config['snr_cycle'][int(group.split('_')[-1]) % len(config['snr_cycle'])]
    source_ids = {pool: sorted(rows.SOURCE_ID.tolist()) for pool, rows in selected.groupby('POOL')}
    for seconds in config['durations']:
        start = geometry_start(config['seed'], group, seconds, config['insertion_seconds'])
        variants = []
        for layout in config['layouts']:
            for case in config['component_cases']:
                audio, labels = render_case(streams, seconds, layout, case, snr, start,
                                           config['insertion_seconds'], fade)
                variants.append((layout, case, audio, labels))
        for pure in config['pure_cases']:
            component = 'V' if pure.endswith('voice') else 'M'
            fake = int(pure.startswith('fake'))
            pool = component + ('F' if fake else 'R')
            audio = streams[pool][:seconds * SR]
            labels = dict(FILE_FAKE=fake, VOICE_FAKE=fake if component == 'V' else 0,
                          MUSIC_FAKE=fake if component == 'M' else 0,
                          VOICE_PRESENT=int(component == 'V'), MUSIC_PRESENT=int(component == 'M'),
                          VOICE_INTERVALS=[[0., float(seconds)]] if component == 'V' else [],
                          MUSIC_INTERVALS=[[0., float(seconds)]] if component == 'M' else [],
                          VOICE_FAKE_INTERVALS=[[0., float(seconds)]] if component == 'V' and fake else [],
                          MUSIC_FAKE_INTERVALS=[[0., float(seconds)]] if component == 'M' and fake else [])
            variants.append((pure, pool, audio, labels))
        for layout, case, audio, labels in variants:
            base = f'{group}__{seconds}s__{layout}__{case}'
            for channel in config['channels']:
                rendered = apply_channel(audio, channel, ffmpeg=ffmpeg)
                if len(rendered) != seconds * SR or not np.isfinite(rendered).all():
                    raise ValueError('codec changed the required duration or produced invalid audio')
                identity = f'{base}__{channel}'
                path = audio_directory / f'{identity}.flac'
                sf.write(path, rendered, SR, format='FLAC', subtype='PCM_16')
                info = sf.info(path)
                if info.frames != seconds * SR or info.samplerate != SR or info.channels != 1:
                    raise ValueError('saved audio format does not match truth')
                output_hashes.append(dict(ID=identity, SHA256=sha256(path), BYTES=path.stat().st_size))
                record = dict(ID=identity, BASE_ID=base, PAIR_GROUP=group, GROUP_ID=group,
                    DATASET='long_component_stress_v70', DURATION=seconds, CHANNEL=channel,
                    MIX_MODE=layout, COMPONENT_CASE=case, SNR_DB=snr,
                    AUDIO_TYPE='mixed' if labels['VOICE_PRESENT'] and labels['MUSIC_PRESENT'] else ('voice' if labels['VOICE_PRESENT'] else 'music'),
                    SOURCE_DATASET='codec_mixed_dev_v4', PROVENANCE_MODE='development_parent_stress',
                    INSERTION_START=start / SR if layout.startswith('sparse_') else np.nan,
                    RESERVED_SOURCE_IDS=json.dumps(source_ids, sort_keys=True), **labels)
                for key in ['VOICE_INTERVALS', 'MUSIC_INTERVALS', 'VOICE_FAKE_INTERVALS', 'MUSIC_FAKE_INTERVALS']:
                    record[key] = json.dumps(record[key])
                records.append(record)
    print(json.dumps(dict(group=group, rendered=len(records))), flush=True)
    return records, output_hashes, source_meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/long_component_stress_v70.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    parent, selected, audits = reserve(config)
    expected = config['groups'] * len(config['durations']) * len(config['channels']) * (
        len(config['layouts']) * len(config['component_cases']) + len(config['pure_cases']))
    if expected != config['expected_rows']:
        raise ValueError('configured sample count mismatch')
    staging, publish = atomic_output_dir(args.output)
    selected.to_csv(staging / 'source_manifest.csv', index=False)
    parent.to_csv(staging / 'parent_identity_manifest.csv', index=False)
    report = dict(schema=config['schema'], config=config, stage='planned',
        expected_rows=expected, training_identity_audit=audits,
        source_sha256=sha256(ROOT / config['parent_sources']), parent_sha256=sha256(ROOT / config['parent_truth']),
        builder_sha256=sha256(Path(__file__)), renderer_sha256=sha256(ROOT / 'src/long_component_stress.py'),
        config_sha256=sha256(args.config), selection_allowed=False, source_ood_claim=False,
        limitations=['development-parent derived; not new-source or new-generator evidence',
          'fixed voice/music pieces are looped to reach long duration; not natural conversations',
          'repetition and synthetic source-switch seams may bias absolute accuracy',
          'instrumental semantics inherit prior development source screening',
          '10 source groups reused across lengths/layouts/classes/channels, not 2160 independent sources'])
    if not args.plan_only:
        directory = staging / 'audio'
        directory.mkdir()
        ffmpeg = resolve_ffmpeg()
        def work(item):
            return build_group(item[0], item[1], config, directory, ffmpeg)
        with ThreadPoolExecutor(max_workers=config['workers']) as pool:
            results = list(pool.map(work, list(selected.groupby('GROUP', sort=True))))
        truth = pd.DataFrame([row for result in results for row in result[0]])
        if len(truth) != expected or truth.ID.duplicated().any():
            raise ValueError('rendered output count or IDs invalid')
        truth.to_csv(staging / 'truth.csv', index=False)
        pd.DataFrame([row for result in results for row in result[1]]).to_csv(staging / 'audio_hashes.csv', index=False)
        metadata = pd.DataFrame([row for result in results for row in result[2]])
        if metadata.PCM_SHA256.duplicated().any():
            raise ValueError('decoded source duplicates across reserved groups')
        metadata.to_csv(staging / 'source_audio_metadata.csv', index=False)
        report.update(stage='built_unscored', rows=len(truth), truth_sha256=sha256(staging / 'truth.csv'),
                      audio_hashes_sha256=sha256(staging / 'audio_hashes.csv'), duration_header_checks_passed=True)
    (staging / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    publish()
    print(json.dumps(dict(stage=report['stage'], expected_rows=expected, output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
