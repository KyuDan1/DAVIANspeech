#!/usr/bin/env python3
"""Reserve fresh voices for long sparse-spoof confirmation, without detector scores.

Protect every prior truth and reservation, including retired banks. Real replay
is real; recording condition is crossed with authenticity, never a label proxy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from io import BytesIO
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import build_codec_mixed_blind_v7 as v7
from build_codec_mixed_blind_v8 import balanced_voice
from build_prospective_mixed_phone_v3 import sha256_file

LABELS = {'bonafide': 0, 'replay_bonafide': 0, 'fake': 1, 'replay_fake': 1}
IDENTITIES = ('VOICE_SOURCE_ID', 'VOICE_SPEAKER', 'VOICE_CONTENT_ID',
              'VOICE_REFERENCE_ID', 'VOICE_REFERENCE_SPEAKER')


def candidate_record(row, shard, protection):
    label = v7.clean(row.label)
    if label not in LABELS:
        return None, 'unsupported_label'
    fake = LABELS[label]
    details = row.synthesis_details if isinstance(row.synthesis_details, dict) else {}
    source, speaker = v7.clean(row.source), v7.clean(row.source_speaker_id)
    reference = v7.clean(details.get('reference'))
    reference_speaker = v7.clean(details.get('reference_speaker_id'))
    model = v7.clean(details.get('model')) if fake else 'bonafide'
    if not source or not speaker or (fake and (not model or not reference or not reference_speaker)):
        return None, 'incomplete'
    tokens = set().union(*(v7.identity_variants(value) for value in
                          (row.utt_id, source, speaker, reference, reference_speaker)))
    if tokens & protection.exact:
        return None, 'protected'
    return dict(ID=f'echofake:{row.utt_id}', VOICE_SOURCE_ID=f'echofake:{row.utt_id}',
                VOICE_ARCHIVE_ID=v7.clean(row.utt_id), VOICE_SPEAKER=speaker,
                VOICE_CONTENT_ID=source, VOICE_REFERENCE_ID=reference,
                VOICE_REFERENCE_SPEAKER=reference_speaker, VOICE_GENERATOR=model,
                VOICE_PARQUET_SHARD=shard, SOURCE_LABEL=label,
                RECORDING='replay' if label.startswith('replay_') else 'direct',
                FILE_FAKE=fake, VOICE_FAKE=fake, MUSIC_FAKE='',
                VOICE_PRESENT=1, MUSIC_PRESENT=0, AUDIO_TYPE='voice'), 'eligible'


def choose_roles(pools, count, seed):
    chosen, forbidden = [], set()
    # Reserve scarce fake identities first; each role gets distinct identities.
    roles = [('fake_insert', 'fake'), ('replay_fake_insert', 'replay_fake'),
             ('background', 'bonafide'), ('real_insert', 'bonafide'),
             ('replay_real_insert', 'replay_bonafide')]
    for offset, (role, label) in enumerate(roles):
        selected = balanced_voice(pools[label], count, seed + offset, forbidden)
        for group, row in enumerate(selected):
            chosen.append({**row, 'ROLE': role, 'SOURCE_GROUP': f'lv61_{group:03d}'})
            for column in IDENTITIES:
                if row[column]:
                    forbidden.add(row[column])
    return pd.DataFrame(chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--groups', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20260905)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.groups < 1:
        raise ValueError('positive groups required')
    truths = set((ROOT / 'data').glob('**/truth*.csv'))
    for directory in ('data', 'reports'):
        truths.update((ROOT / directory).glob('**/reservation.csv'))
    hashes = set((ROOT / 'data').glob('**/source_hashes.csv'))
    hashes.update((ROOT / 'data').glob('**/audio_hashes.csv'))
    protection = v7.build_protection(sorted(truths), sorted(hashes))
    pools = {label: [] for label in LABELS}
    counts, parquet_records = Counter(), []
    for path in sorted((ROOT / 'data/external/echofake').glob('open_set_eval-*.parquet')):
        frame = pd.read_parquet(path, columns=['utt_id', 'label', 'source', 'source_speaker_id', 'synthesis_details'])
        parquet_records.append(dict(path=str(path.relative_to(ROOT)), rows=len(frame)))
        for row in frame.itertuples(index=False):
            record, reason = candidate_record(row, path.name, protection)
            counts[f'{row.label}:{reason}'] += 1
            if record is not None:
                pools[str(row.label)].append(record)
    # Duration eligibility is fixed before scoring and checked from actual
    # payload headers. Do not tile a too-short insertion and call it 2 s speech.
    candidates = {r['VOICE_ARCHIVE_ID']: r for rows in pools.values() for r in rows}
    for metadata in parquet_records:
        path = ROOT / metadata['path']
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=['utt_id', 'path']):
            for row in batch.to_pylist():
                key = str(row['utt_id'])
                if key in candidates:
                    header = sf.info(BytesIO(row['path']['bytes']))
                    candidates[key]['SOURCE_DURATION'] = header.frames / header.samplerate
        print(json.dumps(dict(header_checked=path.name)), flush=True)
    for label in pools:
        valid = []
        for row in pools[label]:
            if 'SOURCE_DURATION' not in row:
                raise ValueError('candidate audio payload missing')
            if row['SOURCE_DURATION'] >= 2.0:
                valid.append(row)
            else:
                counts[f'{label}:excluded_shorter_than_2s'] += 1
        pools[label] = valid
    selected = choose_roles(pools, args.groups, args.seed)
    if selected.ID.duplicated().any():
        raise ValueError('source reuse across roles')
    staging, publish = v7.atomic_output_dir(args.output)
    selected.to_csv(staging / 'truth_sources.csv', index=False)
    protection.snapshot.to_csv(staging / 'protected_inputs.csv', index=False)
    plan = dict(stage='metadata_reserved_not_rendered_not_scored', seed=args.seed,
                groups=args.groups, source_rows=len(selected), eligible_counts=dict(counts),
                source_manifest_sha256=sha256_file(staging / 'truth_sources.csv'),
                protection_sha256=sha256_file(staging / 'protected_inputs.csv'),
                builder_sha256=sha256_file(Path(__file__)), parquet_inputs=parquet_records,
                source='https://huggingface.co/datasets/EchoFake/EchoFake',
                purpose='held_out_long_voice_sparse_interval_confirmation',
                never_train=True, authenticity_scores_used=False,
                full_bank_complete=False, durations=[30, 45, 60],
                insertion_seconds=2.0, positions=['early', 'middle', 'late'],
                channels=['clean', 'g711_ulaw', 'opus_nb_8k'],
                limitation='Synthetic temporal mechanism audit, not natural long calls or all-type generalization.')
    (staging / 'reservation.json').write_text(json.dumps(plan, indent=2) + '\n')
    publish()
    print(json.dumps(plan, indent=2), flush=True)


if __name__ == '__main__':
    main()
