#!/usr/bin/env python3
"""Fetch official CtrSVDD METADATA ONLY; no audio, model download, or training."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RECORD = 'https://zenodo.org/api/records/10467648'


def parse_protocol(text, split):
    records, seen = [], set()
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(f'line {number}: expected six metadata fields')
        source, singer, identity, unused, attack, label = fields
        if label not in {'bonafide', 'deepfake'} or identity in seen:
            raise ValueError(f'line {number}: unknown label or duplicate ID')
        if (label == 'bonafide' and attack != '-') or (label == 'deepfake' and not attack.startswith('A')):
            raise ValueError(f'line {number}: authenticity/attack contradiction')
        seen.add(identity)
        records.append(dict(ID=identity, SOURCE_DATASET=source, SINGER_ID=singer,
            ATTACK=attack, VOICE_FAKE=int(label == 'deepfake'), SPLIT=split,
            AUTHENTICITY=label, UNUSED_PROTOCOL_FIELD=unused))
    if not records:
        raise ValueError('empty official protocol')
    return records


def main():
    import pandas as pd
    import requests
    output = ROOT / 'reports/ctrsvdd_metadata_v75'
    if output.exists():
        raise FileExistsError(output)
    response = requests.get(RECORD, timeout=(10, 30))
    response.raise_for_status()
    metadata = response.json()
    if metadata['id'] != 10467648 or metadata['metadata']['license']['id'] != 'cc-by-nc-nd-4.0':
        raise ValueError('unexpected official record or license; review before proceeding')
    output.mkdir(parents=True)
    (output / 'zenodo_record.json').write_text(json.dumps(metadata, indent=2) + '\n')
    files = {entry['key']: entry for entry in metadata['files']}
    frames, provenance = {}, {}
    for split in ['train', 'dev']:
        name = split + '.txt'
        record = files[name]
        if record['size'] > 10_000_000:
            raise ValueError('metadata size guard exceeded; never fetch audio archives')
        url = record['links']['self']
        payload_response = requests.get(url, timeout=(10, 30))
        payload_response.raise_for_status()
        payload = payload_response.content
        expected = record['checksum']
        if len(payload) != record['size'] or 'md5:' + hashlib.md5(payload).hexdigest() != expected:
            raise ValueError('official metadata checksum/size mismatch')
        (output / name).write_bytes(payload)
        frame = pd.DataFrame(parse_protocol(payload.decode('utf-8'), split))
        frame.to_csv(output / (split + '_metadata.csv'), index=False)
        frames[split] = frame
        provenance[split] = dict(url=url, size=len(payload), official_checksum=expected,
            sha256=hashlib.sha256(payload).hexdigest(), rows=len(frame), singers=frame.SINGER_ID.nunique(),
            source_counts=frame.SOURCE_DATASET.value_counts().to_dict(),
            label_counts={str(k): int(v) for k, v in frame.VOICE_FAKE.value_counts().items()},
            attack_counts=frame.ATTACK.value_counts().to_dict())
    singer_overlap = set(frames['train'].SINGER_ID) & set(frames['dev'].SINGER_ID)
    file_overlap = set(frames['train'].ID) & set(frames['dev'].ID)
    report = dict(status='metadata_complete_not_training_ready', fetched_at=datetime.now(timezone.utc).isoformat(),
        official_record=RECORD, license='CC BY-NC-ND 4.0', protocols=provenance,
        train_dev_singer_overlap=sorted(singer_overlap), train_dev_file_overlap=sorted(file_overlap),
        audio_downloaded=False, model_downloaded=False, registered_for_training=False,
        automatic_submission_allowed=False,
        warnings=['Official audio archives are incomplete for some real source datasets due to licensing',
                  'Dataset-specific original licenses must be checked; do not silently drop only missing real clips',
                  'NoDerivative restrictions and downstream use/distribution require separate review',
                  'Protocol IDs do not establish decoded-PCM or song/content disjointness',
                  'Before any training, reserve new dev/eval IDs and check all existing protected data',
                  'Official baseline pretrained score orientation has a documented label-flip caveat'])
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
