#!/usr/bin/env python3
"""Audit the ACTUAL 600 voice-absent training targets; never change labels."""
import json
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
import torch
import yaml
from src.eat_presence import EatPresence
from src.presence import PannsPresence
from src.pipeline import load_audio
from src.common_encoder_probe_inference import CommonProbePredictor
from train_paired_wpt_file_v60 import load_data, sha256
from train_common_encoder_probe import audit_protected_sources


def main():
    output = ROOT / 'reports/train_presence_negatives_v75'
    if output.exists():
        raise FileExistsError(output)
    config = yaml.safe_load((ROOT / 'configs/common_eat_adaptation.yaml').read_text())
    train, _, audit = load_data(config)
    audit['all_protected_roles'] = audit_protected_sources(train, ROOT / config['partition_config'])
    selected = train.loc[train.VOICE_PRESENT.eq(0)].copy()
    if len(selected) != 600 or set(selected.DATASET) != {'channel_invariant_factorial_train_v1'}:
        raise ValueError('voice-negative population changed; re-audit explicitly')
    checkpoint = ROOT / 'reports/common_eat_adaptation/full/adapted/head.pt'
    raw_path = ROOT / 'data/eval/multigen_music_presence_train_v1/truth.csv'
    raw = pd.read_csv(raw_path, dtype={'ID': str}).set_index('ID')
    if not set(selected.MUSIC_SOURCE_ID).issubset(raw.index):
        raise ValueError('missing original music source metadata')
    for key in ['SOURCE', 'GENERATOR', 'GROUP_ID']:
        selected['MUSIC_' + key] = selected.MUSIC_SOURCE_ID.map(raw[key])
    hashes = {str(path): sha256(path) for path in [Path(__file__), checkpoint, raw_path,
        ROOT / 'data/eval/channel_invariant_factorial_train_v1/truth.csv']}
    output.mkdir(parents=True)
    frozen = dict(scope='TRAIN-only review of voice-absent supervision; NOT ground-truth correction',
        rows=len(selected), artifacts_sha256=hashes, thresholds=dict(review=.5, learned_low=.5),
        automatic_training_changes=False, automatic_submission_allowed=False)
    (output / 'frozen.json').write_text(json.dumps(frozen, indent=2) + '\n')
    (output / 'identity_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    warnings.filterwarnings('ignore', category=FutureWarning)
    torch.set_num_threads(2)
    models = dict(eat=EatPresence(ROOT / 'models/eat-base-as2m', ROOT / 'models/panns'),
                  panns=PannsPresence(ROOT / 'models/panns'),
                  learned=CommonProbePredictor(checkpoint, ROOT))
    records, started = [], time.monotonic()
    for index, row in enumerate(selected.itertuples(index=False)):
        path = Path(row.PATH)
        digest = sha256(path)
        audio = load_audio(path)
        ev, em = models['eat'].predict_audio_set(audio)
        pv, pm = models['panns'].predict(audio)
        learned = models['learned'](audio)
        record = {key: getattr(row, key) for key in ['DATASET', 'ID', 'MIXTURE_ID', 'MUSIC_SOURCE_ID',
            'CHANNEL', 'MUSIC_FAKE', 'VOICE_PRESENT', 'MUSIC_SOURCE', 'MUSIC_GENERATOR', 'MUSIC_GROUP_ID', 'PATH']}
        record.update(AUDIO_SHA256=digest, EAT_VOICE=ev, PANNS_VOICE=pv,
            EAT_MUSIC=em, PANNS_MUSIC=pm, LEARNED_VOICE_PRESENT=float(learned[3]),
            BOTH_VOICE_REVIEW=ev >= .5 and pv >= .5,
            SEMANTIC_LEARNED_DISAGREEMENT=ev >= .5 and pv >= .5 and learned[3] < .5)
        if sha256(path) != digest:
            raise ValueError('training waveform changed')
        records.append(record)
        if (index + 1) % 100 == 0:
            print(json.dumps(dict(files=index + 1, seconds=time.monotonic() - started)), flush=True)
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError('frozen input changed')
    frame = pd.DataFrame(records)
    frame.to_csv(output / 'review.csv', index=False)
    aggregate = frame.groupby(['MUSIC_SOURCE', 'MUSIC_FAKE']).agg(rows=('ID', 'size'),
        unique_sources=('MUSIC_SOURCE_ID', 'nunique'), semantic_flags=('BOTH_VOICE_REVIEW', 'sum'),
        semantic_learned_disagreements=('SEMANTIC_LEARNED_DISAGREEMENT', 'sum')).reset_index()
    aggregate.to_csv(output / 'by_source.csv', index=False)
    result = dict(**frozen, status='complete', summary=aggregate.to_dict('records'),
        semantic_review_flags=int(frame.BOTH_VOICE_REVIEW.sum()),
        semantic_learned_disagreements=int(frame.SEMANTIC_LEARNED_DISAGREEMENT.sum()),
        independent_music_sources=frame.MUSIC_SOURCE_ID.nunique(),
        seconds=time.monotonic() - started,
        limitation='Correlated channel variants and semantic model flags are not verified vocal annotations.')
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
