#!/usr/bin/env python3
"""TRAIN-only semantic review, not automatic relabeling or held-out tuning."""
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
from train_paired_wpt_file_v60 import load_data, sha256
from train_common_encoder_probe import audit_protected_sources
from src.data_guard import identity_tokens, PROTECTED_FROM_TRAIN_ROLES


def main():
    output = ROOT / 'reports/train_music_semantics_v75'
    if output.exists():
        raise FileExistsError(output)
    config_path = ROOT / 'configs/dense_component_v71.yaml'
    config = yaml.safe_load(config_path.read_text())
    train, _, audit = load_data(config)
    audit['all_protected_roles'] = audit_protected_sources(train, ROOT / config['partition_config'])
    catalog_path = ROOT / config['source_catalog']
    catalog_report = json.loads((catalog_path.parent / 'report.json').read_text())
    if sha256(catalog_path) != catalog_report['source_catalog_sha256']:
        raise ValueError('original catalog changed')
    catalog = pd.read_csv(catalog_path, dtype={'ID': str, 'SOURCE_ID': str, 'GROUP_ID': str})
    catalog = catalog.loc[catalog.COMPONENT.eq('MUSIC')].copy()
    eligible = set(train.MUSIC_SOURCE_ID.dropna().astype(str))
    if not set(catalog.SOURCE_ID).issubset(eligible):
        raise ValueError('catalog includes sources outside current strict TRAIN')
    roles = yaml.safe_load((ROOT / config['partition_config']).read_text())
    protected = set()
    for role in PROTECTED_FROM_TRAIN_ROLES:
        for relative in roles.get(role, []):
            protected.update(identity_tokens(pd.read_csv(ROOT / relative, dtype=str)))
    tokens = catalog[['SOURCE_ID', 'GROUP_ID']].rename(columns={'SOURCE_ID': 'ID'})
    if identity_tokens(tokens) & protected:
        raise ValueError('protected source/group found in review selection')
    hashes = {str(path): sha256(path) for path in [catalog_path, config_path, Path(__file__),
        ROOT / 'models/panns/Cnn14_mAP=0.431.pth', ROOT / 'models/panns/component_labels.json']}
    for bank, block in catalog.groupby('SOURCE_BANK'):
        truth = ROOT / 'data/eval' / bank / 'truth.csv'
        expected = set(block.RAW_TRUTH_SHA256)
        if expected != {sha256(truth)}:
            raise ValueError('source truth changed')
        hashes[str(truth)] = sha256(truth)
    output.mkdir(parents=True)
    frozen = dict(scope='authorized TRAIN raw-music semantics review; NOT accuracy/automatic labels',
        rows=len(catalog), model_names=['pretrained AudioSet EAT-base', 'pretrained AudioSet PANNs'],
        review_threshold=.5, decision_rule='both voice evidence >= .5 creates review flag, never ground truth',
        protected_source_overlap=0, artifacts_sha256=hashes, automatic_training_changes=False,
        automatic_submission_allowed=False)
    (output / 'frozen.json').write_text(json.dumps(frozen, indent=2) + '\n')
    (output / 'training_identity_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    # Metadata-only review of sources whose actual background semantics are not
    # established by the MixFake foreground/background authenticity fields.
    mixed = train.loc[train.DATASET.eq('mixfake_music_train_v1')].copy()
    ambiguous = mixed.loc[mixed.MUSIC_GENERATOR.str.lower().isin(['suno', 'udio'])
                          & mixed.MUSIC_FAKE.eq(1) & mixed.VOICE_FAKE.eq(0)]
    ambiguous[['DATASET', 'ID', 'VOICE_SOURCE_ID', 'MUSIC_SOURCE_ID', 'MUSIC_GENERATOR']].to_csv(
        output / 'mixfake_background_vocal_semantics_unverified.csv', index=False)
    warnings.filterwarnings('ignore', category=FutureWarning)
    torch.set_num_threads(2)
    eat = EatPresence(ROOT / 'models/eat-base-as2m', ROOT / 'models/panns')
    panns = PannsPresence(ROOT / 'models/panns')
    torch.cuda.reset_peak_memory_stats()
    started, records = time.monotonic(), []
    for index, row in enumerate(catalog.itertuples(index=False)):
        path = Path(row.PATH)
        digest = sha256(path)
        audio = load_audio(path)
        ev, em = eat.predict_audio_set(audio)
        pv, pm = panns.predict(audio)
        if sha256(path) != digest:
            raise ValueError('source audio changed during review')
        record = row._asdict()
        record.update(AUDIO_SHA256=digest, EAT_VOICE=ev, EAT_MUSIC=em, PANNS_VOICE=pv, PANNS_MUSIC=pm,
            BOTH_VOICE_REVIEW=ev >= .5 and pv >= .5, EITHER_VOICE_REVIEW=ev >= .5 or pv >= .5)
        records.append(record)
        if (index + 1) % 100 == 0:
            print(json.dumps(dict(files=index + 1, seconds=time.monotonic() - started)), flush=True)
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError('frozen review inputs changed')
    frame = pd.DataFrame(records)
    frame.to_csv(output / 'raw_music_review.csv', index=False)
    summary = frame.groupby(['SOURCE_BANK', 'LABEL']).agg(
        rows=('ID', 'size'), both_voice_flags=('BOTH_VOICE_REVIEW', 'sum'),
        either_voice_flags=('EITHER_VOICE_REVIEW', 'sum')).reset_index()
    summary.to_csv(output / 'by_source.csv', index=False)
    report = dict(**frozen, status='complete', by_source=summary.to_dict('records'),
        mixfake_rf_suno_udio_rows_requiring_background_semantics_review=len(ambiguous),
        seconds=time.monotonic() - started, peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
        limitations=['Presence detectors can make false positives; no automatic corrections',
                     'A Suno/Udio generator name alone does not prove vocals exist',
                     'This catalog covers raw training components, not every original training mixture'])
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
