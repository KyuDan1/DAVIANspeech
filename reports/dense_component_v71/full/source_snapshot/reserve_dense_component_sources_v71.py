#!/usr/bin/env python3
"""Find pure raw components already represented in strict authorized TRAIN.

No evaluation audio, model inference or new source downloads. Reserved/retired
and development source/group identities are all excluded before audio reads.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import pandas as pd
import yaml

from src.data_guard import identity_tokens, PROTECTED_FROM_TRAIN_ROLES
from src.pipeline import find_audio_files
from train_paired_wpt_file_v60 import load_data, sha256
from train_common_encoder_probe import audit_protected_sources
from build_temporal_mixed_train import SOURCE_RECIPES, first_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/dense_component_v71.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = yaml.safe_load(args.config.read_text())
    train, _, audit = load_data(config)
    audit['all_protected_roles'] = audit_protected_sources(train, ROOT / config['partition_config'])
    roles = yaml.safe_load((ROOT / config['partition_config']).read_text())
    protected = set()
    for role in PROTECTED_FROM_TRAIN_ROLES:
        for relative in roles.get(role, []):
            protected.update(identity_tokens(pd.read_csv(ROOT / relative, dtype=str)))
    records, seen, excluded, maps = [], set(), {}, {}
    for mixture_name, voice_bank, music_bank in SOURCE_RECIPES:
        mixtures = train.loc[train.DATASET.eq(mixture_name)]
        for component, bank in [('VOICE', voice_bank), ('MUSIC', music_bank)]:
            raw_truth_path = ROOT / 'data/eval' / bank / 'truth.csv'
            raw = pd.read_csv(raw_truth_path, dtype=str).set_index('ID')
            labels = mixtures[[component + '_SOURCE_ID', component + '_FAKE']].dropna().drop_duplicates()
            if labels.groupby(component + '_SOURCE_ID')[component + '_FAKE'].nunique().gt(1).any():
                raise ValueError('inconsistent source labels in authorized train')
            if bank not in maps:
                audio = list(find_audio_files(raw_truth_path.parent / 'audio'))
                maps[bank] = {p.stem: p for p in audio}
                if len(maps[bank]) != len(audio):
                    raise ValueError('ambiguous raw audio IDs')
            other = 'MUSIC' if component == 'VOICE' else 'VOICE'
            for source_id, label in labels.itertuples(index=False, name=None):
                source_id = str(source_id)
                key = (bank, source_id)
                if key in seen:
                    continue
                seen.add(key)
                if source_id not in raw.index:
                    raise ValueError('authorized source missing raw truth')
                row = raw.loc[source_id]
                if isinstance(row, pd.DataFrame):
                    raise ValueError('ambiguous raw truth')
                if (pd.isna(row.get(component + '_PRESENT')) or int(float(row[component + '_PRESENT'])) != 1
                        or (pd.notna(row.get(other + '_PRESENT')) and int(float(row[other + '_PRESENT'])) != 0)):
                    excluded['not_pure'] = excluded.get('not_pure', 0) + 1
                    continue
                if pd.isna(row[component + '_FAKE']) or int(float(row[component + '_FAKE'])) != int(label):
                    raise ValueError('component source label mismatch')
                group = first_value(row, ('SPEAKER_ID', 'SPEAKER', 'GROUP_ID', 'SOURCE_ID'), source_id)
                token_frame = pd.DataFrame([dict(ID=source_id, GROUP_ID=group, **{component + '_SOURCE_ID': source_id})])
                if identity_tokens(token_frame) & protected:
                    excluded['protected_identity'] = excluded.get('protected_identity', 0) + 1
                    continue
                path = maps[bank].get(source_id)
                if path is None:
                    raise FileNotFoundError(f'{bank}/{source_id}')
                records.append(dict(ID=f'{bank}:{source_id}', SOURCE_ID=source_id, SOURCE_BANK=bank,
                    GROUP_ID=group, COMPONENT=component, LABEL=int(label), PATH=str(path.resolve()),
                    GENERATOR=first_value(row, ('GENERATOR',), 'unknown'),
                    RAW_TRUTH_SHA256=sha256(raw_truth_path)))
    frame = pd.DataFrame(records)
    counts = frame.groupby(['COMPONENT', 'LABEL']).size()
    if len(counts) != 4 or counts.min() < 4:
        raise ValueError('insufficient pure training sources after exclusion')
    args.output.mkdir(parents=True)
    frame.to_csv(args.output / 'sources.csv', index=False)
    report = dict(stage='authorized train source catalog; not rendered evaluation',
        counts={f'{a}/{b}': int(n) for (a, b), n in counts.items()}, excluded=excluded,
        protected_source_identity_overlap=0, audit=audit, config_sha256=sha256(args.config),
        source_catalog_sha256=sha256(args.output / 'sources.csv'), code_sha256=sha256(Path(__file__)))
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(counts=report['counts'], excluded=excluded)), flush=True)


if __name__ == '__main__':
    main()
