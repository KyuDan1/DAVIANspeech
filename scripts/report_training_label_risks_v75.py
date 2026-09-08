#!/usr/bin/env python3
"""Trace review flags to TRAIN rows without editing any labels or manifests."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import pandas as pd
import yaml
from train_three_stream_anchor_residual import authorized_partitions, load_train_identity_exclusions, filter_train_pair
from run_exact_anchor_v74 import sha256


def flag_rows(train, reviewed_sources):
    flagged = reviewed_sources.loc[reviewed_sources.BOTH_VOICE_REVIEW.eq(True)]
    # These are source-identity matches, not claims of confirmed label errors.
    ids = set(flagged.SOURCE_ID.astype(str))
    positive_fake_ids = set(flagged.loc[flagged.LABEL.eq(1), 'SOURCE_ID'].astype(str))
    result = train[['DATASET', 'ID']].copy()
    music_ids = train.MUSIC_SOURCE_ID.fillna('').astype(str)
    result['LINKED_MUSIC_SOURCE_HAS_VOICE_REVIEW_FLAG'] = music_ids.isin(ids)
    result['VOICE_ABSENCE_TARGET_REQUIRES_REVIEW'] = music_ids.isin(ids) & train.VOICE_PRESENT.eq(0)
    result['VOICE_REAL_TARGET_WITH_FLAGGED_FAKE_BACKGROUND'] = (
        music_ids.isin(positive_fake_ids) & train.VOICE_FAKE.eq(0) & train.MUSIC_FAKE.eq(1))
    generators = train.get('MUSIC_GENERATOR', pd.Series('', index=train.index)).fillna('').str.lower()
    result['MIXFAKE_RF_SUNO_UDIO_BACKGROUND_UNVERIFIED'] = (
        train.DATASET.eq('mixfake_music_train_v1') & generators.isin(['suno', 'udio'])
        & train.VOICE_FAKE.eq(0) & train.MUSIC_FAKE.eq(1))
    result['MUSIC_SOURCE_ID'] = music_ids
    return result


def main():
    directory = ROOT / 'reports/train_music_semantics_v75'
    output = directory / 'label_risk_trace'
    if output.exists():
        raise FileExistsError(output)
    report = json.loads((directory / 'report.json').read_text())
    if report['status'] != 'complete' or report['automatic_training_changes']:
        raise ValueError('completed review-only input required')
    reviewed = pd.read_csv(directory / 'raw_music_review.csv', dtype={'SOURCE_ID': str})
    config_path = ROOT / 'configs/music_specialist_v58.yaml'
    config = yaml.safe_load(config_path.read_text())
    exclusions = load_train_identity_exclusions(ROOT / config['train_identity_exclusions'])
    pieces, hashes = [], {}
    for partition in authorized_partitions(ROOT / 'configs/data_partitions.yaml', 'train', config[config['train_set']]):
        path = partition.truth_path
        frame = pd.read_csv(path, dtype={'ID': str})
        frame['DATASET'] = partition.name
        (frame, _, _), _ = filter_train_pair((frame, {}, {}), exclusions)
        pieces.append(frame)
        hashes[str(path)] = sha256(path)
    train = pd.concat(pieces, ignore_index=True)
    if len(train) != 18738:
        raise ValueError('current training population changed; re-audit explicitly')
    flags = flag_rows(train, reviewed)
    columns = [c for c in flags if c not in {'DATASET', 'ID', 'MUSIC_SOURCE_ID'}]
    summary = flags.groupby('DATASET')[columns].sum().astype(int)
    output.mkdir()
    flags.to_csv(output / 'training_row_review_flags.csv', index=False)
    summary.to_csv(output / 'by_dataset.csv')
    result = dict(status='complete_review_flags_only', train_rows=len(train),
        raw_source_rows=len(reviewed), raw_both_voice_review_flags=int(reviewed.BOTH_VOICE_REVIEW.sum()),
        counts={c: int(flags[c].sum()) for c in columns},
        limitations=['Detectors are not ground truth; all flags require source/semantic verification',
                    'Only explicit MUSIC_SOURCE_ID links are counted; not exhaustive descendant tracing',
                    'No labels were changed, no samples removed, no new model fitted'],
        source_hashes={**hashes, str(directory / 'raw_music_review.csv'): sha256(directory / 'raw_music_review.csv')},
        automatic_training_changes=False, automatic_submission_allowed=False)
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'train_rows', 'raw_both_voice_review_flags', 'counts']}), flush=True)


if __name__ == '__main__':
    main()
