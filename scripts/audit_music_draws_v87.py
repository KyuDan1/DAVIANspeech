"""Read-only snapshot of actual v87 draws and music source-bank confounding."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'reports/music_counterfactual_v87/full_v2')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    catalog_path = ROOT / 'reports/dense_component_v71/source_catalog/sources.csv'
    catalog_bytes = catalog_path.read_bytes()
    catalog = {r['ID']: r for r in csv.DictReader(catalog_bytes.decode().splitlines())}
    counts = {k: Counter() for k in ['kind', 'music_label', 'synthetic_bank_label',
                                   'synthetic_generator_label', 'layout', 'channel', 'replay_corpus']}
    unique = defaultdict(set)
    snapshots = []
    total = 0
    for path in sorted(args.run.glob('epoch_*_traces.jsonl')):
        payload = path.read_bytes()
        lines = payload.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b'\n'):
            lines.pop()
        snapshots.append(dict(path=str(path), bytes=len(payload), complete_rows=len(lines),
                              sha256=hashlib.sha256(payload).hexdigest()))
        for line in lines:
            row = json.loads(line)
            total += 1
            counts['kind'][row['kind']] += 1
            counts['music_label'][str(row['music_label'])] += 1
            if row['kind'] == 'TRAIN_music_counterfactual':
                source = row['sources'][0]
                raw = catalog[source['id']]
                assert raw['COMPONENT'] == 'MUSIC'
                assert int(raw['LABEL']) == row['music_label'] == source['label']
                assert raw['GROUP_ID'] == source['group']
                label = str(row['music_label'])
                counts['synthetic_bank_label'][label + ':' + raw['SOURCE_BANK']] += 1
                counts['synthetic_generator_label'][label + ':' + raw['GENERATOR']] += 1
                counts['layout'][row['layout']] += 1
                counts['channel'][row['channel']] += 1
                unique[label].add((raw['SOURCE_BANK'], raw['GROUP_ID']))
            elif row['kind'] == 'original_TRAIN_music_replay':
                counts['replay_corpus'][row['dataset']] += 1
            else:
                raise ValueError('unknown training trace')
    pool = Counter((r['LABEL'] + ':' + r['SOURCE_BANK']) for r in catalog.values() if r['COMPONENT'] == 'MUSIC')
    result = dict(status='complete_snapshot_audit', rows=total, snapshots=snapshots,
                  training_complete=(args.run / 'report.json').exists(),
                  catalog_sha256=hashlib.sha256(catalog_bytes).hexdigest(),
                  counts={k: dict(v) for k, v in counts.items()}, catalog_music_bank_label=dict(pool),
                  synthetic_unique_bank_group_by_label={k: len(v) for k, v in unique.items()},
                  caveat='Snapshot, not final counts. Generator names are not independent families; bank/group IDs are not PCM uniqueness proof. No labels, sampler, or weights changed.')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
