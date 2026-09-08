#!/usr/bin/env python3
"""Read declared development audio headers only; no detector or test access."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pandas as pd
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.wav', '.flac', '.mp3', '.ogg', '.opus', '.m4a', '.aac', '.wma'}


def inspect(row):
    info = sf.info(row['path'])
    duration = info.frames / info.samplerate
    # A coverage lower bound, not measured detector errors or an assumption
    # about where a fake interval occurs. Overlapping crops may cover less.
    return {**row, 'seconds': duration, 'sample_rate': info.samplerate, 'channels': info.channels,
            'uncovered_seconds_3view': max(0., duration - 3 * 64600 / 16000),
            'uncovered_seconds_5view': max(0., duration - 5 * 64600 / 16000)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    matrix = yaml.safe_load((ROOT / 'configs/music_specialist_v58.yaml').read_text())
    partitions = yaml.safe_load((ROOT / 'configs/data_partitions.yaml').read_text())
    requested = set(matrix[matrix['development_set']])
    rows = []
    for relative in partitions['development']:
        truth_path = ROOT / relative
        name = truth_path.parent.name
        if name not in requested:
            continue
        files = [p for p in (truth_path.parent / 'audio').iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS]
        paths = {p.stem: p for p in files}
        if len(paths) != len(files):
            raise ValueError(f'duplicate audio IDs in {name}')
        frame = pd.read_csv(truth_path, dtype={'ID': str})
        for row in frame.itertuples():
            rows.append(dict(DATASET=name, ID=row.ID, path=str(paths[row.ID])))
    if {r['DATASET'] for r in rows} != requested:
        raise ValueError('declared development coverage differs')
    with ThreadPoolExecutor(max_workers=4) as executor:
        result = pd.DataFrame(executor.map(inspect, rows))
    summary = []
    for name, frame in [('all', result), *list(result.groupby('DATASET'))]:
        summary.append(dict(dataset=name, n=len(frame), min_seconds=float(frame.seconds.min()),
                            median_seconds=float(frame.seconds.median()), max_seconds=float(frame.seconds.max()),
                            n_lt_4=int(frame.seconds.lt(4).sum()), n_gt_60=int(frame.seconds.gt(60).sum()),
                            n_ge_30=int(frame.seconds.ge(30).sum()), n_ge_40=int(frame.seconds.ge(40).sum()),
                            n_ge_59=int(frame.seconds.ge(59).sum()),
                            n_with_5view_gaps=int(frame.uncovered_seconds_5view.gt(0).sum()),
                            n_with_3view_gaps=int(frame.uncovered_seconds_3view.gt(0).sum())))
    args.output.mkdir(parents=True)
    result.to_csv(args.output / 'headers.csv', index=False)
    (args.output / 'summary.json').write_text(json.dumps(dict(
        scope='Declared development header durations; not unseen test composition or detection accuracy',
        coverage_note='Uncovered seconds are lower bounds from total crop budget, not detector error rates.',
        summaries=summary), indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
