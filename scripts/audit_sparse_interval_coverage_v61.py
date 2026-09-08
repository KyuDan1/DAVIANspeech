#!/usr/bin/env python3
"""Check insertion/crop geometry without loading or scoring a detector."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

SR = 16000
WINDOW = 64600


def interval_coverage(seconds, start, end, views):
    length = round(seconds * SR)
    left, right = round(start * SR), round(end * SR)
    if not 0 <= left < right <= length or views < 1:
        raise ValueError('invalid interval or view count')
    starts = np.rint(np.linspace(0, max(0, length - WINDOW), views)).astype(int)
    covered = np.zeros(right - left, dtype=bool)
    for crop in starts:
        a, b = max(crop, left), min(crop + WINDOW, right)
        if a < b:
            covered[a - left:b - left] = True
    return float(covered.mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--truth', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frame = pd.read_csv(args.truth)
    # Geometry is identical across label/recording/channel variants.
    frame = frame.drop_duplicates(['GROUP_ID', 'DURATION', 'POSITION']).copy()
    summaries = []
    for views in (3, 5):
        coverage = [interval_coverage(r.DURATION, r.INSERTION_START, r.INSERTION_END, views)
                    for r in frame.itertuples()]
        frame[f'fraction_seen_{views}'] = coverage
        for duration, group in frame.groupby('DURATION'):
            values = group[f'fraction_seen_{views}']
            summaries.append(dict(views=views, duration=int(duration), n=len(group),
                entirely_missed=int(values.eq(0).sum()), fully_seen=int(values.eq(1).sum()),
                partial=int((values.gt(0) & values.lt(1)).sum()), mean_fraction_seen=float(values.mean())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(scope='Temporal geometry, not detector accuracy',
        rows=summaries), indent=2) + '\n')
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
