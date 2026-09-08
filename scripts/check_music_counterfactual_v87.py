"""Completed Music heads: saved-probability correspondence and file-order check."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_file_local_v86 import sha


def main():
    import numpy as np
    import pandas as pd
    import torch
    from src.music_counterfactual_inference_v87 import MusicCounterfactualPredictor
    from src.pipeline import load_audio
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT / 'reports/music_counterfactual_v87/full_v2')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert json.loads((args.run / 'report.json').read_text())['status'] == 'complete'
    frozen = json.loads((args.run / 'frozen.json').read_text())
    assert all(sha(p) == digest for p, digest in frozen['artifacts_sha256'].items())
    names = ['bce', 'invariant']
    frames = {n: pd.read_csv(args.run / f'{n}_development.csv', low_memory=False) for n in names}
    pd.testing.assert_frame_equal(frames['bce'][['DATASET', 'ID', 'PATH']], frames['invariant'][['DATASET', 'ID', 'PATH']])
    indices = np.unique(np.rint(np.linspace(0, len(frames['bce'])-1, 80)).astype(int))
    paths = [Path(__file__), ROOT / 'src/music_counterfactual_inference_v87.py',
             ROOT / 'scripts/check_file_local_v86.py', args.run / 'report.json', args.run / 'frozen.json',
             *[args.run / f'{n}.pt' for n in names], *[args.run / f'{n}_development.csv' for n in names]]
    paths.extend(Path(frames['bce'].iloc[i].PATH) for i in indices)
    hashes = {str(p): sha(p) for p in paths}
    args.output.mkdir(parents=True)
    (args.output / 'frozen.json').write_text(json.dumps(dict(artifacts_sha256=hashes,
        indices=indices.tolist(), tolerance=2e-6, scope='native correspondence, not accuracy'), indent=2))
    torch.set_num_threads(2)
    models = {n: MusicCounterfactualPredictor(args.run, ROOT, n) for n in names}
    started = time.monotonic()
    observed = {}
    maximum = {n: 0. for n in names}
    for order in [indices, indices[::-1]]:
        for index in order:
            row = frames['bce'].iloc[index]
            audio = load_audio(Path(row.PATH))
            values = {n: model(audio) for n, model in models.items()}
            if index in observed:
                assert values == observed[index], 'file order changed predictions'
            else:
                observed[index] = values
                for n, value in values.items():
                    diff = abs(value-float(frames[n].iloc[index].MUSIC_FAKE_PROB))
                    assert diff <= 2e-6, 'native/training probability mismatch'
                    maximum[n] = max(maximum[n], diff)
    assert all(sha(p) == digest for p, digest in hashes.items())
    assert all(sha(p) == digest for p, digest in frozen['artifacts_sha256'].items())
    result = dict(status='complete_native_equivalence', files=len(indices), calls=4*len(indices),
                  max_probability_difference=maximum, reverse_order_bit_exact=True,
                  seconds=time.monotonic()-started, official_improvement_verified=False)
    (args.output / 'report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
