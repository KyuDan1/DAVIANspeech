"""Frozen global/local File heads on previously exposed long-audio diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_file_local_v86 import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'worker', 'score'])
    parser.add_argument('--shard', type=int, choices=range(4))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = ROOT / 'reports/file_local_readout_v86/full'
    old = ROOT / 'reports/long_voice_v61/v73_one_shot/frozen.json'
    anchor_path = ROOT / 'reports/long_voice_v61/v73_one_shot/comparison/anchor_predictions.csv'
    if args.mode == 'prepare':
        assert json.loads((run / 'report.json').read_text())['status'] == 'complete'
        protocol = json.loads(old.read_text())
        training = json.loads((run / 'frozen.json').read_text())
        hashes = training['artifacts_sha256'].copy()
        assert all(sha(p) == digest for p, digest in hashes.items())
        extra = [Path(__file__), old, anchor_path, run / 'report.json', run / 'frozen.json',
                 run / 'global.pt', run / 'local.pt', ROOT / 'src/file_local_inference_v86.py',
                 ROOT / 'scripts/check_file_local_v86.py']
        extra.extend(Path(b['truth']) for b in protocol['banks'].values())
        hashes.update({str(p): sha(p) for p in extra})
        args.output.mkdir(parents=True, exist_ok=False)
        frozen = dict(inputs=protocol['inputs'], banks=protocol['banks'], shards=4,
                      artifacts_sha256=hashes,
                      scope='PREVIOUSLY EXPOSED diagnostic; fixed final heads; no tuning or submission')
        (args.output / 'frozen.json').write_text(json.dumps(frozen, indent=2))
        print(json.dumps(dict(status='prepared', rows=len(frozen['inputs']))), flush=True)
        return
    frozen = json.loads((args.output / 'frozen.json').read_text())
    assert all(sha(p) == digest for p, digest in frozen['artifacts_sha256'].items())
    if args.mode == 'worker':
        if args.shard is None:
            raise ValueError('--shard required')
        import torch
        from src.file_local_inference_v86 import FileLocalPredictor
        from src.pipeline import load_audio
        torch.set_num_threads(2)
        destination = args.output / f'shard_{args.shard}'
        destination.mkdir(exist_ok=False)
        predictor = FileLocalPredictor(run, ROOT)
        selected = frozen['inputs'][args.shard::4]
        started = time.monotonic()
        with (destination / 'predictions.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['ID', 'BANK', 'parent', 'global', 'local'])
            writer.writeheader()
            for index, row in enumerate(selected):
                path = Path(row['PATH'])
                assert sha(path) == row['SHA256']
                values = predictor(load_audio(path))
                assert sha(path) == row['SHA256']
                writer.writerow(dict(ID=row['ID'], BANK=row['BANK'], **values))
                if index % 100 == 0:
                    print(json.dumps(dict(shard=args.shard, files=index, seconds=time.monotonic()-started)), flush=True)
        assert all(sha(p) == digest for p, digest in frozen['artifacts_sha256'].items())
        report = dict(status='complete', rows=len(selected), seconds=time.monotonic()-started,
                      predictions_sha256=sha(destination / 'predictions.csv'),
                      frozen_sha256=sha(args.output / 'frozen.json'))
        (destination / 'report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        return
    import pandas as pd
    from src.evaluate_diagnostic import official_eer
    parts = []
    for shard in range(4):
        directory = args.output / f'shard_{shard}'
        report = json.loads((directory / 'report.json').read_text())
        assert report['status'] == 'complete'
        assert report['predictions_sha256'] == sha(directory / 'predictions.csv')
        assert report['frozen_sha256'] == sha(args.output / 'frozen.json')
        parts.append(pd.read_csv(directory / 'predictions.csv'))
    scores = pd.concat(parts, ignore_index=True)
    assert len(scores) == len(frozen['inputs']) and not scores.ID.duplicated().any()
    assert set(zip(scores.ID, scores.BANK)) == {(r['ID'], r['BANK']) for r in frozen['inputs']}
    anchor = pd.read_csv(anchor_path)[['ID', 'FILE_FAKE_PROB']].rename(columns={'FILE_FAKE_PROB': 'anchor'})
    results = []
    for bank, spec in frozen['banks'].items():
        truth = pd.read_csv(spec['truth'])
        frame = truth.merge(scores[scores.BANK.eq(bank)], on='ID', validate='one_to_one').merge(anchor, on='ID', validate='one_to_one')
        assert len(frame) == spec['rows']
        for name in ['anchor', 'parent', 'global', 'local']:
            assert frame[name].between(0, 1).all()
            results.append(dict(bank=bank, model=name, rows=len(frame), FILE_EER=official_eer(frame.FILE_FAKE, frame[name])))
    result = dict(status='complete_exposed_diagnostic', results=results, automatic_submission_allowed=False)
    (args.output / 'report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
