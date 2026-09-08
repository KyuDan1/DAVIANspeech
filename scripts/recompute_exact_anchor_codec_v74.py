#!/usr/bin/env python3
"""Recompute authorized codec development with four unchanged archive workers."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from run_exact_anchor_v74 import COLUMNS, sha256, stage_inputs, verify_inventory

OUTPUT = ROOT / 'reports/exact_anchor_v74/codec_recomputed'


def main():
    import numpy as np
    import pandas as pd
    from verify_paired_checkpoint_inference_v68 import resolve_development
    from src.evaluate_diagnostic import score_frame
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = ROOT / 'reports/three_stream_all_type_v57_strict/v50_authorized_v2/v1_strict__music_only_predictions.csv'
    candidate_path = ROOT / 'reports/component_composition_v73/equal_voice_pair_eat_music_predictions.csv'
    smoke = ROOT / 'reports/exact_anchor_v74/development_smoke'
    initial = json.loads((smoke / 'frozen.json').read_text())
    completion = json.loads((smoke / 'completed.json').read_text())
    if completion['frozen_sha256'] != sha256(smoke / 'frozen.json'):
        raise ValueError('exact entrypoint smoke provenance changed')
    attestation = initial['package_attestation']
    verify_inventory(attestation)
    frame = pd.read_csv(source, dtype={'DATASET': str, 'ID': str})
    if len(frame) != 600 or set(frame.DATASET) != {'codec_mixed_dev_v4'}:
        raise ValueError('only authorized full codec development is supported')
    paths = resolve_development(frame, ROOT / 'configs/data_partitions.yaml')
    rows = [dict(ID=row.ID, DATASET=row.DATASET, PATH=str(path.resolve()), SHA256=sha256(path))
            for (_, row), path in zip(frame.iterrows(), paths)]
    gpu = '4'
    status = subprocess.check_output(['nvidia-smi', '-i', gpu,
        '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    free, utilization = map(int, status.strip().split(','))
    if free < 96 * 1024 or utilization > 5:
        raise ValueError('GPU 4 must be idle with >=96GiB free')
    OUTPUT.mkdir(parents=True)
    hashes = {str(path): sha256(path) for path in [source, candidate_path,
        ROOT / 'scripts/run_exact_anchor_v74.py', Path(__file__),
        ROOT / 'data/eval/codec_mixed_dev_v4/truth.csv']}
    frozen = dict(scope='development-only fresh anchor audit; no model changes',
        artifacts_sha256=hashes, package_attestation=attestation,
        gpu=gpu, shards=4, inputs=rows, automatic_submission_allowed=False)
    root_frozen = OUTPUT / 'frozen.json'
    root_frozen.write_text(json.dumps(frozen, indent=2) + '\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', PYTHONWARNINGS='ignore::FutureWarning')
    processes, logs = [], []
    started = time.monotonic()
    try:
        for shard in range(4):
            stage = OUTPUT / f'shard_{shard}'
            selected = rows[shard::4]
            stage_inputs(stage, Path(attestation['package']), selected)
            local = dict(package_attestation=attestation, inputs=selected,
                root_frozen_sha256=sha256(root_frozen), scope='development recomputation',
                automatic_submission_allowed=False)
            (stage / 'frozen.json').write_text(json.dumps(local, indent=2) + '\n')
            log = (stage / 'run.log').open('x')
            logs.append(log)
            process = subprocess.Popen([sys.executable, str(ROOT / 'scripts/run_exact_anchor_v74.py'),
                'worker', '--stage', str(stage)], env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            print(json.dumps(dict(shard=shard, pid=process.pid, gpu=gpu, rows=len(selected))), flush=True)
        while any(p.poll() is None for p in processes):
            failed = [(p.pid, p.returncode) for p in processes if p.returncode not in {None, 0}]
            if failed:
                raise RuntimeError(f'workers failed: {failed}; no automatic restart')
            time.sleep(5)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError('worker failed')
        pieces = []
        for shard in range(4):
            stage = OUTPUT / f'shard_{shard}'
            report = json.loads((stage / 'completed.json').read_text())
            predictions = stage / 'output/submission.csv'
            if (report['frozen_sha256'] != sha256(stage / 'frozen.json')
                    or report['predictions_sha256'] != sha256(predictions)):
                raise ValueError('changed runtime artifacts')
            pieces.append(pd.read_csv(predictions, dtype={'ID': str}))
        actual = pd.concat(pieces, ignore_index=True)
        if actual.ID.duplicated().any() or set(actual.ID) != set(frame.ID):
            raise ValueError('wrong output IDs')
        actual = actual.set_index('ID').loc[frame.ID].reset_index()
        values = actual[COLUMNS].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError('invalid actual probabilities')
        truth = pd.read_csv(ROOT / 'data/eval/codec_mixed_dev_v4/truth.csv', dtype={'ID': str})
        candidate = pd.read_csv(candidate_path, dtype={'ID': str, 'DATASET': str})
        candidate = candidate.loc[candidate.DATASET.eq('codec_mixed_dev_v4'), ['ID', *COLUMNS]]
        summaries = []
        for name, predictions in [('cached_anchor', frame), ('actual_anchor', actual), ('v73', candidate)]:
            joined = truth.merge(predictions[['ID', *COLUMNS]], on='ID', validate='one_to_one')
            if len(joined) != 600:
                raise ValueError('missing comparison rows')
            summaries.append(dict(model=name, channel='all', **score_frame(joined)))
            summaries.extend(dict(model=name, channel=channel, **score_frame(part))
                             for channel, part in joined.groupby('CHANNEL'))
        table = pd.DataFrame(summaries)
        old_new_difference = np.abs(values - frame[COLUMNS].to_numpy(float))
        first = pd.read_csv(smoke / 'output/submission.csv', dtype={'ID': str}).set_index('ID')
        new = actual.set_index('ID').loc[first.index]
        repeat_difference = np.abs(first[COLUMNS].to_numpy(float) - new[COLUMNS].to_numpy(float))
        if any(sha256(path) != expected for path, expected in hashes.items()):
            raise ValueError('frozen comparison input changed')
        verify_inventory(attestation)
        actual['DATASET'] = 'codec_mixed_dev_v4'
        actual.to_csv(OUTPUT / 'predictions.csv', index=False)
        table.to_csv(OUTPUT / 'metrics.csv', index=False)
        report = dict(status='complete', rows=600, seconds=time.monotonic() - started,
            frozen_sha256=sha256(root_frozen), predictions_sha256=sha256(OUTPUT / 'predictions.csv'),
            maximum_cache_difference=float(old_new_difference.max()),
            eight_file_shard_repeat_maximum_difference=float(repeat_difference.max()),
            by_column_cache_difference=dict(zip(COLUMNS, old_new_difference.max(0).tolist())),
            overall=table.loc[table.channel.eq('all')].to_dict('records'),
            automatic_submission_allowed=False)
        (OUTPUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
