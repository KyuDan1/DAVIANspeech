#!/usr/bin/env python3
"""Measure same-GPU process packing on authorized development, not accuracy.

All replicas stay resident throughout timing. Each file is predicted independently;
no cross-file padded encoder batches or model/ensemble settings are changed.
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]


def shard_indices(rows, workers, rank):
    if rows < 1 or workers < 1 or not 0 <= rank < workers:
        raise ValueError('positive rows/workers and valid rank required')
    return list(range(rank, rows, workers))


def assemble(parts, rows):
    import numpy as np
    result = np.full((rows, 5), np.nan)
    seen = set()
    for part in parts:
        indices = part['indices']
        if len(set(indices)) != len(indices) or any(i in seen or not 0 <= i < rows for i in indices):
            raise ValueError('overlapping or invalid shard indices')
        values = np.asarray(part['probabilities'], dtype=float).reshape(-1, 5)
        if len(values) != len(indices):
            raise ValueError('prediction count mismatch')
        result[indices] = values
        seen.update(indices)
    if len(seen) != rows or not np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
        raise ValueError('incomplete or invalid probabilities')
    return result


def worker(rank, commands, results, frozen, paths, threads):
    try:
        warnings.filterwarnings('ignore', category=FutureWarning)
        import torch
        from src.component_composition_inference_v73 import ComponentCompositionPredictor
        from src.pipeline import load_audio
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        model = ComponentCompositionPredictor(frozen, ROOT)
        # Decode before timing so the experiment isolates inference concurrency.
        waveforms = [load_audio(Path(path)) for path in paths]
        model(waveforms[rank % len(waveforms)])
        torch.cuda.synchronize()
        results.put(dict(kind='ready', rank=rank, gpu=torch.cuda.get_device_name(),
                         allocated_mb=torch.cuda.memory_allocated() / 2**20))
        while True:
            command = commands.get()
            if command is None:
                return
            indices = shard_indices(len(paths), command['workers'], rank)
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            predictions = [model(waveforms[i]).tolist() for i in indices]
            torch.cuda.synchronize()
            results.put(dict(kind='result', rank=rank, trial=command['trial'],
                indices=indices, probabilities=predictions,
                seconds=time.perf_counter() - started,
                peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mb=torch.cuda.max_memory_reserved() / 2**20))
    except BaseException:
        results.put(dict(kind='error', rank=rank, traceback=traceback.format_exc()))


def receive(results, processes):
    while True:
        try:
            message = results.get(timeout=5)
        except queue.Empty:
            failed = [(p.pid, p.exitcode) for p in processes if p.exitcode is not None]
            if failed:
                raise RuntimeError(f'worker exited unexpectedly: {failed}')
            continue
        if message['kind'] == 'error':
            raise RuntimeError(message['traceback'])
        return message


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'reports/component_composition_v73')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads-per-worker', type=int, default=2)
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or ',' in visible or visible == '-1':
        raise ValueError('explicitly select ONE idle GPU with CUDA_VISIBLE_DEVICES')
    if (1 not in args.workers or min(args.workers) < 1 or len(set(args.workers)) != len(args.workers)
            or args.repeats < 2 or args.threads_per_worker < 1):
        raise ValueError('unique positive workers including 1; repeats >= 2; positive threads required')
    if args.output.exists():
        raise FileExistsError(args.output)
    import numpy as np
    import pandas as pd
    from src.component_composition_inference_v73 import checksum
    from src.evaluate_diagnostic import PREDICTION_COLUMNS
    from verify_common_encoder_probe_inference import select_rows
    from verify_paired_checkpoint_inference_v68 import resolve_development
    frozen = args.run / 'frozen.json'
    source = args.run / 'equal_voice_pair_eat_music_predictions.csv'
    hashes = {str(p): checksum(p) for p in [frozen, source, Path(__file__)]}
    frame = select_rows(pd.read_csv(source, dtype={'DATASET': str, 'ID': str}))
    paths = [str(p) for p in resolve_development(frame, ROOT / 'configs/data_partitions.yaml')]
    expected = frame[PREDICTION_COLUMNS].to_numpy(float)
    args.output.mkdir(parents=True)
    settings = dict(scope='development-only throughput and equivalence; NOT accuracy or L4 benchmark',
        visible_gpu=visible, workers=args.workers, repeats=args.repeats,
        threads_per_worker=args.threads_per_worker, files=len(frame),
        waveforms_predecoded=True, all_replicas_resident=True, artifacts_sha256=hashes,
        automatic_submission_allowed=False)
    (args.output / 'frozen.json').write_text(json.dumps(settings, indent=2) + '\n')
    frame[['DATASET', 'ID']].to_csv(args.output / 'input_ids.csv', index=False)
    context = mp.get_context('spawn')
    results = context.Queue()
    commands = [context.Queue() for _ in range(max(args.workers))]
    processes = [context.Process(target=worker, args=(i, commands[i], results,
        str(frozen), paths, args.threads_per_worker)) for i in range(max(args.workers))]
    records, ready, reference = [], [], None
    try:
        for process in processes:
            process.start()
        for _ in processes:
            message = receive(results, processes)
            if message['kind'] != 'ready':
                raise ValueError('expected worker readiness')
            ready.append(message)
            print(json.dumps(message), flush=True)
        # Alternate ascending/descending order to reduce fixed-order warmup bias.
        for repeat in range(args.repeats):
            order = sorted(args.workers, reverse=bool(repeat % 2))
            for count in order:
                trial = f'{repeat}:{count}'
                started = time.perf_counter()
                for command in commands[:count]:
                    command.put(dict(workers=count, trial=trial))
                parts = [receive(results, processes) for _ in range(count)]
                seconds = time.perf_counter() - started
                if (any(p.get('trial') != trial for p in parts)
                        or {p['rank'] for p in parts} != set(range(count))):
                    raise ValueError('wrong trial or worker set')
                actual = assemble(parts, len(frame))
                if reference is None:
                    reference = actual.copy()
                difference = float(np.abs(actual - expected).max())
                exact = bool(np.array_equal(actual, reference))
                record = dict(trial=trial, workers=count, seconds=seconds,
                    files_per_second=len(frame) / seconds,
                    maximum_cached_probability_difference=difference,
                    serial_bit_exact=exact, worker_measurements=[{k: v for k, v in p.items()
                        if k not in {'probabilities', 'indices'}} for p in parts])
                records.append(record)
                with (args.output / 'trials.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
                print(json.dumps(record), flush=True)
                if difference > 5e-4 or not exact:
                    raise ValueError('concurrency changes frozen file-local predictions')
        if any(checksum(path) != digest for path, digest in hashes.items()):
            raise ValueError('frozen benchmark inputs changed')
        medians = {count: float(np.median([r['seconds'] for r in records if r['workers'] == count]))
                   for count in sorted(args.workers)}
        best = min(medians, key=medians.get)
        report = dict(**settings, status='complete', workers_ready=ready, trials=records,
            median_seconds=medians, fastest_measured_workers=best,
            speedup_vs_serial=medians[1] / medians[best],
            limitation='Predecoded short development files on one B200; not long-file/end-to-end/L4 throughput.')
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({k: report[k] for k in ['status', 'median_seconds', 'fastest_measured_workers', 'speedup_vs_serial']}), flush=True)
    finally:
        for command in commands:
            command.put(None)
        for process in processes:
            if process.pid is not None:
                process.join(timeout=10)
                if process.is_alive():
                    # Only terminate this benchmark's own child processes.
                    process.terminate()
                    process.join(timeout=5)


if __name__ == '__main__':
    main()
