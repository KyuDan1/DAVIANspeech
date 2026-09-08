#!/usr/bin/env python3
"""One frozen candidate vs exact submitted anchor on protected v61 banks."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from run_exact_anchor_v74 import COLUMNS, sha256, stage_inputs, verify_inventory

BANKS = {
    'uniform': ('long_voice_uniform_v61', 'uniform_construction_validation.json', 720),
    'fixed': ('long_voice_sparse_v61', 'construction_validation.json', 2160),
}
RUN = ROOT / 'reports/long_voice_v61/v73_one_shot'


def prepare():
    import pandas as pd
    import yaml
    from src.data_guard import assert_development_eval_separation, assert_no_locked_eval_leakage, TRAIN_ROLES
    if RUN.exists():
        raise FileExistsError('one-shot comparison already reserved; never overwrite/reselect')
    smoke = ROOT / 'reports/exact_anchor_v74/development_smoke'
    # The historical cache failed correspondence. Do not loosen its tolerance;
    # require a complete fresh exact-package development comparison instead.
    fresh = ROOT / 'reports/exact_anchor_v74/codec_recomputed'
    fresh_report = json.loads((fresh / 'report.json').read_text())
    if (fresh_report['status'] != 'complete' or fresh_report['rows'] != 600
            or fresh_report['frozen_sha256'] != sha256(fresh / 'frozen.json')
            or fresh_report['predictions_sha256'] != sha256(fresh / 'predictions.csv')
            or fresh_report['eight_file_shard_repeat_maximum_difference'] > 5e-4):
        raise ValueError('fresh anchor correspondence incomplete or repeated outputs differ')
    fresh_freeze = json.loads((fresh / 'frozen.json').read_text())
    for path, expected in fresh_freeze['artifacts_sha256'].items():
        if sha256(path) != expected:
            raise ValueError('fresh development comparison changed')
    metrics = pd.read_csv(fresh / 'metrics.csv')
    old = metrics.loc[metrics.model.eq('actual_anchor')].set_index('channel').ADS
    new = metrics.loc[metrics.model.eq('v73')].set_index('channel').ADS
    if (len(old) != 6 or not old.index.equals(new.index)
            or new['all'] - old['all'] < .01
            or (new.drop('all') - old.drop('all') < -1e-12).any()):
        raise ValueError('fixed v73 does not clear fresh full/each-channel development comparison')
    smoke_freeze = json.loads((smoke / 'frozen.json').read_text())
    completion = json.loads((smoke / 'completed.json').read_text())
    if (completion['frozen_sha256'] != sha256(smoke / 'frozen.json')
            or completion['predictions_sha256'] != sha256(smoke / 'output/submission.csv')):
        raise ValueError('anchor smoke artifacts changed')
    attestation = smoke_freeze['package_attestation']
    verify_inventory(attestation)
    partition = ROOT / 'configs/data_partitions.yaml'
    roles = yaml.safe_load(partition.read_text())
    assert_development_eval_separation(partition)
    for role in TRAIN_ROLES:
        for manifest in roles.get(role, []):
            assert_no_locked_eval_leakage(ROOT / manifest, partition)
    artifacts = {str(path): sha256(path) for path in [Path(__file__),
        ROOT / 'scripts/run_exact_anchor_v74.py', ROOT / 'scripts/score_long_voice_v74.py',
        ROOT / 'docs/long-voice-v74-evaluation.md',
        ROOT / 'docs/long-voice-v61-protocol.md', partition, smoke / 'comparison.json',
        ROOT / 'reports/component_composition_v73/frozen.json',
        ROOT / 'reports/component_composition_v73/integrated_verified/report.json',
        ROOT / 'src/component_composition_inference_v73.py', ROOT / 'src/component_composition_v73.py',
        ROOT / 'src/common_encoder_probe.py', ROOT / 'src/common_encoder_probe_inference.py',
        ROOT / 'src/common_eat_adaptation_inference.py', ROOT / 'src/common_eat_adaptation.py',
        fresh / 'report.json', fresh / 'metrics.csv', fresh / 'frozen.json', fresh / 'predictions.csv']}
    candidate_freeze = ROOT / 'reports/component_composition_v73/frozen.json'
    candidate_config = json.loads(candidate_freeze.read_text())
    for source, expected in candidate_config['sources_sha256'].items():
        if sha256(source) != expected:
            raise ValueError('frozen candidate development attestation changed')
    all_rows, bank_records = [], {}
    for name, (directory, validation_name, count) in BANKS.items():
        bank = ROOT / 'data/eval' / directory
        truth = bank / 'truth.csv'
        memberships = {role for role, manifests in roles.items() if str(truth.relative_to(ROOT)) in manifests}
        if memberships != {'locked_eval'}:
            raise ValueError('bank must be exclusively protected locked_eval')
        validation_path = ROOT / 'reports/long_voice_v61' / validation_name
        validation = json.loads(validation_path.read_text())
        hashes_path = bank / 'audio_hashes.csv'
        if (validation['status'] != 'PASS_CONSTRUCTION_NOT_DETECTION'
                or sha256(truth) != validation['truth_sha256']
                or sha256(hashes_path) != validation['audio_hashes_sha256']
                or validation['source_identity_overlap'] != 0
                or validation['known_protected_source_hash_overlap'] != 0):
            raise ValueError('construction verification changed/failed')
        provenance = json.loads((bank / 'provenance.json').read_text())
        if provenance['stage'] != 'locked_unscored_integrity_verified' or not provenance['full_bank_complete']:
            raise ValueError('not an unexposed complete bank')
        ids = pd.read_csv(truth, usecols=['ID'], dtype=str).ID.tolist()
        hashes = pd.read_csv(hashes_path, dtype=str)
        if len(ids) != count or len(set(ids)) != count or hashes.ID.duplicated().any() or set(ids) != set(hashes.ID):
            raise ValueError('bank identity/count mismatch')
        expected = hashes.set_index('ID').SHA256.to_dict()
        for identity in ids:
            path = bank / 'audio' / (identity + '.flac')
            if sha256(path) != expected[identity]:
                raise ValueError('protected waveform changed')
            all_rows.append(dict(ID=identity, BANK=name, PATH=str(path), SHA256=expected[identity]))
        for path in [truth, hashes_path, validation_path, bank / 'provenance.json']:
            artifacts[str(path)] = sha256(path)
        bank_records[name] = dict(truth=str(truth), rows=count, source_groups=20)
    if len({r['ID'] for r in all_rows}) != len(all_rows):
        raise ValueError('cross-bank IDs must be unique')
    RUN.mkdir(parents=True)
    frozen = dict(schema='long_voice_v74_one_shot', banks=bank_records, inputs=all_rows,
        shards=4, candidate='equal_voice_pair_eat_music', candidate_frozen=str(candidate_freeze),
        package_attestation=attestation, artifacts_sha256=artifacts,
        gates=dict(uniform_min_pooled_file_eer_improvement=.01,
                   max_length_channel_regression=.025, fixed_max_pooled_regression=.025),
        no_retuning=True, automatic_submission_allowed=False)
    frozen_path = RUN / 'frozen.json'
    frozen_path.write_text(json.dumps(frozen, indent=2) + '\n')
    for model in ['anchor', 'candidate']:
        for shard in range(frozen['shards']):
            rows = all_rows[shard::frozen['shards']]
            stage = RUN / model / f'shard_{shard}'
            stage_inputs(stage, Path(attestation['package']), rows)
            stage_freeze = dict(package_attestation=attestation, inputs=rows,
                root_frozen_sha256=sha256(frozen_path), model=model, shard=shard,
                automatic_submission_allowed=False)
            (stage / 'frozen.json').write_text(json.dumps(stage_freeze, indent=2) + '\n')
    print(json.dumps(dict(status='prepared_not_scored', rows=len(all_rows), shards=4)), flush=True)


def frozen_run():
    path = RUN / 'frozen.json'
    frozen = json.loads(path.read_text())
    for artifact, expected in frozen['artifacts_sha256'].items():
        if sha256(artifact) != expected:
            raise ValueError(f'one-shot artifact changed: {artifact}')
    return frozen


def candidate_worker(shard):
    warnings.filterwarnings('ignore', category=FutureWarning)
    import numpy as np
    import torch
    from src.component_composition_inference_v73 import ComponentCompositionPredictor
    from src.pipeline import load_audio
    frozen = frozen_run()
    stage = RUN / 'candidate' / f'shard_{shard}'
    stage_file = stage / 'frozen.json'
    local = json.loads(stage_file.read_text())
    stage_hash = sha256(stage_file)
    if local['root_frozen_sha256'] != sha256(RUN / 'frozen.json'):
        raise ValueError('stage differs from frozen comparison')
    output = stage / 'output'
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    model = ComponentCompositionPredictor(frozen['candidate_frozen'], ROOT, frozen['candidate'])
    torch.cuda.reset_peak_memory_stats()
    predictions = []
    started = time.monotonic()
    for index, row in enumerate(local['inputs']):
        path = Path(row['PATH'])
        if sha256(path) != row['SHA256']:
            raise ValueError('waveform changed')
        predictions.append(model(load_audio(path)))
        if (index + 1) % 50 == 0:
            print(json.dumps(dict(shard=shard, files=index + 1, seconds=time.monotonic() - started)), flush=True)
    for index in [len(predictions) - 1, len(predictions) // 2, 0]:
        np.testing.assert_array_equal(predictions[index], model(load_audio(Path(local['inputs'][index]['PATH']))))
    frozen_run()
    if sha256(stage_file) != stage_hash:
        raise ValueError('stage changed while running')
    with (output / 'submission.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['ID', *COLUMNS])
        writer.writeheader()
        for row, values in zip(local['inputs'], predictions):
            writer.writerow(dict(ID=row['ID'], **dict(zip(COLUMNS, values))))
    report = dict(status='complete', frozen_sha256=stage_hash,
        predictions_sha256=sha256(output / 'submission.csv'), rows=len(predictions),
        repeat_bit_exact=True, repeat_files=3, peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
        seconds=time.monotonic() - started, automatic_submission_allowed=False)
    (stage / 'completed.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


def launch(model, gpu):
    frozen = frozen_run()
    if ',' in gpu or not gpu.isdigit():
        raise ValueError('one numeric GPU ID required')
    status = subprocess.check_output(['nvidia-smi', '-i', gpu,
        '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
    free, utilization = map(int, status.split(','))
    if free < 96 * 1024 or utilization > 5:
        raise ValueError('choose an idle GPU with >=96GiB free; do not disturb other work')
    marker = RUN / f'launch_{model}.json'
    with marker.open('x') as stream:
        json.dump(dict(gpu=gpu, frozen_sha256=sha256(RUN / 'frozen.json'), status='starting'), stream)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', PYTHONWARNINGS='ignore::FutureWarning',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    processes, logs = [], []
    try:
        for shard in range(frozen['shards']):
            stage = RUN / model / f'shard_{shard}'
            log = (stage / 'run.log').open('x')
            logs.append(log)
            if model == 'anchor':
                command = [sys.executable, str(ROOT / 'scripts/run_exact_anchor_v74.py'), 'worker', '--stage', str(stage)]
            else:
                command = [sys.executable, str(Path(__file__).resolve()), 'candidate-worker', '--shard', str(shard)]
            process = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            processes.append(process)
            print(json.dumps(dict(model=model, shard=shard, pid=process.pid, gpu=gpu)), flush=True)
        while any(p.poll() is None for p in processes):
            failed = [(p.pid, p.returncode) for p in processes if p.returncode not in {None, 0}]
            if failed:
                raise RuntimeError(f'failed workers: {failed}; logs retained, no automatic restart')
            time.sleep(5)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError('worker failed')
        print(json.dumps(dict(status='all_workers_completed', model=model)), flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='mode', required=True)
    subs.add_parser('prepare')
    run = subs.add_parser('launch')
    run.add_argument('--model', choices=['anchor', 'candidate'], required=True)
    run.add_argument('--gpu', required=True)
    worker = subs.add_parser('candidate-worker')
    worker.add_argument('--shard', type=int, choices=range(4), required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'launch':
        launch(args.model, args.gpu)
    else:
        candidate_worker(args.shard)


if __name__ == '__main__':
    main()
