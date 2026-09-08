#!/usr/bin/env python3
"""Paired voice-presence readout continuation on audited TRAIN-only features.

Four processes share one explicitly selected research GPU. The existing encoder,
adapters, projection, pooling and four other task outputs stay frozen. Never
updates the v73/v74 candidate or reads protected evaluation audio.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def verify_frozen(output):
    from run_exact_anchor_v74 import sha256
    frozen = json.loads((output / 'frozen.json').read_text())
    for path, digest in frozen['artifacts_sha256'].items():
        if sha256(path) != digest:
            raise ValueError(f'frozen artifact changed: {path}')
    return frozen


def worker(output, rank):
    import numpy as np
    import pandas as pd
    import torch
    from src.common_eat_adaptation_inference import CommonEatAdaptationPredictor
    from src.common_encoder_probe import complete_windows
    from src.full_coverage_wpt import masked_lme
    from src.pipeline import load_audio
    from src.presence_repair_v76 import head_statistics
    from run_exact_anchor_v74 import sha256
    frozen = verify_frozen(output)
    config = frozen['config']
    torch.set_num_threads(config['threads_per_worker'])
    torch.set_num_interop_threads(1)
    directory = output / f'worker_{rank}'
    directory.mkdir(exist_ok=False)
    frame = pd.read_csv(output / 'cache_inputs.csv', dtype={'DATASET': str, 'ID': str})
    frame = frame.iloc[rank::config['workers']]
    model = CommonEatAdaptationPredictor(ROOT / config['parent_checkpoint'], ROOT)
    model.encoder.requires_grad_(False)
    model.head.requires_grad_(False)
    statistics, counts, probabilities, indices, hashes = [], [], [], [], []
    started = time.monotonic()
    with torch.no_grad():
        for position, row in enumerate(frame.itertuples(index=False)):
            digest = sha256(row.PATH)
            windows, lengths, _ = complete_windows(load_audio(Path(row.PATH)))
            chunks, native = [], []
            size = model.config['encoder_chunk']
            for start in range(0, len(windows), size):
                tokens, mask = model.encoder(torch.from_numpy(windows[start:start + size]),
                    torch.from_numpy(lengths[start:start + size]))
                values = head_statistics(model.head, tokens['adapted'], mask)
                reconstructed = (values * model.head.output_weight).sum(-1) + model.head.output_bias
                reference = model.head(tokens['adapted'], mask)
                torch.testing.assert_close(reconstructed, reference, rtol=0, atol=0)
                chunks.append(values[:, 3].cpu().numpy())
                native.append(reference)
            scores = torch.cat(native)[None]
            valid = torch.ones(scores.shape[:2], dtype=torch.bool, device=scores.device)
            prob = masked_lme(scores, valid, config['temperature']).sigmoid()[0]
            if sha256(row.PATH) != digest:
                raise ValueError('audio changed during feature extraction')
            statistics.extend(chunks)
            counts.append(len(windows))
            probabilities.append(prob.cpu().numpy())
            indices.append(row.CACHE_INDEX)
            hashes.append(digest)
            if (position + 1) % 200 == 0:
                print(json.dumps(dict(rank=rank, files=position + 1, total=len(frame),
                    seconds=time.monotonic() - started)), flush=True)
    verify_frozen(output)
    path = directory / 'features.npz'
    np.savez_compressed(path, statistics=np.concatenate(statistics), counts=np.asarray(counts),
        probabilities=np.asarray(probabilities), indices=np.asarray(indices), audio_sha256=np.asarray(hashes))
    dump(directory / 'completed.json', dict(status='complete', rows=len(frame),
        seconds=time.monotonic() - started, features_sha256=sha256(path),
        native_head_reconstruction_bit_exact=True,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20))


def prepare(output, config_path, smoke):
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from src.presence_repair_v76 import training_review_mask
    from train_common_encoder_probe import audit_protected_sources
    from train_paired_wpt_file_v60 import load_data, balanced_weights, sha256
    config = yaml.safe_load(config_path.read_text())
    if output.exists():
        raise FileExistsError(output)
    train, dev, audit = load_data(config)
    train, dev = train.reset_index(drop=True), dev.reset_index(drop=True)
    audit['all_protected_roles'] = audit_protected_sources(train, ROOT / config['partition_config'])
    review = pd.read_csv(ROOT / config['review'], dtype={'DATASET': str, 'ID': str})
    excluded = training_review_mask(train, review)
    if len(train) != 18738 or len(dev) != 4577 or len(review) != 600 or excluded.sum() != 44:
        raise ValueError('population/review changed; require explicit re-audit')
    if smoke:
        chosen = np.r_[np.flatnonzero(excluded)[:2], np.flatnonzero(train.VOICE_PRESENT.eq(0) & ~excluded)[:2],
                       np.flatnonzero(train.VOICE_PRESENT.eq(1))[:4]]
        train = train.iloc[chosen].reset_index(drop=True)
        excluded = excluded[chosen]
        dev = dev.head(8).copy()
        config.update(epochs=1, samples_per_epoch=8, batch_size=4, workers=2)
    generator = torch.Generator().manual_seed(config['seed'])
    weights = balanced_weights(train)
    draws = np.stack([torch.multinomial(weights, config['samples_per_epoch'], replacement=True,
        generator=generator).numpy() for _ in range(config['epochs'])])
    used = np.unique(draws)
    # Feature extraction need only cover the predeclared TRAIN draws. The full
    # TRAIN population and all exclusion flags remain recorded for provenance.
    cache_train = train.iloc[used].copy()
    cache_train['ROLE'], cache_train['ORIGINAL_INDEX'] = 'train', used
    cache_dev = dev.copy()
    cache_dev['ROLE'], cache_dev['ORIGINAL_INDEX'] = 'development', np.arange(len(dev))
    cache = pd.concat([cache_train, cache_dev], ignore_index=True)
    cache['CACHE_INDEX'] = np.arange(len(cache))
    output.mkdir(parents=True)
    train.assign(EXCLUDE_VOICE_PRESENCE=excluded).to_csv(output / 'train.csv', index=False)
    dev.to_csv(output / 'development.csv', index=False)
    cache[['CACHE_INDEX', 'ROLE', 'ORIGINAL_INDEX', 'DATASET', 'ID', 'PATH']].to_csv(output / 'cache_inputs.csv', index=False)
    np.save(output / 'draws.npy', draws)
    paths = [Path(__file__), config_path, ROOT / 'src/presence_repair_v76.py',
        ROOT / 'src/common_encoder_probe.py', ROOT / 'src/common_eat_adaptation.py',
        ROOT / 'src/common_eat_adaptation_inference.py', ROOT / config['parent_checkpoint'],
        ROOT / 'src/full_coverage_wpt.py', ROOT / 'src/pipeline.py',
        ROOT / 'src/eat_large_aasist_inference.py',
        ROOT / 'docs/presence-repair-v76-protocol.md',
        ROOT / config['review'], ROOT / config['partition_config'],
        ROOT / 'scripts/train_common_encoder_probe.py', ROOT / 'scripts/train_paired_wpt_file_v60.py',
        ROOT / 'reports/component_composition_v73/equal_voice_pair_eat_music_predictions.csv',
        output / 'train.csv', output / 'development.csv', output / 'cache_inputs.csv', output / 'draws.npy']
    parent = torch.load(ROOT / config['parent_checkpoint'], map_location='cpu', weights_only=False)
    encoder_directory = ROOT / parent['config']['encoder_path']
    paths.extend(encoder_directory / name for name in ['model.safetensors', 'config.json',
        'configuration_eat.py', 'eat_model.py', 'model_core.py', 'modeling_eat.py'])
    hashes = {str(p.resolve()): sha256(p) for p in paths}
    for record in audit['manifests']:
        hashes[record['path']] = record['sha256']
    dump(output / 'frozen.json', dict(config=config, smoke=smoke, audit=audit,
        train_rows=len(train), development_rows=len(dev), cached_train_rows=len(used),
        excluded_train_rows=int(excluded.sum()), excluded_training_draws=int(excluded[draws].sum()),
        artifacts_sha256=hashes, automatic_submission_allowed=False,
        selection='final epoch fixed in advance; no protected evaluation consumption',
        intervention='mask reviewed negative presence loss ONLY; do not relabel'))
    print(json.dumps(dict(prepared=True, train=len(train), cached_train=len(used), development=len(dev),
        masked_draws=int(excluded[draws].sum()), smoke=smoke)), flush=True)


def train_cached(output):
    import numpy as np
    import pandas as pd
    import torch
    from src.full_coverage_wpt import masked_lme
    from src.presence_repair_v76 import presence_loss
    from src.evaluate_diagnostic import PREDICTION_COLUMNS
    from train_common_encoder_probe import metrics
    from run_exact_anchor_v74 import sha256
    frozen = verify_frozen(output)
    config = frozen['config']
    torch.set_num_threads(2)
    torch.manual_seed(config['seed'])
    cache = pd.read_csv(output / 'cache_inputs.csv')
    features, probabilities = {}, {}
    hashes = {}
    for rank in range(config['workers']):
        directory = output / f'worker_{rank}'
        report = json.loads((directory / 'completed.json').read_text())
        path = directory / 'features.npz'
        if report['status'] != 'complete' or sha256(path) != report['features_sha256']:
            raise ValueError('incomplete/changed feature shard')
        hashes[str(path)] = report['features_sha256']
        with np.load(path, allow_pickle=False) as part:
            offset = 0
            for index, count, prob in zip(part['indices'], part['counts'], part['probabilities']):
                if int(index) in features or count < 1:
                    raise ValueError('duplicate/empty feature bag')
                features[int(index)] = part['statistics'][offset:offset + count].copy()
                probabilities[int(index)] = prob.copy()
                offset += count
            if offset != len(part['statistics']):
                raise ValueError('unassigned cached windows')
    if set(features) != set(range(len(cache))):
        raise ValueError('incomplete feature population')
    maximum, dimension = max(map(len, features.values())), features[0].shape[1]
    array = np.zeros((len(cache), maximum, dimension), dtype=np.float32)
    valid = np.zeros((len(cache), maximum), dtype=bool)
    for index, value in features.items():
        array[index, :len(value)] = value
        valid[index, :len(value)] = True
    x, valid = torch.from_numpy(array).cuda(), torch.from_numpy(valid).cuda()
    parent = torch.load(ROOT / config['parent_checkpoint'], map_location='cpu', weights_only=False)
    initial_weight = parent['state_dict']['output_weight'][3].clone().cuda()
    initial_bias = parent['state_dict']['output_bias'][3].clone().cuda()
    train = pd.read_csv(output / 'train.csv')
    draws = np.load(output / 'draws.npy')
    mapping = dict(zip(cache.loc[cache.ROLE.eq('train'), 'ORIGINAL_INDEX'], cache.loc[cache.ROLE.eq('train'), 'CACHE_INDEX']))
    target = torch.tensor(train.VOICE_PRESENT.to_numpy(float), dtype=torch.float32, device='cuda')
    exclusion = torch.tensor(train.EXCLUDE_VOICE_PRESENCE.to_numpy(bool), device='cuda')
    heads = {name: [torch.nn.Parameter(initial_weight.clone()), torch.nn.Parameter(initial_bias.clone())]
             for name in config['variants']}
    optimizers = {name: torch.optim.AdamW(params, lr=config['learning_rate'], weight_decay=config['weight_decay'])
                  for name, params in heads.items()}

    def forward(indices, weight, bias):
        logits = (x[indices] * weight).sum(-1) + bias
        return masked_lme(logits[..., None], valid[indices], config['temperature'])[:, 0]

    with torch.no_grad():
        initial = forward(torch.arange(len(cache), device='cuda'), initial_weight, initial_bias).sigmoid().cpu().numpy()
    native = np.stack([probabilities[i] for i in range(len(cache))])
    difference = float(np.max(np.abs(initial - native[:, 3])))
    if difference > 1e-6:
        raise ValueError(f'cached initial readout mismatch: {difference}')
    history, started = [], time.monotonic()
    for epoch, order in enumerate(draws, 1):
        losses = {name: [] for name in heads}
        for start in range(0, len(order), config['batch_size']):
            original = order[start:start + config['batch_size']]
            index = torch.tensor([mapping[int(i)] for i in original], device='cuda')
            labels, mask = target[original], exclusion[original]
            for name, (weight, bias) in heads.items():
                omitted = mask if name == 'mask_reviewed_negatives' else torch.zeros_like(mask)
                optimizers[name].zero_grad()
                loss = presence_loss(forward(index, weight, bias), labels, omitted)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite training loss')
                # Avoid AdamW weight decay on a completely unsupervised batch.
                if not omitted.all():
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([weight, bias], 5., error_if_nonfinite=True)
                    optimizers[name].step()
                losses[name].append(float(loss.detach()))
        row = dict(epoch=epoch, losses={k: float(np.mean(v)) for k, v in losses.items()})
        history.append(row)
        print(json.dumps(row), flush=True)
    dump(output / 'training_history.json', history)
    # Commit both final-epoch weights BEFORE opening any development scores.
    for name, params in heads.items():
        torch.save(dict(model_type='presence_repair_v76', voice_weight=params[0].detach().cpu(),
            voice_bias=params[1].detach().cpu(), parent_checkpoint=config['parent_checkpoint'],
            config=config, variant=name, smoke=frozen['smoke']), output / f'{name}.pt')
    dev = pd.read_csv(output / 'development.csv', dtype={'DATASET': str, 'ID': str})
    dev_indices = cache.loc[cache.ROLE.eq('development'), 'CACHE_INDEX'].to_numpy()
    base = dev.copy()
    base[PREDICTION_COLUMNS] = native[dev_indices]
    composed = pd.read_csv(ROOT / 'reports/component_composition_v73/equal_voice_pair_eat_music_predictions.csv',
        dtype={'DATASET': str, 'ID': str}).set_index(['DATASET', 'ID'])
    composed = composed.loc[pd.MultiIndex.from_frame(dev[['DATASET', 'ID']])].reset_index()
    consistency = float(np.max(np.abs(composed[PREDICTION_COLUMNS[2:]].to_numpy() - native[dev_indices, 2:])))
    if consistency > 5e-4:
        raise ValueError(f'cached v73 EAT correspondence failed: {consistency}')
    results = []
    all_heads = {'unchanged': [initial_weight, initial_bias], **heads}
    with torch.no_grad():
        for name, params in all_heads.items():
            voice = forward(torch.tensor(dev_indices, device='cuda'), *params).sigmoid().cpu().numpy()
            for pipeline in ['eat_parent', 'v73_composition']:
                frame = (base if pipeline == 'eat_parent' else composed).copy()
                original = frame[PREDICTION_COLUMNS].to_numpy().copy()
                frame['VOICE_PRESENT_PROB'] = voice
                if pipeline == 'v73_composition':
                    frame['FILE_FAKE_PROB'] = 1 - (1 - frame.VOICE_FAKE_PROB * voice) * (
                        1 - frame.MUSIC_FAKE_PROB * frame.MUSIC_PRESENT_PROB)
                fixed = [0, 1, 2, 4] if pipeline == 'eat_parent' else [1, 2, 4]
                if not np.array_equal(frame[PREDICTION_COLUMNS].to_numpy()[:, fixed], original[:, fixed]):
                    raise ValueError('unrelated output changed')
                if frozen['smoke']:
                    summary = {'smoke_only': True}
                else:
                    summary, slices = metrics(frame)
                    slices.to_csv(output / f'{pipeline}_{name}_slices.csv', index=False)
                frame.to_csv(output / f'{pipeline}_{name}_predictions.csv', index=False)
                results.append(dict(pipeline=pipeline, variant=name, **summary))
    verify_frozen(output)
    report = dict(status='complete', smoke=frozen['smoke'], results=results,
        cache_initial_maximum_difference=difference, v73_correspondence_maximum_difference=consistency,
        features_sha256=hashes, trainable_parameters_per_variant=dimension + 1,
        training_seconds=time.monotonic() - started,
        final_epoch_only=True, automatic_submission_allowed=False, protected_evaluation_used=False,
        frozen_sha256=sha256(output / 'frozen.json'))
    dump(output / 'report.json', report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/presence_repair_v76.yaml')
    parser.add_argument('--worker', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.worker is not None:
        worker(args.output, args.worker)
        return
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not gpu or ',' in gpu or gpu == '-1':
        raise ValueError('select one idle research GPU explicitly')
    prepare(args.output, args.config.resolve(), args.smoke)
    frozen = verify_frozen(args.output)
    processes, logs = [], []
    try:
        for rank in range(frozen['config']['workers']):
            log = (args.output / f'worker_{rank}.log').open('x')
            logs.append(log)
            processes.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                '--output', str(args.output), '--worker', str(rank)], stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            bad = [p for p in processes if p.poll() not in (None, 0)]
            if bad:
                raise RuntimeError(f'feature worker failed: {[(p.pid, p.returncode) for p in bad]}')
            time.sleep(2)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError('feature extraction failed')
        train_cached(args.output)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
