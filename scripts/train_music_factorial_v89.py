"""Matched EAT Music adaptation on bank-balanced 2x2 factorial scenes."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
CONFIG = dict(seed=20260908, epochs=4, draws_per_epoch=768, replay_probability=.5,
              rank_weight=.2, invariance_weight=.02, adapter_lr=1e-4, head_lr=1e-4,
              weight_decay=.01, temperature=2., selection='fixed_final_epoch',
              parent_checkpoint='reports/common_eat_adaptation/full/adapted/head.pt',
              source_catalog='reports/dense_component_v71/source_catalog/sources.csv',
              partition_config='configs/data_partitions.yaml')


def main():
    import numpy as np
    import pandas as pd
    import torch
    from src.common_eat_adaptation_inference import CommonEatAdaptationPredictor
    from src.common_eat_adaptation import paired_file_logits
    from src.common_encoder_probe import complete_windows
    from src.music_factorial_data_v89 import MusicFactorialQuadruplets
    from src.pipeline import load_audio
    from src.full_coverage_wpt import resolve_ffmpeg
    from src.evaluate_diagnostic import official_eer
    from train_common_encoder_probe import audit_protected_sources
    from train_dense_component_v71 import audited_catalog
    from train_paired_wpt_file_v60 import balanced_weights, sha256
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-report', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    inventory_path = ROOT / 'reports/training_inventory_v78/report.json'
    inventory = json.loads(inventory_path.read_text())
    assert inventory['status'] == 'complete_metadata_cache'
    assert all(sha256(p) == digest for p, digest in inventory['artifacts_sha256'].items())
    frames = {}
    for name, record in inventory['tables'].items():
        assert sha256(record['path']) == record['sha256']
        frames[name] = pd.read_csv(record['path'], low_memory=False)
    train, dev = frames['train'], frames['development']
    audit = audit_protected_sources(train, ROOT / CONFIG['partition_config'])
    catalog = audited_catalog(CONFIG, train)
    replay = train[train.MUSIC_PRESENT.eq(1) & train.MUSIC_FAKE.isin([0, 1])].reset_index(drop=True)
    replay_weights = np.asarray(balanced_weights(replay), np.float64)
    replay_weights /= replay_weights.sum()
    ffmpeg = resolve_ffmpeg()
    artifacts = [Path(__file__), inventory_path, ROOT / CONFIG['parent_checkpoint'],
        ROOT / 'reports/common_eat_adaptation/full/completed.json', ROOT / CONFIG['partition_config'],
        ROOT / CONFIG['source_catalog'], ffmpeg,
        *[ROOT / 'src' / n for n in ['music_factorial_data_v89.py', 'music_bank_sampler_v88.py',
          'music_counterfactual_data_v87.py', 'dense_component_data_v71.py', 'common_eat_adaptation.py',
          'common_eat_adaptation_inference.py', 'common_encoder_probe.py', 'eat_large_aasist_inference.py',
          'eat_timm_compat.py', 'eat_music_adapter.py', 'pipeline.py', 'telephone_channel.py',
          'long_component_stress.py']],
        *[ROOT / 'scripts' / n for n in ['train_common_encoder_probe.py', 'train_dense_component_v71.py',
                                         'train_paired_wpt_file_v60.py']],
        *[p for p in (ROOT / 'models/eat-large-as2m-v56').iterdir() if p.suffix in ['.py', '.json', '.safetensors']]]
    hashes = {str(p): sha256(p) for p in artifacts}
    if not args.smoke:
        if args.smoke_report is None:
            raise ValueError('matching completed smoke required')
        prior = json.loads(args.smoke_report.read_text())
        frozen = json.loads((args.smoke_report.parent / 'frozen.json').read_text())
        assert prior['status'] == 'complete_smoke' and frozen['config'] == CONFIG
        assert frozen['artifacts_sha256'] == hashes
    args.output.mkdir(parents=True)
    (args.output / 'frozen.json').write_text(json.dumps(dict(config=CONFIG, artifacts_sha256=hashes,
        source_audit=audit, smoke=args.smoke, train_rows=len(train), replay_rows=len(replay),
        scope='Music only; TRAIN only; fixed final epoch; no automatic submission'), indent=2))
    torch.set_num_threads(2)
    torch.manual_seed(CONFIG['seed'])
    names = ['factorial_bce', 'factorial_ranked']
    models = {name: CommonEatAdaptationPredictor(ROOT / CONFIG['parent_checkpoint'], ROOT) for name in names}
    for key, value in models[names[0]].head.state_dict().items():
        torch.testing.assert_close(value, models[names[1]].head.state_dict()[key], rtol=0, atol=0)
    for key, value in models[names[0]].encoder.adapters.state_dict().items():
        torch.testing.assert_close(value, models[names[1]].encoder.adapters.state_dict()[key], rtol=0, atol=0)
    initial, opts, parameters = {}, {}, {}
    for name, model in models.items():
        model.encoder.eval().requires_grad_(False)
        model.encoder.adapters.requires_grad_(True)
        model.head.eval().requires_grad_(True)
        initial[name] = {k: v.detach().cpu().clone() for k, v in model.encoder.adapters.state_dict().items()}
        parameters[name] = list(model.encoder.adapters.parameters()) + list(model.head.parameters())
        opts[name] = torch.optim.AdamW([dict(params=model.encoder.adapters.parameters(), lr=CONFIG['adapter_lr']),
            dict(params=model.head.parameters(), lr=CONFIG['head_lr'])], weight_decay=CONFIG['weight_decay'])

    def music_logits(model, audios):
        outputs = []
        for audio in audios:
            windows, lengths, _ = complete_windows(audio)
            logits = paired_file_logits(model.encoder, {'adapted': model.head}, torch.from_numpy(windows),
                torch.from_numpy(lengths), [len(windows)], chunk_size=1,
                temperature=CONFIG['temperature'])['adapted']
            outputs.append(logits[0, 2])
        return torch.stack(outputs)

    sampler = MusicFactorialQuadruplets(catalog, CONFIG['seed'], ffmpeg)
    started = time.monotonic()
    history = []
    for epoch in range(1, (1 if args.smoke else CONFIG['epochs']) + 1):
        losses = {name: [] for name in names}
        parts = {name: [] for name in names}
        with (args.output / f'epoch_{epoch}_traces.jsonl').open('w') as stream:
            for index in range(4 if args.smoke else CONFIG['draws_per_epoch']):
                rng = np.random.default_rng(np.random.SeedSequence([CONFIG['seed'], epoch, index, 990]))
                use_replay = index % 2 == 0 if args.smoke else rng.random() < CONFIG['replay_probability']
                if use_replay:
                    row_index = int(rng.choice(len(replay), p=replay_weights))
                    row = replay.iloc[row_index]
                    path = Path(row.PATH)
                    digest = sha256(path)
                    audios, target = [load_audio(path)], np.asarray([row.MUSIC_FAKE], np.float32)
                    assert sha256(path) == digest
                    trace = dict(kind='original_TRAIN_music_replay', id=row.ID, dataset=row.DATASET,
                                 sha256=digest, music_label=int(row.MUSIC_FAKE), epoch=epoch, draw=index)
                else:
                    audios, target, trace = sampler.draw_factorial(epoch, index)
                target_tensor = None
                for name, model in models.items():
                    logits = music_logits(model, audios)
                    target_tensor = torch.as_tensor(target, device=logits.device)
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target_tensor)
                    rank = logits.new_zeros(())
                    invariant = logits.new_zeros(())
                    if len(logits) == 4:
                        rank = torch.nn.functional.softplus(-(logits[2:] - logits[:2])).mean()
                        invariant = ((logits[0]-logits[1]).square() + (logits[2]-logits[3]).square()) / 2
                    loss = bce
                    if name == 'factorial_ranked':
                        loss = loss + CONFIG['rank_weight']*rank + CONFIG['invariance_weight']*invariant
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite loss')
                    opts[name].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters[name], 5., error_if_nonfinite=True)
                    assert all(p.grad is None for p in model.encoder.base.parameters())
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.encoder.adapters.parameters())
                    opts[name].step()
                    losses[name].append(float(loss.detach()))
                    parts[name].append(dict(bce=float(bce.detach()), rank=float(rank.detach()),
                                            invariant=float(invariant.detach())))
                stream.write(json.dumps(trace)+'\n')
                stream.flush()
                if index % 100 == 0:
                    print(json.dumps(dict(epoch=epoch, draw=index, seconds=time.monotonic()-started)), flush=True)
        history.append(dict(epoch=epoch, losses={n: float(np.mean(v)) for n, v in losses.items()},
            loss_parts={n: {k: float(np.mean([x[k] for x in v])) for k in ['bce', 'rank', 'invariant']}
                        for n, v in parts.items()}))
    for name, model in models.items():
        assert any(not torch.equal(value.cpu(), initial[name][key])
                   for key, value in model.encoder.adapters.state_dict().items())
        torch.save(dict(model_type='music_factorial_v89', config=CONFIG, variant=name, smoke=args.smoke,
            state_dict=model.head.state_dict(), adapters=model.encoder.adapters.state_dict(),
            parent_checkpoint_sha256=hashes[str(ROOT / CONFIG['parent_checkpoint'])]), args.output / f'{name}.pt')
    results = []
    if not args.smoke:
        predictions = {name: [] for name in names}
        with torch.no_grad():
            for index, row in dev.iterrows():
                audio = load_audio(Path(row.PATH))
                for name, model in models.items():
                    predictions[name].append(float(music_logits(model, [audio]).sigmoid()[0]))
                if index % 300 == 0:
                    print(json.dumps(dict(stage='development', files=index)), flush=True)
        for name, scores in predictions.items():
            frame = dev.copy()
            frame['MUSIC_FAKE_PROB'] = scores
            frame.to_csv(args.output / f'{name}_development.csv', index=False)
            music = frame[frame.MUSIC_PRESENT.eq(1)]
            results.append(dict(model=name, MUSIC_EER=official_eer(music.MUSIC_FAKE, music.MUSIC_FAKE_PROB)))
    assert all(sha256(p) == digest for p, digest in hashes.items())
    result = dict(status='complete_smoke' if args.smoke else 'complete', history=history, results=results,
        seconds=time.monotonic()-started, peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
        protected_evaluation_consumed=False, automatic_submission_allowed=False)
    (args.output / 'report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
