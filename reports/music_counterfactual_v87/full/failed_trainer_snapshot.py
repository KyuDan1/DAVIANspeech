"""Matched EAT Music adaptation: BCE vs BCE + foreground-swap consistency."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
CONFIG = dict(seed=20260907, epochs=4, draws_per_epoch=1024, replay_probability=.5,
              consistency_weight=.1, adapter_lr=1e-4, head_lr=1e-4, weight_decay=.01,
              temperature=2., selection='fixed_final_epoch',
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
    from src.music_counterfactual_data_v87 import MusicCounterfactualPairs
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
    weights = np.asarray(balanced_weights(replay), np.float64)
    weights /= weights.sum()
    ffmpeg = resolve_ffmpeg()
    artifacts = [Path(__file__), inventory_path, ROOT / CONFIG['parent_checkpoint'],
                 ROOT / 'reports/common_eat_adaptation/full/completed.json',
                 ROOT / CONFIG['partition_config'], ROOT / CONFIG['source_catalog'], ffmpeg,
                 *[ROOT / 'src' / n for n in ['music_counterfactual_data_v87.py', 'dense_component_data_v71.py',
                    'common_eat_adaptation.py', 'common_eat_adaptation_inference.py', 'common_encoder_probe.py',
                    'eat_large_aasist_inference.py', 'eat_timm_compat.py', 'eat_music_adapter.py',
                    'pipeline.py', 'telephone_channel.py', 'long_component_stress.py']],
                 *[ROOT / 'scripts' / n for n in ['train_common_encoder_probe.py', 'train_dense_component_v71.py',
                                                 'train_paired_wpt_file_v60.py']],
                 *[p for p in (ROOT / 'models/eat-large-as2m-v56').iterdir() if p.suffix in ['.py', '.json', '.safetensors']]]
    hashes = {str(p): sha256(p) for p in artifacts}
    if not args.smoke:
        if args.smoke_report is None:
            raise ValueError('matching completed CUDA smoke required')
        assert json.loads(args.smoke_report.read_text())['status'] == 'complete_smoke'
        frozen = json.loads((args.smoke_report.parent / 'frozen.json').read_text())
        assert frozen['config'] == CONFIG and frozen['artifacts_sha256'] == hashes
    args.output.mkdir(parents=True)
    (args.output / 'frozen.json').write_text(json.dumps(dict(config=CONFIG, artifacts_sha256=hashes,
        source_audit=audit, smoke=args.smoke, train_rows=len(train), replay_rows=len(replay),
        scope='Music only; TRAIN only; fixed final epoch; no automatic submission'), indent=2))
    torch.set_num_threads(2)
    torch.manual_seed(CONFIG['seed'])
    models = {name: CommonEatAdaptationPredictor(ROOT / CONFIG['parent_checkpoint'], ROOT)
              for name in ['bce', 'invariant']}
    for key, value in models['bce'].head.state_dict().items():
        torch.testing.assert_close(value, models['invariant'].head.state_dict()[key], rtol=0, atol=0)
    for key, value in models['bce'].encoder.adapters.state_dict().items():
        torch.testing.assert_close(value, models['invariant'].encoder.adapters.state_dict()[key], rtol=0, atol=0)
    opts, parameters = {}, {}
    initial = {name: {k: v.detach().cpu().clone() for k, v in model.encoder.adapters.state_dict().items()}
               for name, model in models.items()}
    for name, model in models.items():
        model.encoder.eval().requires_grad_(False)
        model.encoder.adapters.requires_grad_(True)
        model.head.eval().requires_grad_(True)
        parameters[name] = list(model.head.parameters()) + list(model.encoder.adapters.parameters())
        opts[name] = torch.optim.AdamW([
            dict(params=model.encoder.adapters.parameters(), lr=CONFIG['adapter_lr']),
            dict(params=model.head.parameters(), lr=CONFIG['head_lr'])], weight_decay=CONFIG['weight_decay'])

    def logits(model, audios):
        values = []
        for audio in audios:
            windows, lengths, _ = complete_windows(audio)
            output = paired_file_logits(model.encoder, {'adapted': model.head},
                torch.from_numpy(windows), torch.from_numpy(lengths), [len(windows)],
                chunk_size=1, temperature=CONFIG['temperature'])['adapted']
            values.append(output[0, 2])
        return torch.stack(values)

    sampler = MusicCounterfactualPairs(catalog, CONFIG['seed'], ffmpeg)
    started = time.monotonic()
    history = []
    for epoch in range(1, (1 if args.smoke else CONFIG['epochs']) + 1):
        losses = {name: [] for name in models}
        with (args.output / f'epoch_{epoch}_traces.jsonl').open('w') as stream:
            for index in range(4 if args.smoke else CONFIG['draws_per_epoch']):
                rng = np.random.default_rng(np.random.SeedSequence([CONFIG['seed'], epoch, index, 99]))
                # Smoke explicitly covers both paths without changing full-run RNG.
                use_replay = index % 2 == 0 if args.smoke else rng.random() < CONFIG['replay_probability']
                if use_replay:
                    row_index = int(rng.choice(len(replay), p=weights))
                    row = replay.iloc[row_index]
                    path = Path(row.PATH)
                    digest = sha256(path)
                    audios = [load_audio(path)]
                    assert sha256(path) == digest
                    label = int(row.MUSIC_FAKE)
                    trace = dict(kind='original_TRAIN_music_replay', id=row.ID, dataset=row.DATASET,
                                 sha256=digest, music_label=label, epoch=epoch, draw=index)
                else:
                    audios, label, trace = sampler.draw(epoch, index)
                for name, model in models.items():
                    output = logits(model, audios)
                    target = torch.full_like(output, float(label))
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, target)
                    if name == 'invariant' and len(output) == 2:
                        loss = loss + CONFIG['consistency_weight'] * (output[0]-output[1]).square()
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite loss')
                    opts[name].zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters[name], 5., error_if_nonfinite=True)
                    assert all(p.grad is None for p in model.encoder.base.parameters())
                    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.adapters.parameters())
                    opts[name].step()
                    losses[name].append(float(loss.detach()))
                stream.write(json.dumps(trace) + '\n')
                stream.flush()
                if index % 100 == 0:
                    print(json.dumps(dict(epoch=epoch, draw=index, seconds=time.monotonic()-started)), flush=True)
        history.append(dict(epoch=epoch, losses={n: float(np.mean(v)) for n, v in losses.items()}))
    for name, model in models.items():
        assert any(not torch.equal(v.cpu(), initial[name][k]) for k, v in model.encoder.adapters.state_dict().items())
        torch.save(dict(model_type='music_counterfactual_v87', config=CONFIG, variant=name,
                        smoke=args.smoke, state_dict=model.head.state_dict(), adapters=model.encoder.adapters.state_dict(),
                        parent_checkpoint_sha256=hashes[str(ROOT / CONFIG['parent_checkpoint'])]), args.output / f'{name}.pt')
    results = []
    if not args.smoke:
        values = {name: [] for name in models}
        with torch.no_grad():
            for index, row in dev.iterrows():
                audio = load_audio(Path(row.PATH))
                for name, model in models.items():
                    values[name].append(float(logits(model, [audio]).sigmoid()[0]))
                if index % 300 == 0:
                    print(json.dumps(dict(stage='development', files=index)), flush=True)
        for name, scores in values.items():
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
