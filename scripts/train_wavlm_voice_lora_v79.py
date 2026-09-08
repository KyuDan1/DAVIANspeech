#!/usr/bin/env python3
"""Paired Voice-only WavLM head continuation vs last-four-block LoRA.

Uses only the source-audited TRAIN metadata cache. Evaluates development once
after both final-epoch weights have been saved. No Suno/locked evaluation.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]


def main():
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    import yaml
    from sklearn.metrics import roc_curve
    from src.common_encoder_probe import CommonTokenHead
    from src.common_eat_adaptation import paired_file_logits
    from src.wavlm_voice_lora_v79 import PairedWavLMVoice, known_voice_targets
    from src.lossless_weights_v77 import sha256
    from train_common_encoder_probe import FileBags, collate
    from train_paired_wpt_file_v60 import balanced_weights
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/wavlm_voice_lora_v79.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = yaml.safe_load(args.config.read_text())
    inventory_path = ROOT / config['inventory']
    inventory = json.loads(inventory_path.read_text())
    if inventory['status'] != 'complete_metadata_cache' or not inventory['source_manifests_equal_completed_parent']:
        raise ValueError('complete, source-matched metadata inventory required')
    for path, digest in inventory['artifacts_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('inventory provenance changed')
    tables = {}
    for name, record in inventory['tables'].items():
        if sha256(record['path']) != record['sha256']:
            raise ValueError('inventory table changed')
        tables[name] = pd.read_csv(record['path'], dtype={'DATASET': str, 'ID': str}, low_memory=False)
    full_train = tables['train']
    train = full_train.loc[known_voice_targets(full_train)].reset_index(drop=True)
    dev = tables['development'].loc[tables['development'].VOICE_PRESENT.eq(1)].reset_index(drop=True)
    if train.VOICE_FAKE.nunique() != 2 or dev.VOICE_FAKE.nunique() != 2:
        raise ValueError('both Voice classes required')
    if not train.VOICE_PRESENT.eq(1).all() or not train.VOICE_FAKE.isin([0, 1]).all():
        raise ValueError('invalid Voice training labels')
    paths = [Path(__file__), args.config, ROOT / 'src/wavlm_voice_lora_v79.py',
        ROOT / 'src/common_encoder_probe.py', ROOT / 'src/common_eat_adaptation.py',
        ROOT / 'scripts/train_common_encoder_probe.py', ROOT / 'scripts/train_paired_wpt_file_v60.py',
        ROOT / 'models/wavlm-large/pytorch_model.bin', ROOT / config['initial_head'], inventory_path,
        ROOT / 'reports/wavlm_voice_lora_v79/smoke/report.json']
    hashes = {str(p.resolve()): sha256(p) for p in paths}
    smoke = json.loads((ROOT / 'reports/wavlm_voice_lora_v79/smoke/report.json').read_text())
    if (smoke['status'] != 'complete_smoke_only' or not smoke['initial_control_adapted_bit_exact']
            or smoke['artifacts_sha256'][str(ROOT / 'src/wavlm_voice_lora_v79.py')] != sha256(ROOT / 'src/wavlm_voice_lora_v79.py')):
        raise ValueError('native gradient/initial-equivalence smoke incomplete or changed')
    args.output.mkdir(parents=True)
    train[['DATASET', 'ID']].to_csv(args.output / 'train_ids.csv', index=False)
    dev[['DATASET', 'ID']].to_csv(args.output / 'development_ids.csv', index=False)
    frozen = dict(config=config, train_rows=len(train), excluded_train_rows=len(full_train)-len(train),
        development_voice_rows=len(dev), train_voice_classes=train.VOICE_FAKE.value_counts().to_dict(),
        artifacts_sha256=hashes, selection='fixed final epoch, no early stopping',
        protected_evaluation_consumed=False, automatic_submission_allowed=False)
    (args.output / 'frozen.json').write_text(json.dumps(frozen, indent=2) + '\n')
    print(json.dumps(dict(stage='prepared', train=len(train), development_voice=len(dev),
        excluded=len(full_train)-len(train))), flush=True)
    torch.set_num_threads(2)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    model = PairedWavLMVoice(ROOT / config['encoder_path'], rank=config['rank'], last_blocks=config['last_blocks']).cuda()
    initial = torch.load(ROOT / config['initial_head'], map_location='cpu', weights_only=False)
    template = CommonTokenHead(model.base.config.hidden_size, width=config['head_width'], pooling=config['pooling']).cuda()
    template.load_state_dict(initial['state_dict'], strict=True)
    heads = {name: copy.deepcopy(template) for name in ['control', 'adapted']}
    lora_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizers = dict(control=torch.optim.AdamW(heads['control'].parameters(), lr=config['head_learning_rate'],
        weight_decay=config['weight_decay']), adapted=torch.optim.AdamW([
            dict(params=heads['adapted'].parameters(), lr=config['head_learning_rate']),
            dict(params=lora_parameters, lr=config['adapter_learning_rate'])], weight_decay=config['weight_decay']))
    sampler = WeightedRandomSampler(balanced_weights(train), config['samples_per_epoch'], replacement=True,
        generator=torch.Generator().manual_seed(config['seed']))
    kwargs = dict(batch_size=config['batch_size'], num_workers=config['workers'], collate_fn=collate,
        multiprocessing_context='spawn', persistent_workers=True)
    training = DataLoader(FileBags(train), sampler=sampler, **kwargs)
    development = DataLoader(FileBags(dev), shuffle=False, **kwargs)
    started, history = time.monotonic(), []
    for epoch in range(1, config['epochs']+1):
        model.train()
        for head in heads.values():
            head.train()
        losses, draws = {name: [] for name in heads}, []
        for step, (windows, lengths, counts, targets, indices) in enumerate(training):
            outputs = paired_file_logits(model, heads, windows, lengths, counts,
                config['encoder_chunk'], config['temperature'])
            if epoch == 1 and step == 0:
                torch.testing.assert_close(outputs['control'], outputs['adapted'], rtol=0, atol=0)
            for name, values in outputs.items():
                loss = torch.nn.functional.binary_cross_entropy_with_logits(values[:, 1], targets[:, 1].cuda())
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite Voice loss')
                optimizers[name].zero_grad()
                loss.backward()
                parameters = list(heads[name].parameters()) + (lora_parameters if name == 'adapted' else [])
                torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
                optimizers[name].step()
                losses[name].append(float(loss.detach()))
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise ValueError('frozen WavLM backbone accumulated gradients')
            draws.extend(indices)
            if step % 100 == 0:
                print(json.dumps(dict(epoch=epoch, step=step,
                    losses={k: float(np.mean(v)) for k, v in losses.items()}, seconds=time.monotonic()-started)), flush=True)
        np.save(args.output / f'epoch_{epoch}_draws.npy', np.asarray(draws))
        history.append(dict(epoch=epoch, losses={k: float(np.mean(v)) for k, v in losses.items()}))
        (args.output / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    # Save final paired checkpoints before looking at any development score.
    lora_state = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
    for name, head in heads.items():
        torch.save(dict(model_type='wavlm_voice_lora_v79', config=config, variant=name,
            state_dict=head.state_dict(), lora=lora_state if name=='adapted' else None,
            input_dimension=model.base.config.hidden_size, scope='VOICE_FAKE_PROB only'), args.output / f'{name}.pt')
    model.eval()
    for head in heads.values():
        head.eval()
    scores, indices = {name: [] for name in heads}, []
    with torch.no_grad():
        for windows, lengths, counts, _, index in development:
            outputs = paired_file_logits(model, heads, windows, lengths, counts,
                config['encoder_chunk'], config['temperature'])
            for name, values in outputs.items():
                scores[name].extend(values[:, 1].sigmoid().cpu().tolist())
            indices.extend(index)

    def eer(frame):
        if frame.VOICE_FAKE.nunique()!=2:
            return None
        fpr, tpr, _ = roc_curve(frame.VOICE_FAKE, frame.VOICE_FAKE_PROB, pos_label=1, drop_intermediate=False)
        fnr=1-tpr
        index=np.argmin(np.abs(fpr-fnr))
        return float((fpr[index]+fnr[index])/2)

    results=[]
    for name, values in scores.items():
        frame=dev.iloc[indices].copy()
        frame['VOICE_FAKE_PROB']=values
        frame['VOICE_BACKGROUND_LABEL_UNVERIFIED']=~known_voice_targets(frame)
        frame.to_csv(args.output / f'{name}_development.csv', index=False)
        slices=[]
        for axis in ['DATASET', 'CHANNEL_V57', 'LAYOUT_V57', 'VOICE_BACKGROUND_LABEL_UNVERIFIED']:
            for group, selected in frame.groupby(axis, dropna=False):
                slices.append(dict(axis=axis, group=str(group), rows=len(selected), VOICE_EER=eer(selected)))
        pd.DataFrame(slices).to_csv(args.output / f'{name}_slices.csv', index=False)
        results.append(dict(variant=name, rows=len(frame), VOICE_EER=eer(frame)))
    if any(sha256(path)!=digest for path,digest in hashes.items()):
        raise ValueError('frozen source changed during training')
    report=dict(status='complete', results=results, lora_parameters=model.trainable_count,
        seconds=time.monotonic()-started, peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
        frozen_sha256=sha256(args.output/'frozen.json'), final_epoch_only=True,
        protected_evaluation_consumed=False, automatic_submission_allowed=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
