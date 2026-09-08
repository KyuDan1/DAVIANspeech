#!/usr/bin/env python3
"""Real TRAIN waveform/gradient smoke; no accuracy claims or protected scoring."""
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]


def main():
    import librosa
    import numpy as np
    import pandas as pd
    import torch
    from src.common_encoder_probe import CommonTokenHead, complete_windows
    from src.common_eat_adaptation import paired_file_logits
    from src.wavlm_voice_lora_v79 import PairedWavLMVoice, known_voice_targets
    from src.lossless_weights_v77 import sha256
    directory = ROOT / 'reports/wavlm_voice_lora_v79/smoke'
    if directory.exists():
        raise FileExistsError(directory)
    inventory_path = ROOT / 'reports/training_inventory_v78/report.json'
    inventory = json.loads(inventory_path.read_text())
    if inventory['status'] != 'complete_metadata_cache' or not inventory['train_development_order_exact']:
        raise ValueError('completed exact TRAIN inventory required')
    for path, digest in inventory['artifacts_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('training inventory source changed')
    source = Path(inventory['tables']['train']['path'])
    if sha256(source) != inventory['tables']['train']['sha256']:
        raise ValueError('training table changed')
    frame = pd.read_csv(source, dtype={'DATASET': str, 'ID': str})
    frame = frame.loc[known_voice_targets(frame)]
    selected = pd.concat([frame.loc[frame.VOICE_FAKE.eq(label)].head(2) for label in [0, 1]])
    if len(selected) != 4 or selected.VOICE_FAKE.nunique() != 2:
        raise ValueError('four TRAIN rows with both labels required')
    directory.mkdir(parents=True)
    hashes = {str(path): sha256(path) for path in [Path(__file__),
        ROOT / 'src/wavlm_voice_lora_v79.py', inventory_path,
        ROOT / 'models/wavlm-large/pytorch_model.bin']}
    frozen = dict(scope='TRAIN-only CUDA smoke, not generalization evaluation',
        rank=8, last_blocks=4, native_window_batch_size=1, artifacts_sha256=hashes,
        selected_train=selected[['DATASET', 'ID']].to_dict('records'),
        automatic_submission_allowed=False)
    (directory / 'frozen.json').write_text(json.dumps(frozen, indent=2) + '\n')
    torch.set_num_threads(2)
    torch.manual_seed(20260905)
    model = PairedWavLMVoice(ROOT / 'models/wavlm-large').cuda().train()
    template = CommonTokenHead(model.base.config.hidden_size, width=96, pooling='mean').cuda()
    heads = {name: copy.deepcopy(template) for name in ['control', 'adapted']}
    adapters = [p for p in model.parameters() if p.requires_grad]
    optimizers = dict(control=torch.optim.AdamW(heads['control'].parameters(), lr=.001),
        adapted=torch.optim.AdamW([*heads['adapted'].parameters(), *adapters], lr=.001))
    rows, started, initial_exact = [], time.monotonic(), False
    for index, row in enumerate(selected.itertuples(index=False)):
        audio, _ = librosa.load(row.PATH, sr=16000, mono=True, dtype=np.float32)
        windows, lengths, _ = complete_windows(audio)
        outputs = paired_file_logits(model, heads, torch.from_numpy(windows), torch.from_numpy(lengths),
            [len(windows)], chunk_size=2, temperature=2.)
        if index == 0:
            torch.testing.assert_close(outputs['control'], outputs['adapted'], rtol=0, atol=0)
            initial_exact = True
        losses = {}
        target = torch.tensor([row.VOICE_FAKE], dtype=torch.float32, device='cuda')
        for name, values in outputs.items():
            loss = torch.nn.functional.binary_cross_entropy_with_logits(values[:, 1], target)
            optimizers[name].zero_grad()
            loss.backward()
            parameters = list(heads[name].parameters()) + (adapters if name == 'adapted' else [])
            torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
            optimizers[name].step()
            losses[name] = float(loss.detach())
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise ValueError('frozen WavLM parameter accumulated gradients')
        rows.append(dict(ID=row.ID, losses=losses))
    b_norm = sum(float(adapter.b.detach().abs().sum()) for adapter in model.lora)
    if b_norm <= 0:
        raise ValueError('LoRA did not update')
    model.eval()
    with torch.no_grad():
        first, _ = model(torch.from_numpy(windows[:1]), torch.from_numpy(lengths[:1]))
        second, _ = model(torch.from_numpy(windows[:1]), torch.from_numpy(lengths[:1]))
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise ValueError('smoke source changed')
    result = dict(**frozen, status='complete_smoke_only', steps=rows,
        initial_control_adapted_bit_exact=initial_exact, lora_parameters=model.trainable_count,
        frozen_base_gradients_none=True, updated_lora_b_l1=b_norm, eval_repeat_bit_exact=True,
        peak_cuda_mib=torch.cuda.max_memory_allocated() / 2**20,
        seconds=time.monotonic() - started, accuracy_evaluated=False)
    (directory / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
