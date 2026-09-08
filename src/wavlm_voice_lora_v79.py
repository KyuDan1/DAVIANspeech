"""Paired frozen/LoRA WavLM representations for a Voice-only experiment.

Not a trained detector. Native single-window forwards preserve file-local
inference; only Q/V projections in the last four blocks may be adapted.
"""
from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn.utils import parametrize


class LowRankDelta(nn.Module):
    def __init__(self, rows, columns, rank=8):
        super().__init__()
        if min(rows, columns, rank) <= 0:
            raise ValueError('positive matrix dimensions and rank required')
        self.a = nn.Parameter(torch.empty(rank, columns))
        self.b = nn.Parameter(torch.zeros(rows, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.enabled = True
        self.scale = 1.0

    def forward(self, weight):
        return weight + self.scale * (self.b @ self.a) if self.enabled else weight


def known_voice_targets(frame):
    """A fake background with REAL foreground may contain unannotated fake vocals.

Do not manufacture a new label: exclude that ambiguous RF condition from this
Voice-only training experiment. All filtering is on TRAIN provenance/targets.
"""
    return (frame.VOICE_PRESENT.eq(1) & frame.VOICE_FAKE.isin([0, 1]) & (
        frame.VOICE_FAKE.eq(1) | frame.MUSIC_PRESENT.eq(0) | frame.MUSIC_FAKE.eq(0)))


class PairedWavLMVoice(nn.Module):
    def __init__(self, model_dir, device='cuda', rank=8, last_blocks=4):
        super().__init__()
        from transformers import WavLMModel, Wav2Vec2FeatureExtractor
        from .common_encoder_probe import depth_indices
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, local_files_only=True)
        self.base = WavLMModel.from_pretrained(model_dir, local_files_only=True).to(device).eval()
        self.base.requires_grad_(False)
        self.indices = depth_indices(len(self.base.encoder.layers))
        if not 0 < last_blocks <= len(self.base.encoder.layers):
            raise ValueError('invalid number of adapted layers')
        self.lora = []  # Modules are registered under each native parametrization.
        with torch.random.fork_rng(devices=[]):
            for layer in self.base.encoder.layers[-last_blocks:]:
                for name in ['q_proj', 'v_proj']:
                    linear = getattr(layer.attention, name)
                    adapter = LowRankDelta(linear.out_features, linear.in_features, rank).to(device)
                    parametrize.register_parametrization(linear, 'weight', adapter)
                    self.lora.append(adapter)
        self.trainable_count = sum(p.numel() for p in self.base.parameters() if p.requires_grad)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # Never reactivate native dropout or SpecAugment.
        return self

    @contextmanager
    def adapters_enabled(self, enabled):
        previous = [adapter.enabled for adapter in self.lora]
        try:
            for adapter in self.lora:
                adapter.enabled = enabled
            yield
        finally:
            for adapter, value in zip(self.lora, previous):
                adapter.enabled = value

    def _native(self, window, length, adapted):
        device = next(self.base.parameters()).device
        actual = window[:int(length)].float().cpu().numpy()
        inputs = self.processor([actual], sampling_rate=16000, padding='max_length',
            max_length=163840, return_attention_mask=True, return_tensors='pt').to(device)
        gradient = adapted and self.training and torch.is_grad_enabled()
        with self.adapters_enabled(adapted), torch.set_grad_enabled(gradient), \
                torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            output = self.base(**inputs, output_hidden_states=True)
            tokens = torch.stack([output.hidden_states[i + 1] for i in self.indices], 1).float()
        valid = self.base._get_feat_extract_output_lengths(torch.tensor([length], device=device))
        mask = torch.arange(tokens.shape[2], device=device)[None] < valid[:, None]
        return tokens, mask

    def forward(self, windows, lengths):
        if windows.ndim != 2 or windows.shape[1] != 163840 or lengths.shape != windows.shape[:1]:
            raise ValueError('matching native-window shapes required')
        if not torch.isfinite(windows).all() or not ((lengths > 0) & (lengths <= 163840)).all():
            raise ValueError('finite windows and valid lengths required')
        pieces = {'control': [], 'adapted': []}
        masks = []
        for window, length in zip(windows, lengths.tolist()):
            control, mask = self._native(window, length, False)
            adapted, other_mask = self._native(window, length, True)
            if not torch.equal(mask, other_mask):
                raise ValueError('paired masks differ')
            pieces['control'].append(control)
            pieces['adapted'].append(adapted)
            masks.append(mask)
        return {name: torch.cat(values) for name, values in pieces.items()}, torch.cat(masks)
