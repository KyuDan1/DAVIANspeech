"""Matched frozen-encoder readouts; preserves tokens until task pooling.

This is an experiment primitive, not a validated submission model. Outputs
are File, Voice, Music, Voice-presence, Music-presence logits, in that order.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def complete_windows(audio: np.ndarray, window: int = 163840):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all() or window <= 0:
        raise ValueError('nonempty finite mono audio and positive window required')
    count = max(1, math.ceil(len(audio) / window))
    starts = np.rint(np.linspace(0, max(0, len(audio) - window), count)).astype(np.int64)
    values = np.zeros((count, window), dtype=np.float32)
    lengths = []
    for index, start in enumerate(starts):
        part = audio[start:start + window]
        values[index, :len(part)] = part
        lengths.append(len(part))
    return values, np.asarray(lengths), starts


def depth_indices(depth: int):
    if depth < 3:
        raise ValueError('at least three encoder blocks required')
    return (max(0, math.ceil(depth / 4) - 1), math.ceil(depth / 2) - 1, depth - 1)


def encode_windows_independently(encoder, windows, lengths):
    """Native one-window forwards; pad only already-computed tokens for heads.

This avoids SPEAR's variable-length convolution padding and batch-shape numeric
dependence. It never modifies native checkpoint code or actual audio length.
"""
    if len(windows) == 0 or len(windows) != len(lengths):
        raise ValueError('matching nonempty windows and lengths required')
    pieces = [encoder(windows[i:i + 1], lengths[i:i + 1]) for i in range(len(windows))]
    maximum = max(tokens.shape[2] for tokens, _ in pieces)
    tokens = torch.cat([F.pad(t, (0, 0, 0, maximum - t.shape[2])) for t, _ in pieces])
    mask = torch.cat([F.pad(m, (0, maximum - m.shape[1]), value=False) for _, m in pieces])
    return tokens, mask


class CommonTokenHead(nn.Module):
    """Same downstream architecture for all encoders; native width is explicit.

The input projection parameter count varies with encoder width. Both pooling
conditions share exactly the same state schema, including inactive queries
in the mean condition. Neither averages EAT frequency tokens in advance.
"""
    def __init__(self, dimension: int, layers: int = 3, width: int = 96,
                 pooling: str = 'attention'):
        super().__init__()
        if min(dimension, layers, width) <= 0 or pooling not in {'mean', 'attention'}:
            raise ValueError('invalid dimensions or pooling')
        self.pooling = pooling
        self.layers = layers
        self.input_norm = nn.LayerNorm(dimension)
        self.projection = nn.Linear(dimension, width)
        self.layer_logits = nn.Parameter(torch.zeros(5, layers))
        self.queries = nn.Parameter(torch.empty(5, width))
        self.output_weight = nn.Parameter(torch.empty(5, width * 2))
        self.output_bias = nn.Parameter(torch.zeros(5))
        nn.init.normal_(self.queries, std=.02)
        nn.init.normal_(self.output_weight, std=.02)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        if (tokens.ndim != 4 or tokens.shape[1] != self.layers
                or mask.shape != (tokens.shape[0], tokens.shape[2])
                or mask.dtype != torch.bool or not mask.any(-1).all()):
            raise ValueError('tokens [B,L,N,D], nonempty boolean mask [B,N] required')
        clean = tokens.float().masked_fill(~mask[:, None, :, None], 0)
        values = F.gelu(self.projection(self.input_norm(clean)))
        values = torch.einsum('ql,blnd->bqnd', self.layer_logits.softmax(-1), values)
        if self.pooling == 'attention':
            scores = torch.einsum('bqnd,qd->bqn', values, self.queries) / math.sqrt(values.shape[-1])
        else:
            scores = values.new_zeros(values.shape[:3])
        attention = scores.masked_fill(~mask[:, None], -torch.inf).softmax(-1)
        mean = (attention[..., None] * values).sum(-2)
        variance = (attention[..., None] * (values - mean[..., None, :]).square()).sum(-2)
        statistics = torch.cat((mean, (variance + 1e-5).sqrt()), -1)
        return (statistics * self.output_weight).sum(-1) + self.output_bias


def component_loss(logits, targets):
    """File labels supervise files/bags, never each arbitrary crop.

Targets are [File, Voice, Music, V-present, M-present]. Absent component
targets may be NaN and are excluded, not silently treated as Real.
"""
    if logits.ndim != 2 or logits.shape != targets.shape or logits.shape[1] != 5:
        raise ValueError('matching [B,5] logits and targets required')
    valid = torch.ones_like(targets, dtype=torch.bool)
    valid[:, 1] = targets[:, 3].eq(1)
    valid[:, 2] = targets[:, 4].eq(1)
    if not torch.isfinite(targets[valid]).all() or not ((targets[valid] == 0) | (targets[valid] == 1)).all():
        raise ValueError('active targets must be finite binary labels')
    cleaned = targets.masked_fill(~valid, 0)
    losses = F.binary_cross_entropy_with_logits(logits, cleaned, reduction='none')
    per_task = (losses * valid).sum(0) / valid.sum(0).clamp_min(1)
    return (per_task * logits.new_tensor([.45, .18, .27, .05, .05])).sum()


class FrozenEncoderTokens:
    """Local checkpoints only, matched 10.24-second raw waveform views.

EAT keeps the full joint time-frequency grid. Native frontends and native
normalization differ by encoder; this is intentional and recorded, not a
claim of identical preprocessing. All returned tensors are ordinary detached
tensors, so a downstream head can learn from them.
"""
    def __init__(self, name: str, model_dir: Path, device='cuda'):
        self.name, self.device = name, torch.device(device)
        if name == 'xlsr':
            from .xlsr_antideepfake import XlsrAntiDeepfake
            self.model = XlsrAntiDeepfake.from_checkpoint(model_dir, device=self.device)
            depth = len(self.model.ssl.encoder.layers)
        elif name in {'spear', 'spear_independent'}:
            from .spear_detector import _load_local_model
            self.model = _load_local_model(Path(model_dir), self.device)
            depth = 13  # This experiment pins the XLarge speech-audio v2 checkpoint.
        elif name == 'wavlm':
            from transformers import WavLMModel, Wav2Vec2FeatureExtractor
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, local_files_only=True)
            self.model = WavLMModel.from_pretrained(model_dir, local_files_only=True).to(self.device)
            depth = len(self.model.encoder.layers)
        elif name in {'eat_base', 'eat_large'}:
            from .eat_large_aasist_inference import _load_local_model
            self.model = _load_local_model(Path(model_dir), self.device)
            depth = len(self.model.model.blocks)
        else:
            raise ValueError(f'unknown encoder {name}')
        self.indices = depth_indices(depth)
        self.model.eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, windows: torch.Tensor, lengths: torch.Tensor):
        if (windows.ndim != 2 or windows.shape[1] != 163840
                or lengths.shape != windows.shape[:1]
                or not ((lengths > 0) & (lengths <= windows.shape[1])).all()
                or not torch.isfinite(windows).all()):
            raise ValueError('finite [B,163840] windows and valid [B] lengths required')
        if self.name == 'spear_independent' and len(windows) > 1:
            return encode_windows_independently(self, windows, lengths)
        windows = windows.to(self.device)
        lengths = lengths.to(self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            if self.name == 'xlsr':
                # Normalize only actual audio, excluding zero-padding.
                normalized = torch.zeros_like(windows)
                for i, length in enumerate(lengths.tolist()):
                    normalized[i, :length] = self.model.normalize(windows[i, :length])
                sample_mask = torch.arange(windows.shape[1], device=self.device)[None] < lengths[:, None]
                output = self.model.ssl(normalized, attention_mask=sample_mask.long(), output_hidden_states=True)
                tokens = torch.stack([output.hidden_states[i + 1] for i in self.indices], 1)
                valid_lengths = self.model.ssl._get_feat_extract_output_lengths(lengths)
                mask = torch.arange(tokens.shape[2], device=self.device)[None] < valid_lengths[:, None]
            elif self.name == 'wavlm':
                # Official processor normalizes actual samples, then pads.
                # Never normalize the padded tail or synthesize/separate audio.
                actual = [w[:int(n)].float().cpu().numpy() for w, n in zip(windows, lengths)]
                inputs = self.processor(actual, sampling_rate=16000, padding='max_length',
                                        max_length=163840, return_attention_mask=True,
                                        return_tensors='pt').to(self.device)
                output = self.model(**inputs, output_hidden_states=True)
                tokens = torch.stack([output.hidden_states[i + 1] for i in self.indices], 1)
                valid_lengths = self.model._get_feat_extract_output_lengths(lengths)
                mask = torch.arange(tokens.shape[2], device=self.device)[None] < valid_lengths[:, None]
            elif self.name in {'spear', 'spear_independent'}:
                output = self.model(windows, lengths)
                states = output['hidden_states']
                if len(states) != 13:
                    raise ValueError('unexpected SPEAR depth')
                tokens = torch.stack([states[i] for i in self.indices], 1)
                mask = torch.arange(tokens.shape[2], device=self.device)[None] < output['encoder_out_lens'][:, None]
            else:
                from .eat_large_aasist_inference import fbank
                features = torch.stack([fbank(w[:int(n)].cpu().numpy()) for w, n in zip(windows, lengths)])[:, None].to(self.device)
                core = self.model.model
                grid = core.local_encoder.proj(features)
                batch, dim, times, frequencies = grid.shape
                values = grid.flatten(2).transpose(1, 2)
                positional = getattr(core, 'fixed_positional_encoder', None)
                if positional is not None:
                    values = values + positional(values, None)[:, :values.shape[1]]
                values = torch.cat((core.extra_tokens.expand(batch, -1, -1), values), 1)
                values = core.pos_drop(core.pre_norm(values))
                selected = []
                for i, block in enumerate(core.blocks):
                    values = block(values)
                    if isinstance(values, tuple):
                        values = values[0]
                    if i in self.indices:
                        selected.append(values[:, 1:])
                tokens = torch.stack(selected, 1)
                frame_count = ((lengths - 400).clamp_min(0) // 160 + 1)
                valid_time = ((frame_count + 15) // 16).clamp_max(times)
                mask = (torch.arange(times, device=self.device)[None] < valid_time[:, None])
                mask = mask[:, :, None].expand(-1, -1, frequencies).reshape(batch, -1)
        if not torch.isfinite(tokens).all():
            raise RuntimeError('nonfinite encoder tokens')
        return tokens.float().detach(), mask
