"""Matched frozen/control and trainable-adapter EAT representation experiment.

Both branches share the exact frozen prefix and frontend. No separation, no
cross-file statistics, and no labels or probabilities passed between files.
"""
from contextlib import nullcontext

import torch
from torch import nn

from .common_encoder_probe import depth_indices
from .eat_music_adapter import BottleneckAdapter
from .full_coverage_wpt import masked_lme


class PairedEatRepresentation(nn.Module):
    def __init__(self, base, adapter_blocks=(20, 21, 22, 23), bottleneck=32):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        self.indices = depth_indices(len(base.model.blocks))
        self.adapter_blocks = tuple(adapter_blocks)
        if (not self.adapter_blocks or sorted(set(self.adapter_blocks)) != list(self.adapter_blocks)
                or min(self.adapter_blocks) < 0 or max(self.adapter_blocks) >= len(base.model.blocks)):
            raise ValueError('valid increasing adapter blocks required')
        width = base.model.extra_tokens.shape[-1]
        # Adapter initialization must not perturb the matched head/sampler RNG.
        with torch.random.fork_rng(devices=[]):
            self.adapters = nn.ModuleDict({str(i): BottleneckAdapter(width, bottleneck, dropout=0.)
                                           for i in self.adapter_blocks})

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def from_features(self, features, valid_frames):
        """features [B,1,1024,128], frontend identical to frozen benchmark."""
        core = self.base.model
        with torch.no_grad():
            grid = core.local_encoder.proj(features)
            batch, dim, times, frequencies = grid.shape
            values = grid.flatten(2).transpose(1, 2)
            positional = getattr(core, 'fixed_positional_encoder', None)
            if positional is not None:
                values = values + positional(values, None)[:, :values.shape[1]]
            values = torch.cat((core.extra_tokens.expand(batch, -1, -1), values), 1)
            values = core.pos_drop(core.pre_norm(values))
        control, adapted = [], []
        original = values
        adapted_values = values
        for index, block in enumerate(core.blocks):
            with torch.no_grad():
                original = block(original)
                if isinstance(original, tuple):
                    original = original[0]
            if index < self.adapter_blocks[0]:
                adapted_values = original
            else:
                # The first adapted block has the same frozen input: reuse its
                # output. Later frozen blocks retain gradients to earlier adapters.
                if index == self.adapter_blocks[0]:
                    adapted_values = original
                else:
                    adapted_values = block(adapted_values)
                    if isinstance(adapted_values, tuple):
                        adapted_values = adapted_values[0]
                if str(index) in self.adapters:
                    adapted_values = self.adapters[str(index)](adapted_values)
            if index in self.indices:
                control.append(original[:, 1:])
                adapted.append(adapted_values[:, 1:])
        valid_time = ((valid_frames + 15) // 16).clamp_max(times)
        mask = torch.arange(times, device=features.device)[None] < valid_time[:, None]
        mask = mask[:, :, None].expand(-1, -1, frequencies).reshape(batch, -1)
        result = {'control': torch.stack(control, 1).float(),
                  'adapted': torch.stack(adapted, 1).float()}
        if not all(torch.isfinite(t).all() for t in result.values()):
            raise RuntimeError('nonfinite representation')
        return result, mask

    def forward(self, windows, lengths):
        from .eat_large_aasist_inference import fbank
        device = next(self.base.parameters()).device
        features = torch.stack([fbank(w[:int(n)].cpu().numpy()) for w, n in zip(windows, lengths)])[:, None].to(device)
        valid_frames = ((lengths.to(device) - 400).clamp_min(0) // 160 + 1)
        context = torch.autocast(device.type, dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
        with context:
            return self.from_features(features, valid_frames)


def paired_file_logits(encoder, heads, windows, lengths, counts, chunk_size=2, temperature=2.):
    pieces = {name: [] for name in heads}
    for start in range(0, len(windows), chunk_size):
        tokens, mask = encoder(windows[start:start + chunk_size], lengths[start:start + chunk_size])
        for name, head in heads.items():
            pieces[name].append(head(tokens[name], mask))
    results = {}
    for name in heads:
        values = torch.cat(pieces[name])
        if sum(counts) != len(values) or min(counts) < 1:
            raise ValueError('valid counts matching all windows required')
        padded = values.new_zeros((len(counts), max(counts), 5))
        mask = torch.zeros(padded.shape[:2], device=values.device, dtype=torch.bool)
        offset = 0
        for index, count in enumerate(counts):
            padded[index, :count] = values[offset:offset + count]
            mask[index, :count] = True
            offset += count
        results[name] = masked_lme(padded, mask, temperature)
    return results
