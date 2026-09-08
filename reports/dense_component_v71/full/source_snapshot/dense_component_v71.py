"""Time-resolved readout and exact synthetic-interval supervision.

EAT's time-frequency grid is retained; no waveform separation or cross-file
statistics. Time labels describe known source placement, not oracle VAD.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common_encoder_probe import CommonTokenHead
from .full_coverage_wpt import masked_lme

TIME_BINS = 64
FREQUENCY_BINS = 8
BIN_SECONDS = .16
TASK_WEIGHTS = [.45, .18, .27, .05, .05]


class DenseComponentHead(nn.Module):
    def __init__(self, dimension=1024, width=96, pooling='mean'):
        super().__init__()
        self.readout = CommonTokenHead(dimension, width=width, pooling=pooling)

    def forward(self, tokens, mask):
        batch, layers, patches, width = tokens.shape
        if patches != TIME_BINS * FREQUENCY_BINS or mask.shape != (batch, patches):
            raise ValueError('native EAT 64-time by 8-frequency grid required')
        values = tokens.reshape(batch, layers, TIME_BINS, FREQUENCY_BINS, width)
        values = values.permute(0, 2, 1, 3, 4).reshape(batch * TIME_BINS, layers, FREQUENCY_BINS, width)
        frequency_mask = mask.reshape(batch * TIME_BINS, FREQUENCY_BINS)
        valid = frequency_mask.any(-1)
        if not valid.any():
            raise ValueError('empty temporal representation')
        logits = tokens.new_zeros((batch * TIME_BINS, 5), dtype=torch.float32)
        logits[valid] = self.readout(values[valid], frequency_mask[valid])
        return logits.reshape(batch, TIME_BINS, 5), valid.reshape(batch, TIME_BINS)


def interval_targets(metadata, starts, lengths, margin=.08):
    """Centers of known source intervals; ignore ambiguous transition bins.

All task labels are undefined for unknown interval metadata. Fake-component
loss is ignored when that component is absent, including silent gaps.
"""
    target = np.zeros((len(starts), TIME_BINS, 5), np.float32)
    valid = np.zeros_like(target, dtype=bool)
    if metadata is None:
        return target, valid
    keys = ['VOICE_INTERVALS', 'MUSIC_INTERVALS', 'VOICE_FAKE_INTERVALS', 'MUSIC_FAKE_INTERVALS']
    intervals = [metadata[k] for k in keys]
    for spans in intervals:
        if any(not (0 <= a < b) for a, b in spans):
            raise ValueError('nonempty increasing nonnegative intervals required')
    for window, (start, length) in enumerate(zip(starts, lengths)):
        local = (np.arange(TIME_BINS) + .5) * BIN_SECONDS
        centers = start / 16000 + local
        known = local < length / 16000
        inside = [np.zeros(TIME_BINS, bool) for _ in keys]
        for index, spans in enumerate(intervals):
            for a, b in spans:
                inside[index] |= (centers >= a) & (centers < b)
                known &= (np.abs(centers - a) > margin) & (np.abs(centers - b) > margin)
        vp, mp, vf, mf = inside
        if np.any(vf & ~vp) or np.any(mf & ~mp):
            raise ValueError('fake interval must lie inside component presence')
        target[window] = np.stack([vf | mf, vf, mf, vp, mp], -1)
        valid[window] = known[:, None]
        valid[window, :, 1] &= vp
        valid[window, :, 2] &= mp
    return target, valid


def dense_component_loss(logits, target, valid):
    if logits.shape != target.shape or valid.shape != target.shape or valid.dtype != torch.bool:
        raise ValueError('matching logits, labels and boolean validity required')
    if not torch.isfinite(target[valid]).all() or not ((target[valid] == 0) | (target[valid] == 1)).all():
        raise ValueError('finite binary active labels required')
    losses = F.binary_cross_entropy_with_logits(logits, target.masked_fill(~valid, 0), reduction='none')
    # Equal positive/negative mass per task when both are available. A short
    # fake interval must not vanish in the much larger number of real bins.
    total = logits.sum() * 0
    for task, weight in enumerate(TASK_WEIGHTS):
        pieces = []
        for label in (0, 1):
            selected = valid[..., task] & target[..., task].eq(label)
            if selected.any():
                pieces.append(losses[..., task][selected].mean())
        if pieces:
            total = total + weight * torch.stack(pieces).mean()
    return total


def aggregate_files(values, valid, counts, temperature):
    if sum(counts) != len(values) or not counts or min(counts) < 1:
        raise ValueError('counts must cover every window once')
    results, offset = [], 0
    for count in counts:
        scores = values[offset:offset + count].reshape(1, -1, 5)
        mask = valid[offset:offset + count].reshape(1, -1)
        results.append(masked_lme(scores, mask, temperature))
        offset += count
    return torch.cat(results)


def paired_dense_logits(encoder, heads, windows, lengths, counts, chunk_size=2, temperature=5.):
    """Two trainable heads share ONE fully frozen encoder forward per chunk."""
    scores, masks = {name: [] for name in heads}, []
    for start in range(0, len(windows), chunk_size):
        with torch.no_grad():
            representations, mask = encoder(windows[start:start + chunk_size], lengths[start:start + chunk_size])
            tokens = representations['adapted'].detach()
        for name, head in heads.items():
            values, temporal_mask = head(tokens, mask)
            scores[name].append(values)
        masks.append(temporal_mask)
    valid = torch.cat(masks)
    dense = {name: torch.cat(parts) for name, parts in scores.items()}
    pooled = {name: aggregate_files(values, valid, counts, temperature) for name, values in dense.items()}
    return pooled, dense, valid
