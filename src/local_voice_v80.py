"""Frozen forensic projection, local statistics, and known TRAIN intervals.

XLS-R tokens retain contextual attention: local pooling is NOT a claim that
the encoder itself has a one-second receptive field or performs separation.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def voice_projection(parent, tokens, mask):
    clean = tokens.float().masked_fill(~mask[:, None, :, None], 0)
    values = F.gelu(parent.projection(parent.input_norm(clean)))
    return torch.einsum('l,bltd->btd', parent.layer_logits[1].softmax(-1), values)


def local_statistics(values, mask, kernel=51, stride=10):
    if values.ndim != 3 or mask.shape != values.shape[:2] or mask.dtype != torch.bool:
        raise ValueError('values [windows,time,width], boolean mask required')
    if kernel <= 0 or kernel % 2 != 1 or stride <= 0 or not mask.any(-1).all():
        raise ValueError('positive odd kernel, positive stride, nonempty windows required')
    clean = values.float().masked_fill(~mask[..., None], 0).transpose(1, 2)
    count = F.avg_pool1d(mask[:, None].float(), kernel, 1, kernel//2)*kernel
    mean = F.avg_pool1d(clean, kernel, 1, kernel//2)*kernel/count.clamp_min(1)
    second = F.avg_pool1d(clean.square(), kernel, 1, kernel//2)*kernel/count.clamp_min(1)
    std = (second-mean.square()).clamp_min(0).add(1e-5).sqrt()
    return torch.cat([mean, std], 1).transpose(1, 2)[:, ::stride], mask[:, ::stride]


def temporal_geometry(starts, lengths, tokens=511, stride=10):
    starts, lengths = np.asarray(starts), np.asarray(lengths)
    if len(starts) == 0 or len(starts) != len(lengths) or np.any(np.diff(starts) <= 0):
        raise ValueError('increasing unique starts and matching lengths required')
    # Native XLS-R convolution: stride 320, receptive field 400 samples.
    local = 200 + 320*np.arange(0, tokens, stride)
    centers = starts[:, None]+local
    valid = (local[None]+200 <= lengths[:, None])
    window_centers = starts + lengths/2
    boundaries = (window_centers[:-1]+window_centers[1:])/2
    for index in range(len(starts)):
        if index:
            valid[index] &= centers[index] >= boundaries[index-1]
        if index+1 < len(starts):
            valid[index] &= centers[index] < boundaries[index]
    return centers/16000., valid


def interval_labels(centers, ownership, spans, margin=.08):
    if margin < 0 or any(not (0 <= a < b) for a,b in spans):
        raise ValueError('valid nonnegative intervals and margin required')
    labels, valid = np.zeros_like(centers, dtype=np.float32), ownership.copy()
    for a,b in spans:
        labels[(centers >= a) & (centers < b)] = 1
        valid &= (np.abs(centers-a) > margin) & (np.abs(centers-b) > margin)
    return labels, valid


class LocalVoiceHead(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone())
        self.bias = nn.Parameter(bias.detach().clone())

    def forward(self, features):
        return (features*self.weight).sum(-1)+self.bias


def file_logit(logits, valid, temperature=5.):
    if logits.shape != valid.shape or not valid.any() or temperature <= 0:
        raise ValueError('matching logits and nonempty valid mask required')
    values = logits[valid].float()
    return (torch.logsumexp(values*temperature, 0)-np.log(values.numel()))/temperature


def balanced_interval_loss(logits, target, valid):
    if logits.shape != target.shape or valid.shape != target.shape:
        raise ValueError('matching temporal tensors required')
    if not torch.isfinite(target[valid]).all() or not ((target[valid]==0)|(target[valid]==1)).all():
        raise ValueError('finite binary active labels required')
    parts=[]
    for label in (0,1):
        selected=valid & target.eq(label)
        if selected.any():
            parts.append(F.binary_cross_entropy_with_logits(logits[selected], target[selected]))
    return torch.stack(parts).mean() if parts else logits.sum()*0
