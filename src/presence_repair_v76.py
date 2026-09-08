"""A narrowly controlled TRAIN-label sensitivity experiment, not relabeling."""
import math

import numpy as np
import torch
from torch.nn import functional as F


def head_statistics(head, tokens, mask):
    """Expose existing head statistics without changing its native operations."""
    if (tokens.ndim != 4 or tokens.shape[1] != head.layers
            or mask.shape != (tokens.shape[0], tokens.shape[2])
            or mask.dtype != torch.bool or not mask.any(-1).all()):
        raise ValueError('invalid token/mask shape')
    clean = tokens.float().masked_fill(~mask[:, None, :, None], 0)
    values = F.gelu(head.projection(head.input_norm(clean)))
    values = torch.einsum('ql,blnd->bqnd', head.layer_logits.softmax(-1), values)
    scores = (torch.einsum('bqnd,qd->bqn', values, head.queries) / math.sqrt(values.shape[-1])
              if head.pooling == 'attention' else values.new_zeros(values.shape[:3]))
    attention = scores.masked_fill(~mask[:, None], -torch.inf).softmax(-1)
    mean = (attention[..., None] * values).sum(-2)
    variance = (attention[..., None] * (values - mean[..., None, :]).square()).sum(-2)
    return torch.cat((mean, (variance + 1e-5).sqrt()), -1)


def training_review_mask(train, review):
    """Only explicit reviewed TRAIN keys; no propagation from other channel rows."""
    keys = ['DATASET', 'ID']
    if train.duplicated(keys).any() or review.duplicated(keys).any():
        raise ValueError('duplicate TRAIN/review keys')
    if not review.BOTH_VOICE_REVIEW.isin([True, False]).all():
        raise ValueError('review flags must be booleans')
    lookup = {tuple(row): i for i, row in enumerate(train[keys].to_numpy())}
    result = np.zeros(len(train), dtype=bool)
    for row in review.itertuples(index=False):
        key = (row.DATASET, row.ID)
        if key not in lookup:
            raise ValueError('review contains a non-TRAIN key')
        index = lookup[key]
        if train.iloc[index].VOICE_PRESENT != 0 or row.VOICE_PRESENT != 0:
            raise ValueError('review is exclusively for voice-negative TRAIN targets')
        # Only unadapted semantic detectors determine the flag; learned scores
        # are deliberately excluded from the intervention rule.
        expected = row.EAT_VOICE >= .5 and row.PANNS_VOICE >= .5
        if bool(row.BOTH_VOICE_REVIEW) != expected:
            raise ValueError('flag disagrees with frozen semantic review threshold')
        result[index] = expected
    return result


def presence_loss(logits, targets, excluded):
    if logits.ndim != 1 or targets.shape != logits.shape or excluded.shape != logits.shape:
        raise ValueError('matching one-dimensional tensors required')
    if excluded.dtype != torch.bool or not torch.isfinite(logits).all():
        raise ValueError('finite logits and boolean exclusions required')
    if not ((targets == 0) | (targets == 1)).all():
        raise ValueError('binary presence targets required')
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    # An all-masked batch has zero gradient rather than NaN or an invented label.
    return (losses * ~excluded).sum() / (~excluded).sum().clamp_min(1)
