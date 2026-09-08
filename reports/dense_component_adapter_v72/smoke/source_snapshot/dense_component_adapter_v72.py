"""Dense-time supervision reaching the existing small EAT representation adapters."""
import torch

from .dense_component_v71 import aggregate_files


def adapted_dense_logits(encoder, head, windows, lengths, counts, chunk_size=16, temperature=5.):
    scores, masks = [], []
    for start in range(0, len(windows), chunk_size):
        # PairedEatRepresentation itself freezes the frontend, prefix and all
        # original EAT parameters, retaining gradients through adapted tail blocks.
        representations, mask = encoder(windows[start:start + chunk_size], lengths[start:start + chunk_size])
        values, valid = head(representations['adapted'], mask)
        scores.append(values)
        masks.append(valid)
    dense, valid = torch.cat(scores), torch.cat(masks)
    return aggregate_files(dense, valid, counts, temperature), dense, valid
