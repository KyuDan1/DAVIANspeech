"""Matched global/local File statistics from the same frozen XLS-R projection."""
import torch
from torch.nn import functional as F
from .local_voice_v80 import local_statistics


def file_statistics(parent,tokens,mask):
    clean=tokens.float().masked_fill(~mask[:,None,:,None],0)
    projected=F.gelu(parent.projection(parent.input_norm(clean)))
    values=torch.einsum('l,bltd->btd',parent.layer_logits[0].softmax(-1),projected)
    mean=(values*mask[...,None]).sum(1)/mask.sum(1)[:,None]
    var=((values-mean[:,None]).square()*mask[...,None]).sum(1)/mask.sum(1)[:,None]
    global_stats=torch.cat([mean,(var+1e-5).sqrt()],-1)
    local,valid=local_statistics(values,mask,kernel=51,stride=10)
    return global_stats,local,valid


def lme(values,temperature=5.):
    values=values.reshape(-1)
    if not values.numel() or temperature<=0:raise ValueError('nonempty logits required')
    return (torch.logsumexp(values*temperature,0)-values.new_tensor(values.numel()).log())/temperature
