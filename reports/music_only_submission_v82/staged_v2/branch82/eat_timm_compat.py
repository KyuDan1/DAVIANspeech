"""Minimal timm compatibility surface required by the bundled EAT model.

The grader already ships a matching torch/torchaudio CUDA pair. Installing
timm from pip can resolve a different torch build and break libtorchaudio.so,
so provide the four tiny utilities EAT imports without a package dependency.
"""

from __future__ import annotations

import sys
import types

import torch
from torch import nn


def to_2tuple(value):
    return value if isinstance(value, tuple) else (value, value)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    return x * random_tensor.div_(keep_prob)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        linear = nn.Conv2d if use_conv else nn.Linear
        self.fc1 = linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.fc2 = linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return self.drop2(x)


def install_timm_compat() -> None:
    """Register import-compatible modules before loading EAT remote code."""
    timm = types.ModuleType("timm")
    models = types.ModuleType("timm.models")
    layers = types.ModuleType("timm.models.layers")
    vision_transformer = types.ModuleType("timm.models.vision_transformer")
    layers.to_2tuple = to_2tuple
    layers.trunc_normal_ = torch.nn.init.trunc_normal_
    vision_transformer.DropPath = DropPath
    vision_transformer.Mlp = Mlp
    timm.models = models
    models.layers = layers
    models.vision_transformer = vision_transformer
    sys.modules.update({
        "timm": timm,
        "timm.models": models,
        "timm.models.layers": layers,
        "timm.models.vision_transformer": vision_transformer,
    })
