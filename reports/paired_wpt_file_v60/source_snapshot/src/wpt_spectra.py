"""Wavelet-prompt XLS-R+AASIST for all-type component authenticity.

This is an independent reimplementation of the WPT idea from Xie et al.
(``arXiv:2504.06753``).  The frozen XLS-R encoder receives fresh prompt tokens
at every transformer layer.  Four of the ten prompts are differentiable 2-D
Haar sub-bands; previous-layer prompts are discarded while audio tokens flow
through the network.  A pretrained Spectra-AASIST backend is then co-trained
for Voice, Music, and File authenticity.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def haar_2d_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """Apply a critically sampled 2-D Haar transform without new parameters.

    Input is ``[..., wavelet_tokens, width]``.  Every sub-band is flattened
    back to tokens, so the result has exactly the same shape as the input.
    """
    if tokens.ndim < 2:
        raise ValueError("wavelet prompts must have at least two dimensions")
    count, width = tokens.shape[-2:]
    if count % 4 or width % 2:
        raise ValueError("Haar prompt count must divide by 4 and width by 2")
    prefix = tokens.shape[:-2]
    values = tokens.reshape(*prefix, count // 2, 2, width // 2, 2)
    a = values[..., 0, :, 0]
    b = values[..., 0, :, 1]
    c = values[..., 1, :, 0]
    d = values[..., 1, :, 1]
    # Orthonormal separable Haar: each 2x2 block uses a factor of 1/2.
    bands = (
        (a + b + c + d) * .5,
        (-a - b + c + d) * .5,
        (-a + b - c + d) * .5,
        (a - b - c + d) * .5,
    )
    return torch.cat([
        band.reshape(*prefix, count // 4, width) for band in bands
    ], dim=-2)


class DeepWaveletPrompts(nn.Module):
    """Layer-specific standard and wavelet prompts."""

    def __init__(
        self, layers: int, width: int, prompt_tokens: int = 6,
        wavelet_tokens: int = 4,
    ) -> None:
        super().__init__()
        if layers <= 0 or width <= 0 or prompt_tokens < 0:
            raise ValueError("invalid prompt dimensions")
        if wavelet_tokens <= 0 or wavelet_tokens % 4:
            raise ValueError("wavelet token count must be a positive multiple of 4")
        self.layers = int(layers)
        self.width = int(width)
        self.prompt_tokens = int(prompt_tokens)
        self.wavelet_tokens = int(wavelet_tokens)
        self.standard = nn.Parameter(torch.empty(layers, prompt_tokens, width))
        self.wavelet_initial = nn.Parameter(
            torch.empty(layers, wavelet_tokens, width)
        )
        nn.init.xavier_uniform_(self.standard)
        nn.init.xavier_uniform_(self.wavelet_initial)

    @property
    def count(self) -> int:
        return self.prompt_tokens + self.wavelet_tokens

    def forward(self, layer: int, batch: int) -> torch.Tensor:
        if not 0 <= layer < self.layers:
            raise IndexError(layer)
        wavelet = haar_2d_tokens(self.wavelet_initial[layer])
        values = torch.cat((wavelet, self.standard[layer]), dim=0)
        return values.unsqueeze(0).expand(batch, -1, -1)


class PromptedWav2Vec2Encoder(nn.Module):
    """Run a frozen Hugging Face Wav2Vec2Model with deep WPT tokens."""

    def __init__(
        self, backbone: nn.Module, prompt_tokens: int = 6,
        wavelet_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        layers = len(backbone.encoder.layers)
        width = int(backbone.config.hidden_size)
        self.prompts = DeepWaveletPrompts(
            layers, width, prompt_tokens, wavelet_tokens
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.config.apply_spec_augment = False
        self.backbone.config.layerdrop = 0.0
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # The frozen encoder must not add dropout noise or LayerDrop. Gradients
        # still flow from later audio tokens into the prompt parameters.
        self.backbone.eval()
        return self

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 2:
            raise ValueError("waveforms must have shape [batch, samples]")
        model = self.backbone
        with torch.no_grad():
            features = model.feature_extractor(waveforms).transpose(1, 2)
            hidden, _ = model.feature_projection(features)
        hidden = hidden.detach()
        encoder = model.encoder
        hidden = hidden + encoder.pos_conv_embed(hidden)
        hidden = encoder.dropout(hidden)
        count = self.prompts.count
        for index, layer in enumerate(encoder.layers):
            prompt = self.prompts(index, len(hidden)).to(hidden.dtype)
            combined = torch.cat((prompt, hidden), dim=1)
            combined = layer(
                combined, attention_mask=None, output_attentions=False
            )[0]
            hidden = combined if index == len(encoder.layers) - 1 else combined[:, count:]
        return encoder.layer_norm(hidden)


class WPTSpectraMultitask(nn.Module):
    """Compact-trainable WPT frontend over a pretrained Spectra-AASIST model."""

    TASKS = ("VOICE", "MUSIC", "FILE")

    def __init__(
        self, spectra_model: nn.Module, prompt_tokens: int = 6,
        wavelet_tokens: int = 4, temperature: float = 5.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.prompt_encoder = PromptedWav2Vec2Encoder(
            spectra_model.ssl_encoder.model, prompt_tokens, wavelet_tokens
        )
        self.bridge = spectra_model.bridge
        self.aasist = spectra_model.aasist
        self.temperature = float(temperature)
        embedding_width = self.aasist.out_layer.in_features
        self.task_head = nn.Linear(embedding_width, len(self.TASKS))
        with torch.no_grad():
            margin_weight = (
                self.aasist.out_layer.weight[0]
                - self.aasist.out_layer.weight[1]
            )
            margin_bias = (
                self.aasist.out_layer.bias[0]
                - self.aasist.out_layer.bias[1]
            )
            self.task_head.weight.zero_()
            self.task_head.bias.zero_()
            self.task_head.weight[0].copy_(margin_weight)
            self.task_head.bias[0].copy_(margin_bias)
            self.task_head.weight[2].copy_(margin_weight)
            self.task_head.bias[2].copy_(margin_bias)
            nn.init.normal_(self.task_head.weight[1], std=.01)
        for parameter in self.aasist.out_layer.parameters():
            parameter.requires_grad_(False)
        self._embedding: torch.Tensor | None = None
        self.aasist.out_layer.register_forward_pre_hook(self._capture_embedding)

    def _capture_embedding(self, _module, inputs) -> None:
        self._embedding = inputs[0]

    def train(self, mode: bool = True):
        super().train(mode)
        self.prompt_encoder.backbone.eval()
        return self

    def forward_windows_with_embedding(
        self, waveforms: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-view task logits and the pre-classifier AASIST latent."""
        if waveforms.ndim != 3:
            raise ValueError("window input must be [batch, views, samples]")
        batch, views, samples = waveforms.shape
        hidden = self.prompt_encoder(waveforms.reshape(batch * views, samples))
        hidden = self.bridge(hidden)
        self._embedding = None
        self.aasist(hidden)
        if self._embedding is None:
            raise RuntimeError("failed to capture AASIST embedding")
        embedding = self._embedding.reshape(batch, views, -1)
        logits = self.task_head(self._embedding).reshape(
            batch, views, len(self.TASKS)
        )
        return logits, embedding

    def forward_windows(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Score ``[batch, views, samples]`` and return view logits."""
        return self.forward_windows_with_embedding(waveforms)[0]

    def aggregate(self, view_logits: torch.Tensor) -> torch.Tensor:
        """Smooth max over views; one fake interval is enough for a fake file."""
        if view_logits.ndim != 3:
            raise ValueError("view logits must have shape [batch, views, tasks]")
        temperature = self.temperature
        return torch.logsumexp(temperature * view_logits, dim=1) / temperature - (
            math.log(view_logits.shape[1]) / temperature
        )

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self.aggregate(self.forward_windows(waveforms))

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Exclude the unchanged 300M backbone from compact checkpoints."""
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if not name.startswith("prompt_encoder.backbone.")
        }

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        missing, unexpected = self.load_state_dict(state, strict=False)
        allowed = [
            name for name in missing if name.startswith("prompt_encoder.backbone.")
        ]
        if len(allowed) != len(missing) or unexpected:
            raise ValueError(
                f"compact WPT state mismatch: missing={missing[:5]}, "
                f"unexpected={unexpected[:5]}"
            )


def component_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    presence: torch.Tensor,
    sample_weight: torch.Tensor,
    task_weights: tuple[float, float, float] = (.20, .35, .45),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Balanced component-conditional BCE for Voice, Music, and File."""
    if logits.shape != targets.shape or logits.shape[1] != 3:
        raise ValueError("WPT logits and targets must have shape [batch, 3]")
    if presence.shape != (len(logits), 2):
        raise ValueError("presence must have shape [batch, 2]")
    masks = (
        presence[:, 0].bool(), presence[:, 1].bool(),
        torch.ones(len(logits), dtype=torch.bool, device=logits.device),
    )
    terms = []
    for index, mask in enumerate(masks):
        raw = F.binary_cross_entropy_with_logits(
            logits[mask, index], targets[mask, index], reduction="none"
        )
        weight = sample_weight[mask]
        terms.append((raw * weight).sum() / weight.sum().clamp_min(1e-8))
    if len(task_weights) != 3 or any(value < 0 for value in task_weights):
        raise ValueError("task weights must contain three non-negative values")
    total = sum(task_weights)
    if total <= 0:
        raise ValueError("at least one task weight must be positive")
    loss = sum(weight * term for weight, term in zip(task_weights, terms)) / total
    return loss, dict(zip(("voice", "music", "file"), terms))


def component_ranking_loss(
    logits: torch.Tensor, targets: torch.Tensor, presence: torch.Tensor,
) -> torch.Tensor:
    """Conditional pairwise EER surrogate for small training batches."""
    masks = (
        presence[:, 0].bool(), presence[:, 1].bool(),
        torch.ones(len(logits), dtype=torch.bool, device=logits.device),
    )
    terms = []
    for task, mask in enumerate(masks):
        positive = logits[mask & targets[:, task].eq(1), task]
        negative = logits[mask & targets[:, task].eq(0), task]
        if len(positive) and len(negative):
            terms.append(F.softplus(
                -(positive[:, None] - negative[None, :])
            ).mean())
    return torch.stack(terms).mean() if terms else logits.sum() * 0
