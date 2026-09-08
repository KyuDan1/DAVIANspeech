"""Signal-conditioned multi-stream prompts for mixed-audio forensics.

This module independently implements the three prompt families described in
MixFake (arXiv:2605.23201): a learnable base stream, a multi-scale
instantaneous-frequency stream, and an input-conditioned texture stream.  It
uses the original mixture only; no source separator is involved.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

try:  # package imports in tests; flat imports in the submission archive
    from .wpt_spectra import WPTSpectraMultitask
except ImportError:  # pragma: no cover
    from wpt_spectra import WPTSpectraMultitask


class MultiScaleInstantaneousFrequency(nn.Module):
    """Transform prompt tokens through high/all/low-scale phase changes."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.fusion = nn.Linear(3 * width, width)
        self.norm = nn.LayerNorm(width)

    @staticmethod
    def analytic_signal(values: torch.Tensor) -> torch.Tensor:
        """Return the analytic signal along the prompt-token dimension."""
        if values.ndim != 3:
            raise ValueError("frequency prompts must have shape [batch, tokens, width]")
        count = values.shape[1]
        if count < 2:
            raise ValueError("frequency prompts require at least two tokens")
        spectrum = torch.fft.fft(values.float(), dim=1)
        multiplier = torch.zeros(
            count, device=values.device, dtype=values.dtype
        )
        multiplier[0] = 1
        if count % 2 == 0:
            multiplier[count // 2] = 1
            multiplier[1:count // 2] = 2
        else:
            multiplier[1:(count + 1) // 2] = 2
        return torch.fft.ifft(
            spectrum * multiplier.float()[None, :, None], dim=1
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values32 = values.float()
        high = F.pad(values32[:, 1:] - values32[:, :-1], (0, 0, 1, 0))
        low = F.avg_pool1d(
            values32.transpose(1, 2), kernel_size=3, stride=1, padding=1
        ).transpose(1, 2)
        frequencies = []
        for stream in (high, values32, low):
            phase = torch.angle(self.analytic_signal(stream))
            delta = torch.diff(phase, dim=1, prepend=phase[:, :1])
            frequencies.append(delta.abs())
        return self.norm(values32 + self.fusion(torch.cat(frequencies, dim=-1)))


class SignalTextureConditioner(nn.Module):
    """Modulate prompts with TKEO energy and feature-flux statistics."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(width)

    def statistics(
        self, feature_sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if feature_sequence.ndim != 3:
            raise ValueError("feature sequence must have shape [batch, frames, width]")
        sequence = feature_sequence.float()
        if sequence.shape[1] >= 3:
            tkeo = (
                sequence[:, 1:-1].square()
                - sequence[:, :-2] * sequence[:, 2:]
            ).abs().mean(dim=1, keepdim=True)
        else:
            tkeo = sequence.square().mean(dim=1, keepdim=True)
        if sequence.shape[1] >= 2:
            flux = (sequence[:, 1:] - sequence[:, :-1]).abs().std(
                dim=1, keepdim=True, unbiased=False
            )
        else:
            flux = torch.zeros_like(tkeo)
        gate = self.gate(torch.cat((tkeo, flux), dim=-1))
        return tkeo, gate

    def forward(
        self, prompts: torch.Tensor, tkeo: torch.Tensor, gate: torch.Tensor,
    ) -> torch.Tensor:
        if prompts.ndim != 3 or tkeo.ndim != 3 or gate.ndim != 3:
            raise ValueError("texture inputs must be rank-three tensors")
        return self.norm(prompts.float() * gate + tkeo * (1 - gate))


class DeepMultiStreamPrompts(nn.Module):
    """Layer-specific base, frequency, and signal-conditioned prompts."""

    def __init__(
        self,
        layers: int,
        width: int,
        base_tokens: int = 10,
        frequency_tokens: int = 6,
        texture_tokens: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(layers, width, base_tokens, frequency_tokens, texture_tokens) <= 0:
            raise ValueError("all prompt dimensions must be positive")
        self.layers = int(layers)
        self.width = int(width)
        self.base_tokens = int(base_tokens)
        self.frequency_tokens = int(frequency_tokens)
        self.texture_tokens = int(texture_tokens)
        self.base = nn.Parameter(torch.empty(layers, base_tokens, width))
        self.frequency = nn.Parameter(torch.empty(layers, frequency_tokens, width))
        self.texture = nn.Parameter(torch.empty(layers, texture_tokens, width))
        bound = (3.0 / width) ** 0.5
        for parameter in (self.base, self.frequency, self.texture):
            nn.init.uniform_(parameter, -bound, bound)
        self.frequency_conditioner = MultiScaleInstantaneousFrequency(width)
        self.texture_conditioner = SignalTextureConditioner(width)
        self.dropout = nn.Dropout(dropout)

    @property
    def count(self) -> int:
        return self.base_tokens + self.frequency_tokens + self.texture_tokens

    def texture_statistics(
        self, raw_sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.texture_conditioner.statistics(raw_sequence)

    def forward(
        self,
        layer: int,
        batch: int,
        tkeo: torch.Tensor,
        texture_gate: torch.Tensor,
    ) -> torch.Tensor:
        if not 0 <= layer < self.layers:
            raise IndexError(layer)
        frequency = self.frequency_conditioner(
            self.frequency[layer][None].expand(batch, -1, -1)
        )
        base = self.base[layer][None].expand(batch, -1, -1)
        texture = self.texture_conditioner(
            self.texture[layer][None].expand(batch, -1, -1),
            tkeo,
            texture_gate,
        )
        return self.dropout(torch.cat((frequency, base, texture), dim=1))


class MultiStreamPromptedWav2Vec2Encoder(nn.Module):
    """Frozen XLS-R with fresh signal-conditioned prompts at every layer."""

    def __init__(
        self,
        backbone: nn.Module,
        base_tokens: int = 10,
        frequency_tokens: int = 6,
        texture_tokens: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.prompts = DeepMultiStreamPrompts(
            len(backbone.encoder.layers),
            int(backbone.config.hidden_size),
            base_tokens,
            frequency_tokens,
            texture_tokens,
            dropout,
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.config.apply_spec_augment = False
        self.backbone.config.layerdrop = 0.0
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
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
        tkeo, texture_gate = self.prompts.texture_statistics(hidden)
        count = self.prompts.count
        for index, layer in enumerate(encoder.layers):
            prompt = self.prompts(
                index, len(hidden), tkeo, texture_gate
            ).to(hidden.dtype)
            combined = layer(
                torch.cat((prompt, hidden), dim=1),
                attention_mask=None,
                output_attentions=False,
            )[0]
            # Deep prompts condition every transformer layer but are not audio
            # observations.  Match the released MixFake implementation by
            # stripping them before the next layer *and* before the backend;
            # retaining the final prompt tokens lets AASIST exploit a learned
            # constant shortcut and changes its expected time-axis geometry.
            hidden = combined[:, count:]
        return encoder.layer_norm(hidden)


class MultiStreamSpectraMultitask(WPTSpectraMultitask):
    """Spectra-AASIST backend driven by multi-stream prompted XLS-R."""

    def __init__(
        self,
        spectra_model: nn.Module,
        base_tokens: int = 10,
        frequency_tokens: int = 6,
        texture_tokens: int = 6,
        prompt_dropout: float = 0.1,
        temperature: float = 5.0,
    ) -> None:
        # The parent initializes the released Spectra bridge/backend and the
        # three competition task heads. Replace only its prompt encoder.
        super().__init__(spectra_model, temperature=temperature)
        self.prompt_encoder = MultiStreamPromptedWav2Vec2Encoder(
            spectra_model.ssl_encoder.model,
            base_tokens=base_tokens,
            frequency_tokens=frequency_tokens,
            texture_tokens=texture_tokens,
            dropout=prompt_dropout,
        )


def warm_start_shared_backend(
    model: nn.Module, source_state: dict[str, torch.Tensor],
) -> dict[str, int]:
    """Load only architecture-compatible non-prompt weights.

    Wavelet and multi-stream prompt tensors intentionally have different
    semantics and shapes.  The Spectra bridge, AASIST backend, and competition
    task head are shared exactly, so those weights can provide a channel-robust
    initialization without pretending the two prompt families are compatible.
    """
    target_state = model.state_dict()
    transferable: dict[str, torch.Tensor] = {}
    skipped_prompt = 0
    skipped_incompatible = 0
    for name, value in source_state.items():
        if name.startswith((
            "prompt_encoder.prompts.", "prompt_encoder.backbone.",
        )):
            skipped_prompt += 1
            continue
        if name not in target_state or target_state[name].shape != value.shape:
            skipped_incompatible += 1
            continue
        transferable[name] = value

    required = ("bridge.", "aasist.", "task_head.")
    absent = [
        prefix for prefix in required
        if not any(name.startswith(prefix) for name in transferable)
    ]
    if absent:
        raise ValueError(
            "warm-start checkpoint lacks shared modules: " + ", ".join(absent)
        )
    model.load_state_dict(transferable, strict=False)
    return {
        "tensors_loaded": len(transferable),
        "parameters_loaded": sum(value.numel() for value in transferable.values()),
        "prompt_tensors_skipped": skipped_prompt,
        "incompatible_tensors_skipped": skipped_incompatible,
    }
