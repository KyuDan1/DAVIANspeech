"""Small separation-free low-band magnitude/phase Music MIL research model."""
import math

import torch
from torch import nn

from .full_coverage_wpt import masked_lme


class LowbandMusic(nn.Module):
    def __init__(self, phase: bool = False):
        super().__init__()
        if not isinstance(phase, bool):
            raise ValueError('phase must be a boolean')
        self.phase = phase
        self.register_buffer('hann', torch.hann_window(512))
        layers = []
        incoming = 3
        for outgoing in [16, 32, 64, 128]:
            layers.extend([nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1, bias=False),
                           nn.GroupNorm(4, outgoing), nn.GELU()])
            incoming = outgoing
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(256, 1)

    def features(self, windows):
        if windows.ndim != 2 or windows.shape[-1] < 512:
            raise ValueError('windows must be [batch,samples>=512]')
        # DSP is float32 even under mixed precision; no input reconstruction.
        with torch.autocast(device_type=windows.device.type, enabled=False):
            spectrum = torch.stft(windows.float(), n_fft=512, hop_length=160,
                win_length=512, window=self.hann.float(), center=False, return_complex=True)
            spectrum = spectrum[:, 8:113]  # 16kHz: 250 through 3500 Hz, inclusive.
            power = spectrum.abs().square()
            relative = power / power.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
            magnitude = relative.clamp_min(1e-8).log() / math.log(1e8)
            real, imag = torch.zeros_like(magnitude), torch.zeros_like(magnitude)
            if self.phase:
                cross = spectrum[:, :, 1:] * spectrum[:, :, :-1].conj()
                # Mask inaudible/zero-energy phases using a window-local relative floor.
                usable = (relative[:, :, 1:] > 1e-6) & (relative[:, :, :-1] > 1e-6)
                unit = cross / cross.abs().clamp_min(1e-12)
                real[:, :, 1:] = unit.real * usable
                imag[:, :, 1:] = unit.imag * usable
            return torch.stack((magnitude, real, imag), dim=1)

    def forward_windows(self, windows):
        encoded = self.encoder(self.features(windows))
        summary = torch.cat((encoded.mean(dim=(-2, -1)), encoded.amax(dim=(-2, -1))), dim=1)
        return self.head(summary)

    def forward(self, windows, mask, temperature=2., chunk_size=32):
        if windows.ndim != 3 or windows.shape[:2] != mask.shape or mask.dtype != torch.bool:
            raise ValueError('windows [files,views,samples] and bool mask required')
        if chunk_size <= 0 or not mask.any(dim=1).all():
            raise ValueError('positive chunk size and nonempty bags required')
        valid = windows[mask]
        logits = torch.cat([self.forward_windows(part) for part in valid.split(chunk_size)])
        padded = logits.new_zeros((*mask.shape, 1))
        padded[mask] = logits
        return masked_lme(padded, mask, temperature).squeeze(-1)
