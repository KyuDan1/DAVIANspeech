"""Dependency-free inference for the SONICS SpecTTTra-gamma-5s checkpoint.

This is the minimal MIT-licensed inference architecture needed by
``awsaf49/sonics-spectttra-gamma-5s``. It intentionally uses only torch and
torchaudio, both preinstalled by the competition grader.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram


class MeanStdNorm(nn.Module):
    def forward(self, x):
        mean = x.mean((1, 2), keepdim=True)
        std = x.reshape(x.size(0), -1).std(1, keepdim=True).unsqueeze(-1)
        return (x - mean) / (std + 1e-6)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio2melspec = MelSpectrogram(
            n_fft=2048, hop_length=512, win_length=2048, n_mels=128,
            sample_rate=16_000, f_min=20, f_max=8000, power=2,
        )
        self.amplitude_to_db = AmplitudeToDB(top_db=80)
        self.normalizer = MeanStdNorm()

    def forward(self, audio):
        with torch.autocast(device_type=audio.device.type, enabled=False):
            return self.normalizer(self.amplitude_to_db(self.audio2melspec(audio.float())))


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, dim, tokens):
        super().__init__()
        self.pe = nn.Parameter(torch.empty(1, tokens, dim))

    def forward(self, x):
        return x + self.pe


class Tokenizer1D(nn.Module):
    def __init__(self, input_dim, token_dim, clip_size, num_clips):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, token_dim, clip_size, stride=clip_size, bias=False)
        self.act = nn.GELU()
        self.pos_encoder = LearnedPositionalEncoding(token_dim, num_clips)
        self.norm_pre = nn.LayerNorm(token_dim, eps=1e-6)

    def forward(self, x):
        x = self.conv1d(x).transpose(1, 2)
        return self.norm_pre(self.pos_encoder(self.act(x)))


class STTokenizer(nn.Module):
    def __init__(self, dim=384, t_clip=7, f_clip=5):
        super().__init__()
        temporal_tokens = math.floor((128 - t_clip) / t_clip + 1)
        spectral_tokens = math.floor((128 - f_clip) / f_clip + 1)
        self.temporal_tokenizer = Tokenizer1D(128, dim, t_clip, temporal_tokens)
        self.spectral_tokenizer = Tokenizer1D(128, dim, f_clip, spectral_tokens)

    def forward(self, x):
        return torch.cat([
            self.temporal_tokenizer(x), self.spectral_tokenizer(x.permute(0, 2, 1))
        ], dim=1)


class Attention(nn.Module):
    def __init__(self, dim=384, heads=6):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x):
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj_drop(self.proj(x.transpose(1, 2).reshape(batch, tokens, dim)))


class Mlp(nn.Module):
    def __init__(self, dim=384, ratio=2.67):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(0.0)
        self.norm = nn.Identity()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(0.0)

    def forward(self, x):
        return self.drop2(self.fc2(self.norm(self.drop1(self.act(self.fc1(x))))))


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(384)
        self.attn = Attention()
        self.ls1 = nn.Identity()
        self.drop_path1 = nn.Identity()
        self.norm2 = nn.LayerNorm(384)
        self.mlp = Mlp()
        self.ls2 = nn.Identity()
        self.drop_path2 = nn.Identity()

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(12)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class SpecTTTra(nn.Module):
    def __init__(self):
        super().__init__()
        self.st_tokenizer = STTokenizer()
        self.pos_drop = nn.Dropout(0.1)
        self.transformer = Transformer()

    def forward(self, x):
        return self.transformer(self.pos_drop(self.st_tokenizer(x.squeeze(1))))


class SonicsMusicDetector(nn.Module):
    SAMPLE_RATE = 16_000
    WINDOW = 80_000

    def __init__(self):
        super().__init__()
        self.ft_extractor = FeatureExtractor()
        self.encoder = SpecTTTra()
        self.classifier = nn.Linear(384, 1)

    @classmethod
    def from_checkpoint(cls, model_dir, device="cuda"):
        model = cls()
        state = torch.load(
            Path(model_dir) / "pytorch_model.bin", map_location="cpu", weights_only=True
        )
        model.load_state_dict(state, strict=True)
        return model.to(device).eval()

    def forward(self, audio):
        spec = self.ft_extractor(audio)
        spec = F.interpolate(spec.unsqueeze(1), size=(128, 128), mode="bilinear")
        return self.classifier(self.encoder(spec).mean(dim=1)).flatten()

    @torch.inference_mode()
    def fake_probability(self, audio: np.ndarray, device="cuda") -> float:
        if len(audio) < self.WINDOW:
            audio = np.pad(audio, (0, self.WINDOW - len(audio)))
        starts = list(range(0, max(1, len(audio) - self.WINDOW + 1), self.WINDOW))
        tail = len(audio) - self.WINDOW
        if starts[-1] != tail:
            starts.append(tail)
        windows = np.stack([audio[start:start + self.WINDOW] for start in starts])
        windows /= np.maximum(windows.std(axis=1, keepdims=True), 1e-6)
        logits = self(torch.from_numpy(windows).to(device))
        return float(torch.sigmoid(logits).mean())
