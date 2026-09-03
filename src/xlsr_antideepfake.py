"""XLS-R-2B AntiDeepfake detector, served through HuggingFace transformers.

The published checkpoint (nii-yamagishilab/xls-r-2b-anti-deepfake) stores a
fairseq ``Wav2Vec2Model`` under the ``m_ssl.model.`` prefix plus a single
``proj_fc`` Linear(1920, 2) head.  fairseq 0.12.2 only supports Python 3.9 and
does not run against torch 2.6, so instead of importing it we remap the
fairseq parameter names onto ``transformers.Wav2Vec2Model``.  The two encoders
are architecturally identical once fairseq's ``layer_norm_first=True`` is
matched with HF's ``do_stable_layer_norm=True``.

Reference: https://github.com/nii-yamagishilab/AntiDeepfake (models/W2V.py,
models/W2V_configs.py, dataio/dataio.py).
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import Wav2Vec2Config, Wav2Vec2Model

# Logit order is fixed by the training protocol: dataio.py assigns
# ``logit = 1`` to real audio and ``logit = 0`` to fake, and evaluation.py
# scores with ``softmax(...)[:, 1]`` as the real-class probability.
FAKE_INDEX = 0
REAL_INDEX = 1

# fairseq Wav2Vec2Config for 'xlsr_2b', copied from models/W2V_configs.py.
XLSR_2B_HF_CONFIG = dict(
    hidden_size=1920,
    num_hidden_layers=48,
    num_attention_heads=16,
    intermediate_size=7680,
    hidden_act="gelu",
    feat_extract_activation="gelu",
    # fairseq extractor_mode='layer_norm' + conv_bias=True
    feat_extract_norm="layer",
    conv_bias=True,
    conv_dim=(512,) * 7,
    conv_kernel=(10, 3, 3, 3, 3, 2, 2),
    conv_stride=(5, 2, 2, 2, 2, 2, 2),
    num_conv_pos_embeddings=128,
    num_conv_pos_embedding_groups=16,
    # fairseq layer_norm_first=True -> final LayerNorm after the encoder stack
    do_stable_layer_norm=True,
    layer_norm_eps=1e-5,
    # Inference only: silence every stochastic path.
    hidden_dropout=0.0,
    attention_dropout=0.0,
    activation_dropout=0.0,
    feat_proj_dropout=0.0,
    layerdrop=0.0,
    apply_spec_augment=False,
)


def _fairseq_to_hf_key(key: str) -> str | None:
    """Map one ``m_ssl.model.*`` fairseq name onto its Wav2Vec2Model name."""
    name = key[len("m_ssl.model."):]

    # Objectives used only for pre-training; unused with features_only=True.
    if name.startswith(("final_proj", "project_q", "quantizer")):
        return None

    if name == "mask_emb":
        return "masked_spec_embed"

    # Conv frontend: (0)=conv, (2.1)=LayerNorm inside the TransposeLast block.
    m = re.fullmatch(r"feature_extractor\.conv_layers\.(\d+)\.0\.(weight|bias)", name)
    if m:
        return f"feature_extractor.conv_layers.{m[1]}.conv.{m[2]}"
    m = re.fullmatch(r"feature_extractor\.conv_layers\.(\d+)\.2\.1\.(weight|bias)", name)
    if m:
        return f"feature_extractor.conv_layers.{m[1]}.layer_norm.{m[2]}"

    # fairseq applies `self.layer_norm` to conv features before the projection,
    # which is exactly HF's feature_projection.layer_norm.
    m = re.fullmatch(r"layer_norm\.(weight|bias)", name)
    if m:
        return f"feature_projection.layer_norm.{m[1]}"
    m = re.fullmatch(r"post_extract_proj\.(weight|bias)", name)
    if m:
        return f"feature_projection.projection.{m[1]}"

    # Positional conv. weight_g/weight_v are renamed by torch's parametrization
    # API; the caller fixes that up against the real state_dict.
    m = re.fullmatch(r"encoder\.pos_conv\.0\.(bias|weight_g|weight_v)", name)
    if m:
        suffix = {
            "bias": "bias",
            "weight_g": "parametrizations.weight.original0",
            "weight_v": "parametrizations.weight.original1",
        }[m[1]]
        return f"encoder.pos_conv_embed.conv.{suffix}"

    m = re.fullmatch(r"encoder\.layer_norm\.(weight|bias)", name)
    if m:
        return f"encoder.layer_norm.{m[1]}"

    m = re.fullmatch(r"encoder\.layers\.(\d+)\.(.+)", name)
    if m:
        idx, rest = m[1], m[2]
        rest = re.sub(r"^self_attn\.", "attention.", rest)
        rest = re.sub(r"^self_attn_layer_norm\.", "layer_norm.", rest)
        rest = re.sub(r"^fc1\.", "feed_forward.intermediate_dense.", rest)
        rest = re.sub(r"^fc2\.", "feed_forward.output_dense.", rest)
        return f"encoder.layers.{idx}.{rest}"

    raise KeyError(f"Unmapped fairseq parameter: {key}")


class XlsrAntiDeepfake(nn.Module):
    """AntiDeepfake W2V ``Model``: SSL encoder -> mean pool -> Linear(1920, 2)."""

    SAMPLE_RATE = 16_000

    def __init__(self) -> None:
        super().__init__()
        self.ssl = Wav2Vec2Model(Wav2Vec2Config(**XLSR_2B_HF_CONFIG))
        self.proj_fc = nn.Linear(1920, 2)

    @classmethod
    def from_checkpoint(cls, model_dir, device="cuda", dtype=torch.float32):
        model = cls()
        model_dir = Path(model_dir)
        single = model_dir / "model.safetensors"
        checkpoint_files = [single] if single.is_file() else sorted(
            model_dir.glob("model-*-of-*.safetensors")
        )
        if not checkpoint_files:
            raise FileNotFoundError(f"no XLS-R safetensors checkpoint in {model_dir}")
        raw = {}
        for checkpoint_file in checkpoint_files:
            shard = load_file(checkpoint_file)
            duplicate = raw.keys() & shard.keys()
            if duplicate:
                raise ValueError(f"duplicate XLS-R keys across shards: {sorted(duplicate)[:5]}")
            raw.update(shard)

        target = model.ssl.state_dict()
        converted, unmapped = {}, []
        for key, tensor in raw.items():
            if key.startswith("proj_fc."):
                continue
            hf_key = _fairseq_to_hf_key(key)
            if hf_key is None:
                continue
            if hf_key not in target:
                # Older torch keeps the plain weight_norm names.
                legacy = hf_key.replace("parametrizations.weight.original0", "weight_g")
                legacy = legacy.replace("parametrizations.weight.original1", "weight_v")
                if legacy in target:
                    hf_key = legacy
                else:
                    unmapped.append((key, hf_key))
                    continue
            if tuple(tensor.shape) != tuple(target[hf_key].shape):
                raise ValueError(
                    f"Shape mismatch {key} -> {hf_key}: "
                    f"{tuple(tensor.shape)} vs {tuple(target[hf_key].shape)}"
                )
            converted[hf_key] = tensor

        if unmapped:
            raise KeyError(f"Keys with no destination in Wav2Vec2Model: {unmapped[:5]}")

        missing, unexpected = model.ssl.load_state_dict(converted, strict=False)
        # masked_spec_embed is only read when SpecAugment is on, which it is not.
        missing = [k for k in missing if k != "masked_spec_embed"]
        if missing or unexpected:
            raise RuntimeError(f"Incomplete load. missing={missing} unexpected={unexpected}")

        model.proj_fc.weight.data.copy_(raw["proj_fc.weight"])
        model.proj_fc.bias.data.copy_(raw["proj_fc.bias"])
        return model.to(device=device, dtype=dtype).eval()

    @staticmethod
    def normalize(waveform: torch.Tensor) -> torch.Tensor:
        """Zero-mean/unit-variance over the whole utterance.

        Mirrors ``F.layer_norm(wav, wav.shape)`` in AntiDeepfake's dataio.py,
        applied per example rather than per batch.
        """
        return F.layer_norm(waveform, waveform.shape[-1:])

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (B, T) already normalized -> logits (B, 2)."""
        pooled = self.embedding(waveform)
        return self.proj_fc(pooled)

    def embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return the paper's mean-pooled 1920-D AntiDeepfake representation."""
        hidden = self.ssl(waveform).last_hidden_state       # (B, frames, 1920)
        return hidden.mean(dim=1)                            # AdaptiveAvgPool1d(1)

    @torch.inference_mode()
    def fake_probability(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (B, T) raw audio -> P(fake) per item, as float32."""
        logits = self(self.normalize(waveform))
        return torch.softmax(logits.float(), dim=-1)[:, FAKE_INDEX]
