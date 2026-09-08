#!/usr/bin/env python3
"""Evaluate the public FakeMusicCaps EAT--AASIST checkpoint out of domain.

This is a development-only experiment.  It recreates DeepFense's deterministic
validation transform (four-second repeat padding) and retains three fixed crops
for long recordings.  The checkpoint includes the complete fine-tuned EAT
frontend, so no Hugging Face/network lookup and no base checkpoint load occurs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.metrics import roc_curve
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from eat_timm_compat import install_timm_compat  # noqa: E402
from pipeline import load_audio  # noqa: E402


SAMPLES = 64_000


def official_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2)


def fixed_starts(length: int) -> tuple[int, int, int]:
    maximum = max(0, length - SAMPLES)
    return 0, int(round(maximum / 2)), maximum


def crop_or_repeat(audio: np.ndarray, start: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if not len(audio):
        raise ValueError("empty audio")
    if len(audio) >= SAMPLES:
        return np.ascontiguousarray(audio[start : start + SAMPLES])
    repeats = int(math.ceil(SAMPLES / len(audio)))
    return np.tile(audio, repeats)[:SAMPLES].astype(np.float32, copy=False)


def exact_fbank(audio: np.ndarray) -> torch.Tensor:
    waveform = torch.from_numpy(audio).float()
    waveform = waveform - waveform.mean()
    return torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0), htk_compat=True, sample_frequency=16_000,
        use_energy=False, window_type="hanning", num_mel_bins=128,
        dither=0.0, frame_shift=10,
    )


class _EATModel(nn.Module):
    def __init__(self, core: nn.Module):
        super().__init__()
        self.model = core

    def extract_features(self, values: torch.Tensor) -> torch.Tensor:
        return self.model.extract_features(values)


class _Frontend(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model


class _Detector(nn.Module):
    def __init__(self, eat: nn.Module, backend: nn.Module, loss: nn.Module):
        super().__init__()
        self.frontend = _Frontend(eat)
        self.backend = backend
        self.losses = nn.ModuleList([loss])

    def forward(self, fbank: torch.Tensor) -> torch.Tensor:
        features = self.frontend.model.extract_features(fbank[:, None])
        embedding = self.backend(features)
        return self.losses[0].fc(embedding)


def load_detector(
    eat_dir: Path, checkpoint_path: Path, deepfense_repo: Path,
) -> nn.Module:
    sys.path.insert(0, str(deepfense_repo))
    install_timm_compat()
    package = "davianspeech_deepfense_eat_v103"
    spec = importlib.util.spec_from_file_location(
        package, eat_dir / "__init__.py",
        submodule_search_locations=[str(eat_dir)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    eat_spec = importlib.util.spec_from_file_location(
        f"{package}.eat_model", eat_dir / "eat_model.py",
    )
    eat_module = importlib.util.module_from_spec(eat_spec)
    sys.modules[eat_spec.name] = eat_module
    eat_spec.loader.exec_module(eat_module)
    config_dict = json.loads((eat_dir / "config.json").read_text())
    config_dict["model_variant"] = "pretrain"
    core = eat_module.EAT(SimpleNamespace(**config_dict))

    from deepfense.models.backends.aasist import AASIST
    from deepfense.models.losses.cross_entropy import CrossEntropy

    detector = _Detector(
        _EATModel(core), AASIST({}),
        CrossEntropy({"embedding_dim": 160, "n_classes": 2}),
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False,
    )
    detector.load_state_dict(checkpoint["model_state"], strict=True)
    return detector


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--eat-dir", type=Path, default=ROOT / "models/eat-large-as2m-v56",
    )
    parser.add_argument(
        "--deepfense-repo", type=Path, default=Path("/tmp/deepfense-framework"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")

    manifest = pd.read_csv(args.manifest, dtype={"ID": str})
    required = {"ID", "filepath", "target", "DATASET"}
    if missing := required.difference(manifest.columns):
        raise ValueError(f"missing columns: {sorted(missing)}")
    if manifest.ID.duplicated().any():
        raise ValueError("IDs must be unique")
    device = torch.device(args.device)
    model = load_detector(
        args.eat_dir, args.checkpoint, args.deepfense_repo,
    ).eval().to(device)

    rows: list[dict[str, object]] = []
    started = time.monotonic()
    for offset in range(0, len(manifest), args.batch_size):
        block = manifest.iloc[offset : offset + args.batch_size]
        features: list[torch.Tensor] = []
        metadata: list[tuple[str, int, str, int]] = []
        for row in block.itertuples(index=False):
            audio = load_audio(Path(row.filepath))
            for view, start in enumerate(fixed_starts(len(audio))):
                features.append(exact_fbank(crop_or_repeat(audio, start)))
                metadata.append((str(row.ID), int(row.target), str(row.DATASET), view))
        values = torch.stack(features).to(device)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(values)
        probability = logits.float().softmax(-1)[:, 0].cpu().numpy()
        margin = (logits[:, 0] - logits[:, 1]).float().cpu().numpy()
        for meta, score, logit in zip(metadata, probability, margin):
            rows.append({
                "ID": meta[0], "label": meta[1], "DATASET": meta[2],
                "view": meta[3], "fake_probability": float(score),
                "fake_margin": float(logit),
            })
        if offset == 0 or offset + len(block) >= len(manifest) or offset % 256 == 0:
            print(json.dumps({
                "done": min(offset + len(block), len(manifest)),
                "total": len(manifest), "seconds": time.monotonic() - started,
            }), flush=True)

    frame = pd.DataFrame(rows)
    wide = frame.pivot(index=["ID", "label", "DATASET"], columns="view",
                       values="fake_margin").reset_index()
    wide.columns = [
        "ID", "label", "DATASET", "margin_head", "margin_mid", "margin_tail",
    ]
    labels = wide.label.to_numpy(np.int64)
    metrics = {
        "n": len(wide),
        "eer_head": official_eer(labels, wide.margin_head),
        "eer_mid": official_eer(labels, wide.margin_mid),
        "eer_tail": official_eer(labels, wide.margin_tail),
        "eer_mean_margin": official_eer(
            labels, wide[["margin_head", "margin_mid", "margin_tail"]].mean(1),
        ),
        "by_dataset": {},
    }
    for dataset, block in wide.groupby("DATASET"):
        if block.label.nunique() == 2:
            metrics["by_dataset"][str(dataset)] = {
                "n": len(block),
                "eer_mean_margin": official_eer(
                    block.label.to_numpy(np.int64),
                    block[["margin_head", "margin_mid", "margin_tail"]].mean(1),
                ),
            }
    args.output.mkdir(parents=True)
    frame.to_csv(args.output / "view_scores.csv", index=False)
    wide.to_csv(args.output / "scores.csv", index=False)
    (args.output / "report.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
