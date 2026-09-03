#!/usr/bin/env python3
"""Score an audio bank with the vendored Spectra-AASIST checkpoint.

The scorer is deterministic and internet-free.  It reports several temporal
aggregations so their fusion weight can be selected on development data while
the locked evaluation banks remain untouched.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import find_audio_files, load_audio  # noqa: E402


def load_model(model_dir: Path, device: torch.device) -> torch.nn.Module:
    spec = importlib.util.spec_from_file_location(
        "vendored_spectra_aasist", model_dir / "model.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Spectra-AASIST from {model_dir}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.SpectraAASIST()
    missing, unexpected = model.load_state_dict(
        load_file(model_dir / "model.safetensors"), strict=False
    )
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    return model.eval().to(device)


def fixed_windows(audio: np.ndarray, length: int, count: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if not len(audio):
        audio = np.zeros(1, dtype=np.float32)
    if len(audio) < length:
        audio = np.tile(audio, math.ceil(length / len(audio)))[:length]
        return np.repeat(audio[None], count, axis=0)
    if count == 1:
        starts = np.asarray([(len(audio) - length) // 2])
    else:
        starts = np.rint(np.linspace(0, len(audio) - length, count)).astype(int)
    return np.stack([audio[start:start + length] for start in starts])


def preemphasis(waveforms: torch.Tensor, coefficient: float = 0.97) -> torch.Tensor:
    output = waveforms.clone()
    output[:, 1:] = waveforms[:, 1:] - coefficient * waveforms[:, :-1]
    return output


def logmeanexp(values: np.ndarray, temperature: float = 5.0) -> float:
    scaled = temperature * np.asarray(values, dtype=np.float64)
    maximum = float(scaled.max())
    return float(
        (maximum + np.log(np.exp(scaled - maximum).mean())) / temperature
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--audio-subdir", default="audio",
        help="Audio directory below --data-dir (use 'voice' for cached stems).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-dir", type=Path,
        default=ROOT / "models/external/spectra_aasist",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--windows", type=int, default=1)
    parser.add_argument("--window-samples", type=int, default=64_600)
    parser.add_argument("--file-batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.windows < 1:
        parser.error("--windows must be positive")
    if args.file_batch_size < 1:
        parser.error("--file-batch-size must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    device = torch.device(args.device)
    model = load_model(args.model_dir, device)
    paths = find_audio_files(args.data_dir / args.audio_subdir)
    records = []
    batches = range(0, len(paths), args.file_batch_size)
    for offset in tqdm(batches, desc="Spectra-AASIST"):
        batch_paths = paths[offset:offset + args.file_batch_size]
        batch_windows = np.concatenate([
            fixed_windows(load_audio(path), args.window_samples, args.windows)
            for path in batch_paths
        ])
        windows = torch.from_numpy(batch_windows).to(device)
        windows = preemphasis(windows)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(windows).float().reshape(
                len(batch_paths), args.windows, 2
            )
        margins = (logits[:, :, 0] - logits[:, :, 1]).cpu().numpy()
        for path, values in zip(batch_paths, margins):
            records.append({
                "ID": path.stem,
                "FAKE_LOGIT_MEAN": float(values.mean()),
                "FAKE_LOGIT_MAX": float(values.max()),
                "FAKE_LOGIT_LME5": logmeanexp(values),
                "WINDOWS": len(values),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output, index=False)
    print(f"Saved {len(records)} scores to {args.output}")


if __name__ == "__main__":
    main()
