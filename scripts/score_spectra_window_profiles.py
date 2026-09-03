#!/usr/bin/env python3
"""Cache sliding Spectra-AASIST vocal-stem margins for temporal audits."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import find_audio_files, load_audio  # noqa: E402
from presence import extract_segment, segment_starts  # noqa: E402
from spectra_aasist_detector import _load_model, _preemphasis  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-dir", type=Path,
        default=ROOT / "models" / "external" / "spectra_aasist",
    )
    parser.add_argument("--window", type=int, default=64_600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    files = find_audio_files(args.audio_dir)
    device = torch.device(args.device)
    model = _load_model(args.model_dir, device)
    ids: list[str] = []
    offsets = [0]
    starts_flat: list[int] = []
    margins: list[float] = []
    durations: list[float] = []
    for path in tqdm(files, desc="Spectra sliding windows"):
        audio = load_audio(path)
        starts = segment_starts(audio.size, args.window)
        windows = np.stack([
            extract_segment(audio, start, args.window) for start in starts
        ])
        item_margins: list[float] = []
        for offset in range(0, len(windows), args.batch_size):
            waveform = torch.from_numpy(
                windows[offset:offset + args.batch_size]
            ).to(device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(_preemphasis(waveform)).float()
            item_margins.extend((logits[:, 0] - logits[:, 1]).cpu().tolist())
        ids.append(path.stem)
        starts_flat.extend(starts)
        margins.extend(item_margins)
        offsets.append(len(margins))
        durations.append(audio.size / 16_000)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ids=np.asarray(ids), offsets=np.asarray(offsets, dtype=np.int64),
        starts=np.asarray(starts_flat, dtype=np.int64),
        fake_margins=np.asarray(margins, dtype=np.float32),
        durations=np.asarray(durations, dtype=np.float32),
        window=np.asarray(args.window, dtype=np.int64),
    )
    print(
        f"Saved {len(ids)} files / {len(margins)} Spectra windows to "
        f"{args.output}", flush=True,
    )


if __name__ == "__main__":
    main()
