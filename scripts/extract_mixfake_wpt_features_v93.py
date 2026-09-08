#!/usr/bin/env python3
"""Extract frozen current-anchor WPT window embeddings for MixFake v93.

The extractor is deliberately label-blind during neural inference.  Labels and
source metadata are copied into the final cache only to make the subsequent
readout training auditable.  Rows are assigned to shards by manifest order so
that independently produced GPU shards concatenate deterministically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeline import load_audio  # noqa: E402
from telephone_channel import apply_channel  # noqa: E402
from full_coverage_wpt import resolve_ffmpeg  # noqa: E402
from wpt_spectra_inference import (  # noqa: E402
    _preemphasis, fixed_windows, load_wpt_model,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(item: str) -> int:
    return int.from_bytes(hashlib.sha256(item.encode()).digest()[:8], "little")


def resolve_audio(root: Path, member: str) -> Path:
    path = root / str(member)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path,
                        default=ROOT / "models/external/spectra_aasist")
    parser.add_argument("--checkpoint", type=Path, default=(
        ROOT / "v50_v57m_challenger_v2/model/spectra-aasist/"
        "wpt_spectra_multitask.pt"
    ))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--channel", default="clean",
        choices=("clean", "resample8k", "mulaw_numpy", "fft_narrowband",
                 "lowpass_5k", "g711_ulaw", "g722_wb", "opus_nb_8k",
                 "transcode_g711_opus", "paired_fast"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not (0 <= args.shard_index < args.shard_count):
        raise ValueError("invalid shard")
    if args.views <= 0 or args.batch_size <= 0:
        raise ValueError("views and batch size must be positive")

    manifest = pd.read_csv(args.manifest, dtype={"ID": str})
    required = {
        "ID", "ARCHIVE_MEMBER", "FILE_FAKE", "VOICE_FAKE", "MUSIC_FAKE",
        "COMPONENT_CASE", "VOICE_SOURCE_ID", "MUSIC_GROUP_ID",
        "MUSIC_GENERATOR",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"manifest misses {sorted(missing)}")
    if manifest.ID.duplicated().any():
        raise ValueError("manifest IDs are not unique")
    positions = np.arange(len(manifest), dtype=np.int64)
    positions = positions[positions % args.shard_count == args.shard_index]
    frame = manifest.iloc[positions].reset_index(drop=True)
    paths = [resolve_audio(args.audio_root, value) for value in frame.ARCHIVE_MEMBER]

    device = torch.device(args.device)
    model, config = load_wpt_model(args.model_dir, args.checkpoint, device)
    window = int(config["window"])
    ffmpeg_channels = {
        "g711_ulaw", "g722_wb", "opus_nb_8k", "transcode_g711_opus",
    }
    ffmpeg = resolve_ffmpeg() if args.channel in ffmpeg_channels else None
    fast_channels = ("resample8k", "mulaw_numpy", "fft_narrowband", "lowpass_5k")
    all_embeddings: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    started = time.monotonic()
    for offset in range(0, len(paths), args.batch_size):
        batch_paths = paths[offset:offset + args.batch_size]
        arrays = []
        for path, item in zip(batch_paths, frame.ID.iloc[offset:offset + args.batch_size]):
            audio = load_audio(path)
            if args.channel != "clean":
                channel = (
                    fast_channels[stable_key(str(item)) % len(fast_channels)]
                    if args.channel == "paired_fast" else args.channel
                )
                audio = apply_channel(
                    audio, channel, ffmpeg=ffmpeg,
                    key=stable_key(str(item)) % (2 ** 31),
                )
            arrays.append(fixed_windows(audio, window, args.views))
        tensor = _preemphasis(torch.from_numpy(np.stack(arrays)).to(device))
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits, embeddings = model.forward_windows_with_embedding(tensor)
        all_embeddings.append(embeddings.float().cpu().numpy().astype(np.float16))
        all_logits.append(logits.float().cpu().numpy().astype(np.float16))
        if offset == 0 or offset + len(batch_paths) == len(paths) or offset % 600 == 0:
            print(json.dumps({
                "shard": args.shard_index, "channel": args.channel,
                "done": offset + len(batch_paths), "total": len(paths),
                "seconds": time.monotonic() - started,
            }), flush=True)

    embeddings = np.concatenate(all_embeddings) if all_embeddings else np.empty(
        (0, args.views, 160), dtype=np.float16
    )
    logits = np.concatenate(all_logits) if all_logits else np.empty(
        (0, args.views, 3), dtype=np.float16
    )
    if len(embeddings) != len(frame) or logits.shape[:2] != embeddings.shape[:2]:
        raise RuntimeError("feature shape mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    np.savez_compressed(
        temporary,
        positions=positions,
        ids=frame.ID.astype(str).to_numpy(dtype=str),
        embeddings=embeddings,
        base_logits=logits,
        file_fake=frame.FILE_FAKE.to_numpy(np.int8),
        voice_fake=frame.VOICE_FAKE.to_numpy(np.int8),
        music_fake=frame.MUSIC_FAKE.to_numpy(np.int8),
        component_case=frame.COMPONENT_CASE.astype(str).to_numpy(dtype=str),
        voice_source_id=frame.VOICE_SOURCE_ID.astype(str).to_numpy(dtype=str),
        music_group_id=frame.MUSIC_GROUP_ID.astype(str).to_numpy(dtype=str),
        music_generator=frame.MUSIC_GENERATOR.astype(str).to_numpy(dtype=str),
        channel=np.asarray(args.channel),
        manifest_sha256=np.asarray(sha256_file(args.manifest)),
        checkpoint_sha256=np.asarray(sha256_file(args.checkpoint)),
        views=np.asarray(args.views, dtype=np.int64),
        window=np.asarray(window, dtype=np.int64),
    )
    # numpy appends .npz when the filename itself does not end in .npz.
    written = temporary if temporary.is_file() else Path(str(temporary) + ".npz")
    written.replace(args.output)
    print(json.dumps({
        "status": "complete", "rows": len(frame), "shape": list(embeddings.shape),
        "output": str(args.output), "sha256": sha256_file(args.output),
        "seconds": time.monotonic() - started,
    }), flush=True)


if __name__ == "__main__":
    main()
