#!/usr/bin/env python3
"""Render one deterministic real telephony codec partner per v93 mixture."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import librosa
import pandas as pd
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from telephone_channel import apply_channel  # noqa: E402


CHANNELS = ("g711_ulaw", "g722_wb", "opus_nb_8k", "transcode_g711_opus")


def stable_key(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_audio(path: Path):
    waveform, _ = librosa.load(path, sr=16_000, mono=True, dtype="float32")
    return waveform


def render_one(job: tuple[str, str, str, str, str]) -> dict:
    identity, source_text, output_text, channel, ffmpeg_text = job
    source, output = Path(source_text), Path(output_text)
    key = stable_key(identity)
    waveform = apply_channel(
        load_audio(source), channel, ffmpeg=Path(ffmpeg_text), key=key % (2 ** 31)
    )
    sf.write(output, waveform, 16_000, subtype="PCM_16")
    return {
        "ID": identity, "CHANNEL": channel, "PATH": str(output),
        "SHA256": sha256_file(output), "SAMPLES": len(waveform),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional deterministic stratum-balanced subset size.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if "locked" in " ".join(map(str, (args.manifest, args.output))).lower():
        raise ValueError("locked data are forbidden during training augmentation")
    frame = pd.read_csv(args.manifest, dtype={"ID": str})
    if frame.ID.duplicated().any() or "ARCHIVE_MEMBER" not in frame:
        raise ValueError("manifest must contain unique ID and ARCHIVE_MEMBER")
    if args.limit is not None:
        if not 0 < args.limit <= len(frame):
            raise ValueError("limit must lie within the manifest")
        required = {"COMPONENT_CASE", "MUSIC_GENERATOR"}
        if missing := required.difference(frame.columns):
            raise ValueError(f"balanced subset misses {sorted(missing)}")
        working = frame.copy()
        working["_STRATUM"] = (
            working.COMPONENT_CASE.astype(str) + "|"
            + working.MUSIC_GENERATOR.astype(str)
        )
        working["_ORDER"] = working.ID.map(stable_key)
        groups = [
            group.sort_values("_ORDER").index.tolist()
            for _, group in working.groupby("_STRATUM", sort=True)
        ]
        chosen = []
        offset = 0
        while len(chosen) < args.limit:
            progress = False
            for group in groups:
                if offset < len(group) and len(chosen) < args.limit:
                    chosen.append(group[offset]); progress = True
            if not progress:
                raise RuntimeError("could not construct balanced subset")
            offset += 1
        frame = frame.loc[chosen].reset_index(drop=True)
    ffmpeg = shutil.which("ffmpeg")
    fallback = Path(sys.executable).parent / "ffmpeg"
    if ffmpeg is None and fallback.is_file():
        ffmpeg = str(fallback)
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is missing")
    args.output.mkdir(parents=True)
    jobs = []
    for row in frame.itertuples(index=False):
        key = stable_key(str(row.ID))
        channel = CHANNELS[key % len(CHANNELS)]
        jobs.append((
            str(row.ID), str(args.audio_root / str(row.ARCHIVE_MEMBER)),
            str(args.output / f"{row.ID}.wav"), channel, ffmpeg,
        ))
    started = time.monotonic()
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, record in enumerate(executor.map(render_one, jobs), 1):
            records.append(record)
            if index == 1 or index % 500 == 0 or index == len(jobs):
                print(json.dumps({"done": index, "total": len(jobs),
                                  "seconds": time.monotonic() - started}), flush=True)
    result = pd.DataFrame(records)
    if len(result) != len(frame) or set(result.ID) != set(frame.ID):
        raise RuntimeError("rendered row mismatch")
    result.to_csv(args.output / "render_manifest.csv", index=False)
    frame.to_csv(args.output / "selected_manifest.csv", index=False)
    report = {
        "status": "complete", "rows": len(result),
        "channel_counts": result.CHANNEL.value_counts().sort_index().to_dict(),
        "manifest": str(args.manifest), "manifest_sha256": sha256_file(args.manifest),
        "selected_manifest_sha256": sha256_file(args.output / "selected_manifest.csv"),
        "render_manifest_sha256": sha256_file(args.output / "render_manifest.csv"),
        "seconds": time.monotonic() - started, "locked_scores_read": False,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
