#!/usr/bin/env python3
"""Run the exact submitted v50+v57 anchor on a labeled manifest audit.

This is an offline diagnostic wrapper: it stages only the selected rows in an
isolated DACON-shaped directory, imports the frozen submitted entrypoint, and
preserves its five output columns unchanged for later candidate fusion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

import librosa
import pandas as pd
import soundfile as sf


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "reports/music_only_submission_v82/full_package_smoke"

# Load the audit transform under a private module name.  Importing repository
# modules such as ``pipeline`` or ``telephone_channel`` by their public names
# here would populate sys.modules before the frozen package inserts its own
# vendored source directory, silently breaking the exact-anchor guarantee.
_channel_spec = importlib.util.spec_from_file_location(
    "audit_telephone_channel_v95", ROOT / "src/telephone_channel.py"
)
if _channel_spec is None or _channel_spec.loader is None:
    raise ImportError("cannot load audit telephone channel")
_channel_module = importlib.util.module_from_spec(_channel_spec)
_channel_spec.loader.exec_module(_channel_module)
apply_channel = _channel_module.apply_channel


CHANNELS = (
    "clean", "paired_fast", "g711_ulaw", "g722_wb", "opus_nb_8k",
    "transcode_g711_opus",
)
FAST_CHANNELS = ("resample8k", "mulaw_numpy", "fft_narrowband", "lowpass_5k")
COLUMNS = (
    "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")


def load_audio(path: Path):
    waveform, _ = librosa.load(path, sr=16_000, mono=True, dtype="float32")
    return waveform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channel", choices=CHANNELS, default="clean")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if "locked" in " ".join(map(str, (args.manifest, args.output))).lower():
        raise ValueError("locked data are forbidden during candidate selection")
    frame = pd.read_csv(args.manifest, dtype={"ID": str})
    if frame.ID.duplicated().any():
        raise ValueError("manifest IDs are not unique")
    required = {"ID", "ARCHIVE_MEMBER", "VOICE_FAKE", "MUSIC_FAKE", "FILE_FAKE"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"manifest misses {sorted(missing)}")
    runner = args.output / "runner"
    test_dir = runner / "data/test"
    test_dir.mkdir(parents=True)
    (runner / "output").mkdir()
    os.symlink(PACKAGE.joinpath("model").resolve(), runner / "model")
    ffmpeg = None
    if args.channel in {
        "g711_ulaw", "g722_wb", "opus_nb_8k", "transcode_g711_opus",
    }:
        executable = shutil.which("ffmpeg")
        fallback = Path(sys.executable).parent / "ffmpeg"
        if executable is None and fallback.is_file():
            executable = str(fallback)
        if executable is None:
            raise FileNotFoundError("ffmpeg is missing")
        ffmpeg = Path(executable)
    staged = []
    for row in frame.itertuples(index=False):
        source = (args.audio_root / str(row.ARCHIVE_MEMBER)).resolve()
        target = test_dir / f"{row.ID}.wav"
        if args.channel == "clean":
            os.symlink(source, target)
        else:
            key = stable_key(str(row.ID))
            channel = (
                FAST_CHANNELS[key % len(FAST_CHANNELS)]
                if args.channel == "paired_fast" else args.channel
            )
            waveform = apply_channel(
                load_audio(source), channel, ffmpeg=ffmpeg, key=key % (2 ** 31)
            )
            sf.write(target, waveform, 16_000, subtype="PCM_16")
        staged.append(target)
    sample = runner / "data/sample_submission.csv"
    with sample.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for identity in frame.ID:
            writer.writerow({"ID": identity, **{key: .5 for key in COLUMNS[1:]}})
    frozen_script = PACKAGE / "script_anchor.py"
    frozen = {
        "schema": "frozen_anchor_manifest_v95", "channel": args.channel,
        "rows": len(frame), "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "script": str(frozen_script.resolve()), "script_sha256": sha256_file(frozen_script),
        "sample_sha256": sha256_file(sample), "locked_scores_read": False,
    }
    args.output.mkdir(exist_ok=True)
    (args.output / "frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
    spec = importlib.util.spec_from_file_location("frozen_anchor_v95", frozen_script)
    if spec is None or spec.loader is None:
        raise ImportError(frozen_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BASE_DIR = runner
    sys.argv = [str(frozen_script)]
    started = time.monotonic()
    module.main()
    prediction = runner / "output/submission.csv"
    output = pd.read_csv(prediction, dtype={"ID": str})
    if len(output) != len(frame) or set(output.ID) != set(frame.ID):
        raise ValueError("invalid anchor output")
    report = {
        "status": "complete", "channel": args.channel, "rows": len(output),
        "seconds": time.monotonic() - started,
        "predictions_sha256": sha256_file(prediction),
        "frozen_sha256": sha256_file(args.output / "frozen.json"),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
