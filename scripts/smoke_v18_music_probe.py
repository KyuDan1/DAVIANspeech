#!/usr/bin/env python3
"""Run exact-v18 and its Music probe offline on exactly three unlabeled files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

from build_v18_music_probe import (
    BASE_ARCHIVE_SHA256,
    sha256_file,
    validate_probe_archive,
    verify_base_archive,
    verify_prediction_pair,
)


def run_entrypoint(package: Path, *, cuda_device: str) -> float:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": cuda_device,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "NO_PROXY": "*",
            "no_proxy": "*",
        }
    )
    started = time.monotonic()
    subprocess.run(
        [sys.executable, "script.py"],
        cwd=package,
        env=environment,
        check=True,
    )
    return time.monotonic() - started


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-zip", type=Path, required=True)
    parser.add_argument("--probe-zip", type=Path, required=True)
    parser.add_argument("--audio", type=Path, action="append", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--cuda-device", default="0")
    args = parser.parse_args()
    if len(args.audio) != 3:
        parser.error("exactly three --audio arguments are required")
    if any("codec_mixed_blind_v8" in str(path.resolve()) for path in args.audio):
        parser.error("blind v8 audio is forbidden in this smoke test")
    if any(not path.is_file() for path in args.audio):
        parser.error("all three smoke audio files must exist")

    base_zip = args.base_zip.resolve()
    probe_zip = args.probe_zip.resolve()
    verify_base_archive(base_zip)
    validate_probe_archive(base_zip, probe_zip, verify_crc=True)
    args.work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".v18-music-probe-smoke-", dir=args.work_root.resolve()
    ) as temporary_name:
        package = Path(temporary_name) / "package"
        with zipfile.ZipFile(probe_zip) as handle:
            handle.extractall(package)
        patched_script = (package / "script.py").read_bytes()
        with zipfile.ZipFile(base_zip) as handle:
            anchor_script = handle.read("script.py")

        test_dir = package / "data" / "test"
        test_dir.mkdir(parents=True)
        identifiers = []
        for index, source in enumerate(args.audio):
            identifier = f"offline_smoke_{index:02d}"
            identifiers.append(identifier)
            destination = test_dir / f"{identifier}{source.suffix.lower()}"
            destination.symlink_to(source.resolve())
        sample = package / "data" / "sample_submission.csv"
        with sample.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                [
                    "ID",
                    "FILE_FAKE_PROB",
                    "VOICE_FAKE_PROB",
                    "MUSIC_FAKE_PROB",
                    "VOICE_PRESENT_PROB",
                    "MUSIC_PRESENT_PROB",
                ]
            )
            for identifier in identifiers:
                writer.writerow([identifier, 0.5, 0.5, 0.5, 0.5, 0.5])

        (package / "script.py").write_bytes(anchor_script)
        anchor_seconds = run_entrypoint(package, cuda_device=args.cuda_device)
        anchor_output = package / "anchor_submission.csv"
        shutil.copy2(package / "output" / "submission.csv", anchor_output)
        (package / "script.py").write_bytes(patched_script)
        probe_seconds = run_entrypoint(package, cuda_device=args.cuda_device)
        probe_output = package / "output" / "submission.csv"
        comparison = verify_prediction_pair(anchor_output, probe_output)
        report = {
            "base_sha256": BASE_ARCHIVE_SHA256,
            "probe_sha256": sha256_file(probe_zip),
            "audio_count": 3,
            "anchor_seconds": round(anchor_seconds, 3),
            "probe_seconds": round(probe_seconds, 3),
            "comparison": comparison,
            "offline_environment": True,
        }
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
