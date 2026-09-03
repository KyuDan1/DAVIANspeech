#!/usr/bin/env python3
"""Render deterministic codecs after a final audio mixture is assembled."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import soundfile as sf
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import find_audio_files, load_audio  # noqa: E402
from telephone_channel import apply_channel  # noqa: E402


def render_one(task: tuple[str, str, str, tuple[str, ...]]) -> str:
    input_path_s, output_root_s, ffmpeg_s, channels = task
    input_path = Path(input_path_s)
    output_root = Path(output_root_s)
    audio = load_audio(input_path)
    key = sum(input_path.stem.encode("utf-8"))
    for channel in channels:
        transformed = apply_channel(
            audio, channel, ffmpeg=Path(ffmpeg_s), key=key,
        )
        sf.write(
            output_root / channel / f"{input_path.stem}.flac",
            transformed, 16_000, subtype="PCM_16",
        )
    return input_path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--channels", nargs="+", required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("ffmpeg"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    files = find_audio_files(args.input_dir)
    for channel in args.channels:
        (args.output_dir / channel).mkdir(parents=True)
    tasks = [
        (str(path), str(args.output_dir), str(args.ffmpeg), tuple(args.channels))
        for path in files
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        completed = list(tqdm(
            executor.map(render_one, tasks), total=len(tasks),
            desc="channel variants",
        ))
    if len(completed) != len(files) or len(set(completed)) != len(files):
        raise RuntimeError("Channel rendering did not preserve unique audio IDs")
    print(
        f"Rendered {len(files)} files x {len(args.channels)} channels in "
        f"{args.output_dir}", flush=True,
    )


if __name__ == "__main__":
    main()
