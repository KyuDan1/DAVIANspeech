"""Cache separator stems for controlled raw-versus-separated diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeline import AUDIO_SR, find_audio_files  # noqa: E402
from separation import build_separator  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--htdemucs-repo", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    files = find_audio_files(args.test_dir)[args.shard_index::args.num_shards]
    voice_dir, music_dir = args.output_dir / "voice", args.output_dir / "music"
    voice_dir.mkdir(parents=True, exist_ok=True)
    music_dir.mkdir(parents=True, exist_ok=True)
    separator = build_separator(
        "htdemucs", device=args.device, repo=args.htdemucs_repo
    )
    for path in tqdm(files, desc="separation stems"):
        voice, music = separator.separate(path)
        sf.write(voice_dir / f"{path.stem}.flac", voice, AUDIO_SR)
        sf.write(music_dir / f"{path.stem}.flac", music, AUDIO_SR)


if __name__ == "__main__":
    main()
