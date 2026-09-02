"""Score EAT voice/music presence without source separation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eat_presence import EatPresence  # noqa: E402
from pipeline import find_audio_files, load_audio  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/eat-base-as2m")
    parser.add_argument("--labels-dir", type=Path, default=ROOT / "models/panns")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    files = find_audio_files(args.test_dir)[args.shard_index::args.num_shards]
    detector = EatPresence(args.model_dir, args.labels_dir, device=args.device)
    rows = []
    for path in tqdm(files, desc="EAT presence"):
        voice, music = detector.predict(load_audio(path))
        rows.append({"ID": path.stem, "EAT_VOICE_PRESENT_PROB": voice,
                     "EAT_MUSIC_PRESENT_PROB": music})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} scores to {args.output}")


if __name__ == "__main__":
    main()
