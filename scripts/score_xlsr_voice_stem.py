"""Score the HTDemucs vocal stem with the released AntiDeepfake head."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import fake_probability, find_audio_files  # noqa: E402
from separation import HTDemucsSeparator  # noqa: E402
from xlsr_antideepfake import XlsrAntiDeepfake  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xlsr-dir", type=Path,
                        default=ROOT / "models/xls-r-2b-anti-deepfake")
    parser.add_argument("--htdemucs-dir", type=Path,
                        default=ROOT / "models/htdemucs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window", type=int, default=64_000)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    files = find_audio_files(args.test_dir)[args.shard_index::args.num_shards]
    device = torch.device(args.device)
    separator = HTDemucsSeparator(device=args.device, repo=args.htdemucs_dir)
    detector = XlsrAntiDeepfake.from_checkpoint(args.xlsr_dir, device=device)
    rows = []
    for path in tqdm(files, desc=f"stem-xlsr-{args.shard_index}"):
        voice, _ = separator.separate(path)
        score = fake_probability(detector, voice, device, args.window, 1)
        rows.append({"ID": path.stem, "XLSR_VOICE_STEM_PROB": round(score, 10)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "XLSR_VOICE_STEM_PROB"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} scores to {args.output}")


if __name__ == "__main__":
    main()
