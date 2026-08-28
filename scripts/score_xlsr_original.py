"""Cache XLS-R anti-deepfake scores on original, unseparated audio.

This isolates the effect of source separation: the same detector, windowing,
and pooling used by the submission pipeline are applied directly to the input
mixture.  The output intentionally contains only ID and XLSR_ORIGINAL_PROB so
it can be joined with any existing diagnostic submission without recomputing
presence or stems.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import fake_probability, find_audio_files, load_audio  # noqa: E402
from xlsr_antideepfake import XlsrAntiDeepfake  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--xlsr-dir", type=Path,
                        default=ROOT / "models/xls-r-2b-anti-deepfake")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window", type=int, default=64_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must satisfy 0 <= index < --num-shards")

    files = find_audio_files(args.test_dir)
    files = files[args.shard_index::args.num_shards]
    device = torch.device(args.device)
    detector = XlsrAntiDeepfake.from_checkpoint(args.xlsr_dir, device=device)

    rows = []
    for path in tqdm(files, desc=f"raw-xlsr-{args.shard_index}"):
        probability = fake_probability(
            detector, load_audio(path), device, args.window, args.batch_size
        )
        rows.append({"ID": path.stem, "XLSR_ORIGINAL_PROB": round(probability, 10)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "XLSR_ORIGINAL_PROB"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} scores to {args.output}")


if __name__ == "__main__":
    main()
