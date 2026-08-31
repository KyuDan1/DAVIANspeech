"""Add conservative SPEAR cross-component evidence to anchor predictions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm

from pipeline import find_audio_files, load_audio, order_by_submission
from spear_detector import SpearCrossComponentDetector, fuse_cross_component_scores


def apply_fusion(
    test_dir: Path, submission_path: Path, spear_dir: Path,
    music_head: Path, joint_head: Path, device: str = "cuda",
    weight: float = 0.10,
) -> None:
    """Rewrite an anchor submission with one original-audio SPEAR pass."""
    with submission_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    required = {
        "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    }
    if not required.issubset(columns):
        raise ValueError(f"Submission is missing {sorted(required - set(columns))}")

    audio_files = order_by_submission(find_audio_files(test_dir), rows)
    detector = SpearCrossComponentDetector(
        spear_dir, music_head, joint_head, device=device
    )
    for row, path in zip(rows, tqdm(audio_files, desc="SPEAR cross-component")):
        expert = detector.component_probabilities(load_audio(path))
        file_score, music_score = fuse_cross_component_scores(
            float(row["FILE_FAKE_PROB"]), float(row["MUSIC_FAKE_PROB"]),
            expert["file"], expert["music_expert"], weight=weight,
        )
        row["FILE_FAKE_PROB"] = round(file_score, 10)
        row["MUSIC_FAKE_PROB"] = round(music_score, 10)
    del detector
    torch.cuda.empty_cache()

    temporary = submission_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--spear-dir", type=Path, required=True)
    parser.add_argument("--music-head", type=Path, required=True)
    parser.add_argument("--joint-head", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weight", type=float, default=0.10)
    args = parser.parse_args()
    apply_fusion(
        args.test_dir, args.submission, args.spear_dir, args.music_head,
        args.joint_head, device=args.device, weight=args.weight,
    )


if __name__ == "__main__":
    main()
