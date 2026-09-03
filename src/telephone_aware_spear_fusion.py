"""Apply conservative SPEAR fusion with a file-local narrowband gate."""

from __future__ import annotations

import csv
import gc
from pathlib import Path

import torch
from tqdm import tqdm

from pipeline import find_audio_files, load_audio, order_by_submission
from spear_detector import SpearCrossComponentDetector, fuse_cross_component_scores
from telephone_router import TelephoneRouter


def apply_telephone_aware_fusion(
    test_dir: Path,
    submission_path: Path,
    spear_dir: Path,
    music_head: Path,
    joint_head: Path,
    telephone_router: Path,
    device: str = "cuda",
    base_weight: float = 0.10,
    telephone_weight: float = 0.20,
) -> None:
    """Use the leaderboard-tested fusion, increasing it only for narrowband.

    Every decision is file-local.  The router never reads scores, statistics,
    or metadata from another evaluation file.
    """
    if not 0 <= base_weight <= telephone_weight <= 1:
        raise ValueError("Require 0 <= base_weight <= telephone_weight <= 1")
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
    router = TelephoneRouter(telephone_router)
    gc.collect()
    torch.cuda.empty_cache()
    detector = SpearCrossComponentDetector(
        spear_dir, music_head, joint_head, device=device
    )
    telephone_count = 0
    for row, path in zip(rows, tqdm(audio_files, desc="telephone-aware SPEAR")):
        audio = load_audio(path)
        is_telephone = router.is_narrowband(audio)
        telephone_count += int(is_telephone)
        weight = telephone_weight if is_telephone else base_weight
        expert = detector.component_probabilities(audio)
        file_score, music_score = fuse_cross_component_scores(
            float(row["FILE_FAKE_PROB"]), float(row["MUSIC_FAKE_PROB"]),
            expert["file"], expert["music_expert"], weight=weight,
        )
        row["FILE_FAKE_PROB"] = round(file_score, 10)
        row["MUSIC_FAKE_PROB"] = round(music_score, 10)
    print(f"telephone-aware SPEAR routed {telephone_count}/{len(rows)} files")
    del detector
    torch.cuda.empty_cache()

    temporary = submission_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)
