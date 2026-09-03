"""Apply verified SPEAR fusion and optionally cache dual-domain statistics."""

from __future__ import annotations

import csv
import gc
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:  # package import in tests; flat import in the offline submission
    from .pipeline import find_audio_files, load_audio, order_by_submission
    from .spear_detector import SpearCrossComponentDetector, fuse_cross_component_scores
except ImportError:  # pragma: no cover - exercised by script.py
    from pipeline import find_audio_files, load_audio, order_by_submission
    from spear_detector import SpearCrossComponentDetector, fuse_cross_component_scores


def apply_fusion_with_stats(
    test_dir: Path, submission_path: Path, spear_dir: Path,
    music_head: Path, joint_head: Path, device: str = "cuda",
    weight: float = 0.10, statistics_output_path: Path | None = None,
) -> None:
    """Preserve the verified SPEAR scores while caching a second exact view pass."""
    with submission_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    audio_files = order_by_submission(find_audio_files(test_dir), rows)
    gc.collect()
    torch.cuda.empty_cache()
    detector = SpearCrossComponentDetector(
        spear_dir, music_head, joint_head, device=device
    )
    statistic_ids, statistics, statistic_masks = [], [], []
    pending_ids, pending_audio = [], []

    def flush_statistics() -> None:
        if not pending_audio:
            return
        values = detector.dual_domain_statistics_batch(pending_audio)
        for item, (matrix, mask) in zip(pending_ids, values):
            statistic_ids.append(item)
            statistics.append(matrix)
            statistic_masks.append(mask)
        pending_ids.clear()
        pending_audio.clear()

    for row, path in zip(rows, tqdm(audio_files, desc="SPEAR cross-component+stats")):
        audio = load_audio(path)
        expert = detector.component_probabilities(audio)
        file_score, music_score = fuse_cross_component_scores(
            float(row["FILE_FAKE_PROB"]), float(row["MUSIC_FAKE_PROB"]),
            expert["file"], expert["music_expert"], weight=weight,
        )
        row["FILE_FAKE_PROB"] = round(file_score, 10)
        row["MUSIC_FAKE_PROB"] = round(music_score, 10)
        if statistics_output_path is not None:
            pending_ids.append(path.stem)
            pending_audio.append(audio)
            if len(pending_audio) == 8:
                flush_statistics()
    flush_statistics()
    del detector
    gc.collect()
    torch.cuda.empty_cache()

    temporary = submission_path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(submission_path)
    if statistics_output_path is not None:
        statistics_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            statistics_output_path,
            ids=np.asarray(statistic_ids),
            statistics=np.stack(statistics),
            view_mask=np.stack(statistic_masks),
            stream=np.asarray("spear"), channel=np.asarray("clean"),
        )
