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
    temporal_bin_output_path: Path | None = None,
    temporal_bin_checkpoint_path: Path | None = None,
    additional_temporal_bin_requests: list[tuple[Path, Path]] | None = None,
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
    if (temporal_bin_output_path is None) != (temporal_bin_checkpoint_path is None):
        raise ValueError("temporal-bin output and checkpoint must be provided together")
    temporal_configuration = None
    if temporal_bin_checkpoint_path is not None:
        checkpoint = torch.load(
            temporal_bin_checkpoint_path, map_location="cpu", weights_only=False
        )
        bins = checkpoint.get("spear_bins", checkpoint.get("bins"))
        if bins is None:
            raise ValueError("temporal-bin checkpoint does not declare its bin count")
        temporal_configuration = (
            np.asarray(
                checkpoint.get("spear_projection", checkpoint.get("projection")),
                dtype=np.float32,
            ),
            tuple(int(value) for value in checkpoint.get(
                "spear_layers", checkpoint.get("layers")
            )),
            int(bins),
        )
    additional_temporal_bin_requests = additional_temporal_bin_requests or []
    additional_outputs, additional_configurations = [], []
    for output_path, checkpoint_path in additional_temporal_bin_requests:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        bins = checkpoint.get("spear_bins", checkpoint.get("bins"))
        projection = checkpoint.get(
            "spear_projection", checkpoint.get("projection")
        )
        layers = checkpoint.get("spear_layers", checkpoint.get("layers"))
        if bins is None or projection is None or layers is None:
            raise ValueError("additional temporal-bin checkpoint is incomplete")
        additional_outputs.append(Path(output_path))
        additional_configurations.append((
            np.asarray(projection, dtype=np.float32),
            tuple(int(value) for value in layers), int(bins),
        ))
    if additional_configurations and temporal_configuration is None:
        raise ValueError(
            "additional temporal-bin exports require the primary export"
        )
    statistic_ids, statistics, statistic_masks = [], [], []
    temporal_features = [[] for _ in (
        ([temporal_configuration] if temporal_configuration is not None else [])
        + additional_configurations
    )]
    temporal_masks = [[] for _ in temporal_features]
    pending_ids, pending_audio = [], []

    def flush_statistics() -> None:
        if not pending_audio:
            return
        configurations = (
            ([] if temporal_configuration is None else [temporal_configuration])
            + additional_configurations
        )
        if not configurations:
            values = detector.dual_domain_statistics_batch(pending_audio)
            temporal_values = None
        else:
            values, temporal_values = (
                detector.dual_domain_statistics_and_multiple_temporal_bins_batch(
                    pending_audio, configurations
                )
            )
        for item, (matrix, mask) in zip(pending_ids, values):
            statistic_ids.append(item)
            statistics.append(matrix)
            statistic_masks.append(mask)
        if temporal_values is not None:
            for output_index, output_values in enumerate(temporal_values):
                for matrix, mask in output_values:
                    temporal_features[output_index].append(matrix)
                    temporal_masks[output_index].append(mask)
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
    output_paths = (
        ([] if temporal_bin_output_path is None else [Path(temporal_bin_output_path)])
        + additional_outputs
    )
    configurations = (
        ([] if temporal_configuration is None else [temporal_configuration])
        + additional_configurations
    )
    for output_path, configuration, features, masks in zip(
        output_paths, configurations, temporal_features, temporal_masks
    ):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            ids=np.asarray(statistic_ids),
            features=np.stack(features), mask=np.stack(masks),
            projection=configuration[0],
            layers=np.asarray(configuration[1], dtype=np.int16),
            bins=np.asarray(configuration[2]),
        )
