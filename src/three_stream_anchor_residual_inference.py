"""Offline inference for the three-stream exact-anchor residual head."""

from __future__ import annotations

import csv
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import torch

try:
    from .three_stream_anchor_residual import ThreeStreamAnchorResidualHead
except ImportError:  # Offline submission imports from model/src on sys.path.
    from three_stream_anchor_residual import ThreeStreamAnchorResidualHead


AUTHENTICITY_COLUMNS = (
    "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "FILE_FAKE_PROB",
)
PRESENCE_COLUMNS = ("VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB")
TASK_MODES = ("full", "voice", "music", "file")
TASK_OUTPUT_INDICES = {
    "full": (0, 1, 2), "voice": (0,), "music": (1,), "file": (2,),
}


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1 - 1e-6)
    return (np.log(clipped) - np.log1p(-clipped)).astype(np.float32)


def _metadata_equal(
    archive: np.lib.npyio.NpzFile,
    archive_key: str,
    expected,
    label: str,
) -> None:
    if archive_key not in archive or not np.array_equal(
        archive[archive_key], np.asarray(expected)
    ):
        raise ValueError(f"{label} cache metadata {archive_key} differs from checkpoint")


def _ids(archive: np.lib.npyio.NpzFile, label: str) -> np.ndarray:
    if "ids" not in archive:
        raise ValueError(f"{label} cache has no IDs")
    values = archive["ids"].astype(str)
    if values.ndim != 1 or len(set(values.tolist())) != len(values):
        raise ValueError(f"{label} cache IDs must be one-dimensional and unique")
    return values


def apply_task_mode(
    anchor_logits: torch.Tensor,
    full_logits: torch.Tensor,
    residuals: torch.Tensor,
    task_mode: str,
) -> torch.Tensor:
    """Return full fusion or a strict one-output residual intervention."""
    if task_mode not in TASK_MODES:
        raise ValueError(f"task_mode must be one of {TASK_MODES}")
    if task_mode == "full":
        return full_logits
    result = anchor_logits.clone()
    if task_mode == "voice":
        result[:, 0] += residuals[:, 0]
    elif task_mode == "music":
        result[:, 1] += residuals[:, 1]
    else:
        result[:, 2] += 0.7 * residuals[:, 2]
    return result


@torch.inference_mode()
def apply_three_stream_anchor_residual(
    submission_path: Path,
    eat_path: Path,
    spear_path: Path,
    xlsr_path: Path,
    checkpoint_path: Path | Sequence[Path],
    device: str = "cuda",
    batch_size: int = 64,
    task_mode: str = "full",
) -> None:
    """Apply learned authenticity residuals and preserve anchor presence."""
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if task_mode not in TASK_MODES:
        raise ValueError(f"task_mode must be one of {TASK_MODES}")
    checkpoint_paths = (
        [checkpoint_path]
        if isinstance(checkpoint_path, (str, Path))
        else list(checkpoint_path)
    )
    if not checkpoint_paths:
        raise ValueError("at least one three-stream checkpoint is required")
    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in checkpoint_paths
    ]
    for checkpoint in checkpoints:
        if checkpoint.get("model_type") != "three_stream_anchor_residual_component_query_v1":
            raise ValueError("not a three-stream anchor-residual checkpoint")
    required_checkpoint = {
        "model", "config", "feature_metadata", "spear_projection",
        "spear_layers", "spear_bins",
    }
    for checkpoint in checkpoints:
        missing_checkpoint = required_checkpoint.difference(checkpoint)
        if missing_checkpoint:
            raise ValueError(f"checkpoint misses fields: {sorted(missing_checkpoint)}")
    checkpoint = checkpoints[0]
    for member in checkpoints[1:]:
        for key in ("spear_projection", "spear_layers", "spear_bins"):
            if not np.array_equal(member[key], checkpoint[key]):
                raise ValueError(f"ensemble checkpoint {key} differs")
        if set(member["feature_metadata"]) != set(checkpoint["feature_metadata"]):
            raise ValueError("ensemble feature metadata keys differ")
        for key, value in checkpoint["feature_metadata"].items():
            if not np.array_equal(member["feature_metadata"][key], value):
                raise ValueError(f"ensemble feature metadata {key} differs")

    eat = np.load(eat_path, allow_pickle=False)
    spear = np.load(spear_path, allow_pickle=False)
    xlsr = np.load(xlsr_path, allow_pickle=False)
    try:
        metadata = checkpoint["feature_metadata"]
        _metadata_equal(eat, "projection", metadata["eat_projection"], "EAT")
        _metadata_equal(eat, "layers", metadata["eat_layers"], "EAT")
        _metadata_equal(spear, "projection", checkpoint["spear_projection"], "SPEAR")
        _metadata_equal(spear, "layers", checkpoint["spear_layers"], "SPEAR")
        _metadata_equal(spear, "bins", checkpoint["spear_bins"], "SPEAR")
        _metadata_equal(xlsr, "window", metadata["xlsr_window"], "XLS-R")
        _metadata_equal(
            xlsr, "sample_rate", metadata["xlsr_sample_rate"], "XLS-R"
        )
        eat_ids, spear_ids, xlsr_ids = (
            _ids(eat, "EAT"), _ids(spear, "SPEAR"), _ids(xlsr, "XLS-R")
        )
        if set(eat_ids) != set(spear_ids) or set(eat_ids) != set(xlsr_ids):
            raise ValueError("three-stream cache ID sets differ")
        spear_lookup = {item: index for index, item in enumerate(spear_ids)}
        xlsr_lookup = {item: index for index, item in enumerate(xlsr_ids)}
        spear_order = np.asarray([spear_lookup[item] for item in eat_ids], np.int64)
        xlsr_order = np.asarray([xlsr_lookup[item] for item in eat_ids], np.int64)

        with submission_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [])
            rows = list(reader)
        required_columns = {"ID", *AUTHENTICITY_COLUMNS, *PRESENCE_COLUMNS}
        if not required_columns.issubset(columns):
            raise ValueError("submission misses required probability columns")
        if len(rows) != len(eat_ids) or len({row["ID"] for row in rows}) != len(rows):
            raise ValueError("submission/cache row count or submission IDs are invalid")
        by_id = {row["ID"]: row for row in rows}
        if set(by_id) != set(eat_ids):
            raise ValueError("submission and three-stream cache IDs differ")

        output_indices = TASK_OUTPUT_INDICES[task_mode]
        selected_columns = tuple(AUTHENTICITY_COLUMNS[index] for index in output_indices)
        protected_columns = tuple(column for column in columns if column not in selected_columns)
        original_protected = {
            row["ID"]: tuple(row[column] for column in protected_columns)
            for row in rows
        }
        original_selected = {
            row["ID"]: tuple(row[column] for column in selected_columns)
            for row in rows
        }

        anchor_probability = np.asarray([
            [float(by_id[item][column]) for column in AUTHENTICITY_COLUMNS]
            for item in eat_ids
        ], dtype=np.float64)
        presence_probability = np.asarray([
            [float(by_id[item][column]) for column in PRESENCE_COLUMNS]
            for item in eat_ids
        ], dtype=np.float64)
        if (
            not np.isfinite(anchor_probability).all()
            or not np.isfinite(presence_probability).all()
            or np.any((anchor_probability < 0) | (anchor_probability > 1))
            or np.any((presence_probability < 0) | (presence_probability > 1))
        ):
            raise ValueError("submission probabilities must be finite and in [0,1]")

        for member in checkpoints:
            if xlsr["embeddings"].shape[1] > int(
                member["config"]["maximum_xlsr_windows"]
            ):
                raise ValueError("XLS-R cache exceeds checkpoint positional capacity")
        target = torch.device(device)
        models = []
        for member in checkpoints:
            model = ThreeStreamAnchorResidualHead(**member["config"]).to(target)
            model.load_state_dict(member["model"], strict=True)
            model.eval()
            models.append(model)
        outputs = []
        anchor_logits = _logit(anchor_probability)
        presence_logits = _logit(presence_probability)
        for offset in range(0, len(eat_ids), batch_size):
            end = min(offset + batch_size, len(eat_ids))
            so = spear_order[offset:end]
            xo = xlsr_order[offset:end]
            arguments = (
                torch.from_numpy(eat["temporal"][offset:end]).to(target),
                torch.from_numpy(eat["spectral"][offset:end]).to(target),
                torch.from_numpy(eat["view_mask"][offset:end]).to(target),
                torch.from_numpy(spear["features"][so]).to(target),
                torch.from_numpy(spear["mask"][so]).to(target),
                torch.from_numpy(anchor_logits[offset:end]).to(target),
                torch.from_numpy(presence_logits[offset:end]).to(target),
                torch.from_numpy(xlsr["embeddings"][xo]).to(target),
                torch.from_numpy(xlsr["mask"][xo]).to(target),
            )
            member_logits = []
            for model in models:
                full, unchanged_presence, residuals = model(*arguments)
                if not torch.equal(unchanged_presence, arguments[6]):
                    raise RuntimeError("three-stream head changed presence logits")
                member_logits.append(apply_task_mode(
                    arguments[5], full, residuals, task_mode,
                ))
            final = torch.stack(member_logits).mean(dim=0)
            outputs.append(final.sigmoid().float().cpu().numpy())
        probabilities = np.concatenate(outputs)
        if not np.isfinite(probabilities).all():
            raise ValueError("three-stream inference produced non-finite values")
        for item, values in zip(eat_ids, probabilities):
            row = by_id[item]
            for index in output_indices:
                row[AUTHENTICITY_COLUMNS[index]] = round(
                    float(values[index]), 10
                )
        for row in rows:
            if tuple(row[column] for column in protected_columns) != original_protected[row["ID"]]:
                raise RuntimeError("three-stream inference changed a protected output column")
        changed_selected_rows = sum(
            tuple(str(row[column]) for column in selected_columns)
            != original_selected[row["ID"]]
            for row in rows
        )
        print(
            "THREE_STREAM_OUTPUT_AUDIT "
            f"task_mode={task_mode} protected_columns={','.join(protected_columns)} "
            f"protected_bit_exact=1 changed_selected_rows={changed_selected_rows}/{len(rows)}"
        )
        temporary = submission_path.with_suffix(".three_stream.tmp.csv")
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(submission_path)
    finally:
        eat.close()
        spear.close()
        xlsr.close()
