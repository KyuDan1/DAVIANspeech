#!/usr/bin/env python3
"""Run a predeclared three-stream experiment matrix without score-driven edits."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts" / "train_three_stream_anchor_residual.py"
CONFIRMATION = "RUN_PREDECLARED_THREE_STREAM_MATRIX"
PARAMETER_ORDER = (
    "width", "heads", "depth", "task_interaction_depth",
    "maximum_xlsr_windows", "dropout",
    "residual_limits", "task_weights", "residual_weight", "ranking_weight",
    "ranking_margin", "quartet_invariance_weight",
    "quartet_latent_invariance_weight", "auc_ranking_weight",
    "auc_ranking_margin", "logit_consistency_weight",
    "latent_consistency_weight", "group_dro_weight",
    "group_dro_temperature", "selection_task", "epochs", "batch_size", "eval_batch_size",
    "learning_rate", "weight_decay", "eval_every", "patience",
    "generator_aware", "train_task_interaction_only", "initial_consistency_weight",
)
CONTROL_KEYS = frozenset((
    "train_set", "xlsr", "seed", "initial_checkpoint",
    "train_identity_exclusions", "selected_split_only",
))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_variant_specs(
    matrix_path: Path, selected: list[str] | None = None,
) -> tuple[dict, list[dict]]:
    matrix = yaml.safe_load(matrix_path.read_text("utf-8")) or {}
    if not isinstance(matrix, dict) or not isinstance(matrix.get("variants"), dict):
        raise ValueError("matrix must contain a variants mapping")
    variants = matrix["variants"]
    names = list(variants) if selected is None else selected
    if not names or len(names) != len(set(names)):
        raise ValueError("selected variants must be nonempty and unique")
    unknown = sorted(set(names).difference(variants))
    if unknown:
        raise ValueError(f"unknown matrix variants: {unknown}")
    fixed = matrix.get("fixed_training", {}) or {}
    if not isinstance(fixed, dict):
        raise ValueError("fixed_training must be a mapping")
    seed = matrix.get("seed")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("matrix seed must be a nonnegative integer")

    specs = []
    for name in names:
        variant = variants[name]
        if not isinstance(variant, dict):
            raise ValueError(f"variant {name!r} must be a mapping")
        merged = {**fixed, **variant}
        train_key = merged.pop("train_set", matrix.get("train_set"))
        if not isinstance(train_key, str) or not train_key:
            raise ValueError(f"variant {name!r} has no train_set reference")
        train_datasets = matrix.get(train_key)
        if (
            not isinstance(train_datasets, list)
            or not train_datasets
            or not all(isinstance(value, str) and value for value in train_datasets)
            or len(train_datasets) != len(set(train_datasets))
        ):
            raise ValueError(f"matrix train set {train_key!r} is invalid")
        development_key = merged.pop(
            "development_set", matrix.get("development_set")
        )
        if development_key is None:
            development_datasets = None
        else:
            if not isinstance(development_key, str) or not development_key:
                raise ValueError(
                    f"variant {name!r} development_set must be a nonempty key"
                )
            development_datasets = matrix.get(development_key)
            if (
                not isinstance(development_datasets, list)
                or not development_datasets
                or not all(
                    isinstance(value, str) and value
                    for value in development_datasets
                )
                or len(development_datasets) != len(set(development_datasets))
            ):
                raise ValueError(
                    f"matrix development set {development_key!r} is invalid"
                )
        xlsr = merged.pop("xlsr", True)
        if not isinstance(xlsr, bool):
            raise ValueError(f"variant {name!r} xlsr must be boolean")
        variant_seed = merged.pop("seed", seed)
        if not isinstance(variant_seed, int) or variant_seed < 0:
            raise ValueError(f"variant {name!r} seed must be a nonnegative integer")
        initial_checkpoint = merged.pop("initial_checkpoint", None)
        if initial_checkpoint is not None and (
            not isinstance(initial_checkpoint, str) or not initial_checkpoint
        ):
            raise ValueError(
                f"variant {name!r} initial_checkpoint must be a nonempty path"
            )
        train_identity_exclusions = merged.pop(
            "train_identity_exclusions", matrix.get("train_identity_exclusions")
        )
        if train_identity_exclusions is not None and (
            not isinstance(train_identity_exclusions, str)
            or not train_identity_exclusions
        ):
            raise ValueError("train_identity_exclusions must be a nonempty path")
        selected_split_only = merged.pop(
            "selected_split_only", matrix.get("selected_split_only", False)
        )
        if not isinstance(selected_split_only, bool):
            raise ValueError("selected_split_only must be boolean")
        unknown_parameters = sorted(set(merged).difference(PARAMETER_ORDER))
        if unknown_parameters:
            raise ValueError(
                f"variant {name!r} has unsupported parameters: {unknown_parameters}"
            )
        specs.append({
            "name": name,
            "train_set": train_key,
            "train_datasets": train_datasets,
            "development_set": development_key,
            "development_datasets": development_datasets,
            "xlsr": xlsr,
            "seed": variant_seed,
            "initial_checkpoint": initial_checkpoint,
            "train_identity_exclusions": train_identity_exclusions,
            "selected_split_only": selected_split_only,
            "parameters": merged,
        })
    return matrix, specs


def build_command(
    python_bin: str,
    spec: dict,
    cache_dir: Path,
    output_dir: Path,
    partition_config: Path,
) -> list[str]:
    command = [
        python_bin, str(TRAINER),
        "--partition-config", str(partition_config),
        "--eat-cache-root", str(cache_dir / "eat"),
        "--spear-cache-root", str(cache_dir / "spear"),
        "--anchor-root", str(cache_dir / "anchor"),
        "--output-dir", str(output_dir),
        "--device", "cuda",
        "--seed", str(spec["seed"]),
        "--train-datasets", *spec["train_datasets"],
    ]
    if spec.get("development_datasets") is not None:
        command.extend((
            "--dev-datasets", *spec["development_datasets"],
        ))
    if spec["xlsr"]:
        command.extend(("--xlsr-cache-root", str(cache_dir / "xlsr")))
    if spec.get("initial_checkpoint") is not None:
        initial_checkpoint = (ROOT / spec["initial_checkpoint"]).resolve()
        if not initial_checkpoint.is_file():
            raise FileNotFoundError(initial_checkpoint)
        command.extend(("--initial-checkpoint", str(initial_checkpoint)))
    if spec.get("train_identity_exclusions") is not None:
        exclusion_path = (ROOT / spec["train_identity_exclusions"]).resolve()
        if not exclusion_path.is_file():
            raise FileNotFoundError(exclusion_path)
        command.extend(("--train-identity-exclusions", str(exclusion_path)))
    if spec.get("selected_split_only"):
        command.append("--selected-split-only")
    parameters = spec["parameters"]
    for name in PARAMETER_ORDER:
        if name not in parameters:
            continue
        value = parameters[name]
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        if value is None:
            raise ValueError(f"parameter {name!r} must be numeric or a numeric list")
        command.append(flag)
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(f"parameter {name!r} cannot be empty")
            command.extend(map(str, value))
        else:
            command.append(str(value))
    return command


def run_matrix(
    matrix_path: Path,
    cache_dir: Path,
    output_root: Path,
    partition_config: Path,
    gpus: list[str],
    python_bin: str,
    selected: list[str] | None,
    confirmation: str,
) -> None:
    if confirmation != CONFIRMATION:
        raise ValueError(f"matrix launch requires --confirm {CONFIRMATION}")
    matrix_path = matrix_path.resolve(strict=True)
    cache_dir = cache_dir.resolve(strict=True)
    partition_config = partition_config.resolve(strict=True)
    output_root = output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite matrix output: {output_root}")
    if not gpus or any(not gpu or "," in gpu for gpu in gpus):
        raise ValueError("at least one valid GPU token is required")
    if len(gpus) != len(set(gpus)):
        raise ValueError("GPU tokens must be unique")
    required_cache = ("anchor", "eat", "spear", "xlsr", "merge_provenance.json")
    missing = [name for name in required_cache if not (cache_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"merged cache is incomplete: {missing}")
    matrix, specs = load_variant_specs(matrix_path, selected)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()
    record = {
        "matrix": str(matrix_path),
        "matrix_sha256": sha256_file(matrix_path),
        "schema_version": matrix.get("schema_version"),
        "cache_dir": str(cache_dir),
        "cache_merge_provenance_sha256": sha256_file(
            cache_dir / "merge_provenance.json"
        ),
        "partition_config": str(partition_config),
        "partition_config_sha256": sha256_file(partition_config),
        "trainer_sha256": sha256_file(TRAINER),
        "python": python_bin,
        "gpus": gpus,
        "variants": [],
    }
    for spec in specs:
        output_dir = output_root / spec["name"]
        command = build_command(
            python_bin, spec, cache_dir, output_dir, partition_config,
        )
        record["variants"].append({
            **spec,
            "output_dir": str(output_dir),
            "command": command,
        })
    (output_root / "matrix_manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    queue = deque(record["variants"])
    free_gpus = deque(gpus)
    active: dict[str, tuple[subprocess.Popen, object, dict, float]] = {}
    results = []
    try:
        while queue or active:
            while queue and free_gpus:
                gpu = free_gpus.popleft()
                item = queue.popleft()
                log_path = output_root / f"{item['name']}.log"
                handle = log_path.open("wb")
                env = os.environ.copy()
                env.update({
                    "CUDA_VISIBLE_DEVICES": gpu,
                    "PYTHONNOUSERSITE": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_DATASETS_OFFLINE": "1",
                })
                process = subprocess.Popen(
                    item["command"], cwd=ROOT, env=env,
                    stdout=handle, stderr=subprocess.STDOUT,
                )
                active[gpu] = (process, handle, item, time.monotonic())
                print(f"launched {item['name']} on GPU {gpu}", flush=True)
            finished = []
            for gpu, (process, handle, item, started) in active.items():
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                results.append({
                    "name": item["name"], "gpu": gpu, "returncode": code,
                    "elapsed_seconds": time.monotonic() - started,
                })
                free_gpus.append(gpu)
                finished.append(gpu)
                print(f"finished {item['name']} on GPU {gpu}: {code}", flush=True)
            for gpu in finished:
                del active[gpu]
            if active and not finished:
                time.sleep(1)
    except BaseException:
        for process, handle, _, _ in active.values():
            process.terminate()
            handle.close()
        raise
    (output_root / "matrix_status.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    failures = [item for item in results if item["returncode"] != 0]
    if failures:
        raise RuntimeError(f"matrix variants failed: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--partition-config", type=Path,
        default=ROOT / "configs" / "data_partitions.yaml",
    )
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--variant", action="append")
    parser.add_argument("--confirm", required=True)
    args = parser.parse_args()
    run_matrix(
        args.matrix, args.cache_dir, args.output_root, args.partition_config,
        args.gpus.split(","), args.python, args.variant, args.confirm,
    )


if __name__ == "__main__":
    main()
