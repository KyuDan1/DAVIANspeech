#!/usr/bin/env python3
"""Prepare, run, and merge label-blind exact-v47 anchor caches safely.

The three phases are deliberately separate.  ``prepare`` performs no model
inference, ``launch`` is the only phase which starts GPU work, and ``merge``
accepts results only after revalidating every declared input and shard row.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_guard import (  # noqa: E402
    TRAIN_ROLES,
    assert_development_eval_separation,
    assert_no_locked_eval_leakage,
)


SCHEMA_VERSION = 2
XLSR_EMBEDDING_DIM = 1_920
AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}
PREDICTION_COLUMNS = (
    "FILE_FAKE_PROB",
    "VOICE_FAKE_PROB",
    "MUSIC_FAKE_PROB",
    "VOICE_PRESENT_PROB",
    "MUSIC_PRESENT_PROB",
)
ALLOWED_CACHE_ROLES = frozenset(("train", "development"))
FORBIDDEN_CACHE_ROLES = frozenset(
    ("locked_eval", "ood_holdout", "stress_eval", "retrospective_diagnostic", "invalid_eval")
)
LOCKED_CACHE_ROLES = frozenset(("locked_eval",))
LOCKED_FORBIDDEN_CACHE_ROLES = frozenset(
    (
        "train", "development", "router_train", "training_validation",
        "ood_holdout", "stress_eval", "retrospective_diagnostic", "invalid_eval",
    )
)
CACHE_POLICIES = {
    "train_development": (ALLOWED_CACHE_ROLES, FORBIDDEN_CACHE_ROLES),
    "one_shot_locked": (LOCKED_CACHE_ROLES, LOCKED_FORBIDDEN_CACHE_ROLES),
}
MAP_COLUMNS = (
    "COMBINED_ID",
    "DATASET_KEY",
    "ROLE",
    "ROW_INDEX",
    "ORIGINAL_ID",
    "AUDIO_FILENAME",
    "SOURCE_AUDIO",
    "AUDIO_SHA256",
)
NPZ_FIELDS = frozenset(("ids", "embeddings", "mask", "starts", "window", "sample_rate"))
EAT_FIELDS = frozenset(("ids", "temporal", "spectral", "view_mask", "projection", "layers"))
SPEAR_FIELDS = frozenset(("ids", "features", "mask", "projection", "layers", "bins"))
EXPORT_FUNCTIONS = (
    "fake_probability_and_embedding",
    "_xlsr_window_embedding_arrays",
    "_write_deterministic_npz",
    "save_xlsr_window_embeddings",
)


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_sha256(ids) -> str:
    digest = hashlib.sha256()
    for item_id in ids:
        encoded = str(item_id).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _load_partition_config(config_path: Path) -> tuple[dict, Path]:
    config_path = config_path.resolve(strict=True)
    config = yaml.safe_load(config_path.read_text("utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Partition config must be a mapping")
    config_dir = config_path.parent
    root = config_dir.parent if config_dir.name == "configs" else config_dir
    return config, root


def _declared_paths(config: dict, root: Path) -> dict[Path, set[str]]:
    result: dict[Path, set[str]] = {}
    for role, values in config.items():
        if values is None:
            values = []
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"Partition role {role!r} must be a list of paths")
        for value in values:
            result.setdefault((root / value).resolve(), set()).add(str(role))
    return result


def _run_data_guard(config_path: Path) -> None:
    """Run the repository guard before any cache-runner mutation."""
    config, root = _load_partition_config(config_path)
    assert_development_eval_separation(config_path)
    guarded = 0
    for role in TRAIN_ROLES:
        for declared in config.get(role, []) or []:
            assert_no_locked_eval_leakage(root / declared, config_path)
            guarded += 1
    print(f"data_guard PASS: {guarded} training manifests and development/eval separation")


def _resolve_selected_datasets(
    config_path: Path, dataset_specs: list[str], cache_policy: str,
) -> list[tuple[str, Path]]:
    if not dataset_specs:
        raise ValueError("At least one --dataset is required")
    config, root = _load_partition_config(config_path)
    if cache_policy not in CACHE_POLICIES:
        raise ValueError(f"Unknown cache policy: {cache_policy}")
    allowed_roles, forbidden_roles = CACHE_POLICIES[cache_policy]
    declarations = _declared_paths(config, root)
    raw_aliases: dict[str, Path] = {}
    for role, values in config.items():
        for value in values or []:
            path = (root / value).resolve()
            raw_aliases[value] = path
            raw_aliases[str(path)] = path

    selected: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for spec in dataset_specs:
        candidate = raw_aliases.get(spec)
        if candidate is None:
            raw = Path(spec)
            candidates = [raw.resolve()] if raw.is_absolute() else [
                (root / raw).resolve(), (Path.cwd() / raw).resolve(),
            ]
            candidate = next((path for path in candidates if path in declarations), None)
        if candidate is None or candidate not in declarations:
            raise ValueError(f"Dataset is not declared in {config_path}: {spec}")
        roles = declarations[candidate]
        forbidden = sorted(roles & forbidden_roles)
        allowed = sorted(roles & allowed_roles)
        if forbidden:
            raise ValueError(
                f"Dataset {spec} is forbidden for caching by role(s): {forbidden}"
            )
        if len(allowed) != 1 or roles != set(allowed):
            policy_label = (
                "train/development" if cache_policy == "train_development"
                else "one-shot locked_eval"
            )
            raise ValueError(
                f"Dataset {spec} must have exactly one authorized {policy_label} role; "
                f"found {sorted(roles)}"
            )
        if candidate in seen:
            raise ValueError(f"Dataset selected more than once: {spec}")
        seen.add(candidate)
        selected.append((allowed[0], candidate))
    return selected


def _read_ids(truth_path: Path) -> list[str]:
    frame = pd.read_csv(truth_path, usecols=["ID"], dtype={"ID": str})
    if frame["ID"].isna().any():
        raise ValueError(f"Missing ID in {truth_path}")
    ids = frame["ID"].str.strip().tolist()
    if any(not item_id for item_id in ids):
        raise ValueError(f"Blank ID in {truth_path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate ID in {truth_path}")
    if not ids:
        raise ValueError(f"Empty truth manifest: {truth_path}")
    return ids


def _audio_by_id(audio_dir: Path) -> dict[str, Path]:
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory is missing: {audio_dir}")
    result: dict[str, Path] = {}
    for path in audio_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        if path.stem in result:
            raise ValueError(f"Duplicate audio ID {path.stem!r} under {audio_dir}")
        result[path.stem] = path.resolve(strict=True)
    return result


def _slug(value: str) -> str:
    result = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_.-")
    return result[:64] or "dataset"


def _dataset_key(index: int, truth_path: Path) -> str:
    del index
    return _slug(truth_path.parent.name)


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"Exact package pipeline {label} marker changed; refusing overlay")
    return source.replace(old, new)


def _extract_current_export_functions(current_pipeline: Path) -> str:
    source = current_pipeline.read_text("utf-8")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    found: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in EXPORT_FUNCTIONS:
            found[node.name] = "".join(lines[node.lineno - 1:node.end_lineno])
    missing = [name for name in EXPORT_FUNCTIONS if name not in found]
    if missing:
        raise ValueError(
            f"Current pipeline is missing XLS-R export function(s): {missing}"
        )
    return "\n\n".join(found[name].rstrip() for name in EXPORT_FUNCTIONS) + "\n"


def build_pipeline_overlay(package_pipeline: Path, current_pipeline: Path) -> str:
    """Return exact package pipeline plus one isolated opt-in cache pass."""
    source = package_pipeline.read_text("utf-8")
    support = _extract_current_export_functions(current_pipeline)
    source = _replace_once(
        source,
        "import argparse\nimport csv\nimport sys\n",
        "import argparse\nimport csv\nimport hashlib\nimport os\nimport sys\n"
        "import tempfile\nimport time\nimport zipfile\n",
        "import",
    )
    source = _replace_once(
        source,
        "AUDIO_SR = 16_000\n",
        "AUDIO_SR = 16_000\nXLSR_EMBEDDING_DIM = 1_920\n",
        "constant",
    )
    source = _replace_once(
        source,
        "\ndef combine(voice_fake, music_fake, voice_present, music_present):\n",
        "\n" + support + "\ndef combine(voice_fake, music_fake, voice_present, music_present):\n",
        "export helper insertion",
    )
    prediction_marker = '    print(f"Saved {len(rows)} predictions to {args.output}")\n'
    cache_block = prediction_marker + '''    xlsr_window_output = getattr(args, "xlsr_window_embeddings_output", None)
    if xlsr_window_output is not None:
        prediction_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()
        cache_ids, cache_embeddings, cache_starts = [], [], []
        encoder_started = time.perf_counter()
        for path in tqdm(audio_files, desc="xlsr-cache-extra-pass"):
            original_audio = load_audio(path)
            (_, _, window_embeddings, window_starts) = (
                fake_probability_and_embedding(
                    detector, original_audio, device, args.window,
                    args.batch_size, return_windows=True,
                )
            )
            cache_ids.append(path.stem)
            cache_embeddings.append(window_embeddings.numpy())
            cache_starts.append(window_starts)
        encoder_seconds = time.perf_counter() - encoder_started
        archive_started = time.perf_counter()
        save_xlsr_window_embeddings(
            xlsr_window_output, cache_ids, cache_embeddings,
            cache_starts, args.window,
        )
        archive_seconds = time.perf_counter() - archive_started
        prediction_sha256_after = hashlib.sha256(args.output.read_bytes()).hexdigest()
        if prediction_sha256_after != prediction_sha256:
            raise RuntimeError("XLS-R cache pass modified the package prediction CSV")
        print(
            "CACHE_EXPORT_TIMING "
            f"encoder_seconds={encoder_seconds:.6f} "
            f"archive_seconds={archive_seconds:.6f} "
            f"files={len(cache_ids)} "
            f"windows={sum(len(value) for value in cache_starts)} "
            f"prediction_sha256={prediction_sha256}",
            flush=True,
        )
'''
    source = _replace_once(
        source, prediction_marker, cache_block, "post-prediction cache pass",
    )
    parser_marker = '    parser.add_argument("--panns-dir", type=Path, default=Path("models/panns"))\n'
    parser_block = '''    parser.add_argument(
        "--xlsr-window-embeddings-output", type=Path, default=None,
        help="Optional label-blind original-mixture XLS-R window cache. "
             "This adds one timed encoder pass and does not change predictions.",
    )
    parser.add_argument(
        "--retain-eat-patch-graph-output", type=Path, default=None,
        help="Optional destination retaining exact-v47 EAT patch graph features.",
    )
    parser.add_argument(
        "--retain-spear-component-bins-output", type=Path, default=None,
        help="Optional destination retaining exact-v47 SPEAR component bins.",
    )
''' + parser_marker
    source = _replace_once(
        source, parser_marker, parser_block, "cache CLI option",
    )
    ast.parse(source)
    return source


def build_entrypoint_overlay(package_script: Path) -> str:
    """Add opt-in cache retention without changing the v47 fusion path."""
    source = package_script.read_text("utf-8")
    source = _replace_once(
        source,
        "import os\nimport sys\n",
        "import hashlib\nimport os\nimport shutil\nimport sys\n",
        "entrypoint import",
    )
    main_marker = "\n\ndef main():\n"
    hash_helper = '''

def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
'''
    source = _replace_once(
        source, main_marker, hash_helper + main_marker, "entrypoint hash helper",
    )
    cleanup = (
        "    for path in (eat_stats, spear_stats, eat_patch_graph, "
        "spear_component_bins):\n"
    )
    retention = '''    final_prediction_sha256 = hashlib.sha256(
        args.output.read_bytes()
    ).hexdigest()
    retained = (
        (eat_patch_graph, args.retain_eat_patch_graph_output, "eat_patch_graph"),
        (
            spear_component_bins, args.retain_spear_component_bins_output,
            "spear_component_bins",
        ),
    )
    for source, destination, cache_name in retained:
        if destination is None:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        if destination.exists() or temporary.exists():
            raise FileExistsError(f"Refusing to overwrite retained cache: {destination}")
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        print(
            f"CACHE_RETAINED name={cache_name} bytes={destination.stat().st_size} "
            f"sha256={_sha256_file(destination)}",
            flush=True,
        )
    if hashlib.sha256(args.output.read_bytes()).hexdigest() != final_prediction_sha256:
        raise RuntimeError("Cache retention modified the exact-v47 prediction CSV")
    print(
        f"EXACT_V47_PREDICTION sha256={final_prediction_sha256}", flush=True,
    )
'''
    source = _replace_once(
        source, cleanup, retention + cleanup, "entrypoint cache retention",
    )
    ast.parse(source)
    return source


def build_post_entrypoint_overlay(package_script: Path) -> str:
    """Return the cache-retaining v47 entrypoint without its base ``run`` call.

    This exists only to recover an already verified base prediction/XLS-R pass
    from the schema-v1 multi-shard layout bug.  The remaining v47 fusion calls
    are kept byte-for-byte identical and run against the corrected isolated
    shard view.
    """
    source = build_entrypoint_overlay(package_script)
    source = _replace_once(
        source,
        "    run(args)\n",
        "    if not args.output.is_file():\n"
        "        raise FileNotFoundError(\n"
        "            f\"Verified base prediction is missing: {args.output}\"\n"
        "        )\n"
        "    print(\"RESUME_VERIFIED_BASE_POST_FUSIONS\", flush=True)\n",
        "base-run recovery marker",
    )
    ast.parse(source)
    return source


def _iter_inventory_files(root: Path):
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.parts:
            continue
        if path.is_file():
            yield path


def _file_inventory(root: Path) -> list[dict]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in _iter_inventory_files(root)
    ]


def _write_csv(path: Path, fieldnames, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_sample_submission(path: Path, ids: list[str]) -> None:
    rows = []
    for item_id in ids:
        row = {"ID": item_id}
        row.update({column: "0.5" for column in PREDICTION_COLUMNS})
        rows.append(row)
    _write_csv(path, ("ID", *PREDICTION_COLUMNS), rows)


def _materialize_shard(
    shard_dir: Path,
    package_dir: Path,
    corpus_dir: Path,
    overlay: str,
    entrypoint_overlay: str,
    mapping: list[dict],
    shard_index: int,
    num_shards: int,
) -> dict:
    """Materialize a complete, independently runnable shard.

    The v47 entrypoint contains several post-pipeline fusion stages which do
    not understand ``--num-shards``.  Consequently each process must see only
    its own round-robin audio files and sample-submission rows.  Passing the
    full corpus and relying on pipeline-level sharding is unsafe: the first
    downstream fusion observes extra files and aborts (or, with a weaker
    implementation, could silently misalign rows).
    """
    shard_dir.mkdir(parents=True)
    (shard_dir / "script.py").write_text(entrypoint_overlay, encoding="utf-8")
    (shard_dir / "data").mkdir()
    data_dir = shard_dir / "data"
    test_dir = data_dir / "test"
    test_dir.mkdir()
    selected = mapping[shard_index::num_shards]
    for row in selected:
        source = corpus_dir / "audio" / row["AUDIO_FILENAME"]
        (test_dir / row["AUDIO_FILENAME"]).symlink_to(
            os.path.relpath(source, test_dir),
        )
    sample_path = data_dir / "sample_submission.csv"
    selected_ids = [row["COMBINED_ID"] for row in selected]
    _write_sample_submission(sample_path, selected_ids)
    (shard_dir / "output").mkdir()

    package_model = package_dir / "model"
    model = shard_dir / "model"
    model.mkdir()
    for entry in sorted(package_model.iterdir()):
        if entry.name in ("src", "__pycache__"):
            continue
        (model / entry.name).symlink_to(
            entry.resolve(), target_is_directory=entry.is_dir(),
        )
    source_dir = model / "src"
    source_dir.mkdir()
    for source in _iter_inventory_files(package_model / "src"):
        relative = source.relative_to(package_model / "src")
        if relative.as_posix() == "pipeline.py":
            continue
        target = source_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source.resolve())
    (source_dir / "pipeline.py").write_text(overlay, encoding="utf-8")
    return {
        "index": shard_index,
        "rows": len(selected),
        "ids_sha256": _ids_sha256(selected_ids),
        "sample_submission_sha256": _sha256_file(sample_path),
    }


def prepare_run(
    config_path: Path,
    package_dir: Path,
    current_pipeline: Path,
    work_dir: Path,
    dataset_specs: list[str],
    num_shards: int,
    cache_policy: str = "train_development",
) -> Path:
    """Validate and materialize a run without starting inference."""
    config_path = config_path.resolve(strict=True)
    _run_data_guard(config_path)
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    package_dir = package_dir.resolve(strict=True)
    current_pipeline = current_pipeline.resolve(strict=True)
    work_dir = work_dir.resolve()
    if work_dir.exists() or work_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite run directory: {work_dir}")
    if not (package_dir / "script.py").is_file():
        raise FileNotFoundError(f"Package script is missing: {package_dir / 'script.py'}")
    if not (package_dir / "model" / "src" / "pipeline.py").is_file():
        raise FileNotFoundError("Package pipeline is missing")

    if cache_policy not in CACHE_POLICIES:
        raise ValueError(f"Unknown cache policy: {cache_policy}")
    allowed_roles, forbidden_roles = CACHE_POLICIES[cache_policy]
    selected = _resolve_selected_datasets(
        config_path, dataset_specs, cache_policy,
    )
    overlay = build_pipeline_overlay(
        package_dir / "model" / "src" / "pipeline.py", current_pipeline,
    )
    entrypoint_overlay = build_entrypoint_overlay(package_dir / "script.py")
    model_inventory = _file_inventory(package_dir / "model")

    mapping: list[dict] = []
    datasets: list[dict] = []
    combined_ids: set[str] = set()
    dataset_keys: set[str] = set()
    selected_ids = [(role, path, _read_ids(path)) for role, path in selected]
    if num_shards > sum(len(ids) for _, _, ids in selected_ids):
        raise ValueError("num_shards cannot exceed the combined row count")
    for dataset_index, (role, truth_path, ids) in enumerate(selected_ids):
        key = _dataset_key(dataset_index, truth_path)
        if key in dataset_keys:
            raise ValueError(
                f"Selected truth manifests do not have unique dataset names: {key}"
            )
        dataset_keys.add(key)
        audio = _audio_by_id(truth_path.parent / "audio")
        missing = [item_id for item_id in ids if item_id not in audio]
        if missing:
            raise ValueError(
                f"{truth_path} has {len(missing)} IDs without audio, e.g. {missing[:5]}"
            )
        datasets.append({
            "key": key,
            "role": role,
            "truth_path": str(truth_path.resolve()),
            "truth_sha256": _sha256_file(truth_path),
            "ids_sha256": _ids_sha256(ids),
            "rows": len(ids),
        })
        for row_index, original_id in enumerate(ids):
            identity = hashlib.sha256(
                f"{truth_path.resolve()}\0{row_index}\0{original_id}".encode("utf-8")
            ).hexdigest()[:20]
            combined_id = f"v47c_{dataset_index:03d}_{row_index:07d}_{identity}"
            if combined_id in combined_ids:
                raise ValueError(f"Combined ID collision: {combined_id}")
            combined_ids.add(combined_id)
            source = audio[original_id]
            filename = combined_id + source.suffix.lower()
            mapping.append({
                "COMBINED_ID": combined_id,
                "DATASET_KEY": key,
                "ROLE": role,
                "ROW_INDEX": row_index,
                "ORIGINAL_ID": original_id,
                "AUDIO_FILENAME": filename,
                "SOURCE_AUDIO": str(source),
                "AUDIO_SHA256": _sha256_file(source),
            })

    work_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{work_dir.name}.preparing-", dir=work_dir.parent,
    ))
    try:
        corpus = temporary / "corpus"
        audio_dir = corpus / "audio"
        audio_dir.mkdir(parents=True)
        for row in mapping:
            (audio_dir / row["AUDIO_FILENAME"]).symlink_to(row["SOURCE_AUDIO"])
        _write_csv(corpus / "dataset_map.csv", MAP_COLUMNS, mapping)
        _write_sample_submission(
            corpus / "sample_submission.csv",
            [row["COMBINED_ID"] for row in mapping],
        )

        shards = temporary / "shards"
        shards.mkdir()
        shard_manifests = []
        for shard_index in range(num_shards):
            shard_manifests.append(_materialize_shard(
                shards / f"shard_{shard_index:03d}", package_dir, corpus, overlay,
                entrypoint_overlay, mapping, shard_index, num_shards,
            ))

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "partition_config": str(config_path),
            "partition_config_sha256": _sha256_file(config_path),
            "cache_policy": cache_policy,
            "allowed_roles": sorted(allowed_roles),
            "forbidden_roles": sorted(forbidden_roles),
            "num_shards": num_shards,
            "rows": len(mapping),
            "datasets": datasets,
            "shards": shard_manifests,
            "dataset_map_sha256": _sha256_file(corpus / "dataset_map.csv"),
            "sample_submission_sha256": _sha256_file(corpus / "sample_submission.csv"),
            "package_dir": str(package_dir),
            "package_script_sha256": _sha256_file(package_dir / "script.py"),
            "entrypoint_overlay_sha256": hashlib.sha256(
                entrypoint_overlay.encode("utf-8")
            ).hexdigest(),
            "package_pipeline_sha256": _sha256_file(
                package_dir / "model" / "src" / "pipeline.py"
            ),
            "package_model_inventory": model_inventory,
            "current_pipeline": str(current_pipeline),
            "current_pipeline_sha256": _sha256_file(current_pipeline),
            "overlay_sha256": hashlib.sha256(overlay.encode("utf-8")).hexdigest(),
            "cache_pass": "one opt-in post-prediction original-mixture XLS-R pass",
            "retained_exact_v47_caches": ["eat_patch_graph", "spear_component_bins"],
        }
        (temporary / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        temporary.rename(work_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        f"Prepared {len(mapping)} label-blind rows from {len(datasets)} datasets "
        f"in {num_shards} shards at {work_dir}; inference was not started"
    )
    return work_dir


def _read_manifest(work_dir: Path) -> dict:
    path = work_dir / "run_manifest.json"
    manifest = json.loads(path.read_text("utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported run manifest schema: {manifest.get('schema_version')}")
    return manifest


def _assert_symlink(path: Path, target: Path) -> None:
    if not path.is_symlink() or path.resolve(strict=True) != target.resolve(strict=True):
        raise ValueError(f"Symlink target mismatch: {path}")


def validate_prepared_run(work_dir: Path) -> tuple[dict, pd.DataFrame]:
    """Revalidate role, hashes, rows, corpus, package, and shard layouts."""
    work_dir = work_dir.resolve(strict=True)
    manifest = _read_manifest(work_dir)
    config_path = Path(manifest["partition_config"])
    _run_data_guard(config_path)
    if _sha256_file(config_path) != manifest["partition_config_sha256"]:
        raise ValueError("Partition config hash changed after prepare")
    cache_policy = manifest.get("cache_policy", "train_development")
    if cache_policy not in CACHE_POLICIES:
        raise ValueError("Run manifest cache policy is unsupported")
    allowed_roles, forbidden_roles = CACHE_POLICIES[cache_policy]
    if set(manifest["allowed_roles"]) != allowed_roles:
        raise ValueError("Run manifest allowed-role policy mismatch")
    if set(manifest["forbidden_roles"]) != forbidden_roles:
        raise ValueError("Run manifest forbidden-role policy mismatch")

    config, root = _load_partition_config(config_path)
    declarations = _declared_paths(config, root)
    for dataset in manifest["datasets"]:
        truth = Path(dataset["truth_path"])
        roles = declarations.get(truth.resolve(), set())
        if roles != {dataset["role"]} or dataset["role"] not in allowed_roles:
            raise ValueError(f"Dataset role declaration changed: {truth}")
        if roles & forbidden_roles:
            raise ValueError(f"Forbidden role now declares selected dataset: {truth}")
        ids = _read_ids(truth)
        if len(ids) != dataset["rows"] or _ids_sha256(ids) != dataset["ids_sha256"]:
            raise ValueError(f"Truth rows/IDs changed after prepare: {truth}")
        if _sha256_file(truth) != dataset["truth_sha256"]:
            raise ValueError(f"Truth hash changed after prepare: {truth}")

    corpus = work_dir / "corpus"
    map_path = corpus / "dataset_map.csv"
    sample_path = corpus / "sample_submission.csv"
    if _sha256_file(map_path) != manifest["dataset_map_sha256"]:
        raise ValueError("Dataset map hash mismatch")
    if _sha256_file(sample_path) != manifest["sample_submission_sha256"]:
        raise ValueError("Sample submission hash mismatch")
    mapping = pd.read_csv(map_path, dtype=str, keep_default_na=False)
    if tuple(mapping.columns) != MAP_COLUMNS or len(mapping) != manifest["rows"]:
        raise ValueError("Dataset map schema/row mismatch")
    if mapping["COMBINED_ID"].duplicated().any():
        raise ValueError("Dataset map contains duplicate combined IDs")
    sample = pd.read_csv(sample_path, dtype={"ID": str})
    if tuple(sample.columns) != ("ID", *PREDICTION_COLUMNS):
        raise ValueError("Sample submission schema mismatch")
    if sample["ID"].tolist() != mapping["COMBINED_ID"].tolist():
        raise ValueError("Sample submission and dataset map row order differ")

    source_hashes: dict[Path, str] = {}
    expected_audio_names = set(mapping["AUDIO_FILENAME"])
    actual_audio_names = {path.name for path in (corpus / "audio").iterdir()}
    if actual_audio_names != expected_audio_names:
        raise ValueError("Combined audio corpus filenames changed")
    for row in mapping.to_dict("records"):
        source = Path(row["SOURCE_AUDIO"])
        link = corpus / "audio" / row["AUDIO_FILENAME"]
        _assert_symlink(link, source)
        if source not in source_hashes:
            source_hashes[source] = _sha256_file(source)
        digest = source_hashes[source]
        if digest != row["AUDIO_SHA256"]:
            raise ValueError(f"Audio hash changed after prepare: {source}")

    package = Path(manifest["package_dir"])
    if _sha256_file(package / "script.py") != manifest["package_script_sha256"]:
        raise ValueError("Exact v47 package script hash changed")
    if _sha256_file(package / "model" / "src" / "pipeline.py") != manifest["package_pipeline_sha256"]:
        raise ValueError("Exact v47 package pipeline hash changed")
    if _file_inventory(package / "model") != manifest["package_model_inventory"]:
        raise ValueError("Exact v47 package model inventory changed")
    current_pipeline = Path(manifest["current_pipeline"])
    if _sha256_file(current_pipeline) != manifest["current_pipeline_sha256"]:
        raise ValueError("Current exporter pipeline changed after prepare")
    expected_overlay = build_pipeline_overlay(
        package / "model" / "src" / "pipeline.py", current_pipeline,
    )
    if hashlib.sha256(expected_overlay.encode("utf-8")).hexdigest() != manifest["overlay_sha256"]:
        raise ValueError("Pipeline overlay provenance mismatch")
    expected_entrypoint = build_entrypoint_overlay(package / "script.py")
    if hashlib.sha256(expected_entrypoint.encode("utf-8")).hexdigest() != manifest["entrypoint_overlay_sha256"]:
        raise ValueError("Entrypoint overlay provenance mismatch")

    for shard_index in range(manifest["num_shards"]):
        shard = work_dir / "shards" / f"shard_{shard_index:03d}"
        script = shard / "script.py"
        if script.is_symlink() or _sha256_file(script) != manifest["entrypoint_overlay_sha256"]:
            raise ValueError(f"Shard {shard_index} entrypoint overlay mismatch")
        selected = mapping.iloc[shard_index::manifest["num_shards"]]
        selected_ids = selected["COMBINED_ID"].tolist()
        shard_metadata = manifest["shards"][shard_index]
        if (
            shard_metadata.get("index") != shard_index
            or shard_metadata.get("rows") != len(selected_ids)
            or shard_metadata.get("ids_sha256") != _ids_sha256(selected_ids)
        ):
            raise ValueError(f"Shard {shard_index} manifest mismatch")
        shard_test = shard / "data" / "test"
        if shard_test.is_symlink() or not shard_test.is_dir():
            raise ValueError(f"Shard {shard_index} test view is not isolated")
        expected_names = set(selected["AUDIO_FILENAME"])
        if {path.name for path in shard_test.iterdir()} != expected_names:
            raise ValueError(f"Shard {shard_index} audio view mismatch")
        for row in selected.to_dict("records"):
            _assert_symlink(
                shard_test / row["AUDIO_FILENAME"],
                corpus / "audio" / row["AUDIO_FILENAME"],
            )
        shard_sample_path = shard / "data" / "sample_submission.csv"
        if shard_sample_path.is_symlink() or not shard_sample_path.is_file():
            raise ValueError(f"Shard {shard_index} sample submission is not isolated")
        if _sha256_file(shard_sample_path) != shard_metadata.get(
            "sample_submission_sha256"
        ):
            raise ValueError(f"Shard {shard_index} sample submission hash mismatch")
        shard_sample = pd.read_csv(shard_sample_path, dtype={"ID": str})
        if (
            tuple(shard_sample.columns) != ("ID", *PREDICTION_COLUMNS)
            or shard_sample["ID"].tolist() != selected_ids
        ):
            raise ValueError(f"Shard {shard_index} sample submission row mismatch")
        overlay_path = shard / "model" / "src" / "pipeline.py"
        if overlay_path.is_symlink() or _sha256_file(overlay_path) != manifest["overlay_sha256"]:
            raise ValueError(f"Shard {shard_index} pipeline overlay mismatch")
        for entry in package.joinpath("model").iterdir():
            if entry.name in ("src", "__pycache__"):
                continue
            _assert_symlink(shard / "model" / entry.name, entry)
        for source in _iter_inventory_files(package / "model" / "src"):
            relative = source.relative_to(package / "model" / "src")
            if relative.as_posix() == "pipeline.py":
                continue
            _assert_symlink(shard / "model" / "src" / relative, source)
    return manifest, mapping


def launch_run(
    work_dir: Path,
    gpus: list[str],
    python_bin: str,
    confirmation: str,
) -> None:
    """Launch exactly one isolated package process per prepared shard."""
    if confirmation != "RUN_EXACT_V47_CACHE":
        raise ValueError("launch requires --confirm RUN_EXACT_V47_CACHE")
    manifest, _ = validate_prepared_run(work_dir)
    if len(gpus) != manifest["num_shards"] or any(not gpu or "," in gpu for gpu in gpus):
        raise ValueError("--gpus must provide exactly one device token per shard")
    work_dir = work_dir.resolve()
    existing = []
    for index in range(manifest["num_shards"]):
        output = work_dir / "shards" / f"shard_{index:03d}" / "output"
        existing.extend(path for path in (
            output / "submission.csv", output / "xlsr_windows.npz",
            output / "eat_patch_graph.npz", output / "spear_component_bins.npz",
        ) if path.exists())
    if existing:
        raise FileExistsError(f"Refusing to overwrite shard results: {existing[:3]}")

    processes = []
    handles = []
    started = time.monotonic()
    try:
        for index, gpu in enumerate(gpus):
            shard = work_dir / "shards" / f"shard_{index:03d}"
            log_path = shard / "output" / "run.log"
            handle = log_path.open("wb")
            handles.append(handle)
            command = [
                python_bin,
                str(shard / "script.py"),
                "--device", "cuda",
                # Every process has a physically isolated round-robin view.
                # Keep the entire downstream v47 entrypoint internally
                # unsharded so every fusion sees exactly the same ID universe.
                "--num-shards", "1",
                "--shard-index", "0",
                "--xlsr-window-embeddings-output",
                str(shard / "output" / "xlsr_windows.npz"),
                "--retain-eat-patch-graph-output",
                str(shard / "output" / "eat_patch_graph.npz"),
                "--retain-spear-component-bins-output",
                str(shard / "output" / "spear_component_bins.npz"),
            ]
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": gpu,
                "PYTHONNOUSERSITE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            })
            processes.append(subprocess.Popen(
                command, cwd=shard, env=env, stdout=handle,
                stderr=subprocess.STDOUT,
            ))
        returncodes = [process.wait() for process in processes]
    finally:
        for handle in handles:
            handle.close()
    status = {
        "elapsed_seconds": time.monotonic() - started,
        "gpus": gpus,
        "returncodes": returncodes,
    }
    (work_dir / "launch_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    if any(returncodes):
        raise RuntimeError(f"v47 cache shard failure(s): {returncodes}")
    # Detect any accidental write through a package-model symlink immediately.
    if _file_inventory(Path(manifest["package_dir"]) / "model") != manifest["package_model_inventory"]:
        raise RuntimeError("Package model changed during inference")
    print(f"All {len(processes)} exact-v47 cache shards completed")


def _validate_prediction_frame(path: Path, expected_ids: list[str]) -> None:
    frame = pd.read_csv(path, dtype={"ID": str})
    if tuple(frame.columns) != ("ID", *PREDICTION_COLUMNS):
        raise ValueError(f"Prediction schema mismatch: {path}")
    if frame["ID"].tolist() != expected_ids or frame["ID"].duplicated().any():
        raise ValueError(f"Prediction ID/order mismatch: {path}")
    numeric = frame.loc[:, PREDICTION_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"Non-finite prediction: {path}")
    if ((numeric < 0) | (numeric > 1)).any().any():
        raise ValueError(f"Prediction outside [0, 1]: {path}")


def adopt_schema_v1_base_outputs(
    base_work_dir: Path,
    target_work_dir: Path,
    confirmation: str,
) -> Path:
    """Adopt only verified base predictions/XLS-R from the known shard bug.

    Schema-v1 ran the base pipeline correctly with round-robin sharding and
    then failed before the first post-pipeline fusion because those fusions
    saw the full audio corpus.  This function accepts precisely that failure
    signature, copies no EAT/SPEAR output, and records every adopted hash.
    """
    if confirmation != "ADOPT_VERIFIED_SCHEMA_V1_BASE":
        raise ValueError(
            "adoption requires --confirm ADOPT_VERIFIED_SCHEMA_V1_BASE"
        )
    target_manifest, target_mapping = validate_prepared_run(target_work_dir)
    base_work_dir = base_work_dir.resolve(strict=True)
    target_work_dir = target_work_dir.resolve(strict=True)
    base_manifest_path = base_work_dir / "run_manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text("utf-8"))
    if base_manifest.get("schema_version") != 1:
        raise ValueError("Recovery source must be the known schema-v1 layout")
    for key in (
        "rows", "num_shards", "datasets", "dataset_map_sha256",
        "sample_submission_sha256", "package_script_sha256",
        "package_pipeline_sha256", "package_model_inventory",
        "current_pipeline_sha256", "overlay_sha256",
        "entrypoint_overlay_sha256",
    ):
        if base_manifest.get(key) != target_manifest.get(key):
            raise ValueError(f"Schema-v1 recovery provenance differs for {key}")
    if _sha256_file(base_work_dir / "corpus" / "dataset_map.csv") != _sha256_file(
        target_work_dir / "corpus" / "dataset_map.csv"
    ):
        raise ValueError("Recovery and target dataset maps differ")
    status_path = base_work_dir / "launch_status.json"
    status = json.loads(status_path.read_text("utf-8"))
    expected_failures = [1] * int(target_manifest["num_shards"])
    if status.get("returncodes") != expected_failures:
        raise ValueError(
            "Schema-v1 source does not have the exact expected shard failures"
        )

    adopted = []
    all_ids = target_mapping["COMBINED_ID"].tolist()
    post_source = build_post_entrypoint_overlay(
        Path(target_manifest["package_dir"]) / "script.py"
    )
    post_sha256 = hashlib.sha256(post_source.encode("utf-8")).hexdigest()
    for index in range(target_manifest["num_shards"]):
        expected_ids = all_ids[index::target_manifest["num_shards"]]
        base_output = base_work_dir / "shards" / f"shard_{index:03d}" / "output"
        target_shard = target_work_dir / "shards" / f"shard_{index:03d}"
        target_output = target_shard / "output"
        if any(target_output.iterdir()):
            raise FileExistsError(f"Recovery target shard is not empty: {target_output}")
        prediction = base_output / "submission.csv"
        xlsr = base_output / "xlsr_windows.npz"
        if any((base_output / name).exists() for name in (
            "eat_patch_graph.npz", "spear_component_bins.npz",
        )):
            raise ValueError("Schema-v1 source unexpectedly contains post-fusion caches")
        _validate_prediction_frame(prediction, expected_ids)
        _load_validated_npz(xlsr, expected_ids)
        log_path = base_output / "run.log"
        log = log_path.read_text("utf-8", errors="replace")
        matches = re.findall(
            r"CACHE_EXPORT_TIMING .*?prediction_sha256=([0-9a-f]{64})", log,
        )
        prediction_sha256 = _sha256_file(prediction)
        if matches != [prediction_sha256]:
            raise ValueError(f"Shard {index} base prediction hash marker mismatch")
        required_failure = "ValueError: Test audio and submission IDs disagree."
        if (
            log.count("Traceback (most recent call last):") != 1
            or log.count(required_failure) != 1
            or "EXACT_V47_PREDICTION" in log
            or "CACHE_RETAINED" in log
        ):
            raise ValueError(f"Shard {index} does not match the known schema-v1 failure")
        shutil.copy2(prediction, target_output / prediction.name)
        shutil.copy2(xlsr, target_output / xlsr.name)
        post_script = target_shard / "post_script.py"
        post_script.write_text(post_source, encoding="utf-8")
        adopted.append({
            "shard": index,
            "rows": len(expected_ids),
            "prediction_sha256": prediction_sha256,
            "xlsr_sha256": _sha256_file(xlsr),
            "source_log_sha256": _sha256_file(log_path),
        })
    record = {
        "schema_version": 1,
        "base_run_manifest_sha256": _sha256_file(base_manifest_path),
        "target_run_manifest_sha256": _sha256_file(
            target_work_dir / "run_manifest.json"
        ),
        "post_entrypoint_sha256": post_sha256,
        "known_failure": "full-corpus view reached unsharded post-pipeline fusion",
        "adopted": adopted,
    }
    record_path = target_work_dir / "adopted_schema_v1_base.json"
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Adopted {sum(item['rows'] for item in adopted)} verified base rows "
        f"from {base_work_dir} into {target_work_dir}"
    )
    return record_path


def launch_post_fusions(
    work_dir: Path,
    gpus: list[str],
    python_bin: str,
    confirmation: str,
) -> None:
    """Run only exact-v47 post-pipeline fusions on adopted verified bases."""
    if confirmation != "RUN_ADOPTED_V47_POST_FUSIONS":
        raise ValueError(
            "post launch requires --confirm RUN_ADOPTED_V47_POST_FUSIONS"
        )
    manifest, mapping = validate_prepared_run(work_dir)
    if len(gpus) != manifest["num_shards"] or any(not gpu or "," in gpu for gpu in gpus):
        raise ValueError("--gpus must provide exactly one device token per shard")
    work_dir = work_dir.resolve()
    record_path = work_dir / "adopted_schema_v1_base.json"
    record = json.loads(record_path.read_text("utf-8"))
    if record.get("target_run_manifest_sha256") != _sha256_file(
        work_dir / "run_manifest.json"
    ):
        raise ValueError("Adopted-base target manifest changed")
    expected_post = hashlib.sha256(build_post_entrypoint_overlay(
        Path(manifest["package_dir"]) / "script.py"
    ).encode("utf-8")).hexdigest()
    if record.get("post_entrypoint_sha256") != expected_post:
        raise ValueError("Adopted post-entrypoint provenance mismatch")
    by_shard = {int(item["shard"]): item for item in record.get("adopted", [])}
    all_ids = mapping["COMBINED_ID"].tolist()
    for index in range(manifest["num_shards"]):
        shard = work_dir / "shards" / f"shard_{index:03d}"
        output = shard / "output"
        post_script = shard / "post_script.py"
        item = by_shard.get(index)
        if item is None or _sha256_file(post_script) != expected_post:
            raise ValueError(f"Shard {index} adopted post script mismatch")
        prediction = output / "submission.csv"
        xlsr = output / "xlsr_windows.npz"
        expected_ids = all_ids[index::manifest["num_shards"]]
        _validate_prediction_frame(prediction, expected_ids)
        _load_validated_npz(xlsr, expected_ids)
        if (
            _sha256_file(prediction) != item["prediction_sha256"]
            or _sha256_file(xlsr) != item["xlsr_sha256"]
        ):
            raise ValueError(f"Shard {index} adopted artifact hash changed")
        if any((output / name).exists() for name in (
            "eat_patch_graph.npz", "spear_component_bins.npz",
        )):
            raise FileExistsError(f"Shard {index} post-fusion output already exists")

    processes, handles = [], []
    started = time.monotonic()
    try:
        for index, gpu in enumerate(gpus):
            shard = work_dir / "shards" / f"shard_{index:03d}"
            output = shard / "output"
            handle = (output / "post_run.log").open("wb")
            handles.append(handle)
            command = [
                python_bin, str(shard / "post_script.py"),
                "--device", "cuda", "--num-shards", "1", "--shard-index", "0",
                "--retain-eat-patch-graph-output",
                str(output / "eat_patch_graph.npz"),
                "--retain-spear-component-bins-output",
                str(output / "spear_component_bins.npz"),
            ]
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": gpu,
                "PYTHONNOUSERSITE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            })
            processes.append(subprocess.Popen(
                command, cwd=shard, env=env, stdout=handle,
                stderr=subprocess.STDOUT,
            ))
        returncodes = [process.wait() for process in processes]
    finally:
        for handle in handles:
            handle.close()
    status = {
        "elapsed_seconds": time.monotonic() - started,
        "gpus": gpus,
        "returncodes": returncodes,
    }
    (work_dir / "post_launch_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if any(returncodes):
        raise RuntimeError(f"v47 post-fusion shard failure(s): {returncodes}")
    if _file_inventory(Path(manifest["package_dir"]) / "model") != manifest[
        "package_model_inventory"
    ]:
        raise RuntimeError("Package model changed during post-fusion inference")
    print(f"All {len(processes)} adopted exact-v47 post-fusion shards completed")


def _load_validated_npz(path: Path, expected_ids: list[str]) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != NPZ_FIELDS:
            raise ValueError(f"XLS-R cache schema mismatch: {path}")
        values = {name: np.asarray(archive[name]) for name in archive.files}
    ids = values["ids"].tolist()
    if ids != expected_ids:
        raise ValueError(f"XLS-R cache ID/order mismatch: {path}")
    embeddings, mask, starts = values["embeddings"], values["mask"], values["starts"]
    if embeddings.ndim != 3 or embeddings.shape[0] != len(ids) or embeddings.shape[2] != XLSR_EMBEDDING_DIM:
        raise ValueError(f"XLS-R embedding shape mismatch: {path}")
    if embeddings.dtype != np.float32 or not np.isfinite(embeddings).all():
        raise ValueError(f"XLS-R embeddings must be finite float32: {path}")
    if mask.dtype != np.bool_ or mask.shape != embeddings.shape[:2]:
        raise ValueError(f"XLS-R mask mismatch: {path}")
    if starts.dtype != np.int64 or starts.shape != mask.shape:
        raise ValueError(f"XLS-R starts mismatch: {path}")
    counts = mask.sum(axis=1)
    expected_mask = np.arange(mask.shape[1])[None, :] < counts[:, None]
    if np.any(counts <= 0) or not np.array_equal(mask, expected_mask):
        raise ValueError(f"XLS-R masks must be non-empty prefixes: {path}")
    if np.any(starts[~mask] != -1) or np.any(embeddings[~mask] != 0):
        raise ValueError(f"XLS-R padding mismatch: {path}")
    for index, count in enumerate(counts):
        valid_starts = starts[index, :count]
        if np.any(valid_starts < 0) or np.any(valid_starts[1:] < valid_starts[:-1]):
            raise ValueError(f"XLS-R starts are unordered: {path}")
    if values["window"].shape != () or int(values["window"]) <= 0:
        raise ValueError(f"XLS-R window metadata mismatch: {path}")
    if values["sample_rate"].shape != () or int(values["sample_rate"]) != 16_000:
        raise ValueError(f"XLS-R sample-rate metadata mismatch: {path}")
    return values


def _load_validated_structured_npz(
    path: Path, expected_ids: list[str], kind: str,
) -> dict:
    if kind == "eat":
        expected_fields = EAT_FIELDS
        row_fields = ("temporal", "spectral", "view_mask")
        metadata_fields = ("projection", "layers")
    elif kind == "spear":
        expected_fields = SPEAR_FIELDS
        row_fields = ("features", "mask")
        metadata_fields = ("projection", "layers", "bins")
    else:  # pragma: no cover - internal contract
        raise ValueError(f"Unknown structured cache kind: {kind}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_fields:
            raise ValueError(f"{kind} cache schema mismatch: {path}")
        values = {name: np.asarray(archive[name]) for name in archive.files}
    if values["ids"].tolist() != expected_ids:
        raise ValueError(f"{kind} cache ID/order mismatch: {path}")
    rows = len(expected_ids)
    for field in row_fields:
        value = values[field]
        if value.ndim < 2 or value.shape[0] != rows:
            raise ValueError(f"{kind} cache row shape mismatch for {field}: {path}")
    if kind == "eat":
        if values["temporal"].ndim != 6 or values["spectral"].ndim != 6:
            raise ValueError(f"EAT graph tensors must be rank 6: {path}")
        if values["temporal"].shape[:4] != values["spectral"].shape[:4]:
            raise ValueError(f"EAT temporal/spectral axes disagree: {path}")
        if values["view_mask"].shape != values["temporal"].shape[:2]:
            raise ValueError(f"EAT view mask shape mismatch: {path}")
        if values["view_mask"].dtype != np.bool_ or not values["view_mask"].any(axis=1).all():
            raise ValueError(f"EAT view masks must contain a valid view: {path}")
    else:
        if values["features"].ndim != 6:
            raise ValueError(f"SPEAR component-bin tensor must be rank 6: {path}")
        if values["mask"].shape != values["features"].shape[:3]:
            raise ValueError(f"SPEAR component-bin mask shape mismatch: {path}")
        if values["mask"].dtype != np.bool_ or not values["mask"].any(axis=(1, 2)).all():
            raise ValueError(f"SPEAR masks must contain a valid bin: {path}")
        if values["bins"].shape != () or int(values["bins"]) != values["features"].shape[2]:
            raise ValueError(f"SPEAR bin metadata mismatch: {path}")
    for field in row_fields + metadata_fields:
        value = values[field]
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ValueError(f"Non-finite {kind} cache field {field}: {path}")
    if values["projection"].ndim != 2 or values["layers"].ndim != 1:
        raise ValueError(f"{kind} projection/layer metadata mismatch: {path}")
    return values


class _StructuredSpool:
    """Disk-backed row merger for the large EAT/SPEAR feature tensors."""

    def __init__(
        self,
        root: Path,
        kind: str,
        first: dict,
        dataset_rows: dict[str, int],
        locations: dict[str, tuple[str, int]],
    ) -> None:
        self.kind = kind
        self.row_fields = (
            ("temporal", "spectral", "view_mask")
            if kind == "eat" else ("features", "mask")
        )
        self.metadata_fields = (
            ("projection", "layers")
            if kind == "eat" else ("projection", "layers", "bins")
        )
        self.metadata = {field: first[field].copy() for field in self.metadata_fields}
        self.locations = locations
        self.arrays: dict[str, dict[str, np.memmap]] = {}
        self.filled = {
            key: np.zeros(rows, dtype=np.bool_) for key, rows in dataset_rows.items()
        }
        for key, rows in dataset_rows.items():
            dataset_root = root / kind / key
            dataset_root.mkdir(parents=True, exist_ok=True)
            self.arrays[key] = {}
            for field in self.row_fields:
                value = first[field]
                self.arrays[key][field] = np.lib.format.open_memmap(
                    dataset_root / f"{field}.npy", mode="w+", dtype=value.dtype,
                    shape=(rows, *value.shape[1:]),
                )

    def add(self, values: dict, ids: list[str]) -> None:
        for field in self.row_fields:
            prototype = next(iter(self.arrays.values()))[field]
            if values[field].shape[1:] != prototype.shape[1:] or values[field].dtype != prototype.dtype:
                raise ValueError(f"{self.kind} shard shape/dtype differs for {field}")
        for field in self.metadata_fields:
            if not np.array_equal(values[field], self.metadata[field]):
                raise ValueError(f"{self.kind} shard metadata differs for {field}")
        grouped: dict[str, list[tuple[int, int]]] = {}
        for source_index, item_id in enumerate(ids):
            if item_id not in self.locations:
                raise ValueError(f"Unexpected {self.kind} cache ID: {item_id}")
            key, target_index = self.locations[item_id]
            grouped.setdefault(key, []).append((source_index, target_index))
        for key, indices in grouped.items():
            sources = np.asarray([item[0] for item in indices], dtype=np.int64)
            targets = np.asarray([item[1] for item in indices], dtype=np.int64)
            if self.filled[key][targets].any():
                raise ValueError(f"Duplicate {self.kind} rows for {key}")
            for field in self.row_fields:
                self.arrays[key][field][targets] = values[field][sources]
            self.filled[key][targets] = True

    def validate_complete(self) -> None:
        incomplete = [key for key, mask in self.filled.items() if not mask.all()]
        if incomplete:
            raise ValueError(f"Incomplete {self.kind} dataset rows: {incomplete}")
        for arrays in self.arrays.values():
            for value in arrays.values():
                value.flush()

    def output_arrays(self, key: str, original_ids: list[str]) -> dict:
        result = {"ids": np.asarray(original_ids)}
        result.update(self.arrays[key])
        result.update(self.metadata)
        return result

    def close(self) -> None:
        self.arrays.clear()


def _write_deterministic_npz(path: Path, arrays: dict) -> None:
    with path.open("w+b") as handle:
        with zipfile.ZipFile(
            handle, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
        ) as archive:
            for name, value in arrays.items():
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                with archive.open(info, mode="w", force_zip64=True) as member:
                    np.lib.format.write_array(member, np.asarray(value), allow_pickle=False)


def _pack_records(ids: list[str], records: dict[str, tuple[np.ndarray, np.ndarray]], window: int) -> dict:
    maximum = max(records[item_id][0].shape[0] for item_id in ids)
    embeddings = np.zeros((len(ids), maximum, XLSR_EMBEDDING_DIM), dtype=np.float32)
    mask = np.zeros((len(ids), maximum), dtype=np.bool_)
    starts = np.full((len(ids), maximum), -1, dtype=np.int64)
    for index, item_id in enumerate(ids):
        matrix, positions = records[item_id]
        count = matrix.shape[0]
        embeddings[index, :count] = matrix
        mask[index, :count] = True
        starts[index, :count] = positions
    return {
        "ids": np.asarray(ids),
        "embeddings": embeddings,
        "mask": mask,
        "starts": starts,
        "window": np.asarray(window, dtype=np.int64),
        "sample_rate": np.asarray(16_000, dtype=np.int64),
    }


def merge_run(work_dir: Path, output_dir: Path | None = None) -> Path:
    """Strictly merge exact predictions and XLS-R archives by dataset."""
    manifest, mapping = validate_prepared_run(work_dir)
    work_dir = work_dir.resolve()
    output_dir = (output_dir or (work_dir / "cache")).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite merged cache: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.merging-", dir=output_dir.parent,
    ))
    all_ids = mapping["COMBINED_ID"].tolist()
    locations = {
        row["COMBINED_ID"]: (row["DATASET_KEY"], int(row["ROW_INDEX"]))
        for row in mapping.to_dict("records")
    }
    dataset_rows = {dataset["key"]: int(dataset["rows"]) for dataset in manifest["datasets"]}
    prediction_parts = []
    records: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    input_hashes = []
    window = None
    eat_spool = None
    spear_spool = None
    try:
        for index in range(manifest["num_shards"]):
            expected_ids = all_ids[index::manifest["num_shards"]]
            shard_output = work_dir / "shards" / f"shard_{index:03d}" / "output"
            prediction_path = shard_output / "submission.csv"
            cache_path = shard_output / "xlsr_windows.npz"
            eat_path = shard_output / "eat_patch_graph.npz"
            spear_path = shard_output / "spear_component_bins.npz"
            if not all(path.is_file() for path in (prediction_path, cache_path, eat_path, spear_path)):
                raise FileNotFoundError(f"Shard {index} output is incomplete")
            frame = pd.read_csv(prediction_path, dtype={"ID": str})
            if tuple(frame.columns) != ("ID", *PREDICTION_COLUMNS):
                raise ValueError(f"Prediction schema mismatch: {prediction_path}")
            if frame["ID"].tolist() != expected_ids or frame["ID"].duplicated().any():
                raise ValueError(f"Prediction ID/order mismatch: {prediction_path}")
            numeric = frame.loc[:, PREDICTION_COLUMNS].apply(pd.to_numeric, errors="coerce")
            if not np.isfinite(numeric.to_numpy()).all():
                raise ValueError(f"Non-finite prediction: {prediction_path}")
            if ((numeric < 0) | (numeric > 1)).any().any():
                raise ValueError(f"Prediction outside [0, 1]: {prediction_path}")
            frame.loc[:, PREDICTION_COLUMNS] = numeric
            prediction_parts.append(frame)

            archive = _load_validated_npz(cache_path, expected_ids)
            archive_window = int(archive["window"])
            if window is None:
                window = archive_window
            elif window != archive_window:
                raise ValueError("Shard XLS-R window sizes differ")
            for row_index, item_id in enumerate(expected_ids):
                count = int(archive["mask"][row_index].sum())
                if item_id in records:
                    raise ValueError(f"Duplicate XLS-R row across shards: {item_id}")
                records[item_id] = (
                    archive["embeddings"][row_index, :count].copy(),
                    archive["starts"][row_index, :count].copy(),
                )

            eat = _load_validated_structured_npz(eat_path, expected_ids, "eat")
            if eat_spool is None:
                eat_spool = _StructuredSpool(
                    temporary / ".spool", "eat", eat, dataset_rows, locations,
                )
            eat_spool.add(eat, expected_ids)
            del eat
            spear = _load_validated_structured_npz(spear_path, expected_ids, "spear")
            if spear_spool is None:
                spear_spool = _StructuredSpool(
                    temporary / ".spool", "spear", spear, dataset_rows, locations,
                )
            spear_spool.add(spear, expected_ids)
            del spear
            input_hashes.append({
                "shard": index,
                "prediction_sha256": _sha256_file(prediction_path),
                "xlsr_sha256": _sha256_file(cache_path),
                "eat_patch_graph_sha256": _sha256_file(eat_path),
                "spear_component_bins_sha256": _sha256_file(spear_path),
                "rows": len(expected_ids),
            })
        if set(records) != set(all_ids) or sum(len(frame) for frame in prediction_parts) != len(all_ids):
            raise ValueError("Merged shard coverage is not exact")
        eat_spool.validate_complete()
        spear_spool.validate_complete()
        merged = pd.concat(prediction_parts, ignore_index=True).set_index("ID").loc[all_ids].reset_index()

        combined_predictions = temporary / "exact_v47_predictions.csv"
        merged.to_csv(combined_predictions, index=False, lineterminator="\r\n")
        shutil.copy2(work_dir / "corpus" / "dataset_map.csv", temporary / "dataset_map.csv")
        dataset_outputs = []
        for dataset in manifest["datasets"]:
            key = dataset["key"]
            selected_map = mapping.loc[mapping["DATASET_KEY"].eq(key)]
            combined = selected_map["COMBINED_ID"].tolist()
            original = selected_map["ORIGINAL_ID"].tolist()
            if len(combined) != dataset["rows"] or _ids_sha256(original) != dataset["ids_sha256"]:
                raise ValueError(f"Per-dataset mapping rows changed: {key}")
            dataset_dir = temporary / "datasets" / key
            dataset_dir.mkdir(parents=True)
            predictions = merged.set_index("ID").loc[combined].reset_index(drop=True)
            predictions.insert(0, "ID", original)
            prediction_output = dataset_dir / "exact_v47_predictions.csv"
            predictions.to_csv(prediction_output, index=False, lineterminator="\r\n")
            arrays = _pack_records(combined, records, int(window))
            arrays["ids"] = np.asarray(original)
            cache_output = dataset_dir / "xlsr_windows.npz"
            _write_deterministic_npz(cache_output, arrays)
            # Load the final artifact through the same strict validator.
            _load_validated_npz(cache_output, original)
            eat_output = dataset_dir / "eat_patch_graph.npz"
            spear_output = dataset_dir / "spear_component_bins.npz"
            _write_deterministic_npz(
                eat_output, eat_spool.output_arrays(key, original),
            )
            _write_deterministic_npz(
                spear_output, spear_spool.output_arrays(key, original),
            )
            compatibility = (
                ("anchor", "predictions.csv", prediction_output),
                ("xlsr", "features.npz", cache_output),
                ("eat", "features.npz", eat_output),
                ("spear", "features.npz", spear_output),
            )
            for root_name, filename, target in compatibility:
                link_dir = temporary / root_name / key
                link_dir.mkdir(parents=True, exist_ok=True)
                (link_dir / filename).symlink_to(
                    os.path.relpath(target, link_dir)
                )
            dataset_outputs.append({
                "key": key,
                "role": dataset["role"],
                "rows": len(original),
                "ids_sha256": _ids_sha256(original),
                "prediction_sha256": _sha256_file(prediction_output),
                "xlsr_sha256": _sha256_file(cache_output),
                "eat_patch_graph_sha256": _sha256_file(eat_output),
                "spear_component_bins_sha256": _sha256_file(spear_output),
            })
        eat_spool.close()
        spear_spool.close()
        shutil.rmtree(temporary / ".spool")
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "run_manifest_sha256": _sha256_file(work_dir / "run_manifest.json"),
            "combined_prediction_sha256": _sha256_file(combined_predictions),
            "dataset_map_sha256": _sha256_file(temporary / "dataset_map.csv"),
            "rows": len(all_ids),
            "window": int(window),
            "inputs": input_hashes,
            "datasets": dataset_outputs,
        }
        (temporary / "merge_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        temporary.rename(output_dir)
    except BaseException:
        if eat_spool is not None:
            eat_spool.close()
        if spear_spool is not None:
            spear_spool.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        f"Merged {manifest['num_shards']} shards and {len(all_ids)} rows into "
        f"{len(manifest['datasets'])} label-blind dataset caches at {output_dir}"
    )
    return output_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Validate and materialize; no inference")
    prepare.add_argument("--config", type=Path, default=ROOT / "configs/data_partitions.yaml")
    prepare.add_argument("--package", type=Path, default=ROOT / "component_query_or_v47")
    prepare.add_argument("--current-pipeline", type=Path, default=ROOT / "src/pipeline.py")
    prepare.add_argument("--work-dir", type=Path, required=True)
    prepare.add_argument(
        "--dataset", action="append", required=True,
        help="Exact config-declared train/development truth path; repeat to combine",
    )
    prepare.add_argument("--num-shards", type=int, required=True)

    locked = subparsers.add_parser(
        "prepare-locked",
        help="Prepare a separately registered locked_eval exactly once",
    )
    locked.add_argument(
        "--config", type=Path, default=ROOT / "configs/data_partitions.yaml",
    )
    locked.add_argument(
        "--package", type=Path, default=ROOT / "component_query_or_v47",
    )
    locked.add_argument(
        "--current-pipeline", type=Path, default=ROOT / "src/pipeline.py",
    )
    locked.add_argument("--work-dir", type=Path, required=True)
    locked.add_argument(
        "--dataset", action="append", required=True,
        help="Exact config-declared locked_eval truth path",
    )
    locked.add_argument("--num-shards", type=int, required=True)
    locked.add_argument("--confirm", required=True)

    launch = subparsers.add_parser("launch", help="Explicitly start the prepared GPU run")
    launch.add_argument("--work-dir", type=Path, required=True)
    launch.add_argument("--gpus", required=True, help="Comma-separated GPU IDs, one per shard")
    launch.add_argument("--python", default=sys.executable)
    launch.add_argument("--confirm", required=True)

    adopt = subparsers.add_parser(
        "adopt-schema-v1-base",
        help="Verify/adopt base predictions and XLS-R from the known v1 shard failure",
    )
    adopt.add_argument("--base-work-dir", type=Path, required=True)
    adopt.add_argument("--target-work-dir", type=Path, required=True)
    adopt.add_argument("--confirm", required=True)

    post = subparsers.add_parser(
        "launch-post-fusions",
        help="Run exact post-pipeline v47 fusions on verified adopted base outputs",
    )
    post.add_argument("--work-dir", type=Path, required=True)
    post.add_argument("--gpus", required=True, help="Comma-separated GPU IDs, one per shard")
    post.add_argument("--python", default=sys.executable)
    post.add_argument("--confirm", required=True)

    merge = subparsers.add_parser("merge", help="Validate and merge completed shards")
    merge.add_argument("--work-dir", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path, default=None)
    return parser


def main(argv=None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare_run(
            args.config, args.package, args.current_pipeline, args.work_dir,
            args.dataset, args.num_shards,
        )
    elif args.command == "prepare-locked":
        if args.confirm != "PREPARE_ONE_SHOT_LOCKED_CACHE":
            raise ValueError(
                "locked preparation requires --confirm "
                "PREPARE_ONE_SHOT_LOCKED_CACHE"
            )
        prepare_run(
            args.config, args.package, args.current_pipeline, args.work_dir,
            args.dataset, args.num_shards, cache_policy="one_shot_locked",
        )
    elif args.command == "launch":
        launch_run(
            args.work_dir, args.gpus.split(","), args.python, args.confirm,
        )
    elif args.command == "adopt-schema-v1-base":
        adopt_schema_v1_base_outputs(
            args.base_work_dir, args.target_work_dir, args.confirm,
        )
    elif args.command == "launch-post-fusions":
        launch_post_fusions(
            args.work_dir, args.gpus.split(","), args.python, args.confirm,
        )
    elif args.command == "merge":
        merge_run(args.work_dir, args.output_dir)


if __name__ == "__main__":
    main()
