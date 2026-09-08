#!/usr/bin/env python3
"""Package one selected three-stream residual on top of exact v47."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import shutil
import zipfile

import torch

try:
    from .run_v47_anchor_cache import build_pipeline_overlay
except ImportError:  # Direct script execution.
    from run_v47_anchor_cache import build_pipeline_overlay


ROOT = Path(__file__).resolve().parents[1]
IMPORT_MARKER = (
    "from component_query_mhfa_inference import "
    "apply_component_query_mhfa_fusion  # noqa: E402\n"
)
IMPORT_INSERT = IMPORT_MARKER + (
    "from three_stream_anchor_residual_inference import "
    "apply_three_stream_anchor_residual  # noqa: E402\n"
)
STATS_MARKER = '''    spear_component_bins = BASE_DIR / "output" / ".spear_component_bins.npz"
'''
STATS_INSERT = STATS_MARKER + '''    xlsr_windows = BASE_DIR / "output" / ".xlsr_original_windows.npz"
'''
RUN_MARKER = "    run(args)\n"
RUN_INSERT = '''    args.xlsr_window_embeddings_output = xlsr_windows
    run(args)
'''
QUERY_MARKER = '''    apply_component_query_mhfa_fusion(
        args.output, eat_patch_graph, spear_component_bins,
        BASE_DIR / "model" / "component-query" / "head.pt",
        device=args.device, file_weight=0.025,
        music_weight=0.05,
        file_or_weight=0.3,
    )
'''
RESIDUAL_INSERT = QUERY_MARKER + '''    apply_three_stream_anchor_residual(
        args.output, eat_patch_graph, spear_component_bins, xlsr_windows,
        sorted((BASE_DIR / "model" / "three-stream-residual").glob("head_*.pt")),
        device=args.device, batch_size=64,
    )
'''
CLEANUP_MARKER = (
    "    for path in (eat_stats, spear_stats, eat_patch_graph, "
    "spear_component_bins):\n"
)
CLEANUP_INSERT = (
    "    for path in (eat_stats, spear_stats, eat_patch_graph, "
    "spear_component_bins, xlsr_windows):\n"
)


def replace_once(text: str, marker: str, replacement: str, label: str) -> str:
    if text.count(marker) != 1:
        raise ValueError(f"exact v47 {label} marker changed; refusing injection")
    return text.replace(marker, replacement)


def patch_entrypoint(text: str) -> str:
    for marker, replacement, label in (
        (IMPORT_MARKER, IMPORT_INSERT, "import"),
        (STATS_MARKER, STATS_INSERT, "cache path"),
        (RUN_MARKER, RUN_INSERT, "base run"),
        (QUERY_MARKER, RESIDUAL_INSERT, "residual call"),
        (CLEANUP_MARKER, CLEANUP_INSERT, "cleanup"),
    ):
        text = replace_once(text, marker, replacement, label)
    ast.parse(text)
    return text


def make_zip(source: Path, archive: Path) -> None:
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED,
        compresslevel=1, allowZip64=True,
    ) as handle:
        for path in sorted(source.rglob("*")):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            handle.write(path, path.relative_to(source))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-zip", type=Path, default=ROOT / "component_query_or_v47.zip",
    )
    parser.add_argument(
        "--checkpoint", type=Path, action="append", required=True,
        help="Repeat to package a fixed equal-logit checkpoint ensemble.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    for path in (args.base_zip, *args.checkpoint):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    archive = args.output.with_suffix(".zip")
    if args.archive and (archive.exists() or archive.is_symlink()):
        raise FileExistsError(f"refusing to overwrite {archive}")
    for checkpoint_path in args.checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_type") != "three_stream_anchor_residual_component_query_v1":
            parser.error("checkpoint is not a three-stream anchor residual")
        config = checkpoint.get("config", {})
        if config.get("xlsr_dimension") is None:
            parser.error("deployable checkpoint must include original-mixture XLS-R")
        if int(config.get("maximum_xlsr_windows", 0)) < 16:
            parser.error("checkpoint does not reserve 60-second XLS-R positions")

    shutil.unpack_archive(str(args.base_zip), str(args.output), format="zip")
    source_dir = args.output / "model" / "src"
    package_pipeline = source_dir / "pipeline.py"
    overlay = build_pipeline_overlay(package_pipeline, ROOT / "src" / "pipeline.py")
    package_pipeline.write_text(overlay, encoding="utf-8")
    for name in (
        "three_stream_anchor_residual.py",
        "three_stream_anchor_residual_inference.py",
    ):
        shutil.copy2(ROOT / "src" / name, source_dir / name)
    checkpoint_dir = args.output / "model" / "three-stream-residual"
    checkpoint_dir.mkdir(parents=True)
    for index, checkpoint_path in enumerate(args.checkpoint):
        shutil.copy2(checkpoint_path, checkpoint_dir / f"head_{index:02d}.pt")

    entrypoint = args.output / "script.py"
    entrypoint.write_text(
        patch_entrypoint(entrypoint.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    if args.archive:
        make_zip(args.output, archive)
        print(f"Built {args.output} and {archive}")
    else:
        print(f"Built {args.output}")


if __name__ == "__main__":
    main()
