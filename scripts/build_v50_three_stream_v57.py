#!/usr/bin/env python3
"""Build an unpacked exact-v50 plus one strict v57 task residual package."""

from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import shutil
import zipfile

import torch

try:
    from .run_v47_anchor_cache import build_pipeline_overlay
except ImportError:
    from run_v47_anchor_cache import build_pipeline_overlay


ROOT = Path(__file__).resolve().parents[1]
TASK_MODES = ("voice", "music", "file")
BASE_ZIP_SHA256 = "887783b6447924fba2bb793831cdd4b7016ec36b11401da02cd98d4b2295d2df"
BASE_SCRIPT_SHA256 = "238cc39a493e9df9e5a9e088c9f97494641e03ddee923df3c04c8f7806897713"
BASE_PIPELINE_SHA256 = "c134b3045ce9943befbca39787ce4e815b6e85e62b468c383d060a6a8a53ed62"
IMPORT_MARKER = (
    "from sparse_call_consensus import apply_sparse_call_consensus  # noqa: E402\n"
)
CACHE_MARKER = '''    spear_component_bins = BASE_DIR / "output" / ".spear_component_bins.npz"
'''
RUN_MARKER = "    run(args)\n"
FINAL_V50_MARKER = '''    apply_sparse_call_consensus(
        args.output, xlsr_consensus, spectra_consensus,
        BASE_DIR / "model" / "sparse-call-consensus.npz",
        voice_weight=0.60,
        file_voice_only_weight=0.0,
    )
'''
CLEANUP_MARKER = '''        eat_stats, spear_stats, eat_patch_graph, spear_component_bins,
        spectra_stats, xlsr_consensus, spectra_consensus,
'''


def replace_once(source: str, marker: str, replacement: str, label: str) -> str:
    if source.count(marker) != 1:
        raise ValueError(f"exact v50 {label} marker changed; refusing patch")
    return source.replace(marker, replacement)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patch_entrypoint(source: str, task_mode: str) -> str:
    if task_mode not in TASK_MODES:
        raise ValueError(f"task_mode must be one of {TASK_MODES}")
    source = replace_once(
        source, IMPORT_MARKER,
        IMPORT_MARKER
        + "from three_stream_anchor_residual_inference import "
        "apply_three_stream_anchor_residual  # noqa: E402\n",
        "runtime import",
    )
    source = replace_once(
        source, CACHE_MARKER,
        CACHE_MARKER
        + '''    xlsr_windows = BASE_DIR / "output" / ".xlsr_original_windows.npz"
''',
        "XLS-R cache path",
    )
    source = replace_once(
        source, RUN_MARKER,
        "    args.xlsr_window_embeddings_output = xlsr_windows\n" + RUN_MARKER,
        "base run",
    )
    source = replace_once(
        source, FINAL_V50_MARKER,
        FINAL_V50_MARKER
        + '''    apply_three_stream_anchor_residual(
        args.output, eat_patch_graph, spear_component_bins, xlsr_windows,
        BASE_DIR / "model" / "three-stream-v57" / "head.pt",
        device=args.device, batch_size=64, task_mode="TASK_MODE",
    )
'''.replace("TASK_MODE", task_mode),
        "post-v50 residual",
    )
    source = replace_once(
        source, CLEANUP_MARKER,
        '''        eat_stats, spear_stats, eat_patch_graph, spear_component_bins,
        xlsr_windows, spectra_stats, xlsr_consensus, spectra_consensus,
''',
        "cache cleanup",
    )
    ast.parse(source)
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-zip", type=Path, default=ROOT / "svwpt_v50_pkg.zip")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-mode", choices=TASK_MODES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    if not args.base_zip.is_file() or sha256(args.base_zip) != BASE_ZIP_SHA256:
        parser.error(f"exact v50 archive missing or changed: {args.base_zip}")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint missing: {args.checkpoint}")
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"refusing to overwrite: {args.output}")
    archive_path = args.output.with_suffix(".zip")
    if args.archive and (archive_path.exists() or archive_path.is_symlink()):
        parser.error(f"refusing to overwrite: {archive_path}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "three_stream_anchor_residual_component_query_v1":
        parser.error("checkpoint is not a v57 three-stream residual")
    if not checkpoint.get("selected_split_only"):
        parser.error("checkpoint did not enforce strict selected split")
    if checkpoint.get("strict_identity_contract", {}).get("overlap") != 0:
        parser.error("checkpoint strict identity audit did not pass")

    with zipfile.ZipFile(args.base_zip) as base:
        if base.testzip() is not None:
            parser.error("exact v50 base archive failed CRC")
        base.extractall(args.output)
    exact_files = {
        args.output / "script.py": BASE_SCRIPT_SHA256,
        args.output / "model" / "src" / "pipeline.py": BASE_PIPELINE_SHA256,
    }
    for path, expected in exact_files.items():
        if not path.is_file() or sha256(path) != expected:
            parser.error(f"exact v50 base file changed: {path}")
    entrypoint = args.output / "script.py"
    entrypoint_source = entrypoint.read_text("utf-8")
    entrypoint.unlink()
    entrypoint.write_text(
        patch_entrypoint(entrypoint_source, args.task_mode), encoding="utf-8"
    )
    source_dir = args.output / "model" / "src"
    pipeline = source_dir / "pipeline.py"
    pipeline_source = build_pipeline_overlay(pipeline, ROOT / "src" / "pipeline.py")
    pipeline.unlink()
    pipeline.write_text(pipeline_source, encoding="utf-8")
    for name in (
        "three_stream_anchor_residual.py",
        "three_stream_anchor_residual_inference.py",
    ):
        destination = source_dir / name
        if destination.exists():
            destination.unlink()
        shutil.copy2(ROOT / "src" / name, destination)
    target = args.output / "model" / "three-stream-v57"
    target.mkdir()
    shutil.copy2(args.checkpoint, target / "head.pt")
    if args.archive:
        with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_DEFLATED,
            compresslevel=1, allowZip64=True,
        ) as archive:
            for path in sorted(args.output.rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                    archive.write(path, path.relative_to(args.output))
        print(f"Built {args.output} and {archive_path}")
    else:
        print(f"Built unpacked package only: {args.output}")


if __name__ == "__main__":
    main()
