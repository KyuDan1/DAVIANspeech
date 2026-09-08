#!/usr/bin/env python3
"""Overlay the frozen channel-robust Music-only v51 expert onto v50."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINTS = (
    ROOT / "reports/music_expert_v51/music_cq_seed11/component_query_mhfa.pt",
    ROOT / "reports/music_expert_v51/music_cq_inv001_seed13/component_query_mhfa.pt",
    ROOT / "reports/music_expert_v51/music_cq_inv005_seed14/component_query_mhfa.pt",
)
EXPECTED_CHECKPOINT_SHA256 = (
    "32d846904629c7a99c253fdd7fbd7931176ea919c20652344a7422d5e4f6a3dd",
    "b32c8fdcef7f4aa34d1d78833fc5a18b35e9664a49a626de4ae2111b78a15316",
    "8cd61a05bfcf974ad396a828a0a06345ff87a959ae0f3b5a0db24a2103b3c0a8",
)
EXPECTED_BASE_SCRIPT_SHA256 = (
    "238cc39a493e9df9e5a9e088c9f97494641e03ddee923df3c04c8f7806897713"
)

IMPORT_MARKER = (
    "from component_query_mhfa_inference import "
    "apply_component_query_mhfa_fusion  # noqa: E402\n"
)
IMPORT_INSERT = IMPORT_MARKER + (
    "from music_only_residual import "
    "apply_component_query_music_residual  # noqa: E402\n"
)
CALL_MARKER = '''    apply_component_query_mhfa_fusion(
        args.output, eat_patch_graph, spear_component_bins,
        BASE_DIR / "model" / "component-query" / "head.pt",
        device=args.device, file_weight=0.025,
        music_weight=0.05,
        file_or_weight=0.3,
    )
'''
CALL_INSERT = CALL_MARKER + '''    # Frozen v51: original-mixture Music only; no extra backbone pass.
    apply_component_query_music_residual(
        args.output, eat_patch_graph, spear_component_bins,
        sorted((BASE_DIR / "model" / "music-only-v51").glob("head_*.pt")),
        device=args.device, music_weight=0.0375,
    )
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inject_script(source: str) -> str:
    replacements = (
        (IMPORT_MARKER, IMPORT_INSERT, "Music residual import"),
        (CALL_MARKER, CALL_INSERT, "Music residual call"),
    )
    for marker, replacement, label in replacements:
        if source.count(marker) != 1:
            raise ValueError(f"v50 {label} marker changed; refusing unsafe overlay")
        source = source.replace(marker, replacement)
    return source


def hardlink_or_copy(source: str, destination: str) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def detached_copy(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir", type=Path,
        default=ROOT / "sparse_voice_wpt_file_v50_frozen",
    )
    parser.add_argument(
        "--checkpoints", type=Path, nargs=3,
        default=list(DEFAULT_CHECKPOINTS),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.base_dir.is_dir():
        parser.error(f"missing v50 base directory: {args.base_dir}")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    base_script = args.base_dir / "script.py"
    if sha256(base_script) != EXPECTED_BASE_SCRIPT_SHA256:
        raise ValueError("v50 base script hash changed; refusing unsafe overlay")
    for path, expected in zip(args.checkpoints, EXPECTED_CHECKPOINT_SHA256):
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Music checkpoint hash mismatch: {path}")

    shutil.copytree(
        args.base_dir, args.output,
        copy_function=hardlink_or_copy,
    )
    source_dir = args.output / "model/src"
    detached_copy(ROOT / "src/music_only_residual.py", source_dir / "music_only_residual.py")
    model_dir = args.output / "model/music-only-v51"
    model_dir.mkdir(parents=True)
    for index, checkpoint in enumerate(args.checkpoints):
        detached_copy(checkpoint, model_dir / f"head_{index:02d}.pt")

    # Break the hardlink before modifying the generated entrypoint.
    entrypoint = args.output / "script.py"
    source = inject_script(entrypoint.read_text("utf-8"))
    entrypoint.unlink()
    entrypoint.write_text(source, encoding="utf-8")
    print(f"Built {args.output}")


if __name__ == "__main__":
    main()
