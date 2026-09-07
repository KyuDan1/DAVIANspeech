#!/usr/bin/env python3
"""Add a Music-only temporal residual to the exact-v18 File-router candidate."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
IMPORT_MARKER = (
    "from spear_temporal_joint_inference import "
    "apply_spear_temporal_joint_fusion  # noqa: E402\n"
)
IMPORT_REPLACEMENT = IMPORT_MARKER + (
    "from spear_temporal_bin_inference import "
    "apply_spear_temporal_bin_music_only  # noqa: E402\n"
)
ATTENTION_CALL = '''    apply_spear_temporal_joint_fusion(
        args.output, temporal_bin_stats,
        BASE_DIR / "model" / "spear-temporal-attention" / "head.pt",
        device=args.device, file_weight=0.20, voice_weight=0.0,
        telephone_ids_path=telephone_ids,
        phone_file_weight=0.25, phone_voice_weight=0.0,
    )
'''
MUSIC_CALL = ATTENTION_CALL + '''    apply_spear_temporal_bin_music_only(
        args.output, temporal_bin_stats,
        BASE_DIR / "model" / "spear-temporal-music" / "head.pt",
        device=args.device, music_weight=0.50,
    )
'''


def replace_file(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)


def metadata(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return (
        np.asarray(checkpoint["projection"]),
        np.asarray(checkpoint["layers"]),
        int(checkpoint["bins"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-zip", type=Path,
        default=ROOT / "v18_channel_attention_router_v32.zip",
    )
    parser.add_argument(
        "--music-checkpoint", type=Path,
        default=ROOT / "reports/spear_temporal_bin_v1/mlp64_seed05/spear_temporal_bin_head.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    for path in (args.base_zip, args.music_checkpoint):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    archive = args.output.with_suffix(".zip")
    if args.archive and archive.exists():
        raise FileExistsError(f"Refusing to overwrite {archive}")

    shutil.unpack_archive(str(args.base_zip), str(args.output), format="zip")
    attention_checkpoint = (
        args.output / "model" / "spear-temporal-attention" / "head.pt"
    )
    attention_metadata = metadata(attention_checkpoint)
    music_metadata = metadata(args.music_checkpoint)
    if (
        attention_metadata[2] != music_metadata[2]
        or not np.array_equal(attention_metadata[0], music_metadata[0])
        or not np.array_equal(attention_metadata[1], music_metadata[1])
    ):
        raise ValueError("attention and Music heads require different SPEAR statistics")

    source_dir = args.output / "model" / "src"
    for name in (
        "spear_temporal_bin_head.py", "spear_temporal_bin_inference.py",
    ):
        replace_file(ROOT / "src" / name, source_dir / name)
    head_dir = args.output / "model" / "spear-temporal-music"
    head_dir.mkdir(parents=True)
    shutil.copy2(args.music_checkpoint, head_dir / "head.pt")

    entrypoint = args.output / "script.py"
    script = entrypoint.read_text(encoding="utf-8")
    for old, new, label in (
        (IMPORT_MARKER, IMPORT_REPLACEMENT, "import"),
        (ATTENTION_CALL, MUSIC_CALL, "Music-only fusion"),
    ):
        if script.count(old) != 1:
            raise ValueError(f"v32 {label} marker changed; refusing unsafe injection")
        script = script.replace(old, new)
    entrypoint.unlink()
    entrypoint.write_text(script, encoding="utf-8")

    if args.archive:
        shutil.make_archive(str(args.output), "zip", root_dir=args.output)
        print(f"Built {args.output} and {archive}")
    else:
        print(f"Built {args.output}")


if __name__ == "__main__":
    main()
