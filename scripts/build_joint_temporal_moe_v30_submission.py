#!/usr/bin/env python3
"""Add the separation-free joint temporal expert to the verified v29 ZIP."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
IMPORT_MARKER = (
    "from spear_temporal_bin_inference import "
    "apply_spear_temporal_bin_fusion  # noqa: E402\n"
)
IMPORT_INSERT = IMPORT_MARKER + (
    "from spear_temporal_joint_inference import "
    "apply_spear_temporal_joint_fusion  # noqa: E402\n"
)
TEMPORAL_CALL = '''    apply_spear_temporal_bin_fusion(
        args.output, temporal_bin_stats,
        BASE_DIR / "model" / "spear-temporal-bin" / "head.pt",
        device=args.device, music_weight=0.40,
        file_consistency_update_weight=0.75,
    )
'''
JOINT_CALL = TEMPORAL_CALL + '''    apply_spear_temporal_joint_fusion(
        args.output, temporal_bin_stats,
        BASE_DIR / "model" / "spear-temporal-joint" / "head.pt",
        device=args.device, file_weight=0.25, voice_weight=0.25,
    )
'''


def copy(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-zip", type=Path, default=ROOT / "temporal_bin_moe_v29.zip"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=(
        ROOT / "reports/spear_temporal_joint_v1/seed09/"
        "spear_temporal_joint_head.pt"
    ))
    parser.add_argument(
        "--archive", action="store_true",
        help="Also create OUTPUT.zip with the required three top-level entries.",
    )
    args = parser.parse_args()
    for path in (args.base_zip, args.checkpoint):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    archive_path = args.output.with_suffix(".zip")
    if args.archive and archive_path.exists():
        raise FileExistsError(f"Refusing to overwrite {archive_path}")

    shutil.unpack_archive(str(args.base_zip), str(args.output), format="zip")
    source_dir = args.output / "model" / "src"
    for name in (
        "spear_temporal_joint_head.py", "spear_temporal_joint_inference.py",
    ):
        copy(ROOT / "src" / name, source_dir / name)
    head_dir = args.output / "model" / "spear-temporal-joint"
    head_dir.mkdir(parents=True)
    shutil.copy2(args.checkpoint, head_dir / "head.pt")

    script_path = args.output / "script.py"
    script = script_path.read_text(encoding="utf-8")
    if script.count(IMPORT_MARKER) != 1 or script.count(TEMPORAL_CALL) != 1:
        raise ValueError("v29 entrypoint markers changed; refusing unsafe injection")
    script = script.replace(IMPORT_MARKER, IMPORT_INSERT)
    script = script.replace(TEMPORAL_CALL, JOINT_CALL)
    script_path.write_text(script, encoding="utf-8")

    if args.archive:
        shutil.make_archive(str(args.output), "zip", root_dir=args.output)
        print(f"Built {args.output} and {archive_path}")
    else:
        print(f"Built {args.output}")


if __name__ == "__main__":
    main()
