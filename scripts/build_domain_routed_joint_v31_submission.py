#!/usr/bin/env python3
"""Build v31: phone-conditioned soft joint-expert weights on exact v30."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
OLD_CALL = '''    apply_spear_temporal_joint_fusion(
        args.output, temporal_bin_stats,
        BASE_DIR / "model" / "spear-temporal-joint" / "head.pt",
        device=args.device, file_weight=0.25, voice_weight=0.25,
    )
'''
NEW_CALL = '''    apply_spear_temporal_joint_fusion(
        args.output, temporal_bin_stats,
        BASE_DIR / "model" / "spear-temporal-joint" / "head.pt",
        device=args.device, file_weight=0.225, voice_weight=0.30,
        telephone_ids_path=telephone_ids,
        phone_file_weight=0.40, phone_voice_weight=0.35,
    )
'''


def hardlink_or_copy(source: str, destination: str) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def replace_file(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path,
                        default=ROOT / "joint_temporal_moe_v30")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    if not args.base_dir.is_dir():
        parser.error(f"missing base directory: {args.base_dir}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    archive = args.output.with_suffix(".zip")
    if args.archive and archive.exists():
        raise FileExistsError(f"Refusing to overwrite {archive}")

    shutil.copytree(args.base_dir, args.output, copy_function=hardlink_or_copy)
    replace_file(
        ROOT / "src" / "spear_temporal_joint_inference.py",
        args.output / "model" / "src" / "spear_temporal_joint_inference.py",
    )
    entrypoint = args.output / "script.py"
    script = entrypoint.read_text(encoding="utf-8")
    if script.count(OLD_CALL) != 1:
        raise ValueError("v30 entrypoint marker changed; refusing unsafe injection")
    entrypoint.unlink()
    entrypoint.write_text(script.replace(OLD_CALL, NEW_CALL), encoding="utf-8")

    if args.archive:
        shutil.make_archive(str(args.output), "zip", root_dir=args.output)
        print(f"Built {args.output} and {archive}")
    else:
        print(f"Built {args.output}")


if __name__ == "__main__":
    main()
