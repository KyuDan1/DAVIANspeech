#!/usr/bin/env python3
"""Build v32: v30 soft MoE plus a conservatively phone-routed attention head."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_CALL = '''    apply_spear_temporal_joint_fusion(
        args.output, temporal_bin_stats,
        BASE_DIR / "model" / "spear-temporal-joint" / "head.pt",
        device=args.device, file_weight=0.25, voice_weight=0.25,
    )
'''
ROUTED_CALL = ANCHOR_CALL + '''    apply_spear_temporal_joint_fusion(
        args.output, temporal_bin_stats,
        BASE_DIR / "model" / "spear-temporal-attention" / "head.pt",
        device=args.device, file_weight=0.125, voice_weight=0.0,
        telephone_ids_path=telephone_ids,
        phone_file_weight=0.20, phone_voice_weight=0.0,
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
    parser.add_argument(
        "--base-dir", type=Path, default=ROOT / "joint_temporal_moe_v30"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=(
            ROOT / "reports/spear_temporal_attention_v1/attention_consistent_s12/"
            "spear_temporal_joint_head.pt"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    for path in (args.base_dir, args.checkpoint):
        if not path.exists():
            parser.error(f"missing input: {path}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    archive = args.output.with_suffix(".zip")
    if args.archive and archive.exists():
        raise FileExistsError(f"Refusing to overwrite {archive}")

    shutil.copytree(args.base_dir, args.output, copy_function=hardlink_or_copy)
    source_dir = args.output / "model" / "src"
    for name in ("spear_temporal_joint_head.py", "spear_temporal_joint_inference.py"):
        replace_file(ROOT / "src" / name, source_dir / name)
    head_dir = args.output / "model" / "spear-temporal-attention"
    head_dir.mkdir(parents=True)
    shutil.copy2(args.checkpoint, head_dir / "head.pt")

    entrypoint = args.output / "script.py"
    script = entrypoint.read_text(encoding="utf-8")
    if script.count(ANCHOR_CALL) != 1:
        raise ValueError("v30 entrypoint marker changed; refusing unsafe injection")
    entrypoint.unlink()
    entrypoint.write_text(script.replace(ANCHOR_CALL, ROUTED_CALL), encoding="utf-8")

    if args.archive:
        shutil.make_archive(str(args.output), "zip", root_dir=args.output)
        print(f"Built {args.output} and {archive}")
    else:
        print(f"Built {args.output}")


if __name__ == "__main__":
    main()
