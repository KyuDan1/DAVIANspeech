#!/usr/bin/env python3
"""Add the maximin component-to-File consistency residual to v34."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
IMPORT_MARKER = (
    "from spear_temporal_bin_inference import "
    "apply_spear_temporal_bin_music_only  # noqa: E402\n"
)
IMPORT_REPLACEMENT = IMPORT_MARKER + (
    "from presence_weighted_file_fusion import "
    "apply_presence_weighted_file_fusion  # noqa: E402\n"
)
INVARIANT_CALL = '''    apply_dual_domain_fusion(
        args.output, eat_stats, spear_stats,
        sorted((BASE_DIR / "model" / "channel-invariant-v4").glob("head_*.pt")),
        device=args.device, file_weight=0.125,
        voice_weight=0.20, music_weight=0.10,
    )
'''
CONSISTENCY_CALL = INVARIANT_CALL + '''    apply_presence_weighted_file_fusion(
        args.output, file_weight=0.50, presence_logit_weight=0.0,
    )
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-zip", type=Path, default=ROOT / "v34_invariant.zip",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    if not args.base_zip.is_file():
        parser.error(f"missing input: {args.base_zip}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    archive = args.output.with_suffix(".zip")
    if args.archive and archive.exists():
        raise FileExistsError(f"Refusing to overwrite {archive}")

    shutil.unpack_archive(str(args.base_zip), str(args.output), format="zip")
    shutil.copy2(
        ROOT / "src" / "presence_weighted_file_fusion.py",
        args.output / "model" / "src" / "presence_weighted_file_fusion.py",
    )
    entrypoint = args.output / "script.py"
    script = entrypoint.read_text(encoding="utf-8")
    if script.count(IMPORT_MARKER) != 1:
        raise ValueError("v34 import marker changed; refusing unsafe injection")
    if script.count(INVARIANT_CALL) != 1:
        raise ValueError("v34 fusion marker changed; refusing unsafe injection")
    script = script.replace(IMPORT_MARKER, IMPORT_REPLACEMENT)
    script = script.replace(INVARIANT_CALL, CONSISTENCY_CALL)
    entrypoint.unlink()
    entrypoint.write_text(script, encoding="utf-8")

    if args.archive:
        shutil.make_archive(str(args.output), "zip", root_dir=args.output)
        print(f"Built {args.output} and {archive}")
    else:
        print(f"Built {args.output}")


if __name__ == "__main__":
    main()
