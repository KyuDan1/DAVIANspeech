#!/usr/bin/env python3
"""Add soft component-to-File consistency to the routed v27 archive."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
IMPORT_MARKER = (
    "from long_horizon_music_inference import "
    "apply_long_horizon_music_fusion  # noqa: E402\n"
)
IMPORT_INSERT = IMPORT_MARKER + (
    "from presence_weighted_file_fusion import "
    "apply_presence_weighted_file_fusion  # noqa: E402\n"
)
CLEANUP_MARKER = "    for path in (eat_stats, spear_stats, telephone_ids):\n"
CALL_INSERT = '''    apply_presence_weighted_file_fusion(
        args.output, file_weight=0.5, presence_logit_weight=0.5,
    )
''' + CLEANUP_MARKER


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-zip", type=Path,
        default=ROOT / "routed_long_horizon_v27.zip",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.base_zip.is_file():
        parser.error(f"missing input: {args.base_zip}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    shutil.unpack_archive(str(args.base_zip), str(args.output), format="zip")
    shutil.copy2(
        ROOT / "src/presence_weighted_file_fusion.py",
        args.output / "model/src/presence_weighted_file_fusion.py",
    )
    script_path = args.output / "script.py"
    script = script_path.read_text(encoding="utf-8")
    if script.count(IMPORT_MARKER) != 1 or script.count(CLEANUP_MARKER) != 1:
        raise ValueError("v27 entrypoint markers changed; refusing unsafe injection")
    script = script.replace(IMPORT_MARKER, IMPORT_INSERT)
    script = script.replace(CLEANUP_MARKER, CALL_INSERT)
    script_path.write_text(script, encoding="utf-8")
    print(f"Built {args.output} from {args.base_zip}")


if __name__ == "__main__":
    main()
