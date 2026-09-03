#!/usr/bin/env python3
"""Build v41 by smoothing the five-view WPT File aggregation."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from build_wpt_fixed_moe_v39_submission import make_zip


ROOT = Path(__file__).resolve().parents[1]
CALL_MARKER = '''        unified_expert, device=args.device, file_batch_size=6,
        file_views=5, voice_outer_weight=0.10, file_outer_weight=0.75,
'''
CALL_INSERT = '''        unified_expert, device=args.device, file_batch_size=6,
        file_views=5, file_temperature=2.0,
        voice_outer_weight=0.10, file_outer_weight=0.75,
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-zip", type=Path,
        default=ROOT / "wpt_multiview_file_v40.zip",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    if not args.base_zip.is_file():
        parser.error(f"missing base archive: {args.base_zip}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    archive = args.output.with_suffix(".zip")
    if args.archive and archive.exists():
        raise FileExistsError(f"Refusing to overwrite {archive}")

    shutil.unpack_archive(str(args.base_zip), str(args.output), format="zip")
    shutil.copy2(
        ROOT / "src/wpt_spectra_inference.py",
        args.output / "model/src/wpt_spectra_inference.py",
    )
    entrypoint = args.output / "script.py"
    script = entrypoint.read_text(encoding="utf-8")
    if script.count(CALL_MARKER) != 1:
        raise ValueError("v40 WPT call marker changed; refusing unsafe injection")
    entrypoint.write_text(
        script.replace(CALL_MARKER, CALL_INSERT), encoding="utf-8"
    )
    if args.archive:
        make_zip(args.output, archive)
        print(f"Built {args.output} and {archive}")
    else:
        print(f"Built {args.output}")


if __name__ == "__main__":
    main()
