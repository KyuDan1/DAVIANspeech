#!/usr/bin/env python3
"""Add the verified separation-free SPEAR temporal-bin expert to v28."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
IMPORT_MARKER = (
    "from presence_weighted_file_fusion import "
    "apply_presence_weighted_file_fusion  # noqa: E402\n"
)
IMPORT_INSERT = IMPORT_MARKER + (
    "from spear_temporal_bin_inference import "
    "apply_spear_temporal_bin_fusion  # noqa: E402\n"
)
STAT_MARKER = '    spear_stats = BASE_DIR / "output" / ".spear_temporal_stats.npz"\n'
STAT_INSERT = STAT_MARKER + (
    '    temporal_bin_stats = BASE_DIR / "output" / ".spear_temporal_bins.npz"\n'
)
SPEAR_CALL_MARKER = (
    "        device=args.device, weight=0.10, statistics_output_path=spear_stats,\n"
)
SPEAR_CALL_INSERT = SPEAR_CALL_MARKER.rstrip("\n") + "\n" + (
    "        temporal_bin_output_path=temporal_bin_stats,\n"
    "        temporal_bin_checkpoint_path=(\n"
    "            BASE_DIR / \"model\" / \"spear-temporal-bin\" / \"head.pt\"\n"
    "        ),\n"
)
PRESENCE_CALL = '''    apply_presence_weighted_file_fusion(
        args.output, file_weight=0.5, presence_logit_weight=0.5,
    )
'''
TEMPORAL_CALL = PRESENCE_CALL + '''    apply_spear_temporal_bin_fusion(
        args.output, temporal_bin_stats,
        BASE_DIR / "model" / "spear-temporal-bin" / "head.pt",
        device=args.device, music_weight=0.40,
        file_consistency_update_weight=0.75,
    )
'''
CLEANUP_MARKER = "    for path in (eat_stats, spear_stats, telephone_ids):\n"
CLEANUP_INSERT = (
    "    for path in (eat_stats, spear_stats, temporal_bin_stats, telephone_ids):\n"
)


def copy(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-zip", type=Path,
                        default=ROOT / "presence_weighted_file_v28.zip")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=(
        ROOT / "reports/spear_temporal_bin_v1/mlp64_seed05/"
        "spear_temporal_bin_head.pt"
    ))
    parser.add_argument(
        "--archive", action="store_true",
        help="Also create OUTPUT.zip with the required three top-level entries.",
    )
    args = parser.parse_args()
    for path in (args.base_zip, args.checkpoint):
        if not path.is_file(): parser.error(f"missing input: {path}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    archive_path = args.output.with_suffix(".zip")
    if args.archive and archive_path.exists():
        raise FileExistsError(f"Refusing to overwrite {archive_path}")

    shutil.unpack_archive(str(args.base_zip), str(args.output), format="zip")
    source_dir = args.output / "model" / "src"
    for name in (
        "anchor_spear_stats_fusion.py", "spear_detector.py",
        "spear_temporal_bins.py", "spear_temporal_bin_head.py",
        "spear_temporal_bin_inference.py",
    ):
        copy(ROOT / "src" / name, source_dir / name)
    head_dir = args.output / "model" / "spear-temporal-bin"
    head_dir.mkdir(parents=True)
    shutil.copy2(args.checkpoint, head_dir / "head.pt")

    script_path = args.output / "script.py"
    script = script_path.read_text(encoding="utf-8")
    markers = (
        IMPORT_MARKER, STAT_MARKER, SPEAR_CALL_MARKER,
        PRESENCE_CALL, CLEANUP_MARKER,
    )
    if any(script.count(marker) != 1 for marker in markers):
        raise ValueError("v28 entrypoint markers changed; refusing unsafe injection")
    script = script.replace(IMPORT_MARKER, IMPORT_INSERT)
    script = script.replace(STAT_MARKER, STAT_INSERT)
    script = script.replace(SPEAR_CALL_MARKER, SPEAR_CALL_INSERT)
    script = script.replace(PRESENCE_CALL, TEMPORAL_CALL)
    script = script.replace(CLEANUP_MARKER, CLEANUP_INSERT)
    script_path.write_text(script, encoding="utf-8")

    if args.archive:
        shutil.make_archive(str(args.output), "zip", root_dir=args.output)
        print(f"Built {args.output} and {archive_path}")
    else:
        print(f"Built {args.output}")


if __name__ == "__main__":
    main()
