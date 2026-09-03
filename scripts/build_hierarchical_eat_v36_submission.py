#!/usr/bin/env python3
"""Add the selected hierarchical EAT music MoE to exact leaderboard v18."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINTS = (
    ROOT / "reports/hierarchical_eat_music_v1/no_dsu_seed00/hierarchical_eat_music.pt",
    ROOT / "reports/hierarchical_eat_music_v1/no_dsu_seed02/hierarchical_eat_music.pt",
)
IMPORT_MARKER = (
    "from modern_fakeprint_detector import apply_modern_fakeprint_fusion  # noqa: E402\n"
)
IMPORT_INSERT = IMPORT_MARKER + (
    "from hierarchical_eat_music_inference import "
    "apply_hierarchical_eat_music_fusion  # noqa: E402\n"
)
VARIABLE_MARKER = (
    '    spear_stats = BASE_DIR / "output" / ".spear_temporal_stats.npz"\n'
)
VARIABLE_INSERT = VARIABLE_MARKER + (
    '    hierarchical_stats = BASE_DIR / "output" / ".eat_hierarchical_stats.npz"\n'
    '    telephone_ids = BASE_DIR / "output" / ".telephone_ids.npz"\n'
)
TELEPHONE_MARKER = "        phone_voice_weight=0.10,\n"
TELEPHONE_INSERT = TELEPHONE_MARKER + (
    "        hierarchical_statistics_output_path=hierarchical_stats,\n"
    "        hierarchical_checkpoint_path=(\n"
    "            BASE_DIR / \"model/hierarchical-eat-music/head_00.pt\"\n"
    "        ),\n"
    "        telephone_ids_output_path=telephone_ids,\n"
)
CLEANUP_MARKER = "    for path in (eat_stats, spear_stats):\n"
CLEANUP_INSERT = '''    apply_hierarchical_eat_music_fusion(
        args.output, hierarchical_stats,
        sorted((BASE_DIR / "model/hierarchical-eat-music").glob("head_*.pt")),
        device=args.device, telephone_ids_path=telephone_ids,
        music_weight=0.20, file_weight=0.10,
        phone_music_weight=0.30, phone_file_weight=0.30,
        file_music_presence_threshold=0.50,
    )
    for path in (eat_stats, spear_stats, hierarchical_stats, telephone_ids):
'''


def copy(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-zip", type=Path,
        default=ROOT / "channel_invariant_moe_v18.zip",
    )
    parser.add_argument(
        "--checkpoints", type=Path, nargs="+", default=list(DEFAULT_CHECKPOINTS),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    for path in (args.base_zip, *args.checkpoints):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if len(args.checkpoints) != 2:
        parser.error("v36 requires the two development-selected seed members")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    archive = args.output.with_suffix(".zip")
    if args.archive and archive.exists():
        raise FileExistsError(f"Refusing to overwrite {archive}")

    shutil.unpack_archive(str(args.base_zip), str(args.output), format="zip")
    source_dir = args.output / "model/src"
    for name in (
        "eat_hierarchical.py",
        "hierarchical_eat_music.py",
        "hierarchical_eat_music_inference.py",
        "eat_presence.py",
        "eat_presence_fusion.py",
    ):
        copy(ROOT / "src" / name, source_dir / name)
    model_dir = args.output / "model/hierarchical-eat-music"
    model_dir.mkdir(parents=True)
    for index, checkpoint in enumerate(args.checkpoints):
        shutil.copy2(checkpoint, model_dir / f"head_{index:02d}.pt")

    entrypoint = args.output / "script.py"
    script = entrypoint.read_text(encoding="utf-8")
    replacements = (
        (IMPORT_MARKER, IMPORT_INSERT),
        (VARIABLE_MARKER, VARIABLE_INSERT),
        (TELEPHONE_MARKER, TELEPHONE_INSERT),
        (CLEANUP_MARKER, CLEANUP_INSERT),
    )
    if any(script.count(marker) != 1 for marker, _ in replacements):
        raise ValueError("v18 entrypoint markers changed; refusing unsafe injection")
    for marker, replacement in replacements:
        script = script.replace(marker, replacement)
    entrypoint.write_text(script, encoding="utf-8")

    if args.archive:
        shutil.make_archive(str(args.output), "zip", root_dir=args.output)
        print(f"Built {args.output} and {archive}")
    else:
        print(f"Built {args.output}")


if __name__ == "__main__":
    main()
