#!/usr/bin/env python3
"""Add one routed long-horizon music expert to the exact submitted v18 ZIP."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]

IMPORT_MARKER = (
    "from modern_fakeprint_detector import apply_modern_fakeprint_fusion  # noqa: E402\n"
)
IMPORT_INSERT = IMPORT_MARKER + (
    "from long_horizon_music_inference import "
    "apply_long_horizon_music_fusion  # noqa: E402\n"
)
TELEPHONE_VARIABLE_MARKER = (
    '    spear_stats = BASE_DIR / "output" / ".spear_temporal_stats.npz"\n'
)
TELEPHONE_VARIABLE_INSERT = TELEPHONE_VARIABLE_MARKER + (
    '    telephone_ids = BASE_DIR / "output" / ".telephone_ids.npz"\n'
)
TELEPHONE_CALL_MARKER = "        phone_voice_weight=0.10,\n"
TELEPHONE_CALL_INSERT = TELEPHONE_CALL_MARKER + (
    "        telephone_ids_output_path=telephone_ids,\n"
)
CLEANUP_MARKER = "    for path in (eat_stats, spear_stats):\n"
CLEANUP_INSERT = '''    apply_long_horizon_music_fusion(
        args.output, eat_stats, spear_stats,
        BASE_DIR / "model" / "long-horizon-music" / "ensemble.npz",
        device=args.device, telephone_ids_path=telephone_ids,
    )
    for path in (eat_stats, spear_stats, telephone_ids):
'''


def replace_copy(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-zip", type=Path,
        default=ROOT / "channel_invariant_moe_v18.zip",
        help="Immutable archive that received the v18 leaderboard score.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "model_heads/long-horizon-music-v1/ensemble.npz",
    )
    args = parser.parse_args()
    for path in (args.base_zip, args.checkpoint):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    shutil.unpack_archive(str(args.base_zip), str(args.output), format="zip")
    source_dir = args.output / "model" / "src"
    replace_copy(
        ROOT / "src/long_horizon_music_inference.py",
        source_dir / "long_horizon_music_inference.py",
    )
    checkpoint_dir = args.output / "model" / "long-horizon-music"
    checkpoint_dir.mkdir(parents=True)
    shutil.copy2(args.checkpoint, checkpoint_dir / "ensemble.npz")

    script_path = args.output / "script.py"
    script = script_path.read_text(encoding="utf-8")
    markers = (
        IMPORT_MARKER, TELEPHONE_VARIABLE_MARKER,
        TELEPHONE_CALL_MARKER, CLEANUP_MARKER,
    )
    if any(script.count(marker) != 1 for marker in markers):
        raise ValueError("v18 entrypoint markers changed; refusing unsafe injection")
    script = script.replace(IMPORT_MARKER, IMPORT_INSERT)
    script = script.replace(TELEPHONE_VARIABLE_MARKER, TELEPHONE_VARIABLE_INSERT)
    script = script.replace(TELEPHONE_CALL_MARKER, TELEPHONE_CALL_INSERT)
    script = script.replace(CLEANUP_MARKER, CLEANUP_INSERT)
    script_path.write_text(script, encoding="utf-8")
    print(f"Built {args.output} from {args.base_zip}")


if __name__ == "__main__":
    main()
