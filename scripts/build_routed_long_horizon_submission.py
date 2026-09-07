#!/usr/bin/env python3
"""Build v26 by adding the routed long-horizon music expert to v25."""

from __future__ import annotations

import argparse
import os
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
CALL_MARKER = "    apply_spectra_voice_fusion(\n"
CALL_INSERT = '''    apply_long_horizon_music_fusion(
        args.output, eat_stats, spear_stats,
        BASE_DIR / "model" / "long-horizon-music" / "ensemble.npz",
        device=args.device, telephone_ids_path=telephone_ids,
    )
''' + CALL_MARKER


def replace_copy(source: Path, destination: Path) -> None:
    """Break a possible hardlink before replacing one packaged file."""
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "temporal_layout_v25")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "model_heads/long-horizon-music-v1/ensemble.npz",
    )
    args = parser.parse_args()
    for path in (args.base, args.checkpoint):
        if not path.exists():
            parser.error(f"missing input: {path}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    shutil.copytree(
        args.base, args.output, copy_function=os.link,
        ignore=shutil.ignore_patterns("data", "open", "output", "__pycache__"),
    )
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
    if script.count(IMPORT_MARKER) != 1 or script.count(CALL_MARKER) != 1:
        raise ValueError("Base entrypoint markers changed; refusing unsafe injection")
    script = script.replace(IMPORT_MARKER, IMPORT_INSERT)
    script = script.replace(CALL_MARKER, CALL_INSERT)
    script_path.unlink()
    script_path.write_text(script, encoding="utf-8")
    print(f"Built {args.output}")


if __name__ == "__main__":
    main()
