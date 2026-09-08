#!/usr/bin/env python3
"""Build a v50-derived package with last-stage File/component consistency."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
IMPORT_MARKER = (
    "from sparse_call_consensus import apply_sparse_call_consensus  # noqa: E402\n"
)
IMPORT_INSERT = IMPORT_MARKER + (
    "from component_consistent_file_fusion import "
    "apply_component_consistent_file_fusion  # noqa: E402\n"
)
CLEANUP_MARKER = "    for path in (\n"
FINAL_FILE_TEMPLATE = '''    # Run last: consume every upstream specialist's final component scores.
    apply_component_consistent_file_fusion(
        args.output, file_weight={file_weight:.8g},
        voice_presence_threshold={voice_threshold:.8g},
        music_presence_threshold={music_threshold:.8g},
    )
''' + CLEANUP_MARKER


def replace_once(text: str, marker: str, replacement: str, label: str) -> str:
    if text.count(marker) != 1:
        raise ValueError(f"v50 {label} marker changed; refusing unsafe injection")
    return text.replace(marker, replacement)


def patch_entrypoint(
    source: str,
    *,
    file_weight: float = .20,
    voice_threshold: float = .10,
    music_threshold: float = .20,
) -> str:
    """Insert one final, model-agnostic File residual after all specialists."""
    if not 0.0 <= file_weight <= .30:
        raise ValueError("cross-development guard limits File residual to 30%")
    for name, value in (
        ("voice_threshold", voice_threshold),
        ("music_threshold", music_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if "from component_consistent_file_fusion import" in source:
        raise ValueError("v50 File consistency marker changed; refusing unsafe injection")
    source = replace_once(source, IMPORT_MARKER, IMPORT_INSERT, "import")
    return replace_once(
        source,
        CLEANUP_MARKER,
        FINAL_FILE_TEMPLATE.format(
            file_weight=file_weight,
            voice_threshold=voice_threshold,
            music_threshold=music_threshold,
        ),
        "final cleanup",
    )


def replace_copy(source: Path, destination: Path) -> None:
    """Replace a possible hardlink without mutating its source inode."""
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)


def build_package(
    base: Path,
    output: Path,
    *,
    file_weight: float = .20,
    voice_threshold: float = .10,
    music_threshold: float = .20,
) -> None:
    base = Path(base).resolve()
    output = Path(output).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"missing v50 package: {base}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    shutil.copytree(
        base, output, copy_function=os.link,
        ignore=shutil.ignore_patterns("data", "open", "output", "__pycache__"),
    )
    script_path = output / "script.py"
    original = base / "script.py"
    replace_copy(original, script_path)
    script_path.write_text(
        patch_entrypoint(
            script_path.read_text(encoding="utf-8"),
            file_weight=file_weight,
            voice_threshold=voice_threshold,
            music_threshold=music_threshold,
        ),
        encoding="utf-8",
    )
    replace_copy(
        ROOT / "src/component_consistent_file_fusion.py",
        output / "model/src/component_consistent_file_fusion.py",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path,
        default=ROOT / "sparse_voice_wpt_file_v50_frozen",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--file-weight", type=float, default=.20)
    parser.add_argument("--voice-threshold", type=float, default=.10)
    parser.add_argument("--music-threshold", type=float, default=.20)
    args = parser.parse_args()
    build_package(
        args.base, args.output,
        file_weight=args.file_weight,
        voice_threshold=args.voice_threshold,
        music_threshold=args.music_threshold,
    )
    print(f"Built {args.output} from {args.base}")


if __name__ == "__main__":
    main()
