#!/usr/bin/env python3
"""Add the hard-cell-safe standalone WPT residual to exact v47."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WPT = (
    ROOT / "reports/wpt_spectra_v1/seed_20260906/wpt_spectra_multitask.pt"
)
IMPORT_MARKER = (
    "from component_query_mhfa_inference import "
    "apply_component_query_mhfa_fusion  # noqa: E402\n"
)
IMPORT_INSERT = IMPORT_MARKER + (
    "from wpt_spectra_inference import "
    "apply_wpt_standalone_fusion  # noqa: E402\n"
)
QUERY_MARKER = '''    apply_component_query_mhfa_fusion(
        args.output, eat_patch_graph, spear_component_bins,
        BASE_DIR / "model" / "component-query" / "head.pt",
        device=args.device, file_weight=0.025,
        music_weight=0.05,
        file_or_weight=0.3,
    )
'''
WPT_INSERT = QUERY_MARKER + '''    apply_wpt_standalone_fusion(
        args.test_dir, args.output, BASE_DIR / "model" / "wpt-spectra",
        BASE_DIR / "model" / "wpt-spectra" / "wpt_spectra_multitask.pt",
        device=args.device, file_batch_size=8,
        voice_weight={voice_weight:.8g}, file_weight={file_weight:.8g},
    )
'''


def replace_once(text: str, marker: str, replacement: str, label: str) -> str:
    if text.count(marker) != 1:
        raise ValueError(f"v47 {label} marker changed; refusing unsafe injection")
    return text.replace(marker, replacement)


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def make_zip(source: Path, archive: Path) -> None:
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED,
        compresslevel=1, allowZip64=True,
    ) as handle:
        for path in sorted(source.rglob("*")):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            handle.write(path, path.relative_to(source))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-zip", type=Path, default=ROOT / "component_query_or_v47.zip"
    )
    parser.add_argument("--wpt-checkpoint", type=Path, default=DEFAULT_WPT)
    parser.add_argument(
        "--spectra-dir", type=Path,
        default=ROOT / "models/external/spectra_aasist",
    )
    # Frozen using only factorial_eval_1200_v2_dev.  The File residual did not
    # satisfy the zero-regression rule, so the deployable change is Voice-only.
    parser.add_argument("--voice-weight", type=float, default=.05)
    parser.add_argument("--file-weight", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.voice_weight <= .10 or not 0 <= args.file_weight <= .025:
        parser.error("hidden-transfer guard rejected the WPT residual weight")
    required = (
        args.base_zip, args.wpt_checkpoint,
        args.spectra_dir / "model.py",
        args.spectra_dir / "model.safetensors",
        args.spectra_dir / "xlsr_config/config.json",
    )
    for path in required:
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    archive = args.output.with_suffix(".zip")
    if args.archive and archive.exists():
        raise FileExistsError(f"Refusing to overwrite {archive}")

    shutil.unpack_archive(str(args.base_zip), str(args.output), format="zip")
    source_dir = args.output / "model/src"
    for name in ("wpt_spectra.py", "wpt_spectra_inference.py"):
        copy(ROOT / "src" / name, source_dir / name)
    wpt_target = args.output / "model/wpt-spectra"
    for name in ("model.py", "model.safetensors", "README.md", "SOURCE.md"):
        path = args.spectra_dir / name
        if path.is_file():
            copy(path, wpt_target / name)
    copy(
        args.spectra_dir / "xlsr_config/config.json",
        wpt_target / "xlsr_config/config.json",
    )
    copy(args.wpt_checkpoint, wpt_target / "wpt_spectra_multitask.pt")

    entrypoint = args.output / "script.py"
    script = entrypoint.read_text(encoding="utf-8")
    script = replace_once(script, IMPORT_MARKER, IMPORT_INSERT, "import")
    script = replace_once(
        script, QUERY_MARKER,
        WPT_INSERT.format(
            voice_weight=args.voice_weight, file_weight=args.file_weight,
        ),
        "fusion",
    )
    entrypoint.write_text(script, encoding="utf-8")
    if args.archive:
        make_zip(args.output, archive)
        print(f"Built {args.output} and {archive}")
    else:
        print(f"Built {args.output}")


if __name__ == "__main__":
    main()
