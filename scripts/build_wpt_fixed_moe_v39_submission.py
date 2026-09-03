#!/usr/bin/env python3
"""Add the validated original-audio WPT fixed MoE to the v38 archive."""

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
    "from unified_dual_ssl_inference import "
    "apply_unified_music_fusion  # noqa: E402\n"
)
IMPORT_INSERT = IMPORT_MARKER + (
    "from wpt_spectra_inference import "
    "apply_wpt_fixed_moe_fusion  # noqa: E402\n"
)
VARIABLE_MARKER = (
    '    telephone_ids = BASE_DIR / "output" / ".telephone_ids.npz"\n'
)
VARIABLE_INSERT = VARIABLE_MARKER + (
    '    unified_expert = BASE_DIR / "output" / ".unified_expert.npz"\n'
)
UNIFIED_MARKER = '''    apply_unified_music_fusion(
        args.output, hierarchical_stats, spear_unified,
        sorted((BASE_DIR / "model" / "unified-dual-ssl").glob("head_*.pt")),
        device=args.device, music_weight=0.20,
    )
'''
UNIFIED_INSERT = '''    apply_unified_music_fusion(
        args.output, hierarchical_stats, spear_unified,
        sorted((BASE_DIR / "model" / "unified-dual-ssl").glob("head_*.pt")),
        device=args.device, music_weight=0.20,
        expert_output_path=unified_expert,
    )
    apply_wpt_fixed_moe_fusion(
        args.test_dir, args.output, BASE_DIR / "model" / "wpt-spectra",
        BASE_DIR / "model" / "wpt-spectra" / "wpt_spectra_multitask.pt",
        unified_expert, device=args.device, file_batch_size=8,
        voice_outer_weight=0.10, file_outer_weight=0.60,
    )
'''
CLEANUP_MARKER = '''        segmental_stats, telephone_ids,
    ):
'''
CLEANUP_INSERT = '''        segmental_stats, telephone_ids, unified_expert,
    ):
'''


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)


def make_zip(source: Path, archive: Path) -> None:
    """Create an explicit ZIP64 archive with exactly three top-level entries."""
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
    parser.add_argument("--base-zip", type=Path, default=ROOT / "v38_unified.zip")
    parser.add_argument("--wpt-checkpoint", type=Path, default=DEFAULT_WPT)
    parser.add_argument(
        "--spectra-dir", type=Path,
        default=ROOT / "models/external/spectra_aasist",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
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
    for name in (
        "unified_dual_ssl_head.py", "unified_dual_ssl_inference.py",
        "wpt_spectra.py", "wpt_spectra_inference.py",
    ):
        copy(ROOT / "src" / name, source_dir / name)

    model_dir = args.output / "model/wpt-spectra"
    for name in ("model.py", "model.safetensors", "README.md", "SOURCE.md"):
        source = args.spectra_dir / name
        if source.is_file():
            copy(source, model_dir / name)
    copy(
        args.spectra_dir / "xlsr_config/config.json",
        model_dir / "xlsr_config/config.json",
    )
    copy(args.wpt_checkpoint, model_dir / "wpt_spectra_multitask.pt")

    entrypoint = args.output / "script.py"
    script = entrypoint.read_text(encoding="utf-8")
    replacements = (
        (IMPORT_MARKER, IMPORT_INSERT),
        (VARIABLE_MARKER, VARIABLE_INSERT),
        (UNIFIED_MARKER, UNIFIED_INSERT),
        (CLEANUP_MARKER, CLEANUP_INSERT),
    )
    if any(script.count(marker) != 1 for marker, _ in replacements):
        raise ValueError("v38 entrypoint markers changed; refusing unsafe injection")
    for marker, replacement in replacements:
        script = script.replace(marker, replacement)
    entrypoint.write_text(script, encoding="utf-8")

    if args.archive:
        make_zip(args.output, archive)
        print(f"Built {args.output} and {archive}")
    else:
        print(f"Built {args.output}")


if __name__ == "__main__":
    main()
