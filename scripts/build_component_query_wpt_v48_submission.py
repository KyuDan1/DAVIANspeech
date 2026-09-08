#!/usr/bin/env python3
"""Add a conservative WPT fixed expert to the exact v47 pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WPT = (
    ROOT / "reports/wpt_spectra_v1/seed_20260906/wpt_spectra_multitask.pt"
)
DEFAULT_UNIFIED = ROOT / "wpt_fixed_moe_v39/model/unified-dual-ssl"

IMPORT_MARKER = (
    "from component_query_mhfa_inference import "
    "apply_component_query_mhfa_fusion  # noqa: E402\n"
)
IMPORT_INSERT = IMPORT_MARKER + (
    "from unified_dual_ssl_inference import "
    "apply_unified_music_fusion  # noqa: E402\n"
    "from wpt_spectra_inference import "
    "apply_wpt_fixed_moe_fusion  # noqa: E402\n"
)
VARIABLE_MARKER = (
    '    spear_component_bins = BASE_DIR / "output" / ".spear_component_bins.npz"\n'
)
VARIABLE_INSERT = VARIABLE_MARKER + (
    '    eat_unified = BASE_DIR / "output" / ".eat_unified.npz"\n'
    '    spear_unified = BASE_DIR / "output" / ".spear_unified.npz"\n'
    '    unified_expert = BASE_DIR / "output" / ".unified_expert.npz"\n'
)
EAT_MARKER = '''        patch_graph_checkpoint_path=(
            BASE_DIR / "model" / "eat-patch-graph" / "head.pt"
        ),
    )
'''
EAT_INSERT = '''        patch_graph_checkpoint_path=(
            BASE_DIR / "model" / "eat-patch-graph" / "head.pt"
        ),
        hierarchical_statistics_output_path=eat_unified,
        hierarchical_checkpoint_path=(
            BASE_DIR / "model" / "unified-dual-ssl" / "head_00.pt"
        ),
    )
'''
SPEAR_MARKER = '''        temporal_bin_checkpoint_path=(
            BASE_DIR / "model" / "component-query" / "head.pt"
        ),
    )
'''
SPEAR_INSERT = '''        temporal_bin_checkpoint_path=(
            BASE_DIR / "model" / "component-query" / "head.pt"
        ),
        additional_temporal_bin_requests=[(
            spear_unified,
            BASE_DIR / "model" / "unified-dual-ssl" / "head_00.pt",
        )],
    )
'''
QUERY_MARKER = '''    apply_component_query_mhfa_fusion(
        args.output, eat_patch_graph, spear_component_bins,
        BASE_DIR / "model" / "component-query" / "head.pt",
        device=args.device, file_weight=0.025,
        music_weight=0.05,
        file_or_weight=0.3,
    )
'''
WPT_INSERT = QUERY_MARKER + '''    # Export the fixed Unified half without changing v47 Music/File.
    apply_unified_music_fusion(
        args.output, eat_unified, spear_unified,
        sorted((BASE_DIR / "model" / "unified-dual-ssl").glob("head_*.pt")),
        device=args.device, music_weight=0.0, file_weight=0.0,
        expert_output_path=unified_expert,
    )
    apply_wpt_fixed_moe_fusion(
        args.test_dir, args.output, BASE_DIR / "model" / "wpt-spectra",
        BASE_DIR / "model" / "wpt-spectra" / "wpt_spectra_multitask.pt",
        unified_expert, device=args.device, file_batch_size=8,
        voice_outer_weight={outer_weight:.8g},
        file_outer_weight={outer_weight:.8g},
    )
'''
CLEANUP_MARKER = (
    "    for path in (eat_stats, spear_stats, eat_patch_graph, "
    "spear_component_bins):\n"
)
CLEANUP_INSERT = (
    "    for path in (eat_stats, spear_stats, eat_patch_graph, "
    "spear_component_bins, eat_unified, spear_unified, unified_expert):\n"
)


def replace_once(text: str, marker: str, replacement: str, label: str) -> str:
    if text.count(marker) != 1:
        raise ValueError(f"v47 {label} marker changed; refusing unsafe injection")
    return text.replace(marker, replacement)


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def make_zip(source: Path, archive: Path) -> None:
    """Build an explicit ZIP64 archive with no wrapper directory."""
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
    parser.add_argument("--unified-head-dir", type=Path, default=DEFAULT_UNIFIED)
    parser.add_argument("--outer-weight", type=float, default=.05)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.outer_weight <= .10:
        parser.error("the hidden-transfer guard limits WPT residuals to 10%")
    required = (
        args.base_zip, args.wpt_checkpoint,
        args.spectra_dir / "model.py",
        args.spectra_dir / "model.safetensors",
        args.spectra_dir / "xlsr_config/config.json",
        args.unified_head_dir / "head_00.pt",
        args.unified_head_dir / "head_01.pt",
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
        "anchor_spear_stats_fusion.py", "spear_detector.py",
        "spear_temporal_bins.py", "eat_presence.py", "eat_presence_fusion.py",
        "unified_dual_ssl_head.py", "unified_dual_ssl_inference.py",
        "wpt_spectra.py", "wpt_spectra_inference.py",
    ):
        copy(ROOT / "src" / name, source_dir / name)

    unified_target = args.output / "model/unified-dual-ssl"
    for path in sorted(args.unified_head_dir.glob("head_*.pt")):
        copy(path, unified_target / path.name)
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
    for marker, replacement, label in (
        (IMPORT_MARKER, IMPORT_INSERT, "imports"),
        (VARIABLE_MARKER, VARIABLE_INSERT, "cache variables"),
        (EAT_MARKER, EAT_INSERT, "EAT multi-export"),
        (SPEAR_MARKER, SPEAR_INSERT, "SPEAR multi-export"),
        (
            QUERY_MARKER, WPT_INSERT.format(outer_weight=args.outer_weight),
            "WPT fusion",
        ),
        (CLEANUP_MARKER, CLEANUP_INSERT, "cache cleanup"),
    ):
        script = replace_once(script, marker, replacement, label)
    entrypoint.write_text(script, encoding="utf-8")

    if args.archive:
        make_zip(args.output, archive)
        print(f"Built {args.output} and {archive}")
    else:
        print(f"Built {args.output}")


if __name__ == "__main__":
    main()
