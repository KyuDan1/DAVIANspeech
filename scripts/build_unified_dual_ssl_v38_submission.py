#!/usr/bin/env python3
"""Add the fixed Music-only unified EAT/SPEAR residual to v37."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINTS = (
    ROOT / "reports/unified_dual_ssl_v1/seed00/unified_dual_ssl_head.pt",
    ROOT / "reports/unified_dual_ssl_v1/seed01/unified_dual_ssl_head.pt",
)
IMPORT_MARKER = (
    "from temporal_dual_domain_inference import "
    "apply_temporal_dual_domain_fusion  # noqa: E402\n"
)
IMPORT_INSERT = IMPORT_MARKER + (
    "from unified_dual_ssl_inference import "
    "apply_unified_music_fusion  # noqa: E402\n"
)
VARIABLE_MARKER = (
    '    spear_stats = BASE_DIR / "output" / ".spear_temporal_stats.npz"\n'
)
VARIABLE_INSERT = VARIABLE_MARKER + (
    '    spear_unified = BASE_DIR / "output" / ".spear_unified_bins.npz"\n'
)
SPEAR_MARKER = '''    apply_fusion_with_stats(
        args.test_dir, args.output, BASE_DIR / "model" / "spear",
        BASE_DIR / "model" / "spear-mixed-music-head.npz",
        BASE_DIR / "model" / "spear-cross-component-joint-v1.npz",
        device=args.device, weight=0.10, statistics_output_path=spear_stats,
    )
'''
SPEAR_INSERT = '''    apply_fusion_with_stats(
        args.test_dir, args.output, BASE_DIR / "model" / "spear",
        BASE_DIR / "model" / "spear-mixed-music-head.npz",
        BASE_DIR / "model" / "spear-cross-component-joint-v1.npz",
        device=args.device, weight=0.10, statistics_output_path=spear_stats,
        temporal_bin_output_path=spear_unified,
        temporal_bin_checkpoint_path=(
            BASE_DIR / "model" / "unified-dual-ssl" / "head_00.pt"
        ),
    )
'''
FUSION_MARKER = '''    for path in (eat_stats, spear_stats, hierarchical_stats, segmental_stats, telephone_ids):
        path.unlink(missing_ok=True)
'''
FUSION_INSERT = '''    apply_unified_music_fusion(
        args.output, hierarchical_stats, spear_unified,
        sorted((BASE_DIR / "model" / "unified-dual-ssl").glob("head_*.pt")),
        device=args.device, music_weight=0.20,
    )
    for path in (
        eat_stats, spear_stats, spear_unified, hierarchical_stats,
        segmental_stats, telephone_ids,
    ):
        path.unlink(missing_ok=True)
'''


def copy(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-zip", type=Path,
        default=ROOT / "segmental_eat_moe_v37_submit.zip",
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
        parser.error("v38 requires the two development-selected unified heads")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    archive = args.output.with_suffix(".zip")
    if args.archive and archive.exists():
        raise FileExistsError(f"Refusing to overwrite {archive}")

    shutil.unpack_archive(str(args.base_zip), str(args.output), format="zip")
    source_dir = args.output / "model/src"
    for name in (
        "anchor_spear_stats_fusion.py", "eat_presence_fusion.py",
        "spear_detector.py", "spear_temporal_bins.py",
        "unified_dual_ssl_head.py", "unified_dual_ssl_inference.py",
    ):
        copy(ROOT / "src" / name, source_dir / name)
    model_dir = args.output / "model/unified-dual-ssl"
    model_dir.mkdir(parents=True)
    for index, checkpoint in enumerate(args.checkpoints):
        shutil.copy2(checkpoint, model_dir / f"head_{index:02d}.pt")

    entrypoint = args.output / "script.py"
    script = entrypoint.read_text(encoding="utf-8")
    replacements = (
        (IMPORT_MARKER, IMPORT_INSERT),
        (VARIABLE_MARKER, VARIABLE_INSERT),
        (SPEAR_MARKER, SPEAR_INSERT),
        (FUSION_MARKER, FUSION_INSERT),
    )
    if any(script.count(marker) != 1 for marker, _ in replacements):
        raise ValueError("v37 entrypoint markers changed; refusing unsafe injection")
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
