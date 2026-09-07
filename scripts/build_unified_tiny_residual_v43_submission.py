#!/usr/bin/env python3
"""Add one 3-seed, five-percent File/Music residual to exact submitted v18."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINTS = (
    ROOT / "reports/unified_dual_ssl_v1/seed01/unified_dual_ssl_head.pt",
    ROOT / "reports/unified_dual_ssl_v1/seed02/unified_dual_ssl_head.pt",
    ROOT / "reports/unified_dual_ssl_v1/seed03/unified_dual_ssl_head.pt",
)
IMPORT_MARKER = (
    "from temporal_dual_domain_inference import "
    "apply_temporal_dual_domain_fusion  # noqa: E402\n"
)
IMPORT_REPLACEMENT = IMPORT_MARKER + (
    "from unified_dual_ssl_inference import "
    "apply_unified_music_fusion  # noqa: E402\n"
)
STATS_MARKER = '''    eat_stats = BASE_DIR / "output" / ".eat_temporal_stats.npz"
    spear_stats = BASE_DIR / "output" / ".spear_temporal_stats.npz"
'''
STATS_REPLACEMENT = STATS_MARKER + '''    hierarchical_stats = BASE_DIR / "output" / ".eat_hierarchical_stats.npz"
    spear_unified = BASE_DIR / "output" / ".spear_unified_bins.npz"
'''
EAT_MARKER = '''        telephone_router_path=BASE_DIR / "model" / "telephone-router.npz",
        phone_voice_weight=0.10,
    )
'''
EAT_REPLACEMENT = '''        telephone_router_path=BASE_DIR / "model" / "telephone-router.npz",
        phone_voice_weight=0.10,
        hierarchical_statistics_output_path=hierarchical_stats,
        hierarchical_checkpoint_path=(
            BASE_DIR / "model" / "unified-dual-ssl" / "head_00.pt"
        ),
    )
'''
SPEAR_MARKER = '''        BASE_DIR / "model" / "spear-cross-component-joint-v1.npz",
        device=args.device, weight=0.10, statistics_output_path=spear_stats,
    )
'''
SPEAR_REPLACEMENT = '''        BASE_DIR / "model" / "spear-cross-component-joint-v1.npz",
        device=args.device, weight=0.10, statistics_output_path=spear_stats,
        temporal_bin_output_path=spear_unified,
        temporal_bin_checkpoint_path=(
            BASE_DIR / "model" / "unified-dual-ssl" / "head_00.pt"
        ),
    )
'''
V18_CALL = '''    apply_dual_domain_fusion(
        args.output, eat_stats, spear_stats,
        [BASE_DIR / "model" / "channel-invariant" / "dual_domain_head.pt"],
        device=args.device, file_weight=0.05,
        voice_weight=0.05, music_weight=0.05,
    )
'''
FUSION_REPLACEMENT = V18_CALL + '''    apply_unified_music_fusion(
        args.output, hierarchical_stats, spear_unified,
        sorted((BASE_DIR / "model" / "unified-dual-ssl").glob("head_*.pt")),
        device=args.device, music_weight=0.05, file_weight=0.05,
    )
'''
CLEANUP_MARKER = "    for path in (eat_stats, spear_stats):\n"
CLEANUP_REPLACEMENT = (
    "    for path in (eat_stats, spear_stats, hierarchical_stats, spear_unified):\n"
)


def replace_file(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)


def replace_once(script: str, old: str, new: str, label: str) -> str:
    if script.count(old) != 1:
        raise ValueError(f"v18 {label} marker changed; refusing unsafe injection")
    return script.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-zip", type=Path, default=ROOT / "channel_invariant_moe_v18.zip"
    )
    parser.add_argument(
        "--checkpoints", type=Path, nargs="+", default=list(DEFAULT_CHECKPOINTS)
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    for path in (args.base_zip, *args.checkpoints):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if len(args.checkpoints) != 3:
        parser.error("v43 requires the selected three unified members")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    archive = args.output.with_suffix(".zip")
    if args.archive and archive.exists():
        raise FileExistsError(f"Refusing to overwrite {archive}")

    shutil.unpack_archive(str(args.base_zip), str(args.output), format="zip")
    source_dir = args.output / "model" / "src"
    for name in (
        "anchor_spear_stats_fusion.py", "dual_domain_stats.py",
        "eat_hierarchical.py", "eat_presence.py", "eat_presence_fusion.py",
        "spear_detector.py", "spear_temporal_bins.py",
        "unified_dual_ssl_head.py", "unified_dual_ssl_inference.py",
    ):
        replace_file(ROOT / "src" / name, source_dir / name)
    head_dir = args.output / "model" / "unified-dual-ssl"
    head_dir.mkdir(parents=True)
    for index, checkpoint in enumerate(args.checkpoints):
        shutil.copy2(checkpoint, head_dir / f"head_{index:02d}.pt")

    entrypoint = args.output / "script.py"
    script = entrypoint.read_text(encoding="utf-8")
    for old, new, label in (
        (IMPORT_MARKER, IMPORT_REPLACEMENT, "import"),
        (STATS_MARKER, STATS_REPLACEMENT, "statistics"),
        (EAT_MARKER, EAT_REPLACEMENT, "EAT extraction"),
        (SPEAR_MARKER, SPEAR_REPLACEMENT, "SPEAR extraction"),
        (V18_CALL, FUSION_REPLACEMENT, "residual call"),
        (CLEANUP_MARKER, CLEANUP_REPLACEMENT, "cleanup"),
    ):
        script = replace_once(script, old, new, label)
    entrypoint.unlink()
    entrypoint.write_text(script, encoding="utf-8")

    if args.archive:
        shutil.make_archive(str(args.output), "zip", root_dir=args.output)
        print(f"Built {args.output} and {archive}")
    else:
        print(f"Built {args.output}")


if __name__ == "__main__":
    main()
