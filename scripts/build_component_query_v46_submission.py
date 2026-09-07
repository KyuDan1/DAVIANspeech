#!/usr/bin/env python3
"""Add one conservative component-query residual to validated v44."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT / "reports/component_query_mhfa_v1/seed04_base/component_query_mhfa.pt"
)
IMPORT_MARKER = (
    "from eat_patch_graph_inference import "
    "apply_eat_patch_graph_fusion  # noqa: E402\n"
)
IMPORT_REPLACEMENT = IMPORT_MARKER + (
    "from component_query_mhfa_inference import "
    "apply_component_query_mhfa_fusion  # noqa: E402\n"
)
STATS_MARKER = '''    eat_patch_graph = BASE_DIR / "output" / ".eat_patch_graph.npz"
'''
STATS_REPLACEMENT = STATS_MARKER + '''    spear_component_bins = BASE_DIR / "output" / ".spear_component_bins.npz"
'''
SPEAR_MARKER = '''        device=args.device, weight=0.10, statistics_output_path=spear_stats,
    )
'''
SPEAR_REPLACEMENT = '''        device=args.device, weight=0.10, statistics_output_path=spear_stats,
        temporal_bin_output_path=spear_component_bins,
        temporal_bin_checkpoint_path=(
            BASE_DIR / "model" / "component-query" / "head.pt"
        ),
    )
'''
PATCH_CALL = '''    apply_eat_patch_graph_fusion(
        args.output, eat_patch_graph,
        [BASE_DIR / "model" / "eat-patch-graph" / "head.pt"],
        device=args.device, file_weight=0.05, music_weight=0.05,
    )
'''
QUERY_CALL = PATCH_CALL + '''    apply_component_query_mhfa_fusion(
        args.output, eat_patch_graph, spear_component_bins,
        BASE_DIR / "model" / "component-query" / "head.pt",
        device=args.device, file_weight={file_weight:.8g},
        music_weight={music_weight:.8g},
        file_or_weight={file_or_weight:.8g},
    )
'''
CLEANUP_MARKER = "    for path in (eat_stats, spear_stats, eat_patch_graph):\n"
CLEANUP_REPLACEMENT = (
    "    for path in (eat_stats, spear_stats, eat_patch_graph, "
    "spear_component_bins):\n"
)


def replace_once(script: str, old: str, new: str, label: str) -> str:
    if script.count(old) != 1:
        raise ValueError(f"v44 {label} marker changed; refusing unsafe injection")
    return script.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-zip", type=Path, default=ROOT / "eat_patch_graph_v44.zip")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    parser.add_argument("--file-weight", type=float, default=.025)
    parser.add_argument("--music-weight", type=float, default=.05)
    parser.add_argument("--file-or-weight", type=float, default=0.)
    args = parser.parse_args()
    for value in (args.file_weight, args.music_weight):
        if not 0 <= value <= .10:
            parser.error("official-ablation guard limits query residuals to 10%")
    if not 0 <= args.file_or_weight <= .30:
        parser.error("component-OR guard limits the File residual to 30%")
    for path in (args.base_zip, args.checkpoint):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    archive = args.output.with_suffix(".zip")
    if args.archive and archive.exists():
        raise FileExistsError(f"Refusing to overwrite {archive}")

    shutil.unpack_archive(str(args.base_zip), str(args.output), format="zip")
    source = args.output / "model" / "src"
    # v44 predates temporal-bin cache export.  These three files upgrade the
    # existing SPEAR statistics pass without adding another encoder forward.
    for name in (
        "component_query_mhfa.py", "component_query_mhfa_inference.py",
        "anchor_spear_stats_fusion.py", "spear_detector.py",
        "spear_temporal_bins.py",
    ):
        shutil.copy2(ROOT / "src" / name, source / name)
    target = args.output / "model" / "component-query"
    target.mkdir(parents=True)
    shutil.copy2(args.checkpoint, target / "head.pt")
    entrypoint = args.output / "script.py"
    script = entrypoint.read_text(encoding="utf-8")
    for old, new, label in (
        (IMPORT_MARKER, IMPORT_REPLACEMENT, "import"),
        (STATS_MARKER, STATS_REPLACEMENT, "statistics"),
        (SPEAR_MARKER, SPEAR_REPLACEMENT, "SPEAR extraction"),
        (PATCH_CALL, QUERY_CALL.format(
            file_weight=args.file_weight, music_weight=args.music_weight,
            file_or_weight=args.file_or_weight,
        ), "query fusion"),
        (CLEANUP_MARKER, CLEANUP_REPLACEMENT, "cleanup"),
    ):
        script = replace_once(script, old, new, label)
    entrypoint.write_text(script, encoding="utf-8")
    if args.archive:
        shutil.make_archive(str(args.output), "zip", root_dir=args.output)
        print(f"Built {args.output} and {archive}")
    else:
        print(f"Built {args.output}")


if __name__ == "__main__":
    main()
