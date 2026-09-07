#!/usr/bin/env python3
"""Add one five-percent EAT patch-graph residual to exact submitted v18."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT / "reports/eat_patch_graph_v1/seed01_balanced_fm"
    / "eat_patch_graph_head.pt"
)
IMPORT_MARKER = (
    "from temporal_dual_domain_inference import "
    "apply_temporal_dual_domain_fusion  # noqa: E402\n"
)
IMPORT_REPLACEMENT = IMPORT_MARKER + (
    "from eat_patch_graph_inference import "
    "apply_eat_patch_graph_fusion  # noqa: E402\n"
)
STATS_MARKER = '''    eat_stats = BASE_DIR / "output" / ".eat_temporal_stats.npz"
    spear_stats = BASE_DIR / "output" / ".spear_temporal_stats.npz"
'''
STATS_REPLACEMENT = STATS_MARKER + '''    eat_patch_graph = BASE_DIR / "output" / ".eat_patch_graph.npz"
'''
EAT_MARKER = '''        telephone_router_path=BASE_DIR / "model" / "telephone-router.npz",
        phone_voice_weight=0.10,
    )
'''
EAT_REPLACEMENT = '''        telephone_router_path=BASE_DIR / "model" / "telephone-router.npz",
        phone_voice_weight=0.10,
        patch_graph_statistics_output_path=eat_patch_graph,
        patch_graph_checkpoint_path=(
            BASE_DIR / "model" / "eat-patch-graph" / "head.pt"
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
FUSION_REPLACEMENT = V18_CALL + '''    apply_eat_patch_graph_fusion(
        args.output, eat_patch_graph,
        [BASE_DIR / "model" / "eat-patch-graph" / "head.pt"],
        device=args.device, file_weight={weight:.8g}, music_weight={weight:.8g},
    )
'''
CLEANUP_MARKER = "    for path in (eat_stats, spear_stats):\n"
CLEANUP_REPLACEMENT = (
    "    for path in (eat_stats, spear_stats, eat_patch_graph):\n"
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
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    parser.add_argument("--residual-weight", type=float, default=.05)
    args = parser.parse_args()
    if not 0 <= args.residual_weight <= .30:
        parser.error("residual weight must lie in [0, 0.30]")
    for path in (args.base_zip, args.checkpoint):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    archive = args.output.with_suffix(".zip")
    if args.archive and archive.exists():
        raise FileExistsError(f"Refusing to overwrite {archive}")

    shutil.unpack_archive(str(args.base_zip), str(args.output), format="zip")
    source_dir = args.output / "model" / "src"
    for name in (
        "dual_domain_stats.py", "eat_hierarchical.py", "eat_patch_graph.py",
        "eat_patch_graph_inference.py", "eat_presence.py",
        "eat_presence_fusion.py",
    ):
        replace_file(ROOT / "src" / name, source_dir / name)
    model_dir = args.output / "model" / "eat-patch-graph"
    model_dir.mkdir(parents=True)
    shutil.copy2(args.checkpoint, model_dir / "head.pt")

    entrypoint = args.output / "script.py"
    script = entrypoint.read_text(encoding="utf-8")
    for old, new, label in (
        (IMPORT_MARKER, IMPORT_REPLACEMENT, "import"),
        (STATS_MARKER, STATS_REPLACEMENT, "statistics"),
        (EAT_MARKER, EAT_REPLACEMENT, "EAT extraction"),
        (
            V18_CALL,
            FUSION_REPLACEMENT.format(weight=args.residual_weight),
            "residual call",
        ),
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
