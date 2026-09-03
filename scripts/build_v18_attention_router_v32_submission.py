#!/usr/bin/env python3
"""Add one phone-routed temporal-attention File expert to exact submitted v18."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
IMPORT_MARKER = (
    "from temporal_dual_domain_inference import "
    "apply_temporal_dual_domain_fusion  # noqa: E402\n"
)
IMPORT_REPLACEMENT = IMPORT_MARKER + (
    "from spear_temporal_joint_inference import "
    "apply_spear_temporal_joint_fusion  # noqa: E402\n"
)
STATS_MARKER = '''    eat_stats = BASE_DIR / "output" / ".eat_temporal_stats.npz"
    spear_stats = BASE_DIR / "output" / ".spear_temporal_stats.npz"
'''
STATS_REPLACEMENT = STATS_MARKER + '''    temporal_bin_stats = BASE_DIR / "output" / ".spear_temporal_bins.npz"
    telephone_ids = BASE_DIR / "output" / ".telephone_ids.npz"
'''
EAT_MARKER = '''        telephone_router_path=BASE_DIR / "model" / "telephone-router.npz",
        phone_voice_weight=0.10,
    )
'''
EAT_REPLACEMENT = '''        telephone_router_path=BASE_DIR / "model" / "telephone-router.npz",
        phone_voice_weight=0.10,
        telephone_ids_output_path=telephone_ids,
    )
'''
SPEAR_MARKER = '''        BASE_DIR / "model" / "spear-cross-component-joint-v1.npz",
        device=args.device, weight=0.10, statistics_output_path=spear_stats,
    )
'''
SPEAR_REPLACEMENT = '''        BASE_DIR / "model" / "spear-cross-component-joint-v1.npz",
        device=args.device, weight=0.10, statistics_output_path=spear_stats,
        temporal_bin_output_path=temporal_bin_stats,
        temporal_bin_checkpoint_path=(
            BASE_DIR / "model" / "spear-temporal-attention" / "head.pt"
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
ATTENTION_CALL = V18_CALL + '''    apply_spear_temporal_joint_fusion(
        args.output, temporal_bin_stats,
        BASE_DIR / "model" / "spear-temporal-attention" / "head.pt",
        device=args.device, file_weight=0.20, voice_weight=0.0,
        telephone_ids_path=telephone_ids,
        phone_file_weight=0.25, phone_voice_weight=0.0,
    )
'''
CLEANUP_MARKER = "    for path in (eat_stats, spear_stats):\n"
CLEANUP_REPLACEMENT = (
    "    for path in (eat_stats, spear_stats, temporal_bin_stats, telephone_ids):\n"
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
        "--checkpoint", type=Path, default=(
            ROOT / "reports/spear_temporal_attention_v1/attention_channel_s12/"
            "spear_temporal_joint_head.pt"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
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
        "anchor_spear_stats_fusion.py", "spear_detector.py", "spear_temporal_bins.py",
        "spear_temporal_joint_head.py", "spear_temporal_joint_inference.py",
    ):
        replace_file(ROOT / "src" / name, source_dir / name)
    head_dir = args.output / "model" / "spear-temporal-attention"
    head_dir.mkdir(parents=True)
    shutil.copy2(args.checkpoint, head_dir / "head.pt")

    entrypoint = args.output / "script.py"
    script = entrypoint.read_text(encoding="utf-8")
    for old, new, label in (
        (IMPORT_MARKER, IMPORT_REPLACEMENT, "import"),
        (STATS_MARKER, STATS_REPLACEMENT, "statistics"),
        (EAT_MARKER, EAT_REPLACEMENT, "telephone output"),
        (SPEAR_MARKER, SPEAR_REPLACEMENT, "temporal extraction"),
        (V18_CALL, ATTENTION_CALL, "attention call"),
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
