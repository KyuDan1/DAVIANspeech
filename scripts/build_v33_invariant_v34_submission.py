#!/usr/bin/env python3
"""Add the four-head invariant residual to the validated exact-v33 stack."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
MUSIC_CALL = '''    apply_spear_temporal_bin_music_only(
        args.output, temporal_bin_stats,
        BASE_DIR / "model" / "spear-temporal-music" / "head.pt",
        device=args.device, music_weight=0.50,
    )
'''
INVARIANT_CALL = MUSIC_CALL + '''    apply_dual_domain_fusion(
        args.output, eat_stats, spear_stats,
        sorted((BASE_DIR / "model" / "channel-invariant-v4").glob("head_*.pt")),
        device=args.device, file_weight=0.125,
        voice_weight=0.20, music_weight=0.10,
    )
'''
DEFAULT_CHECKPOINTS = (
    ROOT / "reports/invariant_dual_domain_v4_paired/seed00_fixedteacher_lr1e4/dual_domain_head.pt",
    ROOT / "reports/invariant_dual_domain_v4_paired/seed00_ch01/dual_domain_head.pt",
    ROOT / "reports/invariant_dual_domain_v4_paired/seed01_ch01/dual_domain_head.pt",
    ROOT / "reports/invariant_dual_domain_v4_paired/seed02_ch01/dual_domain_head.pt",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-zip", type=Path,
        default=ROOT / "v18_attention_music_v33_fixed.zip",
    )
    parser.add_argument(
        "--checkpoints", type=Path, nargs="+",
        default=list(DEFAULT_CHECKPOINTS),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    for path in (args.base_zip, *args.checkpoints):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if len(args.checkpoints) != 4:
        parser.error("v34 requires the locked four-member invariant ensemble")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    archive = args.output.with_suffix(".zip")
    if args.archive and archive.exists():
        raise FileExistsError(f"Refusing to overwrite {archive}")

    shutil.unpack_archive(str(args.base_zip), str(args.output), format="zip")
    model_dir = args.output / "model" / "channel-invariant-v4"
    model_dir.mkdir(parents=True)
    for index, checkpoint in enumerate(args.checkpoints):
        shutil.copy2(checkpoint, model_dir / f"head_{index:02d}.pt")

    entrypoint = args.output / "script.py"
    script = entrypoint.read_text(encoding="utf-8")
    if script.count(MUSIC_CALL) != 1:
        raise ValueError("v33 Music marker changed; refusing unsafe injection")
    script = script.replace(MUSIC_CALL, INVARIANT_CALL)
    entrypoint.unlink()
    entrypoint.write_text(script, encoding="utf-8")

    if args.archive:
        shutil.make_archive(str(args.output), "zip", root_dir=args.output)
        print(f"Built {args.output} and {archive}")
    else:
        print(f"Built {args.output}")


if __name__ == "__main__":
    main()
