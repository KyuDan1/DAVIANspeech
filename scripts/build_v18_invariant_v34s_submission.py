#!/usr/bin/env python3
"""Add only the four-head invariant residual to the verified v18 archive."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_CALL = '''    apply_dual_domain_fusion(
        args.output, eat_stats, spear_stats,
        [BASE_DIR / "model" / "channel-invariant" / "dual_domain_head.pt"],
        device=args.device, file_weight=0.05,
        voice_weight=0.05, music_weight=0.05,
    )
'''
INVARIANT_TEMPLATE = ANCHOR_CALL + '''    apply_dual_domain_fusion(
        args.output, eat_stats, spear_stats,
        sorted((BASE_DIR / "model" / "channel-invariant-v4").glob("head_*.pt")),
        device=args.device, file_weight={file_weight},
        voice_weight={voice_weight}, music_weight={music_weight},
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
        default=ROOT / "channel_invariant_moe_v18.zip",
    )
    parser.add_argument(
        "--checkpoints", type=Path, nargs="+", default=list(DEFAULT_CHECKPOINTS),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    parser.add_argument("--file-weight", type=float, default=.125)
    parser.add_argument("--voice-weight", type=float, default=.20)
    parser.add_argument("--music-weight", type=float, default=.10)
    args = parser.parse_args()
    if any(not 0 <= value <= 1 for value in (
        args.file_weight, args.voice_weight, args.music_weight,
    )):
        parser.error("fusion weights must be in [0, 1]")
    for path in (args.base_zip, *args.checkpoints):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    if len(args.checkpoints) != 4:
        parser.error("the safe candidate requires four locked members")
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
    if script.count(ANCHOR_CALL) != 1:
        raise ValueError("v18 anchor marker changed; refusing unsafe injection")
    invariant_call = INVARIANT_TEMPLATE.format(
        file_weight=args.file_weight,
        voice_weight=args.voice_weight,
        music_weight=args.music_weight,
    )
    script = script.replace(ANCHOR_CALL, invariant_call)
    entrypoint.unlink()
    entrypoint.write_text(script, encoding="utf-8")

    if args.archive:
        shutil.make_archive(str(args.output), "zip", root_dir=args.output)
        print(f"Built {args.output} and {archive}")
    else:
        print(f"Built {args.output}")


if __name__ == "__main__":
    main()
