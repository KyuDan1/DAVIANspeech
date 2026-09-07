#!/usr/bin/env python3
"""Build v42: v41 plus a two-view multistream File-only soft MoE."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from build_wpt_fixed_moe_v39_submission import make_zip


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT
    / "reports/multistream_prompt_spectra_v1/seed_20260908"
    / "wpt_spectra_multitask.pt"
)
IMPORT_MARKER = (
    "from wpt_spectra_inference import "
    "apply_wpt_fixed_moe_fusion  # noqa: E402\n"
)
IMPORT_REPLACEMENT = (
    "from multistream_prompt_inference import "
    "apply_multistream_file_moe_fusion  # noqa: E402\n"
)
CALL_MARKER = '''    apply_wpt_fixed_moe_fusion(
        args.test_dir, args.output, BASE_DIR / "model" / "wpt-spectra",
        BASE_DIR / "model" / "wpt-spectra" / "wpt_spectra_multitask.pt",
        unified_expert, device=args.device, file_batch_size=6,
        file_views=5, file_temperature=2.0,
        voice_outer_weight=0.10, file_outer_weight=0.75,
    )
'''
CALL_REPLACEMENT = '''    apply_multistream_file_moe_fusion(
        args.test_dir, args.output, BASE_DIR / "model" / "wpt-spectra",
        BASE_DIR / "model" / "wpt-spectra" / "wpt_spectra_multitask.pt",
        BASE_DIR / "model" / "multistream-prompt" / "multistream.pt",
        unified_expert, device=args.device,
        wpt_batch_size=6, wpt_file_views=5, wpt_file_temperature=2.0,
        multistream_batch_size=12, multistream_views=2,
        multistream_weight=0.20,
        voice_outer_weight=0.10, file_outer_weight=0.75,
    )
'''


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-zip", type=Path, default=ROOT / "wpt_file_lme_v41.zip",
    )
    parser.add_argument(
        "--multistream-checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    for path in (args.base_zip, args.multistream_checkpoint):
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
        "wpt_spectra.py", "wpt_spectra_inference.py",
        "multistream_prompt_spectra.py", "multistream_prompt_inference.py",
    ):
        copy(ROOT / "src" / name, source_dir / name)
    copy(
        args.multistream_checkpoint,
        args.output / "model/multistream-prompt/multistream.pt",
    )

    entrypoint = args.output / "script.py"
    script = entrypoint.read_text(encoding="utf-8")
    for marker, replacement in (
        (IMPORT_MARKER, IMPORT_REPLACEMENT),
        (CALL_MARKER, CALL_REPLACEMENT),
    ):
        if script.count(marker) != 1:
            raise ValueError("v41 entrypoint marker changed; refusing unsafe injection")
        script = script.replace(marker, replacement)
    entrypoint.write_text(script, encoding="utf-8")

    if args.archive:
        make_zip(args.output, archive)
        print(f"Built {args.output} and {archive}")
    else:
        print(f"Built {args.output}")


if __name__ == "__main__":
    main()
