"""Overlay train-free EAT presence fusion onto the verified LME+SPEAR package."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

ENTRYPOINT = '''#!/usr/bin/env python3
"""DACON entry point: LME anchor + EAT presence + SPEAR fake fusion."""

import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
sys.dont_write_bytecode = True

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "model" / "src"))

from anchor_spear_fusion import apply_fusion  # noqa: E402
from eat_presence_fusion import apply_eat_presence_fusion  # noqa: E402
from pipeline import parse_args, run  # noqa: E402


def main():
    args = parse_args()
    args.pooling = "logmeanexp"
    args.temperature = 5.0
    extensions = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}
    candidates = [BASE_DIR / "data" / "test", BASE_DIR / "open" / "test",
                  BASE_DIR / "data", BASE_DIR / "open"]
    args.test_dir = next((directory for directory in candidates
                          if directory.is_dir() and any(path.is_file()
                          and path.suffix.lower() in extensions
                          for path in directory.iterdir())), candidates[0])
    samples = [BASE_DIR / "data" / "sample_submission.csv",
               BASE_DIR / "open" / "sample_submission.csv",
               BASE_DIR / "sample_submission.csv"]
    args.sample_submission = next((path for path in samples if path.is_file()), samples[0])
    args.output = BASE_DIR / "output" / "submission.csv"
    args.panns_dir = BASE_DIR / "model" / "panns"
    args.xlsr_dir = BASE_DIR / "model" / "xlsr"
    args.artifactnet_dir = BASE_DIR / "model" / "artifactnet"
    args.htdemucs_repo = BASE_DIR / "model" / "htdemucs"

    run(args)
    apply_eat_presence_fusion(
        args.test_dir, args.output, BASE_DIR / "model" / "eat",
        BASE_DIR / "model" / "panns", device=args.device,
        voice_weight=0.30, music_weight=0.90, file_gate=0.60,
    )
    apply_fusion(
        args.test_dir, args.output, BASE_DIR / "model" / "spear",
        BASE_DIR / "model" / "spear-mixed-music-head.npz",
        BASE_DIR / "model" / "spear-cross-component-joint-v1.npz",
        device=args.device, weight=0.10,
    )


if __name__ == "__main__":
    main()
'''


def replace_file(source: Path, target: Path) -> None:
    target.unlink(missing_ok=True)
    shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "submit_lme_spear_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    shutil.copytree(args.base, args.output, copy_function=os.link)
    for name in ("eat_detector.py", "eat_timm_compat.py", "eat_presence.py",
                 "eat_presence_fusion.py"):
        replace_file(ROOT / "src" / name, args.output / "model" / "src" / name)
    shutil.copytree(
        ROOT / "models/eat-base-as2m", args.output / "model" / "eat",
        copy_function=os.link,
        ignore=shutil.ignore_patterns(".gitattributes", "README.md", "__pycache__"),
    )
    script = args.output / "script.py"
    script.unlink(missing_ok=True)
    script.write_text(ENTRYPOINT, encoding="utf-8")
    print(f"Built {args.output}")


if __name__ == "__main__":
    main()
