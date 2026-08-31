"""Build the minimal offline submission for ``src/simple_pipeline.py``."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from build_submission import copy_xlsr_fp16

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [
    "simple_pipeline.py", "presence.py", "fourier_detector.py",
    "xlsr_antideepfake.py", "spear_detector.py", "eat_detector.py",
    "eat_timm_compat.py", "sonics_detector.py", "telephone_router.py",
]

SCRIPT = '''#!/usr/bin/env python3
import os, sys
from pathlib import Path
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True
base = Path(__file__).resolve().parent
sys.path.insert(0, str(base / "model" / "src"))
from simple_pipeline import parse_args, run

def main():
    args = parse_args()
    extensions = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}
    candidates = [base / "data" / "test", base / "open" / "test", base / "data", base / "open"]
    args.test_dir = next((d for d in candidates if d.is_dir() and any(p.is_file() and p.suffix.lower() in extensions for p in d.iterdir())), candidates[0])
    samples = [base / "data" / "sample_submission.csv", base / "open" / "sample_submission.csv", base / "sample_submission.csv"]
    args.sample_submission = next((p for p in samples if p.is_file()), samples[0])
    args.output = base / "output" / "submission.csv"
    args.panns_dir = base / "model" / "panns"
    args.xlsr_dir = base / "model" / "xlsr"
    args.fourier_music_head = base / "model" / "fourier-music-head.npz"
    args.xlsr_mixed_voice_head = base / "model" / "xlsr-mixed-voice-head.npz"
    args.music_segment_weight = 0.7
    args.spear_dir = base / "model" / "spear"
    args.spear_mixture_head = base / "model" / "spear-mixture-present-head.npz"
    args.eat_dir = base / "model" / "eat"
    args.eat_phone_head = base / "model" / "eat-phone-head.npz"
    args.sonics_dir = base / "model" / "sonics"
    args.telephone_router = base / "model" / "telephone-router.npz"
    run(args)
if __name__ == "__main__": main()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsr-dir", type=Path, required=True)
    parser.add_argument("--panns-dir", type=Path, required=True)
    parser.add_argument("--fourier-music-head", type=Path, required=True)
    parser.add_argument("--xlsr-mixed-voice-head", type=Path, required=True)
    parser.add_argument("--spear-dir", type=Path)
    parser.add_argument("--spear-mixture-head", type=Path)
    parser.add_argument("--eat-dir", type=Path)
    parser.add_argument("--eat-phone-head", type=Path)
    parser.add_argument("--sonics-dir", type=Path)
    parser.add_argument("--telephone-router", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()
    out = args.output_dir
    if out.exists():
        shutil.rmtree(out)
    (out / "model" / "src").mkdir(parents=True)
    for name in SOURCES:
        shutil.copy2(ROOT / "src" / name, out / "model" / "src" / name)
    shutil.copytree(args.panns_dir, out / "model" / "panns", ignore=shutil.ignore_patterns("__pycache__"))
    copy_xlsr_fp16(args.xlsr_dir, out / "model" / "xlsr")
    shutil.copy2(args.fourier_music_head, out / "model" / "fourier-music-head.npz")
    shutil.copy2(args.xlsr_mixed_voice_head, out / "model" / "xlsr-mixed-voice-head.npz")
    if args.spear_dir:
        shutil.copytree(args.spear_dir, out / "model" / "spear",
                        ignore=shutil.ignore_patterns(".cache", "__pycache__", "README.md"))
        shutil.copy2(args.spear_mixture_head, out / "model" / "spear-mixture-present-head.npz")
    if args.eat_dir:
        shutil.copytree(args.eat_dir, out / "model" / "eat",
                        ignore=shutil.ignore_patterns(".cache", "__pycache__", "README.md"))
        shutil.copy2(args.eat_phone_head, out / "model" / "eat-phone-head.npz")
    if args.sonics_dir:
        shutil.copytree(args.sonics_dir, out / "model" / "sonics",
                        ignore=shutil.ignore_patterns(".cache", "__pycache__", "README.md"))
    shutil.copy2(args.telephone_router, out / "model" / "telephone-router.npz")
    (out / "script.py").write_text(SCRIPT, encoding="utf-8")
    (out / "requirements.txt").write_text("", encoding="utf-8")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"package: {out} ({total / 2**30:.2f} GiB)")
    if args.zip:
        archive = shutil.make_archive(str(out), "zip", root_dir=out)
        print(f"archive: {archive} ({Path(archive).stat().st_size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
