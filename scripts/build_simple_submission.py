"""Build the minimal offline submission for ``src/simple_pipeline.py``."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from build_submission import copy_xlsr_fp16

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ["simple_pipeline.py", "presence.py", "fourier_detector.py", "xlsr_antideepfake.py"]

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
    run(args)
if __name__ == "__main__": main()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsr-dir", type=Path, required=True)
    parser.add_argument("--panns-dir", type=Path, required=True)
    parser.add_argument("--fourier-music-head", type=Path, required=True)
    parser.add_argument("--xlsr-mixed-voice-head", type=Path, required=True)
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
    (out / "script.py").write_text(SCRIPT, encoding="utf-8")
    (out / "requirements.txt").write_text("", encoding="utf-8")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"package: {out} ({total / 2**30:.2f} GiB)")
    if args.zip:
        archive = shutil.make_archive(str(out), "zip", root_dir=out)
        print(f"archive: {archive} ({Path(archive).stat().st_size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
