"""Assemble the competition submission package.

Produces a directory (and optionally a zip) laid out the way the organisers'
baseline is:

    script.py            entry point, run from the package root
    src/*.py             pipeline source
    model/panns/         Cnn14 checkpoint + AudioSet label groups
    model/htdemucs/      demucs bag, loaded offline
    model/xlsr/          XLS-R-2B-AntiDeepfake weights, stored as fp16
    model/artifactnet/   ArtifactNet ONNX graph + external weights
    requirements.txt

The XLS-R weights are cast to fp16 on the way in: it halves the package
(8.65 GB -> 4.33 GB) and moves P(fake) by ~1e-7, since inference still
upcasts to fp32.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parent.parent

SCRIPT_TEMPLATE = '''#!/usr/bin/env python3
"""Competition entry point: writes output/submission.csv for data/test."""

import os
import sys
from pathlib import Path

# Everything must come from this package -- no network at inference time.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
sys.dont_write_bytecode = True

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from pipeline import parse_args, run  # noqa: E402


def main():
    args = parse_args()
    args.panns_dir = BASE_DIR / "model" / "panns"
    args.xlsr_dir = BASE_DIR / "model" / "xlsr"
    args.artifactnet_dir = BASE_DIR / "model" / "artifactnet"
    args.htdemucs_repo = BASE_DIR / "model" / "htdemucs"
    run(args)


if __name__ == "__main__":
    main()
'''


SUBMISSION_REQUIREMENTS = """\
# ArtifactNet is distributed as an ONNX graph. All other dependencies are
# already part of the competition image.
onnxruntime-gpu==1.23.2
"""


def copy_xlsr_fp16(source_dir: Path, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    tensors = load_file(source_dir / "model.safetensors")
    converted = {
        name: (tensor.half() if tensor.is_floating_point() else tensor)
        for name, tensor in tensors.items()
    }
    save_file(converted, str(destination_dir / "model.safetensors"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsr-dir", type=Path, required=True)
    parser.add_argument("--panns-dir", type=Path, required=True)
    parser.add_argument("--htdemucs-dir", type=Path, required=True)
    parser.add_argument("--artifactnet-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fp32", action="store_true",
                        help="Ship full-precision XLS-R weights instead of fp16.")
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()

    out = args.output_dir
    if out.exists():
        shutil.rmtree(out)
    (out / "model").mkdir(parents=True)

    print("copying src/")
    # evaluate.py is a dev tool; it pulls pandas and scikit-learn, which the
    # pipeline never touches, so keep it out of the grading image's way.
    shutil.copytree(REPO_ROOT / "src", out / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "evaluate.py"))

    print("copying model/panns")
    shutil.copytree(args.panns_dir, out / "model" / "panns",
                    ignore=shutil.ignore_patterns("__pycache__"))
    print("copying model/htdemucs")
    shutil.copytree(args.htdemucs_dir, out / "model" / "htdemucs")

    if args.fp32:
        print("copying model/xlsr (fp32)")
        (out / "model" / "xlsr").mkdir()
        shutil.copy2(args.xlsr_dir / "model.safetensors",
                     out / "model" / "xlsr" / "model.safetensors")
    else:
        print("converting model/xlsr to fp16")
        copy_xlsr_fp16(args.xlsr_dir, out / "model" / "xlsr")

    print("copying model/artifactnet")
    shutil.copytree(
        args.artifactnet_dir, out / "model" / "artifactnet",
        ignore=shutil.ignore_patterns(".cache", "__pycache__"),
    )

    (out / "script.py").write_text(SCRIPT_TEMPLATE, encoding="utf-8")
    (out / "script.py").chmod(0o755)
    # NOT the repo's requirements.txt. That one provisions a dev machine, and
    # pip-installing it over the grading image mixes numpy-2-built wheels into
    # a numpy-1 runtime -- "numpy.dtype size changed, Expected 96 ... got 88"
    # before the first model even loads. The grader preinstalls everything the
    # baseline imports, and this pipeline imports nothing beyond that set, so
    # the right answer is to ask for nothing.
    (out / "requirements.txt").write_text(SUBMISSION_REQUIREMENTS, encoding="utf-8")

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"\npackage: {out}  ({total / 2**30:.2f} GiB)")

    if args.zip:
        archive = shutil.make_archive(str(out), "zip", root_dir=out)
        print(f"archive: {archive}  "
              f"({Path(archive).stat().st_size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
