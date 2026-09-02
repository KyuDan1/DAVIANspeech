#!/usr/bin/env python3
"""Build LME+SPEAR+CPS with a low-weight temporal MIL authenticity expert."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]

ENTRYPOINT = '''#!/usr/bin/env python3
"""DACON entry point: verified LME+SPEAR, decoupled CPS, temporal MIL."""

import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
sys.dont_write_bytecode = True

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "model" / "src"))

from anchor_spear_stats_fusion import apply_fusion_with_stats  # noqa: E402
from eat_presence_fusion import apply_eat_presence_fusion  # noqa: E402
from pipeline import parse_args, run  # noqa: E402
from temporal_dual_domain_inference import apply_temporal_dual_domain_fusion  # noqa: E402


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
    eat_stats = BASE_DIR / "output" / ".eat_temporal_stats.npz"
    spear_stats = BASE_DIR / "output" / ".spear_temporal_stats.npz"

    run(args)
    apply_eat_presence_fusion(
        args.test_dir, args.output, BASE_DIR / "model" / "eat",
        BASE_DIR / "model" / "panns", device=args.device,
        voice_weight=0.35, music_weight=0.90, file_gate=0.60,
        update_file_score=False,
        presence_head_path=BASE_DIR / "model" / "eat-presence-head-v1.npz",
        music_probe_weight=0.40,
        statistics_output_path=eat_stats,
        phone_voice_head_path=BASE_DIR / "model" / "phone-voice-presence-head.npz",
        telephone_router_path=BASE_DIR / "model" / "telephone-router.npz",
        phone_voice_weight=0.10,
    )
    apply_fusion_with_stats(
        args.test_dir, args.output, BASE_DIR / "model" / "spear",
        BASE_DIR / "model" / "spear-mixed-music-head.npz",
        BASE_DIR / "model" / "spear-cross-component-joint-v1.npz",
        device=args.device, weight=0.10, statistics_output_path=spear_stats,
    )
    apply_temporal_dual_domain_fusion(
        args.output, eat_stats, spear_stats,
        sorted((BASE_DIR / "model" / "temporal-domain-v1").glob("seed_*.pt")),
        device=args.device, file_weight=0.05,
        voice_weight=0.05, music_weight=0.05,
        music_checkpoint_paths=sorted(
            (BASE_DIR / "model" / "temporal-domain-v2").glob("seed_*.pt")
        ),
    )
    for path in (eat_stats, spear_stats):
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-zip", type=Path, default=ROOT / "cps_v13.zip")
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--music-checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    with zipfile.ZipFile(args.base_zip) as archive:
        roots = {Path(name).parts[0] for name in archive.namelist() if Path(name).parts}
        if not roots.issubset({"model", "script.py", "requirements.txt"}):
            raise ValueError(f"Unexpected base ZIP roots: {sorted(roots)}")
        archive.extractall(args.output)

    source_dir = args.output / "model" / "src"
    for name in (
        "anchor_spear_stats_fusion.py", "dual_domain_head.py",
        "dual_domain_stats.py", "eat_presence_fusion.py",
        "spear_detector.py",
        "temporal_dual_domain_head.py", "temporal_dual_domain_inference.py",
    ):
        shutil.copy2(ROOT / "src" / name, source_dir / name)
    for name, paths in (
        ("temporal-domain-v1", args.checkpoint),
        ("temporal-domain-v2", args.music_checkpoint),
    ):
        checkpoint_dir = args.output / "model" / name
        checkpoint_dir.mkdir()
        for index, path in enumerate(paths):
            shutil.copy2(path, checkpoint_dir / f"seed_{index:02d}.pt")
    (args.output / "script.py").write_text(ENTRYPOINT, encoding="utf-8")
    (args.output / "script.py").chmod(0o755)

    total = sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file())
    print(f"Built {args.output} ({total / 2**30:.2f} GiB unpacked)")
    if args.zip:
        archive = shutil.make_archive(str(args.output), "zip", root_dir=args.output)
        print(f"Built {archive} ({Path(archive).stat().st_size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
