#!/usr/bin/env python3
"""Build LME+SPEAR with decoupled CPS and phone-only dual File/Music fusion."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]

ENTRYPOINT = '''#!/usr/bin/env python3
"""DACON entry point: LME+SPEAR, decoupled CPS, phone-only dual ADS."""

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
from dual_domain_inference import apply_dual_domain_fusion  # noqa: E402
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
    eat_stats = BASE_DIR / "output" / ".eat_dual_stats.npz"
    spear_stats = BASE_DIR / "output" / ".spear_dual_stats.npz"
    telephone_ids = BASE_DIR / "output" / ".telephone_ids.npz"

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
        telephone_ids_output_path=telephone_ids,
    )
    apply_fusion_with_stats(
        args.test_dir, args.output, BASE_DIR / "model" / "spear",
        BASE_DIR / "model" / "spear-mixed-music-head.npz",
        BASE_DIR / "model" / "spear-cross-component-joint-v1.npz",
        device=args.device, weight=0.10, statistics_output_path=spear_stats,
    )
    # Voice transfer was inconsistent between two independent phone banks.
    # Keep the leaderboard anchor voice score and replace only File/Music.
    apply_dual_domain_fusion(
        args.output, eat_stats, spear_stats,
        sorted((BASE_DIR / "model" / "dual-domain").glob("seed_*.pt")),
        device=args.device, file_weight=0.50,
        voice_weight=0.0, music_weight=0.50,
        routed_ids_path=telephone_ids,
    )
    for path in (eat_stats, spear_stats, telephone_ids):
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=ROOT / "submit_lme_spear_dual_presence_v8"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    shutil.copytree(
        args.base, args.output, copy_function=os.link,
        ignore=shutil.ignore_patterns("data", "open", "output", "__pycache__"),
    )
    for name in (
        "eat_presence_fusion.py", "dual_domain_inference.py", "telephone_router.py",
    ):
        shutil.copy2(ROOT / "src" / name, args.output / "model/src" / name)
    shutil.copy2(
        ROOT / "reports/phone_presence_probe_v2/phone_voice_presence_head.npz",
        args.output / "model/phone-voice-presence-head.npz",
    )
    shutil.copy2(
        ROOT / "model_heads/telephone-router-narrowband-v1.npz",
        args.output / "model/telephone-router.npz",
    )
    (args.output / "script.py").write_text(ENTRYPOINT, encoding="utf-8")
    print(f"Built {args.output}")


if __name__ == "__main__":
    main()
