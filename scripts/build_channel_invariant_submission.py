#!/usr/bin/env python3
"""Build v18/v19 with one or more channel/component-invariant heads."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]

ENTRYPOINT = '''#!/usr/bin/env python3
"""DACON entrypoint: LME+SPEAR, temporal/MERT/fakeprint/invariant MoE, CPS."""

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
from modern_fakeprint_detector import apply_modern_fakeprint_fusion  # noqa: E402
from pipeline import parse_args, run  # noqa: E402
from sofia_mert_detector import apply_sofia_mert_fusion  # noqa: E402
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
    apply_sofia_mert_fusion(
        args.test_dir, args.output, BASE_DIR / "model" / "sofia-mert" / "mert",
        BASE_DIR / "model" / "sofia-mert" / "sofia_g1_mert_head.pt",
        device=args.device, file_weight=0.025, music_weight=0.0125,
    )
    apply_modern_fakeprint_fusion(
        args.test_dir, args.output,
        BASE_DIR / "model" / "modern-fakeprint" / "weights.npz",
        file_weight=0.025, music_weight=0.025,
    )
    apply_dual_domain_fusion(
        args.output, eat_stats, spear_stats,
        sorted((BASE_DIR / "model" / "channel-invariant").glob("seed_*.pt")),
        device=args.device, file_weight=__INVARIANT_WEIGHT__,
        voice_weight=__INVARIANT_WEIGHT__, music_weight=__INVARIANT_WEIGHT__,
    )
    for path in (eat_stats, spear_stats):
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path,
                        default=ROOT / "temporal_mert_fakeprint_v17")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, nargs="+",
        default=[ROOT / "models" / "channel-invariant-v1" / "dual_domain_head.pt"],
    )
    parser.add_argument("--invariant-weight", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 <= args.invariant_weight <= 1:
        parser.error("--invariant-weight must be in [0, 1]")
    missing = [str(path) for path in args.checkpoint if not path.is_file()]
    if missing:
        parser.error(f"checkpoint not found: {', '.join(missing)}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    shutil.copytree(
        args.base, args.output, copy_function=os.link,
        ignore=shutil.ignore_patterns("data", "open", "output", "__pycache__"),
    )
    source = args.output / "model" / "src"
    for name in (
        "dual_domain_inference.py", "invariant_dual_domain_head.py",
    ):
        shutil.copy2(ROOT / "src" / name, source / name)
    invariant = args.output / "model" / "channel-invariant"
    invariant.mkdir(parents=True)
    for index, checkpoint in enumerate(args.checkpoint):
        shutil.copy2(checkpoint, invariant / f"seed_{index:02d}.pt")
    entrypoint = ENTRYPOINT.replace(
        "__INVARIANT_WEIGHT__", repr(float(args.invariant_weight))
    )
    (args.output / "script.py").write_text(entrypoint, encoding="utf-8")
    print(f"Built {args.output}")


if __name__ == "__main__":
    main()
