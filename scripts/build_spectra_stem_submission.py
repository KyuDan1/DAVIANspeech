#!/usr/bin/env python3
"""Build v18 plus a low-weight Spectra-AASIST vocal-stem expert."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]

ENTRYPOINT = '''#!/usr/bin/env python3
"""DACON entrypoint: verified v18 plus Spectra vocal-stem MoE."""

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
from spectra_aasist_detector import apply_spectra_voice_fusion  # noqa: E402
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
    spectra_stats = BASE_DIR / "output" / ".spectra_stem_stats.npz"
    args.spectra_model_dir = BASE_DIR / "model" / "spectra-aasist"
    args.spectra_statistics_output = spectra_stats
    args.spectra_windows = 3
    args.spectra_file_batch_size = 4

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
        [BASE_DIR / "model" / "channel-invariant" / "seed_00.pt"],
        device=args.device, file_weight=0.05,
        voice_weight=0.05, music_weight=0.05,
    )
    apply_spectra_voice_fusion(
        args.output, spectra_stats, voice_weight=0.10, file_weight=0.05,
        voice_presence_gate=0.50,
    )
    for path in (eat_stats, spear_stats, spectra_stats):
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
'''


def replace_copy(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)


def inject_spectra(pipeline_path: Path) -> None:
    """Add stem score collection to the verified legacy pipeline copy."""
    text = pipeline_path.read_text(encoding="utf-8")
    replacements = {
        "from xlsr_antideepfake import XlsrAntiDeepfake  # noqa: E402\n": (
            "from xlsr_antideepfake import XlsrAntiDeepfake  # noqa: E402\n"
            "from spectra_aasist_detector import SpectraStemScorer  # noqa: E402\n"
        ),
        "    artifact_detector = ArtifactNetMusicDetector(args.artifactnet_dir)\n\n": (
            "    artifact_detector = ArtifactNetMusicDetector(args.artifactnet_dir)\n"
            "    spectra_scorer = SpectraStemScorer(\n"
            "        args.spectra_model_dir, device=args.device,\n"
            "        windows=args.spectra_windows,\n"
            "        file_batch_size=args.spectra_file_batch_size,\n"
            "    )\n\n"
        ),
        "        voice_audio, music_audio = separator.separate(path)\n": (
            "        voice_audio, music_audio = separator.separate(path)\n"
            "        spectra_scorer.add(path.stem, voice_audio)\n"
        ),
        "    print(f\"[3/3] Writing {args.output}\", flush=True)\n": (
            "    spectra_scorer.save(args.spectra_statistics_output)\n"
            "    print(f\"[3/3] Writing {args.output}\", flush=True)\n"
        ),
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise RuntimeError(
                f"Expected exactly one injection point in {pipeline_path}: {old!r}"
            )
        text = text.replace(old, new)
    pipeline_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=ROOT / "temporal_mert_fakeprint_v17"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=ROOT / "models/channel-invariant-v1/dual_domain_head.pt",
    )
    parser.add_argument(
        "--spectra-model", type=Path,
        default=ROOT / "models/external/spectra_aasist",
    )
    args = parser.parse_args()
    required = [
        args.base, args.checkpoint, args.spectra_model / "model.py",
        args.spectra_model / "model.safetensors",
        args.spectra_model / "xlsr_config" / "config.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error(f"missing input: {', '.join(missing)}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    shutil.copytree(
        args.base, args.output, copy_function=os.link,
        ignore=shutil.ignore_patterns("data", "open", "output", "__pycache__"),
    )
    source_dir = args.output / "model" / "src"
    for name in (
        "dual_domain_inference.py", "invariant_dual_domain_head.py",
        "spectra_aasist_detector.py",
    ):
        replace_copy(ROOT / "src" / name, source_dir / name)
    # Keep the exact v17/v18 pipeline implementation and inject only score
    # collection around the vocal stem it already computes.
    replace_copy(source_dir / "pipeline.py", source_dir / "pipeline.py.inject")
    source_dir.joinpath("pipeline.py.inject").replace(source_dir / "pipeline.py")
    inject_spectra(source_dir / "pipeline.py")

    invariant_dir = args.output / "model" / "channel-invariant"
    invariant_dir.mkdir(parents=True)
    shutil.copy2(args.checkpoint, invariant_dir / "seed_00.pt")
    spectra_dir = args.output / "model" / "spectra-aasist"
    shutil.copytree(
        args.spectra_model, spectra_dir, copy_function=os.link,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    replace_copy(args.spectra_model / "model.py", spectra_dir / "model.py")

    script_path = args.output / "script.py"
    script_path.unlink(missing_ok=True)
    script_path.write_text(ENTRYPOINT, encoding="utf-8")
    print(f"Built {args.output}")


if __name__ == "__main__":
    main()
