#!/usr/bin/env python3
"""Build v24 plus a non-telephone temporal-mixture consistency route."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]


def replace_copy(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)


def replace_once(text: str, old: str, new: str, source: Path) -> str:
    if text.count(old) != 1:
        raise RuntimeError(
            f"Expected exactly one injection point in {source}: {old!r}"
        )
    return text.replace(old, new)


def inject_layout_statistics(pipeline_path: Path) -> None:
    text = pipeline_path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "from spectra_aasist_detector import SpectraStemScorer  # noqa: E402\n",
        "from spectra_aasist_detector import SpectraStemScorer  # noqa: E402\n"
        "from component_consistency import (  # noqa: E402\n"
        "    save_layout_scores, stem_layout_score,\n"
        ")\n",
        pipeline_path,
    )
    text = replace_once(
        text,
        '    xlsr_profile = {\n'
        '        "ids": [], "offsets": [0], "starts": [], "scores": [],\n'
        '        "durations": [],\n'
        '    }\n\n'
        '    for row, path in zip(rows, tqdm(audio_files, desc="detect")):\n',
        '    xlsr_profile = {\n'
        '        "ids": [], "offsets": [0], "starts": [], "scores": [],\n'
        '        "durations": [],\n'
        '    }\n'
        '    layout_ids, layout_scores = [], []\n\n'
        '    for row, path in zip(rows, tqdm(audio_files, desc="detect")):\n',
        pipeline_path,
    )
    text = replace_once(
        text,
        '        original_audio = load_audio(path)\n'
        '        music_fake_artifact = artifact_detector.fake_probability(original_audio)\n',
        '        original_audio = load_audio(path)\n'
        '        layout_ids.append(path.stem)\n'
        '        layout_scores.append(stem_layout_score(\n'
        '            original_audio, voice_audio, music_audio,\n'
        '        ))\n'
        '        music_fake_artifact = artifact_detector.fake_probability(original_audio)\n',
        pipeline_path,
    )
    text = replace_once(
        text,
        '    spectra_scorer.save(\n'
        '        args.spectra_statistics_output,\n'
        '        args.spectra_consensus_statistics_output,\n'
        '    )\n',
        '    spectra_scorer.save(\n'
        '        args.spectra_statistics_output,\n'
        '        args.spectra_consensus_statistics_output,\n'
        '    )\n'
        '    save_layout_scores(\n'
        '        args.layout_statistics_output, layout_ids, layout_scores,\n'
        '    )\n',
        pipeline_path,
    )
    pipeline_path.write_text(text, encoding="utf-8")


def inject_entrypoint(script_path: Path) -> None:
    text = script_path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "from anchor_spear_stats_fusion import apply_fusion_with_stats  # noqa: E402\n",
        "from anchor_spear_stats_fusion import apply_fusion_with_stats  # noqa: E402\n"
        "from component_consistency import (  # noqa: E402\n"
        "    apply_mixed_component_consistency,\n"
        ")\n",
        script_path,
    )
    text = replace_once(
        text,
        '    spectra_consensus = BASE_DIR / "output" / ".spectra_call_windows.npz"\n',
        '    spectra_consensus = BASE_DIR / "output" / ".spectra_call_windows.npz"\n'
        '    layout_stats = BASE_DIR / "output" / ".stem_layout_stats.npz"\n'
        '    telephone_ids = BASE_DIR / "output" / ".telephone_ids.npz"\n',
        script_path,
    )
    text = replace_once(
        text,
        '    args.spectra_consensus_statistics_output = spectra_consensus\n',
        '    args.spectra_consensus_statistics_output = spectra_consensus\n'
        '    args.layout_statistics_output = layout_stats\n',
        script_path,
    )
    text = replace_once(
        text,
        '        telephone_router_path=BASE_DIR / "model" / "telephone-router.npz",\n'
        '        phone_voice_weight=0.10,\n',
        '        telephone_router_path=BASE_DIR / "model" / "telephone-router.npz",\n'
        '        phone_voice_weight=0.10,\n'
        '        telephone_ids_output_path=telephone_ids,\n',
        script_path,
    )
    text = replace_once(
        text,
        '        file_voice_only_weight=0.40,\n'
        '    )\n'
        '    for path in (\n',
        '        file_voice_only_weight=0.40,\n'
        '    )\n'
        '    apply_mixed_component_consistency(\n'
        '        args.output, layout_stats, telephone_ids,\n'
        '        layout_threshold=0.35, file_weight=0.70,\n'
        '    )\n'
        '    for path in (\n',
        script_path,
    )
    text = replace_once(
        text,
        '        spectra_consensus,\n'
        '    ):\n',
        '        spectra_consensus, layout_stats, telephone_ids,\n'
        '    ):\n',
        script_path,
    )
    script_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=ROOT / "sparse_call_consistency_v24",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.base.is_dir():
        parser.error(f"missing base package: {args.base}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    shutil.copytree(
        args.base, args.output, copy_function=os.link,
        ignore=shutil.ignore_patterns("data", "open", "output", "__pycache__"),
    )
    source_dir = args.output / "model" / "src"
    replace_copy(args.base / "model" / "src" / "pipeline.py", source_dir / "pipeline.py")
    replace_copy(args.base / "script.py", args.output / "script.py")
    replace_copy(ROOT / "src" / "component_consistency.py", source_dir / "component_consistency.py")
    inject_layout_statistics(source_dir / "pipeline.py")
    inject_entrypoint(args.output / "script.py")
    print(f"Built {args.output}")


if __name__ == "__main__":
    main()
