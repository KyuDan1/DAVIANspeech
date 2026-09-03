#!/usr/bin/env python3
"""Build v22 plus sparse-call consensus and voice-only File consistency."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]


def replace_copy(source: Path, destination: Path) -> None:
    """Break a package hardlink before replacing one generated member."""
    destination.unlink(missing_ok=True)
    shutil.copy2(source, destination)


def replace_once(text: str, old: str, new: str, source: Path) -> str:
    if text.count(old) != 1:
        raise RuntimeError(
            f"Expected exactly one injection point in {source}: {old!r}"
        )
    return text.replace(old, new)


def inject_xlsr_profiles(pipeline_path: Path) -> None:
    """Retain vocal XLS-R window scores already computed by the base path."""
    text = pipeline_path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        'def fake_probability(detector, audio, device, window, batch_size,\n'
        '                     pooling="max", temperature=5.0):',
        'def fake_probability(detector, audio, device, window, batch_size,\n'
        '                     pooling="max", temperature=5.0, profile=None,\n'
        '                     profile_id=None):',
        pipeline_path,
    )
    text = replace_once(
        text,
        '    if rms < SILENCE_RMS:\n'
        '        # Nothing to judge: an empty stem must not create fake evidence.\n'
        '        return 0.0\n\n'
        '    windows = np.stack([\n'
        '        extract_segment(audio, start, window)\n'
        '        for start in segment_starts(audio.size, window)\n'
        '    ])\n',
        '    if rms < SILENCE_RMS:\n'
        '        # Keep an aligned invalid row; Spectra validity suppresses fusion.\n'
        '        if profile is not None:\n'
        '            profile["ids"].append(str(profile_id))\n'
        '            profile["starts"].append(0)\n'
        '            profile["scores"].append(0.0)\n'
        '            profile["offsets"].append(len(profile["scores"]))\n'
        '            profile["durations"].append(audio.size / 16000)\n'
        '        return 0.0\n\n'
        '    starts = segment_starts(audio.size, window)\n'
        '    windows = np.stack([\n'
        '        extract_segment(audio, start, window) for start in starts\n'
        '    ])\n',
        pipeline_path,
    )
    text = replace_once(
        text,
        '    return pool_window_scores(np.asarray(scores), pooling, temperature)\n',
        '    values = np.asarray(scores)\n'
        '    if profile is not None:\n'
        '        profile["ids"].append(str(profile_id))\n'
        '        profile["starts"].extend(starts)\n'
        '        profile["scores"].extend(values.tolist())\n'
        '        profile["offsets"].append(len(profile["scores"]))\n'
        '        profile["durations"].append(audio.size / 16000)\n'
        '    return pool_window_scores(values, pooling, temperature)\n',
        pipeline_path,
    )
    text = replace_once(
        text,
        '        file_batch_size=args.spectra_file_batch_size,\n'
        '    )\n\n'
        '    for row, path in zip(rows, tqdm(audio_files, desc="detect")):\n',
        '        file_batch_size=args.spectra_file_batch_size,\n'
        '        collect_sliding=True,\n'
        '    )\n'
        '    xlsr_profile = {\n'
        '        "ids": [], "offsets": [0], "starts": [], "scores": [],\n'
        '        "durations": [],\n'
        '    }\n\n'
        '    for row, path in zip(rows, tqdm(audio_files, desc="detect")):\n',
        pipeline_path,
    )
    text = replace_once(
        text,
        '            args.pooling, args.temperature\n'
        '        )\n'
        '        music_fake_xlsr = fake_probability(',
        '            args.pooling, args.temperature, profile=xlsr_profile,\n'
        '            profile_id=path.stem\n'
        '        )\n'
        '        music_fake_xlsr = fake_probability(',
        pipeline_path,
    )
    text = replace_once(
        text,
        '    spectra_scorer.save(args.spectra_statistics_output)\n',
        '    np.savez_compressed(\n'
        '        args.xlsr_consensus_statistics_output,\n'
        '        ids=np.asarray(xlsr_profile["ids"]),\n'
        '        offsets=np.asarray(xlsr_profile["offsets"], dtype=np.int64),\n'
        '        starts=np.asarray(xlsr_profile["starts"], dtype=np.int64),\n'
        '        scores=np.asarray(xlsr_profile["scores"], dtype=np.float32),\n'
        '        durations=np.asarray(xlsr_profile["durations"], dtype=np.float32),\n'
        '        window=np.asarray(args.window, dtype=np.int64),\n'
        '    )\n'
        '    spectra_scorer.save(\n'
        '        args.spectra_statistics_output,\n'
        '        args.spectra_consensus_statistics_output,\n'
        '    )\n',
        pipeline_path,
    )
    pipeline_path.write_text(text, encoding="utf-8")


def inject_entrypoint(script_path: Path) -> None:
    text = script_path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        'from spectra_aasist_detector import apply_spectra_voice_fusion  # noqa: E402\n',
        'from spectra_aasist_detector import apply_spectra_voice_fusion  # noqa: E402\n'
        'from sparse_call_consensus import apply_sparse_call_consensus  # noqa: E402\n',
        script_path,
    )
    text = replace_once(
        text,
        '    spectra_stats = BASE_DIR / "output" / ".spectra_stem_stats.npz"\n',
        '    spectra_stats = BASE_DIR / "output" / ".spectra_stem_stats.npz"\n'
        '    xlsr_consensus = BASE_DIR / "output" / ".xlsr_call_windows.npz"\n'
        '    spectra_consensus = BASE_DIR / "output" / ".spectra_call_windows.npz"\n',
        script_path,
    )
    text = replace_once(
        text,
        '    args.spectra_statistics_output = spectra_stats\n',
        '    args.spectra_statistics_output = spectra_stats\n'
        '    args.xlsr_consensus_statistics_output = xlsr_consensus\n'
        '    args.spectra_consensus_statistics_output = spectra_consensus\n',
        script_path,
    )
    text = replace_once(
        text,
        '    apply_spectra_voice_fusion(\n'
        '        args.output, spectra_stats, voice_weight=0.10, file_weight=0.05,\n'
        '        voice_presence_gate=0.50,\n'
        '    )\n',
        '    apply_spectra_voice_fusion(\n'
        '        args.output, spectra_stats, voice_weight=0.10, file_weight=0.05,\n'
        '        voice_presence_gate=0.50,\n'
        '    )\n'
        '    apply_sparse_call_consensus(\n'
        '        args.output, xlsr_consensus, spectra_consensus,\n'
        '        BASE_DIR / "model" / "sparse-call-consensus.npz",\n'
        '        voice_weight=0.60,\n'
        '        file_voice_only_weight=0.40,\n'
        '    )\n',
        script_path,
    )
    text = replace_once(
        text,
        '    for path in (eat_stats, spear_stats, spectra_stats):\n',
        '    for path in (\n'
        '        eat_stats, spear_stats, spectra_stats, xlsr_consensus,\n'
        '        spectra_consensus,\n'
        '    ):\n',
        script_path,
    )
    script_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, default=ROOT / "spectra_stem_v22_fixed"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--head", type=Path,
        default=ROOT / "model_heads/sparse-call-consensus-v1.npz",
    )
    args = parser.parse_args()
    required = [args.base, args.head]
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
    # ``copytree(..., copy_function=os.link)`` saves ~9 GB, but every text file
    # initially shares an inode with the immutable v22 base.  Break the two
    # links that the injectors rewrite before calling ``write_text``.
    replace_copy(
        args.base / "model" / "src" / "pipeline.py",
        source_dir / "pipeline.py",
    )
    replace_copy(args.base / "script.py", args.output / "script.py")
    for name in ("spectra_aasist_detector.py", "sparse_call_consensus.py"):
        replace_copy(ROOT / "src" / name, source_dir / name)
    inject_xlsr_profiles(source_dir / "pipeline.py")
    replace_copy(args.head, args.output / "model" / "sparse-call-consensus.npz")
    inject_entrypoint(args.output / "script.py")
    print(f"Built {args.output}")


if __name__ == "__main__":
    main()
