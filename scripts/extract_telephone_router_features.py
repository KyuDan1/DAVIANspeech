"""Extract paired clean/telephone router features without storing audio copies."""

from __future__ import annotations

import argparse
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from telephone_channel import (  # noqa: E402
    NEGATIVE_VARIANTS,
    POSITIVE_VARIANTS,
    WIDEBAND_VARIANTS,
    apply_channel,
)
from telephone_router import extract_telephone_features  # noqa: E402


AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}


def _stable_key(*values: str) -> int:
    digest = hashlib.sha256("|".join(values).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _process(task):
    path, dataset, parent_id, variant, ffmpeg, max_seconds = task
    audio, _ = librosa.load(path, sr=16_000, mono=True, dtype=np.float32)
    maximum = int(max_seconds * 16_000)
    if len(audio) > maximum:
        start = _stable_key(dataset, parent_id) % (len(audio) - maximum + 1)
        audio = audio[start:start + maximum]
    transformed = apply_channel(
        audio, variant, ffmpeg=Path(ffmpeg), key=_stable_key(dataset, parent_id, variant)
    )
    return extract_telephone_features(transformed)


def _choose_rows(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if not maximum or len(frame) <= maximum:
        return frame
    strata = [column for column in ("AUDIO_TYPE", "CONDITION", "SOURCE") if column in frame]
    if not strata:
        return frame.sample(maximum, random_state=seed)
    chosen = []
    grouped = list(frame.groupby(strata, dropna=False))
    for _, group in grouped:
        count = max(1, round(maximum * len(group) / len(frame)))
        chosen.append(group.sample(min(count, len(group)), random_state=seed))
    result = pd.concat(chosen).drop_duplicates("ID")
    if len(result) > maximum:
        result = result.sample(maximum, random_state=seed)
    elif len(result) < maximum:
        remainder = frame[~frame.ID.isin(result.ID)]
        result = pd.concat([
            result, remainder.sample(min(maximum - len(result), len(remainder)), random_state=seed)
        ])
    return result


def _source_is_narrowband(row: dict) -> bool:
    markers = " ".join(str(row.get(column, "")) for column in (
        "CHANNEL", "STRESS_VARIANT", "CODEC", "FORMAT"
    )).lower()
    return any(token in markers for token in (
        "telephone", "phone", "pstn", "g711", "g.711", "g726", "g.726",
        "gsm", "amr_nb", "amr-nb", "narrowband",
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True,
                        help="Relative data/eval dataset directory; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True,
                        choices=sorted(set(
                            POSITIVE_VARIANTS + NEGATIVE_VARIANTS + WIDEBAND_VARIANTS
                        )))
    parser.add_argument("--max-files-per-dataset", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--ffmpeg", type=Path,
        default=Path("/home/nas_main/kyudanjung/conda_envs/envs/davianspeech/bin/ffmpeg"),
    )
    parser.add_argument("--training", action="store_true",
                        help="Apply the locked-data leakage guard to every input truth")
    args = parser.parse_args()

    tasks, metadata = [], []
    for dataset_index, dataset in enumerate(args.dataset):
        root = ROOT / "data" / "eval" / dataset
        truth_path = root / "truth.csv"
        if args.training:
            assert_no_locked_eval_leakage(truth_path, ROOT / "configs/data_partitions.yaml")
        truth = pd.read_csv(truth_path, dtype={"ID": str})
        truth = _choose_rows(
            truth, args.max_files_per_dataset, args.seed + dataset_index
        ).sort_values("ID")
        audio_by_id = {
            path.stem: path for path in (root / "audio").iterdir()
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        }
        for row in truth.to_dict("records"):
            parent_id = str(row["ID"])
            path = audio_by_id[parent_id]
            source_is_narrowband = _source_is_narrowband(row)
            for variant in args.variants:
                tasks.append((
                    path, dataset, parent_id, variant, args.ffmpeg, args.max_seconds
                ))
                metadata.append((
                    dataset, parent_id, variant,
                    int(source_is_narrowband or variant in POSITIVE_VARIANTS),
                    str(row.get("AUDIO_TYPE", "unknown")),
                    int(source_is_narrowband),
                ))

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        features = list(tqdm(
            executor.map(_process, tasks, chunksize=4), total=len(tasks),
            desc="telephone features",
        ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.stack(features),
        dataset=np.asarray([row[0] for row in metadata]),
        parent_id=np.asarray([row[1] for row in metadata]),
        variant=np.asarray([row[2] for row in metadata]),
        label=np.asarray([row[3] for row in metadata], dtype=np.int8),
        audio_type=np.asarray([row[4] for row in metadata]),
        source_is_narrowband=np.asarray([row[5] for row in metadata], dtype=np.int8),
    )
    print(f"saved {len(features)} rows x {features[0].size} features to {args.output}")


if __name__ == "__main__":
    main()
