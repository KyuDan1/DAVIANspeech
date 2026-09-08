#!/usr/bin/env python3
"""Acoustic semantic audit for reserved prospective fake-music sources.

This program is deliberately isolated from authenticity detectors.  It reads
only reservation metadata, source audio, an authorized development manifest's
stored PANNs ``VOICE_SCREEN`` values, the independent PANNs AudioSet tagger,
and optionally HTDemucs vocal-stem energy.  It never reads XLS-R, EAT, SPEAR,
Music fake, File fake, or submission predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


CALIBRATION_RELATIVE = "data/eval/echoes_fma_paired_v3/truth_dev.csv"
VOCAL_LABELS = (
    "Singing", "Male singing", "Female singing", "Child singing",
    "Synthetic singing", "Choir", "Yodeling", "Chant", "Rapping",
    "Humming", "Opera", "Vocal music",
)
SPEECH_LABELS = (
    "Speech", "Male speech, man speaking", "Female speech, woman speaking",
    "Child speech, kid speaking", "Conversation", "Narration, monologue",
    "Speech synthesizer", "Whispering",
    "Hubbub, speech noise, speech babble",
)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def round_up(value: float, step: float) -> float:
    return float(math.ceil((value - 1e-12) / step) * step)


def verify_calibration_role(config_path: Path, calibration_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text("utf-8")) or {}
    allowed = {
        str((ROOT / item).resolve()) for item in config.get("development", [])
    }
    if str(calibration_path.resolve()) not in allowed:
        raise ValueError(
            "VOICE_SCREEN threshold calibration manifest is not an authorized "
            f"development bank: {calibration_path}"
        )


def calibrate_panns_threshold(
    config_path: Path, calibration_path: Path,
) -> tuple[float, dict[str, object]]:
    verify_calibration_role(config_path, calibration_path)
    frame = pd.read_csv(calibration_path)
    if "VOICE_SCREEN" not in frame:
        raise ValueError(f"VOICE_SCREEN missing from {calibration_path}")
    values = frame.VOICE_SCREEN.dropna().to_numpy(np.float64)
    if not len(values):
        raise ValueError("VOICE_SCREEN calibration is empty")
    # Freeze a simple upper-envelope rule before prospective audio is opened.
    # The 0.01 ceiling reproduces the original builder's 0.20 boundary without
    # copying that CLI default or looking at prospective scores.
    threshold = round_up(float(values.max()), 0.01)
    return threshold, {
        "path": str(calibration_path.resolve()),
        "sha256": sha256_file(calibration_path),
        "rows": int(len(values)),
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(values.max()),
        "threshold_rule": "ceil(max authorized-dev VOICE_SCREEN to 0.01)",
        "threshold": threshold,
        "limitation": (
            "This development bank was itself pre-screened at construction, "
            "so its VOICE_SCREEN distribution is right-censored. The threshold "
            "is an upper-envelope anomaly screen, not an estimated vocal "
            "detection operating point."
        ),
    }


def load_audio(path: Path, target_rate: int = 16_000) -> np.ndarray:
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.nan_to_num(audio.mean(axis=1).astype(np.float32))
    if rate != target_rate:
        divisor = math.gcd(int(rate), target_rate)
        mono = resample_poly(
            mono, target_rate // divisor, int(rate) // divisor
        ).astype(np.float32)
    if not len(mono):
        raise ValueError(f"empty audio: {path}")
    return mono


class IndependentPannsVocalScreen:
    """Expose vocal/singing and speech AudioSet groups separately."""

    def __init__(self, model_dir: Path, device: str) -> None:
        from presence import PannsPresence
        from panns_inference import labels

        self.presence = PannsPresence(model_dir, device=device)
        index = {name: offset for offset, name in enumerate(labels)}
        missing = set((*VOCAL_LABELS, *SPEECH_LABELS)).difference(index)
        if missing:
            raise ValueError(f"PANNs labels missing: {sorted(missing)}")
        self.vocal = np.asarray([index[name] for name in VOCAL_LABELS])
        self.speech = np.asarray([index[name] for name in SPEECH_LABELS])
        self.index = index

    def score(self, audio: np.ndarray) -> dict[str, float]:
        clipwise, _ = self.presence.model.inference(
            self.presence._segments_32k(audio)
        )
        clipwise = np.asarray(clipwise, dtype=np.float32)
        result = {
            "PANNS_VOCAL_MAX": float(clipwise[:, self.vocal].max()),
            "PANNS_SPEECH_MAX": float(clipwise[:, self.speech].max()),
        }
        result["PANNS_ANY_VOICE_MAX"] = max(result.values())
        for label in VOCAL_LABELS:
            key = "PANNS_" + label.upper().replace(" ", "_").replace(",", "")
            result[key] = float(clipwise[:, self.index[label]].max())
        return result


def resolve_source(source_root: Path, member: str) -> Path:
    root = source_root.resolve()
    relative = Path(str(member).replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe archive member path: {member}")
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"source escapes root: {member}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def selected_fake_music(
    reservation_path: Path, source_root: Path,
) -> pd.DataFrame:
    frame = pd.read_csv(reservation_path, dtype=str).fillna("")
    required = {
        "MUSIC_FAKE", "MUSIC_SOURCE_ID", "MUSIC_GROUP_ID",
        "MUSIC_GENERATOR", "MUSIC_ARCHIVE_MEMBER",
    }
    if missing := required.difference(frame):
        raise ValueError(f"reservation lacks columns: {sorted(missing)}")
    selected = frame.loc[frame.MUSIC_FAKE.astype(int).eq(1)].copy()
    identity = ["MUSIC_SOURCE_ID", "MUSIC_ARCHIVE_MEMBER"]
    conflicts = selected.groupby("MUSIC_SOURCE_ID").MUSIC_ARCHIVE_MEMBER.nunique()
    if (conflicts > 1).any():
        raise ValueError("one music source ID maps to multiple archive members")
    selected = selected.drop_duplicates(identity).reset_index(drop=True)
    selected["AUDIO_PATH"] = [
        str(resolve_source(source_root, member))
        for member in selected.MUSIC_ARCHIVE_MEMBER
    ]
    return selected


def demucs_energy(separator, path: Path) -> dict[str, float]:
    mixture = load_audio(path)
    vocals, residual = separator.separate(path)
    count = min(len(mixture), len(vocals), len(residual))
    if count <= 0:
        raise ValueError(f"empty Demucs output: {path}")
    mixture = mixture[:count].astype(np.float64)
    vocals = vocals[:count].astype(np.float64)
    residual = residual[:count].astype(np.float64)
    epsilon = 1e-12
    mix_energy = float(np.mean(mixture * mixture) + epsilon)
    vocal_energy = float(np.mean(vocals * vocals) + epsilon)
    residual_energy = float(np.mean(residual * residual) + epsilon)
    return {
        "DEMUCS_VOCAL_TO_MIX_DB": float(10 * np.log10(vocal_energy / mix_energy)),
        "DEMUCS_VOCAL_TO_RESIDUAL_DB": float(
            10 * np.log10(vocal_energy / residual_energy)
        ),
    }


def calibrate_demucs(
    separator, calibration_path: Path, output_dir: Path,
) -> tuple[float, dict[str, object]]:
    cache = output_dir / "demucs_development_calibration.csv"
    if cache.is_file():
        result = pd.read_csv(cache, dtype={"ID": str})
        expected = pd.read_csv(calibration_path, dtype={"ID": str}).ID.astype(str)
        if set(result.ID) != set(expected) or len(result) != len(expected):
            raise ValueError("incomplete/misaligned cached Demucs calibration")
    else:
        frame = pd.read_csv(calibration_path, dtype={"ID": str})
        audio_dir = calibration_path.parent / "audio"
        records = []
        for index, row in enumerate(frame.itertuples(index=False), 1):
            candidates = list(audio_dir.glob(f"{row.ID}.*"))
            if len(candidates) != 1:
                raise FileNotFoundError(f"{row.ID}: expected one development audio")
            score = demucs_energy(separator, candidates[0])
            records.append({"ID": row.ID, **score})
            if index % 10 == 0:
                print(f"Demucs calibration {index}/{len(frame)}", flush=True)
        result = pd.DataFrame(records)
        result.to_csv(cache, index=False)
    values = result.DEMUCS_VOCAL_TO_MIX_DB.to_numpy(np.float64)
    threshold = round_up(float(values.max()), 0.5)
    return threshold, {
        "rows": len(result),
        "minimum_db": float(values.min()),
        "median_db": float(np.median(values)),
        "p95_db": float(np.quantile(values, 0.95)),
        "p99_db": float(np.quantile(values, 0.99)),
        "maximum_db": float(values.max()),
        "threshold_rule": "ceil(max authorized-dev vocal/mix dB to 0.5 dB)",
        "threshold_db": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reservation", type=Path,
        default=ROOT / (
            "reports/prospective_eval_v3_design/"
            "reservation_strict_instrumental/reservation.csv"
        ),
    )
    parser.add_argument(
        "--source-root", type=Path,
        default=ROOT / "data/eval/prospective_mixed_phone_v3_sources/MixFake",
    )
    parser.add_argument(
        "--calibration", type=Path, default=ROOT / CALIBRATION_RELATIVE
    )
    parser.add_argument(
        "--partition-config", type=Path,
        default=ROOT / "configs/data_partitions.yaml",
    )
    parser.add_argument("--panns-dir", type=Path, default=ROOT / "models/panns")
    parser.add_argument(
        "--demucs-repo", type=Path, default=ROOT / "models/htdemucs"
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--with-demucs", action="store_true")
    parser.add_argument(
        "--panns-threshold", type=float,
        help="Use an already frozen PANNs threshold instead of recalibrating.",
    )
    parser.add_argument(
        "--demucs-threshold-db", type=float,
        help="Use an already frozen Demucs threshold instead of recalibrating.",
    )
    parser.add_argument(
        "--only-source-id", action="append",
        help=(
            "Audit only this reserved MUSIC_SOURCE_ID. Repeat for a frozen "
            "replacement set; omitted means audit every selected fake source."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.output_dir / "summary.json").exists() or (
        args.output_dir / "source_scores.csv"
    ).exists():
        raise FileExistsError(f"completed output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.panns_threshold is None:
        panns_threshold, panns_calibration = calibrate_panns_threshold(
            args.partition_config, args.calibration
        )
    else:
        panns_threshold = float(args.panns_threshold)
        panns_calibration = {
            "threshold": panns_threshold,
            "threshold_rule": "frozen CLI threshold",
            "calibration_read": False,
        }
    provenance: dict[str, object] = {
        "purpose": "truth semantic audit; not authenticity-model evaluation",
        "authenticity_detector_scores_read": False,
        "panns_threshold_calibration": panns_calibration,
        "panns_labels": {
            "vocal_singing": list(VOCAL_LABELS),
            "speech": list(SPEECH_LABELS),
            "aggregation": "maximum label probability over all 4.04 s segments",
        },
        "reservation_path": str(args.reservation.resolve()),
        "reservation_present_at_start": args.reservation.is_file(),
        "demucs_requested": args.with_demucs,
    }

    separator = None
    demucs_threshold = None
    if args.with_demucs:
        from separation import HTDemucsSeparator

        separator = HTDemucsSeparator(
            device=args.device, repo=args.demucs_repo, shifts=0, overlap=0.25
        )
        if args.demucs_threshold_db is None:
            demucs_threshold, calibration = calibrate_demucs(
                separator, args.calibration, args.output_dir
            )
        else:
            demucs_threshold = float(args.demucs_threshold_db)
            calibration = {
                "threshold_db": demucs_threshold,
                "threshold_rule": "frozen CLI threshold",
                "calibration_read": False,
            }
        provenance["demucs_threshold_calibration"] = calibration

    if not args.reservation.is_file():
        provenance["status"] = "prepared_waiting_for_reservation"
        provenance["candidate_sources_scored"] = 0
        provenance["selection_bias_limitations"] = [
            "The authorized development reference was pre-screened with PANNs, "
            "so calibration is right-censored and cannot estimate sensitivity.",
            "The future reservation is text-screened before this acoustic audit; "
            "reported pass rates are conditional on that metadata selection.",
            "PANNs and Demucs are weak semantic screens, not human annotations; "
            "borderline or disagreeing sources require listening review.",
        ]
        (args.output_dir / "summary.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(provenance, indent=2))
        return

    reservation_hash_before = sha256_file(args.reservation)
    sources = selected_fake_music(args.reservation, args.source_root)
    if args.only_source_id:
        requested_sources = set(args.only_source_id)
        sources = sources.loc[
            sources.MUSIC_SOURCE_ID.isin(requested_sources)
        ].reset_index(drop=True)
        found_sources = set(sources.MUSIC_SOURCE_ID)
        if found_sources != requested_sources:
            raise ValueError(
                "requested source IDs are not uniquely present in the "
                f"reservation: missing={sorted(requested_sources - found_sources)}"
            )
        expected_sources = len(requested_sources)
        provenance["source_filter"] = {
            "mode": "frozen_explicit_source_ids",
            "source_ids": sorted(requested_sources),
        }
    else:
        expected_sources = int(
            pd.read_csv(args.reservation).loc[
                lambda frame: frame.MUSIC_FAKE.astype(int).eq(1),
                "MUSIC_SOURCE_ID",
            ].nunique()
        )
        provenance["source_filter"] = {"mode": "all_reserved_fake_music"}
    if len(sources) != expected_sources:
        raise ValueError(
            f"expected {expected_sources} unique reserved fake-music sources, "
            f"got {len(sources)}"
        )
    panns = IndependentPannsVocalScreen(args.panns_dir, args.device)
    records = []
    for index, row in enumerate(sources.itertuples(index=False), 1):
        path = Path(row.AUDIO_PATH)
        audio = load_audio(path)
        scores = panns.score(audio)
        if separator is not None:
            scores.update(demucs_energy(separator, path))
        panns_flag = scores["PANNS_ANY_VOICE_MAX"] > panns_threshold
        demucs_flag = bool(
            demucs_threshold is not None
            and scores["DEMUCS_VOCAL_TO_MIX_DB"] > demucs_threshold
        )
        records.append({
            "MUSIC_SOURCE_ID": row.MUSIC_SOURCE_ID,
            "MUSIC_GROUP_ID": row.MUSIC_GROUP_ID,
            "MUSIC_GENERATOR": row.MUSIC_GENERATOR,
            "MUSIC_ARCHIVE_MEMBER": row.MUSIC_ARCHIVE_MEMBER,
            "AUDIO_SHA256": sha256_file(path),
            **scores,
            "PANNS_THRESHOLD": panns_threshold,
            "PANNS_FLAG": panns_flag,
            "DEMUCS_THRESHOLD_DB": demucs_threshold,
            "DEMUCS_FLAG": demucs_flag,
            "ACOUSTIC_SCREEN_PASS": not (panns_flag or demucs_flag),
            "REVIEW_REASON": (
                "panns_and_demucs" if panns_flag and demucs_flag else
                "panns" if panns_flag else "demucs" if demucs_flag else ""
            ),
        })
        if index % 10 == 0:
            print(f"prospective acoustic screen {index}/{len(sources)}", flush=True)
    if sha256_file(args.reservation) != reservation_hash_before:
        raise RuntimeError("reservation changed during acoustic audit")
    result = pd.DataFrame(records)
    result.to_csv(args.output_dir / "source_scores.csv", index=False)
    generator = result.groupby("MUSIC_GENERATOR").agg(
        SOURCES=("MUSIC_SOURCE_ID", "size"),
        PANNS_FLAGS=("PANNS_FLAG", "sum"),
        DEMUCS_FLAGS=("DEMUCS_FLAG", "sum"),
        PASSES=("ACOUSTIC_SCREEN_PASS", "sum"),
        MAX_PANNS_VOICE=("PANNS_ANY_VOICE_MAX", "max"),
    ).reset_index()
    generator.to_csv(args.output_dir / "generator_summary.csv", index=False)
    provenance.update({
        "status": "completed",
        "reservation_sha256": reservation_hash_before,
        "candidate_sources_scored": len(result),
        "panns_flags": int(result.PANNS_FLAG.sum()),
        "demucs_flags": int(result.DEMUCS_FLAG.sum()),
        "acoustic_screen_passes": int(result.ACOUSTIC_SCREEN_PASS.sum()),
        "acoustic_screen_failures": int((~result.ACOUSTIC_SCREEN_PASS).sum()),
        "selection_bias_limitations": [
            "The authorized development reference was pre-screened with PANNs, "
            "so calibration is right-censored and cannot estimate sensitivity.",
            "The reservation was text-screened before this acoustic audit; pass "
            "rates are conditional on that metadata selection.",
            "PANNs and Demucs are weak semantic screens, not human annotations. "
            "Every flagged source and a random sample of passes need listening "
            "review before truth is frozen.",
            "This audit must not be used to choose an authenticity detector or "
            "to estimate fake-detection performance.",
        ],
    })
    (args.output_dir / "summary.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, indent=2))
    print(generator.to_string(index=False))


if __name__ == "__main__":
    main()
