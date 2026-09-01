"""Check whether an eval set can be solved without listening for generation.

Four times now a number in this project turned out to be measuring something
other than authenticity: ArtifactNet read loudness, GTZAN-vs-SONICS read codec
history, and one eval set leaked the voice label through clip duration
(-DURATION alone gave EER 0.2800 against the detector's 0.2633) and the music
label through bandwidth (spectral flatness gave 0.3133 against 0.3567). Each was
caught after the fact, one of them after it had cost a submission.

So: before trusting a set, try to solve it with features that cannot possibly
know whether audio was generated -- how long the clip is, how loud, where its
spectrum sits. Those should land at EER 0.5. Whatever distance they cover below
0.5 is label that leaked into the construction, and it caps what the set can
tell you.

    python scripts/leakage_gate.py --truth <set>/truth.csv --audio-dir <set>/audio
    python scripts/leakage_gate.py --truth <set>/truth.csv --path-column source_path \
        --detector <set>/run/submission.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import roc_curve

# Label -> the prediction column a detector would use for it, when one is given.
LABELS = {
    "VOICE_FAKE": "VOICE_FAKE_PROB",
    "MUSIC_FAKE": "MUSIC_FAKE_PROB",
    "FILE_FAKE": "FILE_FAKE_PROB",
}


def official_eer(labels, scores) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    index = np.argmin(np.abs(fpr - fnr))
    return float((fpr[index] + fnr[index]) / 2)


def trivial_features(path: Path) -> dict[str, float]:
    """Things that say nothing about whether audio was generated."""
    audio, rate = librosa.load(path, sr=16_000, mono=True, dtype=np.float32)
    if audio.size == 0:
        audio = np.zeros(1600, dtype=np.float32)
    spectrum = np.abs(librosa.stft(audio, n_fft=1024)) ** 2
    freqs = librosa.fft_frequencies(sr=16_000, n_fft=1024)
    total = spectrum.sum() + 1e-12
    band = lambda lo, hi: float(spectrum[(freqs >= lo) & (freqs < hi)].sum() / total)

    magnitude = np.abs(audio)
    loud = magnitude > (magnitude.max() * 0.02 + 1e-9)
    lead = int(np.argmax(loud)) if loud.any() else audio.size
    tail = int(audio.size - np.argmax(loud[::-1])) if loud.any() else 0

    return {
        "duration": audio.size / 16_000,
        "rms": float(np.sqrt(np.mean(audio ** 2))),
        "peak": float(magnitude.max()),
        "crest": float(magnitude.max() / (np.sqrt(np.mean(audio ** 2)) + 1e-9)),
        "zcr": float(np.mean(librosa.feature.zero_crossing_rate(audio))),
        "centroid": float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=16_000))),
        "rolloff95": float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=16_000, roll_percent=0.95))),
        "flatness": float(np.median(librosa.feature.spectral_flatness(y=audio))),
        "band_low": band(0, 700),
        "band_speech": band(300, 3400),
        "band_high": band(6000, 8000),
        "lead_silence": lead / 16_000,
        "tail_silence": (audio.size - tail) / 16_000,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, default=None,
                        help="Directory of <ID>.wav; otherwise use --path-column")
    parser.add_argument("--path-column", default="source_path")
    parser.add_argument("--detector", type=Path, default=None,
                        help="submission.csv, to compare against the detector")
    parser.add_argument("--limit", type=int, default=0)
    # A trivial feature within this of the detector means the set is not
    # measuring what it claims to.
    parser.add_argument("--fail-margin", type=float, default=0.03)
    parser.add_argument("--strata", type=int, default=4,
                        help="Quantile bins used to neutralise a leaking feature.")
    parser.add_argument("--min-stratum", type=int, default=40)
    args = parser.parse_args()

    truth = pd.read_csv(args.truth, dtype={"ID": str})
    if args.limit:
        truth = truth.sample(n=min(args.limit, len(truth)), random_state=0)

    rows = []
    for record in truth.itertuples():
        if args.audio_dir:
            matches = list(args.audio_dir.glob(f"{record.ID}.*"))
            if not matches:
                continue
            path = matches[0]
        else:
            path = Path(getattr(record, args.path_column))
        if not path.is_file():
            continue
        rows.append({"ID": record.ID, **trivial_features(path)})
    features = pd.DataFrame(rows)
    frame = truth.merge(features, on="ID")
    print(f"{len(frame)} clips, {len(features.columns) - 1} trivial features\n")

    detector = None
    if args.detector and args.detector.is_file():
        detector = pd.read_csv(args.detector, dtype={"ID": str})
        frame = frame.merge(detector, on="ID", suffixes=("", "_pred"))

    names = [c for c in features.columns if c != "ID"]
    worst_overall = {}
    for label, prediction in LABELS.items():
        if label not in frame:
            continue
        subset = frame.dropna(subset=[label])
        if subset[label].nunique() < 2:
            continue
        scored = []
        for name in names:
            value = official_eer(subset[label], subset[name])
            # A feature that predicts the label inverted leaks just as much.
            scored.append((min(value, 1 - value), name))
        scored.sort()
        best_eer, best_name = scored[0]
        worst_overall[label] = (best_eer, best_name)

        detector_eer = None
        if detector is not None and prediction in subset:
            detector_eer = official_eer(subset[label], subset[prediction])

        print(f"{label}  (n={len(subset)}, positives={int(subset[label].sum())})")
        for value, name in scored[:5]:
            print(f"   {name:<14} EER {value:.4f}")
        if detector_eer is not None:
            verdict = ("LEAKING — trivial feature matches or beats the detector"
                       if best_eer <= detector_eer + args.fail_margin else "ok")
            print(f"   {'detector':<14} EER {detector_eer:.4f}   -> {verdict}")
        else:
            verdict = "LEAKING" if best_eer < 0.40 else ("weak leak" if best_eer < 0.45 else "ok")
            print(f"   best trivial {best_eer:.4f} vs 0.5 expected   -> {verdict}")
        print()

    if detector is not None:
        print("=" * 58)
        print("Stratified check -- is the detector reading the leaking cue?\n")
        print("A cue that cannot be normalised away can still be neutralised by")
        print("comparing only clips that share it. If the detector holds up when")
        print("the cue no longer separates the classes, its signal is independent")
        print("of the cue and the set is usable along that axis.\n")
        for label, prediction in LABELS.items():
            if label not in frame or prediction not in frame:
                continue
            subset = frame.dropna(subset=[label])
            if subset[label].nunique() < 2 or label not in worst_overall:
                continue
            _, cue = worst_overall[label]
            bins = pd.qcut(subset[cue], args.strata, labels=False, duplicates="drop")
            print(f"{label}  (cue: {cue})")
            print(f"   {'stratum':<12}{'n':>5}{'cue EER':>10}{'detector':>11}")
            print(f"   {'all':<12}{len(subset):>5}"
                  f"{min(official_eer(subset[label], subset[cue]), 1 - official_eer(subset[label], subset[cue])):>10.4f}"
                  f"{official_eer(subset[label], subset[prediction]):>11.4f}")
            held = []
            for value in sorted(pd.unique(bins.dropna())):
                part = subset[bins == value]
                if len(part) < args.min_stratum or part[label].nunique() < 2:
                    continue
                cue_eer = official_eer(part[label], part[cue])
                cue_eer = min(cue_eer, 1 - cue_eer)
                det_eer = official_eer(part[label], part[prediction])
                held.append((cue_eer, det_eer))
                print(f"   {'q' + str(int(value)):<12}{len(part):>5}{cue_eer:>10.4f}{det_eer:>11.4f}")
            if held:
                overall = official_eer(subset[label], subset[prediction])
                mean_cue = float(np.mean([c for c, _ in held]))
                mean_det = float(np.mean([d for _, d in held]))
                # The cue has to actually weaken inside strata, or the split did
                # not neutralise anything and the comparison says nothing.
                if mean_cue < 0.40:
                    note = "inconclusive -- cue survives stratification"
                elif mean_det <= overall + args.fail_margin:
                    note = "USABLE -- detector holds with the cue neutralised"
                else:
                    note = "NOT USABLE -- detector was riding the cue"
                print(f"   -> cue {mean_cue:.4f}, detector {mean_det:.4f} "
                      f"(was {overall:.4f}): {note}")
            print()

    print("=" * 58)
    for label, (value, name) in worst_overall.items():
        print(f"  {label:<12} best trivial feature: {name} at EER {value:.4f}")
    print("\n  0.5 is what a feature that knows nothing should score.")
    print("  Distance below 0.5 is label that leaked into how the set was built.")


if __name__ == "__main__":
    main()
