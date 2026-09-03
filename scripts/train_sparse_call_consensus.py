#!/usr/bin/env python3
"""Fit the tiny channel-robust temporal consensus used for long calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from sparse_call_consensus import aligned_features, _logmeanexp  # noqa: E402


DEFAULT_CHANNELS = (
    ("clean", "reports/forensic_call_train_v1/xlsr_voice_window_embeddings.npz",
     "reports/forensic_call_train_v1/spectra_voice_window_profiles.npz"),
    ("g711_ulaw", "reports/forensic_call_train_v1_channels/g711_ulaw_xlsr.npz",
     "reports/forensic_call_train_v1_channels/g711_ulaw_spectra.npz"),
    ("g722_wb", "reports/forensic_call_train_v1_channels/g722_wb_xlsr.npz",
     "reports/forensic_call_train_v1_channels/g722_wb_spectra.npz"),
    ("opus_nb_8k", "reports/forensic_call_train_v1_channels/opus_nb_8k_xlsr.npz",
     "reports/forensic_call_train_v1_channels/opus_nb_8k_spectra.npz"),
    ("transcode_g711_opus",
     "reports/forensic_call_train_v1_channels/transcode_g711_opus_xlsr.npz",
     "reports/forensic_call_train_v1_channels/transcode_g711_opus_spectra.npz"),
)


def official_eer(labels, scores) -> float:
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    fnr = 1 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--truth", type=Path,
        default=ROOT / "data/eval/forensic_call_train_v1/truth.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--channel", nargs=3, action="append", metavar=("NAME", "XLSR", "SPECTRA"),
        help="Repeatable channel profile triple; defaults to the five call channels.",
    )
    parser.add_argument("--local-c", type=float, default=1e-4)
    parser.add_argument("--pool-temperature", type=float, default=2.0)
    parser.add_argument("--minimum-fake-seconds", type=float, default=0.25)
    args = parser.parse_args()
    assert_no_locked_eval_leakage(
        args.truth, ROOT / "configs/data_partitions.yaml"
    )
    truth_all = pd.read_csv(args.truth, dtype={"ID": str})
    definitions = args.channel or DEFAULT_CHANNELS
    channel_data = []
    for name, xlsr_s, spectra_s in definitions:
        xlsr_path = ROOT / xlsr_s
        spectra_path = ROOT / spectra_s
        xlsr = np.load(xlsr_path, allow_pickle=False)
        spectra = np.load(spectra_path, allow_pickle=False)
        ids = xlsr["ids"].astype(str)
        truth = truth_all.set_index("ID").loc[ids].reset_index()
        spectra_index = {
            item: index for index, item in enumerate(spectra["ids"].astype(str))
        }
        features, labels, parents = [], [], []
        for index, row in truth.iterrows():
            local = aligned_features(
                xlsr, spectra, index, spectra_index[row.ID]
            )
            begin, end = xlsr["offsets"][index:index + 2]
            starts = xlsr["starts"][begin:end] / 16_000
            ranges = json.loads(row.VOICE_FAKE_RANGES)
            for start in starts:
                overlap = sum(
                    max(0.0, min(start + 4.0, range_end) - max(start, range_start))
                    for range_start, range_end in ranges
                )
                labels.append(int(overlap >= args.minimum_fake_seconds))
            features.extend(local)
            parents.extend([index] * len(local))
        channel_data.append((
            str(name), truth, np.asarray(features), np.asarray(labels),
            np.asarray(parents),
        ))

    features = np.concatenate([item[2] for item in channel_data])
    labels = np.concatenate([item[3] for item in channel_data])
    train_windows = np.concatenate([
        item[1].SPLIT.eq("train").to_numpy()[item[4]] for item in channel_data
    ])
    scaler = StandardScaler().fit(features[train_windows])
    classifier = LogisticRegression(
        C=args.local_c, class_weight="balanced", max_iter=3_000,
        random_state=20260903,
    ).fit(scaler.transform(features[train_windows]), labels[train_windows])

    pooled_by_channel = {}
    for name, truth, local, _, parents in channel_data:
        margins = classifier.decision_function(scaler.transform(local))
        pooled_by_channel[name] = np.asarray([
            _logmeanexp(margins[parents == index], args.pool_temperature)
            for index in range(len(truth))
        ])
    train_pooled = np.concatenate([
        pooled_by_channel[name][truth.SPLIT.eq("train")]
        for name, truth, *_ in channel_data
    ])
    train_labels = np.concatenate([
        truth.loc[truth.SPLIT.eq("train"), "VOICE_FAKE"]
        for _, truth, *_ in channel_data
    ])
    calibrator = LogisticRegression(
        C=100, class_weight="balanced", max_iter=1_000,
        random_state=20260903,
    ).fit(train_pooled[:, None], train_labels)

    metrics = []
    for name, truth, *_ in channel_data:
        selected = truth.SPLIT.eq("dev").to_numpy()
        score = calibrator.predict_proba(
            pooled_by_channel[name][:, None]
        )[:, 1]
        sparse = (
            selected & truth.CONVERSATION_MODE.eq("sparse_second_speaker")
            & truth.VOICE_CASE.isin(("real_real", "real_fake"))
        )
        metrics.append({
            "CHANNEL": name,
            "VOICE_EER": official_eer(
                truth.loc[selected, "VOICE_FAKE"], score[selected]
            ),
            "SPARSE_FAKE_EER": official_eer(
                truth.loc[sparse, "VOICE_FAKE"], score[sparse]
            ),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        feature_mean=scaler.mean_.astype(np.float32),
        feature_scale=scaler.scale_.astype(np.float32),
        local_weight=classifier.coef_[0].astype(np.float32),
        local_bias=np.asarray(classifier.intercept_[0], dtype=np.float32),
        pool_temperature=np.asarray(args.pool_temperature, dtype=np.float32),
        bag_weight=np.asarray(calibrator.coef_[0, 0], dtype=np.float32),
        bag_bias=np.asarray(calibrator.intercept_[0], dtype=np.float32),
        voice_weight=np.asarray(0.60, dtype=np.float32),
    )
    table = pd.DataFrame(metrics)
    table.to_csv(args.output.with_suffix(".metrics.csv"), index=False)
    print(table.to_string(index=False))
    print(f"Saved sparse-call consensus head to {args.output}")


if __name__ == "__main__":
    main()
