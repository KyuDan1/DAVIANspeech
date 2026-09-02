#!/usr/bin/env python3
"""Learn a label-free SPEAR codec-bias correction from clean/phone pairs.

The target is the score displacement caused by a channel transform on the same
source, not the authenticity label.  Labels are used only to select a small
correction strength on source-disjoint development banks.  The locked phone
factorial bank is reported once after all choices have been frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit, softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_diagnostic import official_eer  # noqa: E402


LAYERS = 13
DIMENSION = 1280
CODECS = ("telephone8k", "g711_ulaw", "g726_24k", "opus_nb_8k")
TRAIN = (
    "external_mixed_train_v1",
    "mixed_devvoice_train_v1",
    "mixed_fmc_music_train_v1",
)
# These banks do not contribute sources to phone_factorial_1200_v1.
DEVELOPMENT = ("external_mixed_v1", "source_disjoint_mixed_v1")
LOCKED_AUDIT = "phone_factorial_1200_v1"


def read_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.is_dir():
        items = [np.load(item) for item in sorted(path.glob("*.npz"))]
    elif path.is_file():
        items = [np.load(path)]
    elif path.with_suffix(".npz").is_file():
        items = [np.load(path.with_suffix(".npz"))]
    else:
        raise FileNotFoundError(path)
    if not items:
        raise FileNotFoundError(path)
    ids = np.concatenate([item["ids"].astype(str) for item in items])
    vectors = np.concatenate([item["embeddings"] for item in items]).reshape(
        -1, LAYERS, DIMENSION
    )
    order = np.argsort(ids)
    return ids[order], vectors[order].astype(np.float32, copy=False)


def clean_path(dataset: str) -> Path:
    return ROOT / "output" / f"spear_{dataset}"


def transformed_path(dataset: str, codec: str) -> Path:
    if codec == "telephone8k":
        special = (
            "source_disjoint_music_telephone_v1"
            if dataset == "source_disjoint_music_v1"
            else f"{dataset}_telephone_v1"
        )
        return ROOT / "output" / f"spear_{special}.npz"
    return ROOT / "output" / f"spear_channel_{codec}_{dataset}.npz"


def canonical_ids(ids: np.ndarray, codec: str) -> np.ndarray:
    suffix = "__telephone8k" if codec == "telephone8k" else ""
    if not suffix:
        return ids
    return np.asarray([
        item[:-len(suffix)] if item.endswith(suffix) else item for item in ids
    ])


def aligned_pair(dataset: str, codec: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean_ids, clean = read_npz(clean_path(dataset))
    phone_ids, phone = read_npz(transformed_path(dataset, codec))
    phone_ids = canonical_ids(phone_ids, codec)
    clean_order = {item: index for index, item in enumerate(clean_ids)}
    if set(phone_ids) != set(clean_ids):
        missing = sorted(set(clean_ids) ^ set(phone_ids))[:5]
        raise ValueError(f"Unpaired embeddings for {dataset}/{codec}: {missing}")
    indices = np.asarray([clean_order[item] for item in phone_ids])
    return phone_ids, clean[indices], phone


class SpearScores:
    def __init__(self) -> None:
        music = np.load(ROOT / "model_heads/spear-mixed-music_fake-head.npz")
        joint = np.load(ROOT / "model_heads/spear-cross-component-joint-v1.npz")
        self.music_weight = music["weight"].reshape(LAYERS, DIMENSION)
        self.music_bias = float(music["bias"])
        self.joint_mean = joint["mean"].reshape(LAYERS, DIMENSION)
        self.joint_std = joint["std"].reshape(LAYERS, DIMENSION)
        self.joint_weight = joint["joint_weight"]
        self.joint_bias = joint["joint_bias"]

    def logits(self, vectors: np.ndarray) -> np.ndarray:
        music = np.einsum("nld,ld->n", vectors, self.music_weight)
        music += self.music_bias
        layer = 2
        normalized = (vectors[:, layer] - self.joint_mean[layer]) / self.joint_std[layer]
        classes = softmax(
            normalized @ self.joint_weight[layer] + self.joint_bias[layer], axis=1
        )
        file_probability = np.clip(1.0 - classes[:, 0], 1e-6, 1 - 1e-6)
        return np.column_stack([logit(file_probability), music])


def truth(dataset: str, ids: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(
        ROOT / "data/eval" / dataset / "truth.csv", dtype={"ID": str}
    ).set_index("ID")
    return frame.loc[ids].reset_index()


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, dtype=np.float64)
    scale = x.std(axis=0, dtype=np.float64).clip(1e-4)
    normalized = np.clip((x - mean) / scale, -8, 8)
    design = np.column_stack([normalized, np.ones(len(normalized))]).astype(np.float64)
    gram = design.T @ design
    penalty = np.eye(gram.shape[0]) * alpha
    penalty[-1, -1] = 0
    coefficients = np.linalg.solve(gram + penalty, design.T @ y)
    # Store normalization separately to make the deployment equation explicit.
    return coefficients.astype(np.float32), np.stack([mean, scale]).astype(np.float32)


def predict_ridge(x: np.ndarray, coef: np.ndarray, norm: np.ndarray) -> np.ndarray:
    normalized = np.clip((x - norm[0]) / norm[1], -8, 8)
    return normalized @ coef[:-1] + coef[-1]


def task_eers(frame: pd.DataFrame, logits_: np.ndarray) -> dict[str, float]:
    return {
        "FILE_EER": official_eer(frame.FILE_FAKE.astype(int), logits_[:, 0]),
        "MUSIC_EER": official_eer(frame.MUSIC_FAKE.astype(int), logits_[:, 1]),
    }


def final_phone_scores(corrected: np.ndarray, ids: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    anchor = pd.concat([
        pd.read_csv(item, dtype={"ID": str})
        for item in sorted((ROOT / "reports/phone_factorial_1200_v1").glob(
            "anchor_shard_*.csv"
        ))
    ]).set_index("ID").loc[ids]
    frame = truth(LOCKED_AUDIT, ids)
    expert = expit(corrected)
    file_score = 0.9 * anchor.FILE_FAKE_PROB.to_numpy() + 0.1 * expert[:, 0]
    music_score = 0.9 * anchor.MUSIC_FAKE_PROB.to_numpy() + 0.1 * expert[:, 1]
    voice_score = anchor.VOICE_FAKE_PROB.to_numpy()
    return frame, np.column_stack([file_score, voice_score, music_score])


def final_phone_metrics(corrected: np.ndarray, ids: np.ndarray) -> dict[str, float]:
    frame, scores = final_phone_scores(corrected, ids)
    file_score, voice_score, music_score = scores.T
    voice_mask = frame.VOICE_PRESENT.eq(1).to_numpy()
    music_mask = frame.MUSIC_PRESENT.eq(1).to_numpy()
    file_eer = official_eer(frame.FILE_FAKE, file_score)
    voice_eer = official_eer(frame.loc[voice_mask, "VOICE_FAKE"], voice_score[voice_mask])
    music_eer = official_eer(frame.loc[music_mask, "MUSIC_FAKE"], music_score[music_mask])
    return {
        "FILE_EER": file_eer,
        "VOICE_EER": voice_eer,
        "MUSIC_EER": music_eer,
        "ADS": .5 * (1 - file_eer) + .2 * (1 - voice_eer) + .3 * (1 - music_eer),
    }


def grouped_phone_metrics(frame: pd.DataFrame, scores: np.ndarray,
                          system: str) -> list[dict[str, object]]:
    rows = []
    for grouping in ("ALL", "CHANNEL", "AUDIO_TYPE"):
        groups = [("ALL", frame)] if grouping == "ALL" else frame.groupby(grouping)
        for value, group in groups:
            indices = group.index.to_numpy()
            subset_scores = scores[indices]
            voice = group.VOICE_PRESENT.eq(1).to_numpy()
            music = group.MUSIC_PRESENT.eq(1).to_numpy()
            file_eer = official_eer(group.FILE_FAKE, subset_scores[:, 0])
            voice_eer = (
                official_eer(group.loc[voice, "VOICE_FAKE"], subset_scores[voice, 1])
                if group.loc[voice, "VOICE_FAKE"].nunique() > 1 else np.nan
            )
            music_eer = (
                official_eer(group.loc[music, "MUSIC_FAKE"], subset_scores[music, 2])
                if group.loc[music, "MUSIC_FAKE"].nunique() > 1 else np.nan
            )
            ads = (
                .5 * (1 - file_eer) + .2 * (1 - voice_eer) + .3 * (1 - music_eer)
                if np.isfinite(voice_eer) and np.isfinite(music_eer) else np.nan
            )
            rows.append({
                "SYSTEM": system, "GROUP": grouping, "VALUE": str(value),
                "N": len(group), "FILE_EER": file_eer,
                "VOICE_EER": voice_eer, "MUSIC_EER": music_eer, "ADS": ads,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "reports/phone_codec_debias_v1")
    parser.add_argument("--projection-dim", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scorer = SpearScores()
    train_blocks: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        codec: [] for codec in CODECS
    }
    for dataset in TRAIN:
        for codec in CODECS:
            _, clean, phone = aligned_pair(dataset, codec)
            displacement = scorer.logits(clean) - scorer.logits(phone)
            train_blocks[codec].append((phone, displacement))

    rng = np.random.default_rng(args.seed)
    projections = rng.normal(
        0, 1 / np.sqrt(args.projection_dim),
        size=(LAYERS, DIMENSION, args.projection_dim),
    ).astype(np.float32)

    dev = {}
    for dataset in DEVELOPMENT:
        for codec in CODECS:
            ids, clean, phone = aligned_pair(dataset, codec)
            dev[(dataset, codec)] = (ids, clean, phone, truth(dataset, ids))

    rows = []
    alphas = (1.0, 10.0, 100.0, 1_000.0, 10_000.0)
    betas = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    fitted = {}
    for layer in range(LAYERS):
        x = np.concatenate([
            phone[:, layer] @ projections[layer]
            for codec in CODECS for phone, _ in train_blocks[codec]
        ])
        y = np.concatenate([
            delta for codec in CODECS for _, delta in train_blocks[codec]
        ])
        for alpha in alphas:
            coef, norm = fit_ridge(x, y, alpha)
            fitted[(layer, alpha)] = (coef, norm)
            for beta in betas:
                bank_metrics = []
                for dataset in DEVELOPMENT:
                    all_truth, all_logits = [], []
                    codec_metrics = []
                    for codec in CODECS:
                        _, _, phone, frame = dev[(dataset, codec)]
                        raw = scorer.logits(phone)
                        delta = predict_ridge(
                            phone[:, layer] @ projections[layer], coef, norm
                        )
                        corrected = raw + beta * delta
                        metric = task_eers(frame, corrected)
                        codec_metrics.append(metric)
                        all_truth.append(frame)
                        all_logits.append(corrected)
                    combined = task_eers(
                        pd.concat(all_truth, ignore_index=True),
                        np.concatenate(all_logits),
                    )
                    bank_metrics.append(combined)
                    rows.append({
                        "LAYER": layer, "ALPHA": alpha, "BETA": beta,
                        "DATASET": dataset, "SCOPE": "all_codecs",
                        **combined,
                    })
                    rows.append({
                        "LAYER": layer, "ALPHA": alpha, "BETA": beta,
                        "DATASET": dataset, "SCOPE": "worst_codec",
                        "FILE_EER": max(item["FILE_EER"] for item in codec_metrics),
                        "MUSIC_EER": max(item["MUSIC_EER"] for item in codec_metrics),
                    })

    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "development_sweep.csv", index=False)
    combined = table[table.SCOPE.eq("all_codecs")].groupby(
        ["LAYER", "ALPHA", "BETA"], as_index=False
    ).agg(FILE_MEAN=("FILE_EER", "mean"), FILE_WORST=("FILE_EER", "max"),
          MUSIC_MEAN=("MUSIC_EER", "mean"), MUSIC_WORST=("MUSIC_EER", "max"))
    combined["SELECTION"] = (
        .5 * combined.FILE_MEAN + .5 * combined.FILE_WORST
        + .3 * combined.MUSIC_MEAN + .3 * combined.MUSIC_WORST
    )
    selected = combined.sort_values(
        ["SELECTION", "FILE_WORST", "MUSIC_WORST", "BETA"]
    ).iloc[0]
    layer, alpha, beta = (
        int(selected.LAYER), float(selected.ALPHA), float(selected.BETA)
    )
    coef, norm = fitted[(layer, alpha)]

    # Locked evaluation: no tuning below this point.
    audit_ids, audit_vectors = read_npz(ROOT / "output/spear_phone_factorial_1200_v1")
    audit_raw = scorer.logits(audit_vectors)
    audit_delta = predict_ridge(
        audit_vectors[:, layer] @ projections[layer], coef, norm
    )
    audit_corrected = audit_raw + beta * audit_delta
    raw_metrics = final_phone_metrics(audit_raw, audit_ids)
    corrected_metrics = final_phone_metrics(audit_corrected, audit_ids)
    audit_truth, raw_scores = final_phone_scores(audit_raw, audit_ids)
    _, corrected_scores = final_phone_scores(audit_corrected, audit_ids)
    grouped = grouped_phone_metrics(audit_truth, raw_scores, "raw")
    grouped += grouped_phone_metrics(audit_truth, corrected_scores, "corrected")
    pd.DataFrame(grouped).to_csv(
        args.output_dir / "locked_group_metrics.csv", index=False
    )
    audit_output = audit_truth.copy()
    for column, values in zip(
        ("RAW_FILE", "RAW_VOICE", "RAW_MUSIC"), raw_scores.T
    ):
        audit_output[column] = values
    for column, values in zip(
        ("CORRECTED_FILE", "CORRECTED_VOICE", "CORRECTED_MUSIC"),
        corrected_scores.T,
    ):
        audit_output[column] = values
    audit_output.to_csv(args.output_dir / "locked_predictions.csv", index=False)

    np.savez_compressed(
        args.output_dir / "spear_phone_codec_debias.npz",
        projection=projections[layer], coefficient=coef, normalization=norm,
        layer=np.asarray(layer), alpha=np.asarray(alpha), beta=np.asarray(beta),
    )
    summary = {
        "protocol": {
            "train": list(TRAIN), "development": list(DEVELOPMENT),
            "locked_audit": LOCKED_AUDIT,
            "target": "clean logit minus paired telephone-channel logit",
        },
        "selected": {key: float(selected[key]) for key in selected.index},
        "locked_phone_raw": raw_metrics,
        "locked_phone_corrected": corrected_metrics,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
