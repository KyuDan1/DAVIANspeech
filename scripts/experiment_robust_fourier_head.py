"""Train and select a domain-balanced Fourier head without touching locked eval.

Selection uses only development domains.  The source-disjoint prospective split
and Suno vocals are reported after the hyperparameters are frozen.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from evaluate_diagnostic import official_eer  # noqa: E402

TRAIN_SETS = [
    "external_mixed_train_v1",
    "mixed_devvoice_train_v1",
    "mixed_fmc_music_train_v1",
]
MIXED_DEVELOPMENT = [
    ("external_mixed_v1", "output/external_mixed_v1_v10_diagnostics.csv"),
    ("source_disjoint_mixed_v1", "output/source_disjoint_mixed_v1_v10_diagnostics.csv"),
    ("source_disjoint_mixed_equal_v1", "output/source_disjoint_mixed_equal_v1_v11_diagnostics.csv"),
]


def load_embeddings(directory: Path) -> dict[str, np.ndarray]:
    vectors = {}
    paths = sorted(directory.glob("*.npz"))
    # A few earlier experiments stored all rows in one adjacent NPZ rather
    # than a shard directory.
    if not paths and directory.with_suffix(".npz").is_file():
        paths = [directory.with_suffix(".npz")]
    for path in paths:
        shard = np.load(path)
        for sample_id, vector in zip(shard["ids"].astype(str), shard["embeddings"]):
            if sample_id in vectors:
                raise ValueError(f"Duplicate embedding: {sample_id}")
            vectors[sample_id] = vector
    return vectors


def dataset(name: str):
    truth = pd.read_csv(ROOT / "data/eval" / name / "truth.csv", dtype={"ID": str})
    vectors = load_embeddings(ROOT / "output" / f"fourier_{name}")
    missing = sorted(set(truth.ID) - set(vectors))
    if missing:
        raise ValueError(f"{name}: missing {len(missing)} embeddings")
    x = np.stack([vectors[sample_id] for sample_id in truth.ID])
    return truth, x


def domain_weights(labels: np.ndarray, domains: np.ndarray):
    """Give every (domain, class) cell the same total training weight."""
    weights = np.zeros(len(labels), dtype=np.float64)
    cells = {(domain, label) for domain, label in zip(domains, labels)}
    for domain, label in cells:
        mask = (domains == domain) & (labels == label)
        weights[mask] = 1.0 / mask.sum()
    return weights * len(weights) / weights.sum()


def mixed_ads(truth, diagnostics, music_score, condition=None):
    frame = truth.merge(diagnostics, on="ID", validate="one_to_one")
    frame["music_score"] = music_score
    if condition is not None:
        frame = frame[frame.CONDITION == condition]
    voice = frame.raw_fake_xlsr.to_numpy()
    music = frame.music_score.to_numpy()
    file_score = np.maximum(voice, music)
    file_eer = official_eer(frame.FILE_FAKE, file_score)
    voice_eer = official_eer(frame.VOICE_FAKE, voice)
    music_eer = official_eer(frame.MUSIC_FAKE, music)
    ads = .5 * (1 - file_eer) + .2 * (1 - voice_eer) + .3 * (1 - music_eer)
    return ads, file_eer, voice_eer, music_eer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-head", type=Path, required=True)
    parser.add_argument("--output-results", type=Path, required=True)
    parser.add_argument("--partition-config", type=Path, default=ROOT / "configs/data_partitions.yaml")
    args = parser.parse_args()

    train_x, train_y, train_domains = [], [], []
    for name in TRAIN_SETS:
        truth_path = ROOT / "data/eval" / name / "truth.csv"
        assert_no_locked_eval_leakage(truth_path, args.partition_config)
        truth, x = dataset(name)
        train_x.append(x)
        train_y.append(truth.MUSIC_FAKE.astype(int).to_numpy())
        train_domains.extend([name] * len(truth))
    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    train_domains = np.asarray(train_domains)
    weights = domain_weights(train_y, train_domains)

    development = []
    for name, diagnostic_path in MIXED_DEVELOPMENT:
        truth, x = dataset(name)
        diagnostic = pd.read_csv(ROOT / diagnostic_path, dtype={"ID": str})
        development.append((name, truth, x, diagnostic))
    music_truth, music_x = dataset("source_disjoint_music_v1")

    original = np.load(ROOT / "model_heads/fourier-echoes-music-head.npz")
    original_weight, original_bias = original["weight"], float(original["bias"])

    records, candidates = [], []
    for c in [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
        model = LogisticRegression(C=c, max_iter=5_000, random_state=20260830)
        model.fit(train_x, train_y, sample_weight=weights)
        for blend in [0.0, .25, .5, .75, 1.0]:
            domain_scores = []
            for name, truth, x, diagnostic in development:
                new_logit = x @ model.coef_[0] + model.intercept_[0]
                old_logit = x @ original_weight + original_bias
                score = expit(blend * new_logit + (1 - blend) * old_logit)
                for condition in ["sequential", "simultaneous"]:
                    metrics = mixed_ads(truth, diagnostic, score, condition)
                    domain_scores.append(metrics[0])
            mask = music_truth.SPLIT.eq("alignment").to_numpy()
            new_logit = music_x @ model.coef_[0] + model.intercept_[0]
            old_logit = music_x @ original_weight + original_bias
            score = expit(blend * new_logit + (1 - blend) * old_logit)
            music_eer = official_eer(music_truth.loc[mask, "MUSIC_FAKE"], score[mask])
            domain_scores.append(1 - music_eer)
            record = {
                "C": c, "BLEND_NEW": blend,
                "DEV_MIN": min(domain_scores), "DEV_MEAN": np.mean(domain_scores),
                "ALIGNMENT_MUSIC_EER": music_eer,
            }
            records.append(record)
            candidates.append((record, model))

    # Maximin first, mean only breaks ties. Prospective and Suno are not used.
    best_record, best_model = max(
        candidates, key=lambda item: (item[0]["DEV_MIN"], item[0]["DEV_MEAN"])
    )
    blend = best_record["BLEND_NEW"]
    final_weight = blend * best_model.coef_[0] + (1 - blend) * original_weight
    final_bias = blend * best_model.intercept_[0] + (1 - blend) * original_bias
    args.output_head.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_head,
        weight=final_weight.astype(np.float32),
        bias=np.asarray(final_bias, dtype=np.float32),
        train_c=np.asarray(best_record["C"]),
        blend_new=np.asarray(blend),
    )

    summary = [{"SECTION": "SELECTED", **best_record}]
    for name, truth, x, diagnostic in development:
        score = expit(x @ final_weight + final_bias)
        for condition in ["sequential", "simultaneous"]:
            ads, fe, ve, me = mixed_ads(truth, diagnostic, score, condition)
            summary.append({
                "SECTION": "DEVELOPMENT", "DATASET": name,
                "CONDITION": condition, "ADS": ads,
                "FILE_EER": fe, "VOICE_EER": ve, "MUSIC_EER": me,
            })
    music_score = expit(music_x @ final_weight + final_bias)
    for split in ["alignment", "prospective"]:
        mask = music_truth.SPLIT.eq(split).to_numpy()
        summary.append({
            "SECTION": "DEVELOPMENT" if split == "alignment" else "FINAL_HOLDOUT",
            "DATASET": "source_disjoint_music_v1", "CONDITION": split,
            "MUSIC_EER": official_eer(music_truth.loc[mask, "MUSIC_FAKE"], music_score[mask]),
        })

    suno_truth, suno_x = dataset("suno_vocals_v1")
    suno_score = expit(suno_x @ final_weight + final_bias)
    summary.append({
        "SECTION": "FINAL_HOLDOUT", "DATASET": "suno_vocals_v1",
        "CONDITION": "all_fake", "N": len(suno_score),
        "RECALL_AT_0.5": np.mean(suno_score >= .5),
        "MIN_SCORE": np.min(suno_score), "MEDIAN_SCORE": np.median(suno_score),
    })
    table = pd.DataFrame(summary)
    args.output_results.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_results, index=False)
    pd.DataFrame(records).to_csv(args.output_results.with_name(args.output_results.stem + "_grid.csv"), index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
