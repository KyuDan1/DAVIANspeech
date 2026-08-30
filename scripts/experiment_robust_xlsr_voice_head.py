"""Fit a domain-balanced mixed-voice head and blend it with the NII head.

Only declared training mixtures are fitted. Hyperparameters are selected by
maximin ADS across six development mixture conditions; locked eval is unused.
"""

from __future__ import annotations

import argparse
import glob
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

TRAIN = ["external_mixed_train_v1", "mixed_devvoice_train_v1", "mixed_fmc_music_train_v1"]
DEVELOPMENT = [
    ("external_mixed_v1", "external_mixed_v1_v10_diagnostics.csv"),
    ("source_disjoint_mixed_v1", "source_disjoint_mixed_v1_v10_diagnostics.csv"),
    ("source_disjoint_mixed_equal_v1", "source_disjoint_mixed_equal_v1_v11_diagnostics.csv"),
]


def load_vectors(prefix, name):
    vectors = {}
    paths = glob.glob(str(ROOT / "output" / f"{prefix}_{name}" / "*.npz"))
    if not paths:
        paths = [str(ROOT / "output" / f"{prefix}_{name}.npz")]
    for path in paths:
        shard = np.load(path)
        vectors.update({i: v for i, v in zip(shard["ids"].astype(str), shard["embeddings"])})
    truth = pd.read_csv(ROOT / "data/eval" / name / "truth.csv", dtype={"ID": str})
    return truth, np.stack([vectors[i] for i in truth.ID])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-head", type=Path, required=True)
    parser.add_argument("--output-grid", type=Path, required=True)
    parser.add_argument("--partition-config", type=Path, default=ROOT / "configs/data_partitions.yaml")
    args = parser.parse_args()
    blocks, domains = [], []
    for name in TRAIN:
        truth_path = ROOT / "data/eval" / name / "truth.csv"
        assert_no_locked_eval_leakage(truth_path, args.partition_config)
        truth, x = load_vectors("xlsr_emb", name)
        blocks.append((truth, x)); domains.extend([name] * len(truth))
    x = np.concatenate([item[1] for item in blocks])
    y = np.concatenate([item[0].VOICE_FAKE.astype(int) for item in blocks])
    domains = np.asarray(domains)
    sample_weight = np.zeros(len(y))
    for domain in np.unique(domains):
        for label in (0, 1):
            mask = (domains == domain) & (y == label)
            sample_weight[mask] = 1 / mask.sum()
    sample_weight *= len(y) / sample_weight.sum()

    evaluation = []
    music_head = np.load(ROOT / "model_heads/fourier-domain-balanced-v13.npz")
    for name, diagnostic in DEVELOPMENT:
        truth, voice_x = load_vectors("xlsr_emb", name)
        _, music_x = load_vectors("fourier", name)
        music = expit(music_x @ music_head["weight"] + float(music_head["bias"]))
        diagnostics = pd.read_csv(ROOT / "output" / diagnostic).set_index("ID").loc[truth.ID]
        evaluation.append((truth, voice_x, diagnostics, music))

    records, candidates = [], []
    for c in [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
        model = LogisticRegression(C=c, max_iter=5_000, random_state=20260830)
        model.fit(x, y, sample_weight=sample_weight)
        for blend in np.arange(0, 1.01, .1):
            scores, voice_eers = [], []
            for truth, voice_x, diagnostics, music in evaluation:
                adapted = model.predict_proba(voice_x)[:, 1]
                released = diagnostics.raw_fake_xlsr.to_numpy()
                voice = expit(
                    (1 - blend) * logit(np.clip(released, 1e-7, 1 - 1e-7))
                    + blend * logit(np.clip(adapted, 1e-7, 1 - 1e-7))
                )
                frame = truth.copy(); frame["voice"] = voice; frame["music"] = music
                for condition in ("sequential", "simultaneous"):
                    group = frame[frame.CONDITION == condition]
                    fe = official_eer(group.FILE_FAKE, np.maximum(group.voice, group.music))
                    ve = official_eer(group.VOICE_FAKE, group.voice)
                    me = official_eer(group.MUSIC_FAKE, group.music)
                    scores.append(.5 * (1 - fe) + .2 * (1 - ve) + .3 * (1 - me))
                    voice_eers.append(ve)
            record = {"C": c, "BLEND_NEW": blend, "DEV_MIN": min(scores),
                      "DEV_MEAN": np.mean(scores), "MAX_VOICE_EER": max(voice_eers)}
            records.append(record); candidates.append((record, model))
    best, model = max(candidates, key=lambda item: (item[0]["DEV_MIN"], item[0]["DEV_MEAN"], -item[0]["MAX_VOICE_EER"]))
    args.output_head.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_head, weight=model.coef_[0].astype(np.float32),
                        bias=np.asarray(model.intercept_[0], dtype=np.float32),
                        blend_new=np.asarray(best["BLEND_NEW"]), train_c=np.asarray(best["C"]))
    pd.DataFrame(records).to_csv(args.output_grid, index=False)
    print(best)


if __name__ == "__main__":
    main()
