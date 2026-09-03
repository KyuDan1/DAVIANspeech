"""Train a codec-robust SPEAR music head with leave-one-codec-out selection.

Only declared training mixtures are fitted.  Hyperparameters are chosen by
holding out each telephone codec in turn; source-disjoint music is reported
only after the model and fusion weight have been frozen.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from evaluate_diagnostic import official_eer  # noqa: E402

LAYERS = 13
DIMENSION = 1280
CODECS = ("telephone8k", "g711_ulaw", "g726_24k", "opus_nb_8k")
TRAIN = (
    "external_mixed_train_v1",
    "mixed_devvoice_train_v1",
    "mixed_fmc_music_train_v1",
)
DEVELOPMENT = ("external_mixed_v1", "source_disjoint_mixed_equal_v1")
FINAL_AUDIT = ("source_disjoint_mixed_v1", "source_disjoint_music_v1")


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.is_file():
        shards = [np.load(path)]
    elif path.with_suffix(".npz").is_file():
        shards = [np.load(path.with_suffix(".npz"))]
    else:
        shards = [np.load(item) for item in sorted(path.glob("*.npz"))]
    if not shards:
        raise FileNotFoundError(path)
    ids = np.concatenate([item["ids"].astype(str) for item in shards])
    vectors = np.concatenate([item["embeddings"] for item in shards])
    return ids, vectors.reshape(-1, LAYERS, DIMENSION)


def load_block(name: str, codec: str) -> tuple[pd.DataFrame, np.ndarray]:
    if codec == "telephone8k":
        dataset = (
            "source_disjoint_music_telephone_v1"
            if name == "source_disjoint_music_v1"
            else f"{name}_telephone_v1"
        )
        path = ROOT / "output" / f"spear_{dataset}.npz"
    else:
        dataset = name
        path = ROOT / "output" / f"spear_channel_{codec}_{name}.npz"
    ids, vectors = load_npz(path)
    truth = pd.read_csv(
        ROOT / "data" / "eval" / dataset / "truth.csv", dtype={"ID": str}
    ).set_index("ID").loc[ids]
    return truth, vectors


def balanced_weights(labels: np.ndarray, cells: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(labels), dtype=np.float64)
    for cell in np.unique(cells):
        for label in (0, 1):
            selected = (cells == cell) & (labels == label)
            weights[selected] = 1.0 / selected.sum()
    return weights * len(weights) / weights.sum()


def fit_head(blocks, layer: int, c_value: float):
    features = np.concatenate([item[1][:, layer] for item in blocks])
    labels = np.concatenate([item[0].MUSIC_FAKE.astype(int) for item in blocks])
    cells = np.concatenate([
        np.repeat(index, len(item[0])) for index, item in enumerate(blocks)
    ])
    mean = features.mean(axis=0)
    scale = features.std(axis=0) + 1e-5
    normalized = (features - mean) / scale
    model = LogisticRegression(
        C=c_value, max_iter=3_000, random_state=20260901
    ).fit(normalized, labels, sample_weight=balanced_weights(labels, cells))
    weight = model.coef_[0] / scale
    bias = float(model.intercept_[0] - np.dot(model.coef_[0], mean / scale))
    return weight.astype(np.float32), np.float32(bias)


def existing_score(vectors: np.ndarray, head) -> np.ndarray:
    return expit(vectors.reshape(len(vectors), -1) @ head["weight"] + head["bias"])


def new_score(vectors: np.ndarray, layer: int, weight, bias) -> np.ndarray:
    return expit(vectors[:, layer] @ weight + bias)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-head", type=Path, required=True)
    parser.add_argument("--output-results", type=Path, required=True)
    parser.add_argument(
        "--partition-config", type=Path,
        default=ROOT / "configs" / "data_partitions.yaml",
    )
    args = parser.parse_args()
    for name in TRAIN:
        assert_no_locked_eval_leakage(
            ROOT / "data" / "eval" / name / "truth.csv",
            args.partition_config,
        )

    train = {(name, codec): load_block(name, codec)
             for name in TRAIN for codec in CODECS}
    development = {(name, codec): load_block(name, codec)
                   for name in DEVELOPMENT for codec in CODECS}
    existing = np.load(ROOT / "model_heads" / "spear-mixed-music_fake-head.npz")

    search = []
    c_values = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    for layer in range(5):
        for c_value in c_values:
            heldout_eers = []
            for heldout in CODECS:
                blocks = [block for (name, codec), block in train.items()
                          if codec != heldout]
                weight, bias = fit_head(blocks, layer, c_value)
                for name in DEVELOPMENT:
                    truth, vectors = development[(name, heldout)]
                    heldout_eers.append(official_eer(
                        truth.MUSIC_FAKE.astype(int),
                        new_score(vectors, layer, weight, bias),
                    ))
            search.append({
                "LAYER": layer, "C": c_value,
                "LOCO_WORST_EER": max(heldout_eers),
                "LOCO_MEAN_EER": float(np.mean(heldout_eers)),
            })
    selection = min(
        search, key=lambda row: (row["LOCO_WORST_EER"], row["LOCO_MEAN_EER"])
    )
    layer, c_value = int(selection["LAYER"]), float(selection["C"])
    weight, bias = fit_head(list(train.values()), layer, c_value)

    # Freeze the fusion on development only. Probability fusion matches the
    # already leaderboard-positive SPEAR post-pass used by v15.
    fusion_rows = []
    for blend in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        eers = []
        for truth, vectors in development.values():
            old = existing_score(vectors, existing)
            new = new_score(vectors, layer, weight, bias)
            eers.append(official_eer(
                truth.MUSIC_FAKE.astype(int), (1 - blend) * old + blend * new
            ))
        fusion_rows.append({
            "BLEND_NEW": blend, "DEV_WORST_EER": max(eers),
            "DEV_MEAN_EER": float(np.mean(eers)),
        })
    fusion = min(
        fusion_rows, key=lambda row: (row["DEV_WORST_EER"], row["DEV_MEAN_EER"])
    )
    blend = float(fusion["BLEND_NEW"])

    records = [{"SECTION": "SELECTION", **selection, **fusion}]
    for section, names in (("DEVELOPMENT", DEVELOPMENT), ("FINAL_AUDIT", FINAL_AUDIT)):
        for name in names:
            for codec in CODECS:
                truth, vectors = load_block(name, codec)
                old = existing_score(vectors, existing)
                new = new_score(vectors, layer, weight, bias)
                fused = (1 - blend) * old + blend * new
                records.append({
                    "SECTION": section, "DATASET": name, "CODEC": codec,
                    "N": len(truth),
                    "EXISTING_EER": official_eer(truth.MUSIC_FAKE.astype(int), old),
                    "NEW_EER": official_eer(truth.MUSIC_FAKE.astype(int), new),
                    "FUSED_EER": official_eer(truth.MUSIC_FAKE.astype(int), fused),
                })

    args.output_head.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_head, weight=weight, bias=np.asarray(bias),
        layer=np.asarray(layer), train_c=np.asarray(c_value),
        blend_new=np.asarray(blend), codecs=np.asarray(CODECS),
    )
    args.output_results.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output_results, index=False)
    pd.DataFrame(search).to_csv(
        args.output_results.with_name(args.output_results.stem + "_search.csv"),
        index=False,
    )
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()
