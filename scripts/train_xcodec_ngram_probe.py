#!/usr/bin/env python3
"""Train a compact X-Codec token-distribution/transition music probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from evaluate_diagnostic import official_eer  # noqa: E402


VOCAB = 1024
BIGRAM_BINS = 4096
CROSS_BINS = 4096
CODEBOOKS = 4
UNIGRAM_DIM = CODEBOOKS * VOCAB
BIGRAM_DIM = CODEBOOKS * BIGRAM_BINS
CROSS_DIM = 2 * CROSS_BINS
FULL_DIM = UNIGRAM_DIM + BIGRAM_DIM + CROSS_DIM


def load_tokens(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    datasets, ids, tokens = [], [], []
    for path in sorted(directory.glob("shard_*.npz")):
        archive = np.load(path)
        datasets.append(archive["datasets"])
        ids.append(archive["ids"])
        tokens.append(archive["tokens"])
    if not tokens:
        raise FileNotFoundError(f"No token shards in {directory}")
    return np.concatenate(datasets), np.concatenate(ids), np.concatenate(tokens)


def _append_hist(rows, cols, values, row, offset, items, bins, scale=1.0):
    counts = np.bincount(items, minlength=bins).astype(np.float32)
    nonzero = np.flatnonzero(counts)
    rows.extend([row] * len(nonzero))
    cols.extend((offset + nonzero).tolist())
    values.extend((scale * counts[nonzero] / max(1, len(items))).tolist())


def token_features(tokens: np.ndarray, mode: str) -> sparse.csr_matrix:
    """Sparse normalized histograms; transitions keep codec-token structure."""
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for row, sample in enumerate(tokens.astype(np.int64)):
        for codebook in range(CODEBOOKS):
            sequence = sample[codebook]
            _append_hist(
                rows, cols, values, row, codebook * VOCAB,
                sequence, VOCAB,
            )
            if mode == "transition":
                # A fixed multiplicative hash preserves exact adjacent token pairs
                # without a 4M-dimensional table per codebook.
                pair = ((sequence[:-1] * 1009) ^ sequence[1:]) % BIGRAM_BINS
                _append_hist(
                    rows, cols, values, row,
                    UNIGRAM_DIM + codebook * BIGRAM_BINS,
                    pair, BIGRAM_BINS,
                )
        if mode == "transition":
            for pair_index, (left, right) in enumerate(((0, 1), (2, 3))):
                cross = ((sample[left] * 1009) ^ sample[right]) % CROSS_BINS
                _append_hist(
                    rows, cols, values, row,
                    UNIGRAM_DIM + BIGRAM_DIM + pair_index * CROSS_BINS,
                    cross, CROSS_BINS,
                )
    dimension = UNIGRAM_DIM if mode == "unigram" else FULL_DIM
    return sparse.csr_matrix(
        (np.asarray(values, np.float32), (rows, cols)),
        shape=(len(tokens), dimension), dtype=np.float32,
    )


def truth_for(dataset: str) -> pd.DataFrame:
    return pd.read_csv(
        ROOT / "data/eval" / dataset / "truth.csv", dtype={"ID": str}
    ).set_index("ID")


def evaluate(
    datasets: np.ndarray, ids: np.ndarray, probabilities: np.ndarray,
    names: list[str], output: Path,
) -> pd.DataFrame:
    rows, summary = [], []
    for name in names:
        selected = np.flatnonzero(datasets == name)
        truth = truth_for(name).loc[ids[selected]]
        valid = truth.MUSIC_PRESENT.eq(1) & truth.MUSIC_FAKE.notna()
        score = probabilities[selected]
        eer = official_eer(
            truth.loc[valid, "MUSIC_FAKE"].astype(int), score[valid.to_numpy()]
        )
        summary.append({"DATASET": name, "ROWS": int(valid.sum()), "MUSIC_EER": eer})
        generator = truth.get("MUSIC_GENERATOR", truth.get("GENERATOR", "unknown"))
        if not isinstance(generator, pd.Series):
            generator = pd.Series(generator, index=truth.index)
        for item, label, present, gen, probability in zip(
            truth.index, truth.MUSIC_FAKE, truth.MUSIC_PRESENT, generator, score
        ):
            rows.append({
                "DATASET": name, "ID": item, "MUSIC_FAKE": label,
                "MUSIC_PRESENT": present, "GENERATOR": gen,
                "XCODEC_FAKE_PROB": probability,
            })
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "predictions.csv", index=False)
    result = pd.DataFrame(summary)
    result.to_csv(output / "summary.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-dataset", default="temporal_mixed_train_v2")
    parser.add_argument("--mode", choices=["unigram", "transition"], default="transition")
    parser.add_argument("--c", type=float, default=10.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()

    assert_no_locked_eval_leakage(
        ROOT / "data/eval" / args.train_dataset / "truth.csv",
        ROOT / "configs/data_partitions.yaml",
    )
    datasets, ids, tokens = load_tokens(args.tokens)
    features = token_features(tokens, args.mode)
    truth = truth_for(args.train_dataset)
    train_mask = datasets == args.train_dataset
    train_ids = ids[train_mask]
    split = truth.loc[train_ids, "SPLIT"].astype(str).to_numpy()
    labels = truth.loc[train_ids, "MUSIC_FAKE"].astype(int).to_numpy()
    train_rows = np.flatnonzero(train_mask)[split == "train"]

    model = LogisticRegression(
        C=args.c, max_iter=args.max_iter, solver="liblinear",
        class_weight="balanced", random_state=20260903,
    )
    model.fit(features[train_rows], labels[split == "train"])
    probabilities = model.predict_proba(features)[:, 1]
    names = sorted(set(datasets.tolist()))
    result = evaluate(datasets, ids, probabilities, names, args.output)
    joblib.dump(model, args.output / "xcodec_ngram.joblib", compress=3)
    metadata = {
        "mode": args.mode, "C": args.c, "train_dataset": args.train_dataset,
        "train_rows": int(len(train_rows)), "feature_dim": int(features.shape[1]),
    }
    (args.output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
