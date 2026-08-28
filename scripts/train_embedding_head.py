"""Fit a deployable linear detector head on cached audio embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, nargs="+", required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--label", default="MUSIC_FAKE")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--c", type=float, required=True)
    args = parser.parse_args()

    vectors: dict[str, np.ndarray] = {}
    for path in args.embeddings:
        shard = np.load(path)
        for sample_id, embedding in zip(shard["ids"].astype(str), shard["embeddings"]):
            if sample_id in vectors:
                raise ValueError(f"Duplicate embedding ID: {sample_id}")
            vectors[sample_id] = embedding

    truth = pd.read_csv(args.truth, dtype={"ID": str})
    truth = truth[truth[args.label].notna()].copy()
    missing = [sample_id for sample_id in truth.ID if sample_id not in vectors]
    if missing:
        raise ValueError(f"Missing {len(missing)} embeddings: {missing[:5]}")
    features = np.stack([vectors[sample_id] for sample_id in truth.ID])
    labels = truth[args.label].astype(int).to_numpy()

    model = LogisticRegression(
        C=args.c, class_weight="balanced", max_iter=3_000, random_state=20260828
    ).fit(features, labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        weight=model.coef_[0].astype(np.float32),
        bias=np.asarray(model.intercept_[0], dtype=np.float32),
    )
    print(
        f"Saved {features.shape[1]}-D head trained on {len(labels)} rows "
        f"({int(labels.sum())} fake) to {args.output}"
    )


if __name__ == "__main__":
    main()
