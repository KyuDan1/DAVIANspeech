#!/usr/bin/env python3
"""Train/select a dual-scale MERT Music head using authorized dev and LOGO only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from evaluate_diagnostic import official_eer  # noqa: E402
from mert_generator_robust_music import (  # noqa: E402
    MertGeneratorRobustMusicHead, asymmetric_codec_kl,
    dual_scale_features, make_projection,
)
from train_dual_domain_head import truth_path  # noqa: E402


TRAIN_DEFAULT = (
    "external_mixed_train_v1", "mixed_devvoice_train_v1",
    "mixed_fmc_music_train_v1", "mixfake_music_train_v1",
    "telephone_mixed_train_v1", "temporal_mixed_train_v2",
    "channel_invariant_factorial_train_v1",
)
DEV_DEFAULT = (
    "mixfake_music_dev_v1", "external_mixed_v1",
    "source_disjoint_mixed_v1", "source_disjoint_mixed_equal_v1",
    "source_disjoint_music_v1", "factorial_eval_1200_v2_dev",
    "telephone_mixed_dev_v1",
)
LOGO_DEFAULT = ("suno", "udio", "musicgen", "audioldm", "musicldm", "mubert")
TRAIN_TRUTH_OVERRIDES = {
    "temporal_mixed_train_v2":
        ROOT / "data/eval/temporal_mixed_train_v2/truth_train.csv",
}


def load_statistics(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    datasets, ids, values = [], [], []
    for path in sorted(root.glob("shard_*.npz")):
        with np.load(path, allow_pickle=False) as shard:
            if str(shard["representation"]) != "temporal_statistics":
                raise ValueError(f"not temporal MERT statistics: {path}")
            datasets.append(shard["datasets"].astype(str))
            ids.append(shard["ids"].astype(str))
            values.append(shard["embeddings"].astype(np.float16, copy=True))
    if not values:
        raise FileNotFoundError(f"no MERT statistics in {root}")
    datasets, ids, values = (
        np.concatenate(datasets), np.concatenate(ids), np.concatenate(values),
    )
    keys = list(zip(datasets, ids))
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate dataset/ID MERT statistics")
    return datasets, ids, values


def load_metadata(datasets: np.ndarray, ids: np.ndarray) -> pd.DataFrame:
    frames = []
    for name in np.unique(datasets):
        truth = pd.read_csv(truth_path(str(name)), dtype={"ID": str})
        truth["DATASET"] = str(name)
        frames.append(truth)
    metadata = pd.concat(frames, ignore_index=True).set_index(["DATASET", "ID"])
    keys = pd.MultiIndex.from_arrays([datasets, ids], names=["DATASET", "ID"])
    missing = keys.difference(metadata.index)
    if len(missing):
        raise ValueError(f"truth missing for {list(missing[:3])}")
    return metadata.loc[keys].reset_index()


def normalized_generator(row: pd.Series) -> str:
    if int(row.MUSIC_FAKE) != 1:
        return "real"
    value = row.get("MUSIC_GENERATOR", row.get("GENERATOR"))
    if pd.notna(value) and str(value).strip():
        value = str(value).strip().lower().replace("-", "_")
        aliases = {
            "musicgen_medium": "musicgen", "stable_audio_open": "stableaudio",
            "stable_audio": "stableaudio", "audioldm2": "audioldm",
        }
        return aliases.get(value, value)
    source = str(row.get("MUSIC_SOURCE_ID", "unknown")).lower()
    if source.startswith("echoes_fake"):
        return "echoes_unknown"
    if source.startswith("mgm_fake"):
        return "multigen_unknown"
    if source.startswith("suite_"):
        return "fmc_unknown"
    return "unknown"


def source_id(row: pd.Series) -> str:
    value = row.get("MUSIC_SOURCE_ID")
    if pd.notna(value) and str(value).strip():
        return str(value)
    return f"{row.DATASET}:{row.ID}"


def real_group(row: pd.Series) -> str:
    source = source_id(row).lower()
    return source.split("_", 1)[0]


def stable_real_holdout(source: str) -> bool:
    return int(hashlib.sha1(source.encode()).hexdigest()[:8], 16) % 5 == 0


def source_balanced_weights(frame: pd.DataFrame) -> np.ndarray:
    keys = frame.apply(source_id, axis=1)
    counts = keys.value_counts()
    weights = np.asarray([1 / counts[key] for key in keys], dtype=np.float32)
    labels = frame.MUSIC_FAKE.astype(int).to_numpy()
    mass = np.bincount(labels, weights=weights, minlength=2)
    weights /= mass[labels].clip(1e-8)
    return weights / weights.mean()


def codec_pairs(frame: pd.DataFrame) -> np.ndarray:
    """Return ``[codec student, clean teacher]`` verified same-source pairs."""
    by_id = {}
    for index, row in frame.iterrows():
        by_id.setdefault(str(row.ID), []).append(index)
    clean_by_mixture = {}
    for index, row in frame.iterrows():
        mixture = row.get("MIXTURE_ID")
        if pd.notna(mixture) and str(row.get("CHANNEL", "")) == "clean":
            clean_by_mixture[(str(row.DATASET), str(mixture))] = index
    pairs = []
    for index, row in frame.iterrows():
        channel = str(row.get("CHANNEL", row.get("STRESS_VARIANT", "clean")))
        if channel in ("", "clean", "nan") and pd.isna(row.get("PARENT_ID")):
            continue
        teacher = None
        parent = row.get("PARENT_ID")
        if pd.notna(parent):
            candidates = by_id.get(str(parent), [])
            if len(candidates) == 1:
                teacher = candidates[0]
        if teacher is None:
            mixture = row.get("MIXTURE_ID")
            if pd.notna(mixture):
                teacher = clean_by_mixture.get((str(row.DATASET), str(mixture)))
        if teacher is None:
            continue
        clean = frame.iloc[teacher]
        if (
            int(clean.MUSIC_FAKE) != int(row.MUSIC_FAKE)
            or source_id(clean) != source_id(row)
        ):
            raise ValueError("codec pair changed Music source or label")
        pairs.append((index, teacher))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def environment_indices(frame: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    fake = frame.MUSIC_FAKE.eq(1)
    generator = frame.apply(normalized_generator, axis=1)
    environments = []
    for name in sorted(generator[fake].unique()):
        indices = frame.index[fake & generator.eq(name)].to_numpy(np.int64)
        if len(indices):
            environments.append((f"fake:{name}", indices))
    real = frame.loc[~fake].copy()
    real["_GROUP"] = real.apply(real_group, axis=1)
    for name, group in real.groupby("_GROUP"):
        environments.append((f"real:{name}", group.index.to_numpy(np.int64)))
    return environments


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def _normalization(values: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = values[indices]
    mean = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = selected.std(axis=0, dtype=np.float64).clip(1e-4).astype(np.float32)
    return mean, std


def _score_frame(frame: pd.DataFrame, scores: np.ndarray, source_level: bool) -> float:
    evaluated = frame[["MUSIC_FAKE", "MUSIC_SOURCE_ID"]].copy()
    evaluated["SCORE"] = scores
    if source_level:
        evaluated["SOURCE"] = frame.apply(source_id, axis=1).to_numpy()
        evaluated = evaluated.groupby("SOURCE", as_index=False).agg(
            MUSIC_FAKE=("MUSIC_FAKE", "first"), SCORE=("SCORE", "mean"),
        )
    return float(official_eer(evaluated.MUSIC_FAKE.astype(int), evaluated.SCORE))


def authorized_metrics(
    frame: pd.DataFrame, scores: np.ndarray, datasets: tuple[str, ...],
) -> tuple[pd.DataFrame, float]:
    rows = []
    for name in datasets:
        mask = frame.DATASET.eq(name).to_numpy()
        eer = _score_frame(frame.loc[mask], scores[mask], False)
        rows.append({"DATASET": name, "MUSIC_EER": eer, "MUSIC_SCORE": 1 - eer})
    result = pd.DataFrame(rows)
    selection = .5 * result.MUSIC_SCORE.mean() + .5 * result.MUSIC_SCORE.min()
    return result, float(selection)


def train_one(
    local: np.ndarray,
    long: np.ndarray,
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    mode: str,
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    eval_every: int,
    patience: int,
    group_dro_temperature: float,
    consistency_weight: float,
    evaluation_function,
) -> tuple[dict, np.ndarray, dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    local_mean, local_std = _normalization(local, train_indices)
    long_mean, long_std = _normalization(long, train_indices)
    normalized_local = np.clip((local - local_mean) / local_std, -8, 8).astype(np.float32)
    normalized_long = np.clip((long - long_mean) / long_std, -8, 8).astype(np.float32)
    train_frame = frame.iloc[train_indices].reset_index(drop=True)
    local_x = torch.from_numpy(normalized_local[train_indices]).to(device)
    long_x = torch.from_numpy(normalized_long[train_indices]).to(device)
    y = torch.from_numpy(train_frame.MUSIC_FAKE.to_numpy(np.float32)).to(device)
    weights = torch.from_numpy(source_balanced_weights(train_frame)).to(device)
    environments = [
        (name, torch.from_numpy(np.asarray(indices).copy()).to(device))
        for name, indices in environment_indices(train_frame)
    ]
    pairs = torch.from_numpy(codec_pairs(train_frame)).to(device)
    model = MertGeneratorRobustMusicHead(
        local.shape[1], long.shape[1], hidden=96, dropout=.15,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    eval_local = torch.from_numpy(normalized_local[evaluation_indices]).to(device)
    eval_long = torch.from_numpy(normalized_long[evaluation_indices]).to(device)
    best_score, best_state, best_epoch, stale = -np.inf, None, -1, 0
    history = []
    for epoch in range(epochs + 1):
        model.train()
        logit, local_logit, long_logit, representation = model(local_x, long_x, mode)
        point = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
        average = _weighted_mean(point, weights)
        group_losses = torch.stack([
            _weighted_mean(point[index], weights[index]) for _, index in environments
        ])
        temperature = group_dro_temperature
        group_dro = temperature * (
            torch.logsumexp(group_losses / temperature, dim=0)
            - np.log(len(group_losses))
        )
        auxiliary = .5 * (
            F.binary_cross_entropy_with_logits(local_logit, y)
            + F.binary_cross_entropy_with_logits(long_logit, y)
        )
        channel = logit.new_zeros(())
        latent = logit.new_zeros(())
        if len(pairs):
            student, teacher = pairs[:, 0], pairs[:, 1]
            model.eval()
            with torch.no_grad():
                teacher_logit, _, _, teacher_representation = model(
                    local_x[teacher], long_x[teacher], mode,
                )
            model.train()
            channel = asymmetric_codec_kl(logit[student], teacher_logit)
            latent = F.smooth_l1_loss(
                representation[student], teacher_representation.detach(),
            )
        loss = (
            .25 * average + .75 * group_dro + .10 * auxiliary
            + consistency_weight * channel + .02 * latent
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch % eval_every:
            continue
        model.eval()
        with torch.inference_mode():
            score = model(eval_local, eval_long, mode)[0].sigmoid().cpu().numpy()
        selection = float(evaluation_function(score))
        history.append({
            "EPOCH": epoch, "LOSS": float(loss.detach()),
            "GROUP_DRO": float(group_dro.detach()),
            "CHANNEL_KL": float(channel.detach()), "SELECTION": selection,
        })
        if selection > best_score + 1e-5:
            best_score, best_epoch = selection, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        score = model(eval_local, eval_long, mode)[0].sigmoid().cpu().numpy()
    state = {
        "model": {name: value.cpu() for name, value in best_state.items()},
        "local_mean": local_mean, "local_std": local_std,
        "long_mean": long_mean, "long_std": long_std,
        "best_epoch": best_epoch, "selection": best_score,
    }
    return state, score, {"history": history, "pairs": int(len(pairs))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--statistics", type=Path, default=ROOT / "output/mert_temporal_v2",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--projection-width", type=int, default=48)
    parser.add_argument("--projection-seed", type=int, default=20260904)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--group-dro-temperature", type=float, default=.10)
    parser.add_argument("--consistency-weight", type=float, default=.35)
    parser.add_argument("--modes", nargs="+", choices=("local", "long", "dual"),
                        default=["local", "long", "dual"])
    parser.add_argument("--logo-generators", nargs="+", default=list(LOGO_DEFAULT))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in TRAIN_DEFAULT:
        assert_no_locked_eval_leakage(
            TRAIN_TRUTH_OVERRIDES.get(name, truth_path(name)),
            ROOT / "configs/data_partitions.yaml",
        )

    datasets, ids, statistics = load_statistics(args.statistics)
    frame = load_metadata(datasets, ids)
    present = frame.MUSIC_PRESENT.eq(1) & frame.MUSIC_FAKE.notna()
    frame, statistics = frame.loc[present].reset_index(drop=True), statistics[present]
    projection = make_projection(args.projection_width, args.projection_seed).to(args.device)
    local_blocks, long_blocks = [], []
    for start in range(0, len(statistics), 128):
        local, long = dual_scale_features(statistics[start:start + 128], projection)
        local_blocks.append(local.cpu().numpy())
        long_blocks.append(long.cpu().numpy())
    local, long = np.concatenate(local_blocks), np.concatenate(long_blocks)
    del statistics

    train_mask = frame.DATASET.isin(TRAIN_DEFAULT).to_numpy(copy=True)
    for name, path in TRAIN_TRUTH_OVERRIDES.items():
        allowed = set(pd.read_csv(path, dtype={"ID": str}).ID)
        train_mask &= ~(frame.DATASET.eq(name).to_numpy() & ~frame.ID.isin(allowed).to_numpy())
    dev_mask = frame.DATASET.isin(DEV_DEFAULT).to_numpy()
    train_indices = np.flatnonzero(train_mask)
    dev_indices = np.flatnonzero(dev_mask)
    dev_frame = frame.iloc[dev_indices].reset_index(drop=True)
    generator = frame.apply(normalized_generator, axis=1)
    real_sources = frame.apply(source_id, axis=1)
    train_series = pd.Series(train_mask, index=frame.index)
    real_holdout = (
        train_series & frame.MUSIC_FAKE.eq(0)
        & real_sources.map(stable_real_holdout)
    )
    print(json.dumps({
        "train": len(train_indices), "dev": len(dev_indices),
        "local_features": local.shape[1], "long_features": long.shape[1],
        "logo_generators": args.logo_generators,
    }), flush=True)

    rows, final_states, final_predictions = [], {}, {}
    for mode_index, mode in enumerate(args.modes):
        def dev_evaluation(score):
            return authorized_metrics(dev_frame, score, DEV_DEFAULT)[1]

        state, dev_score, details = train_one(
            local, long, frame, train_indices, dev_indices, mode,
            device=torch.device(args.device), seed=args.seed + mode_index,
            epochs=args.epochs, eval_every=args.eval_every, patience=args.patience,
            group_dro_temperature=args.group_dro_temperature,
            consistency_weight=args.consistency_weight,
            evaluation_function=dev_evaluation,
        )
        dev_result, dev_selection = authorized_metrics(
            dev_frame, dev_score, DEV_DEFAULT,
        )
        final_states[mode] = state
        final_predictions[mode] = dev_score
        mode_dir = args.output_dir / mode
        mode_dir.mkdir(exist_ok=True)
        pd.DataFrame(details["history"]).to_csv(mode_dir / "history.csv", index=False)
        dev_result.to_csv(mode_dir / "dev_metrics.csv", index=False)

        logo_scores = []
        for fold_index, heldout in enumerate(args.logo_generators):
            heldout_fake = train_series & frame.MUSIC_FAKE.eq(1) & generator.eq(heldout)
            heldout_sources = set(real_sources[heldout_fake])
            heldout_source_rows = real_sources.isin(heldout_sources)
            fold_train = (
                train_mask & ~heldout_source_rows.to_numpy()
                & ~real_holdout.to_numpy()
            )
            fold_eval = heldout_fake | real_holdout
            fold_indices = np.flatnonzero(fold_eval.to_numpy())
            if not heldout_fake.any() or frame.loc[fold_eval, "MUSIC_FAKE"].nunique() != 2:
                raise ValueError(f"invalid LOGO fold {heldout}")
            fold_frame = frame.iloc[fold_indices].reset_index(drop=True)

            def fold_evaluation(score, fold_frame=fold_frame):
                return 1 - _score_frame(fold_frame, score, True)

            _, fold_score, _ = train_one(
                local, long, frame, np.flatnonzero(fold_train), fold_indices, mode,
                device=torch.device(args.device),
                seed=args.seed + 100 + mode_index * 10 + fold_index,
                epochs=args.epochs, eval_every=args.eval_every, patience=args.patience,
                group_dro_temperature=args.group_dro_temperature,
                consistency_weight=args.consistency_weight,
                evaluation_function=fold_evaluation,
            )
            eer = _score_frame(fold_frame, fold_score, True)
            logo_scores.append(1 - eer)
            rows.append({
                "MODE": mode, "ROLE": "LOGO", "FOLD": heldout,
                "MUSIC_SCORE": 1 - eer,
            })
            print(f"mode={mode} LOGO={heldout} score={1-eer:.5f}", flush=True)
        logo_mean, logo_worst = float(np.mean(logo_scores)), float(np.min(logo_scores))
        robust = .5 * dev_selection + .25 * logo_mean + .25 * logo_worst
        rows.append({
            "MODE": mode, "ROLE": "SUMMARY", "FOLD": "authorized+LOGO",
            "MUSIC_SCORE": robust, "AUTHORIZED_SELECTION": dev_selection,
            "LOGO_MEAN": logo_mean, "LOGO_WORST": logo_worst,
        })
        print(
            f"mode={mode} authorized={dev_selection:.5f} LOGO_mean={logo_mean:.5f} "
            f"LOGO_worst={logo_worst:.5f} robust={robust:.5f}", flush=True,
        )

    selection = pd.DataFrame(rows)
    summaries = selection.loc[selection.ROLE.eq("SUMMARY")].sort_values(
        ["MUSIC_SCORE", "MODE"], ascending=[False, True],
    )
    chosen = str(summaries.iloc[0].MODE)
    state = final_states[chosen]
    checkpoint = {
        "model_type": "mert_generator_robust_music",
        "model": state["model"],
        "projection": projection.cpu().numpy(),
        "local_mean": state["local_mean"], "local_std": state["local_std"],
        "long_mean": state["long_mean"], "long_std": state["long_std"],
        "config": {
            "local_dim": local.shape[1], "long_dim": long.shape[1],
            "hidden": 96, "dropout": .15,
        },
        "mode": chosen, "seed": args.seed,
        "best_epoch": state["best_epoch"],
        "authorized_selection": state["selection"],
        "robust_selection": float(summaries.iloc[0].MUSIC_SCORE),
        "logo_generators": args.logo_generators,
        "training_datasets": list(TRAIN_DEFAULT),
        "development_datasets": list(DEV_DEFAULT),
        "selection_uses_codec_v4_v5_v6_v7": False,
    }
    torch.save(checkpoint, args.output_dir / "mert_generator_robust_music.pt")
    selection.to_csv(args.output_dir / "selection.csv", index=False)
    pd.DataFrame({
        "DATASET": dev_frame.DATASET, "ID": dev_frame.ID,
        "MERT_ROBUST_MUSIC_PROB": final_predictions[chosen],
    }).to_csv(args.output_dir / "dev_predictions.csv", index=False)
    summary = {
        "chosen_mode": chosen,
        "robust_selection": float(summaries.iloc[0].MUSIC_SCORE),
        "authorized_selection": float(summaries.iloc[0].AUTHORIZED_SELECTION),
        "logo_mean": float(summaries.iloc[0].LOGO_MEAN),
        "logo_worst": float(summaries.iloc[0].LOGO_WORST),
        "retrospective_banks_used_for_selection": [],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(summaries.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
