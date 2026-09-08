#!/usr/bin/env python3
"""Audit fixed File/component fusion without using blind-v6 for selection.

The script treats ``codec_mixed_blind_v6`` as diagnostic-only.  The selected
formula and its constants are fixed by the training/development banks and the
already-retrospective blind-v5 bank before blind-v6 rows are reported.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from component_consistent_file_fusion import _logit  # noqa: E402
from evaluate_diagnostic import official_eer  # noqa: E402


BANKS = {
    "train_channel_invariant": (
        ROOT / "data/eval/channel_invariant_factorial_train_v1/truth.csv",
        ROOT / "reports/v47_anchor_cache_train_dev_v2/cache/datasets/"
        "channel_invariant_factorial_train_v1/exact_v47_predictions.csv",
        ROOT / "reports/component_consistent_file_v51/"
        "channel_invariant_factorial_train_v1_phone_router.csv",
        "selection",
    ),
    "codec_mixed_dev_v4": (
        ROOT / "data/eval/codec_mixed_dev_v4/truth.csv",
        ROOT / "reports/v50_candidate_codec_dev_v4/predictions.csv",
        ROOT / "reports/component_consistent_file_v51/"
        "codec_mixed_dev_v4_phone_router.csv",
        "selection",
    ),
    "codec_mixed_blind_v5": (
        ROOT / "data/eval/codec_mixed_blind_v5/truth.csv",
        ROOT / "reports/v50_candidate_blind_v5/predictions.csv",
        ROOT / "reports/component_consistent_file_v51/"
        "codec_mixed_blind_v5_phone_router.csv",
        "retrospective_selection",
    ),
    "codec_mixed_blind_v6": (
        ROOT / "data/eval/codec_mixed_blind_v6/truth.csv",
        ROOT / "reports/v50_blind_v6_one_shot/v50_frozen/output/submission.csv",
        ROOT / "reports/component_consistent_file_v51/"
        "codec_mixed_blind_v6_phone_router.csv",
        "diagnostic_only_no_selection",
    ),
}


def read_bank(specification: tuple[Path, Path, Path, str]) -> pd.DataFrame:
    truth_path, prediction_path, phone_path, role = specification
    truth = pd.read_csv(truth_path, dtype={"ID": str}).set_index("ID")
    prediction = pd.read_csv(
        prediction_path, dtype={"ID": str},
    ).set_index("ID").loc[truth.index]
    phone = pd.read_csv(phone_path, dtype={"ID": str}).set_index("ID").loc[truth.index]
    result = truth.join(prediction, validate="one_to_one")
    result["IS_PHONE"] = (
        phone.PHONE_PROB.to_numpy() >= phone.THRESHOLD.to_numpy()
    )
    result.attrs["role"] = role
    return result


def component_evidence(frame: pd.DataFrame, kind: str) -> np.ndarray:
    voice = np.clip(frame.VOICE_FAKE_PROB.to_numpy(np.float64), 1e-5, 1 - 1e-5)
    music = np.clip(frame.MUSIC_FAKE_PROB.to_numpy(np.float64), 1e-5, 1 - 1e-5)
    if kind == "max":
        return np.maximum(voice, music)
    if kind == "noisy_or":
        return 1.0 - (1.0 - voice) * (1.0 - music)
    if kind == "presence_logit_max":
        voice_presence = np.clip(
            frame.VOICE_PRESENT_PROB.to_numpy(np.float64), 1e-5, 1 - 1e-5,
        )
        music_presence = np.clip(
            frame.MUSIC_PRESENT_PROB.to_numpy(np.float64), 1e-5, 1 - 1e-5,
        )
        return expit(np.maximum(
            _logit(voice) + .5 * _logit(voice_presence),
            _logit(music) + .5 * _logit(music_presence),
        ))
    raise ValueError(f"unknown evidence: {kind}")


def fixed_score(
    frame: pd.DataFrame,
    *,
    kind: str = "max",
    weight: float = .20,
    phone_weight: float | None = None,
) -> np.ndarray:
    file_score = np.clip(
        frame.FILE_FAKE_PROB.to_numpy(np.float64), 1e-5, 1 - 1e-5,
    )
    evidence = np.clip(component_evidence(frame, kind), 1e-5, 1 - 1e-5)
    mixed_gate = (
        (frame.VOICE_PRESENT_PROB.to_numpy() >= .10)
        & (frame.MUSIC_PRESENT_PROB.to_numpy() >= .20)
    )
    weights = np.full(len(frame), weight, dtype=np.float64)
    if phone_weight is not None:
        weights[frame.IS_PHONE.to_numpy(bool)] = phone_weight
    weights *= mixed_gate
    return expit((1.0 - weights) * _logit(file_score) + weights * _logit(evidence))


def fit_monotone(frame: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    """Fit non-negative File/component coefficients using regularized logloss."""
    file_score = np.clip(frame.FILE_FAKE_PROB.to_numpy(), 1e-5, 1 - 1e-5)
    component = np.clip(component_evidence(frame, "max"), 1e-5, 1 - 1e-5)
    features = np.stack((_logit(file_score), _logit(component)), axis=1)
    labels = frame.FILE_FAKE.to_numpy(np.float64)

    def objective(parameters: np.ndarray) -> float:
        score = parameters[0] + features[indices] @ parameters[1:]
        loss = np.mean(np.logaddexp(0.0, score) - labels[indices] * score)
        return float(loss + np.sum(parameters[1:] ** 2))

    fitted = minimize(
        objective, np.asarray([0.0, .2, .2]), method="L-BFGS-B",
        bounds=[(None, None), (0.0, None), (0.0, None)],
    )
    if not fitted.success:
        raise RuntimeError(f"monotone optimization failed: {fitted.message}")
    return fitted.x


def monotone_score(frame: pd.DataFrame, parameters: np.ndarray) -> np.ndarray:
    file_score = np.clip(frame.FILE_FAKE_PROB.to_numpy(), 1e-5, 1 - 1e-5)
    component = np.clip(component_evidence(frame, "max"), 1e-5, 1 - 1e-5)
    return parameters[0] + np.stack(
        (_logit(file_score), _logit(component)), axis=1,
    ) @ parameters[1:]


def file_eer(frame: pd.DataFrame, score: np.ndarray) -> float:
    return float(official_eer(frame.FILE_FAKE.astype(int), score))


def source_connected_groups(frame: pd.DataFrame) -> np.ndarray:
    """Keep every repeated Voice/Music source in one connected component."""
    base = frame.reset_index().drop_duplicates("MIXTURE_ID")
    parents: dict[str, str] = {}

    def find(item: str) -> str:
        parents.setdefault(item, item)
        if parents[item] != item:
            parents[item] = find(parents[item])
        return parents[item]

    def union(left: str, right: str) -> None:
        left, right = find(left), find(right)
        if left != right:
            parents[right] = left

    for row in base.itertuples():
        sources = []
        if pd.notna(row.VOICE_SOURCE_ID):
            sources.append(f"voice:{row.VOICE_SOURCE_ID}")
        if pd.notna(row.MUSIC_SOURCE_ID):
            sources.append(f"music:{row.MUSIC_SOURCE_ID}")
        if len(sources) == 2:
            union(*sources)
    mapping = {}
    for row in base.itertuples():
        sources = []
        if pd.notna(row.VOICE_SOURCE_ID):
            sources.append(f"voice:{row.VOICE_SOURCE_ID}")
        if pd.notna(row.MUSIC_SOURCE_ID):
            sources.append(f"music:{row.MUSIC_SOURCE_ID}")
        mapping[row.MIXTURE_ID] = find(sources[0])
    return frame.MIXTURE_ID.map(mapping).to_numpy(str)


def cell_rows(
    bank: str, role: str, frame: pd.DataFrame, model: str, score: np.ndarray,
) -> list[dict]:
    work = frame.copy()
    work["SCORE"] = score
    rows = []
    groupings = ["MIX_MODE", "CHANNEL", "COMPONENT_CASE", "IS_PHONE"]
    for column in groupings:
        if column not in work:
            continue
        for value, group in work.groupby(column, dropna=False):
            if group.FILE_FAKE.nunique() != 2:
                continue
            rows.append({
                "BANK": bank, "ROLE": role, "MODEL": model,
                "GROUP": column, "VALUE": value, "N": len(group),
                "FILE_EER": file_eer(group, group.SCORE.to_numpy()),
            })
    if {"MIX_MODE", "COMPONENT_CASE"}.issubset(work.columns):
        for mix_mode, mixed in work.groupby("MIX_MODE"):
            negative = mixed[mixed.COMPONENT_CASE.eq("RR")]
            for component_case in ("RF", "FR", "FF"):
                positive = mixed[mixed.COMPONENT_CASE.eq(component_case)]
                contrast = pd.concat((negative, positive))
                if len(negative) and len(positive):
                    rows.append({
                        "BANK": bank, "ROLE": role, "MODEL": model,
                        "GROUP": "MIX_MODE_X_FILE_CASE",
                        "VALUE": f"{mix_mode}__RR_vs_{component_case}",
                        "N": len(contrast),
                        "FILE_EER": file_eer(
                            contrast, contrast.SCORE.to_numpy(),
                        ),
                    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "reports/component_consistent_file_v51",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    banks = {name: read_bank(specification) for name, specification in BANKS.items()}

    train = banks["train_channel_invariant"]
    parameters = fit_monotone(train, np.arange(len(train)))
    models = {
        "input_file": lambda frame: frame.FILE_FAKE_PROB.to_numpy(np.float64),
        "max_logit_w20_gated_v51": lambda frame: fixed_score(frame),
        "noisy_or_logit_w20_gated": lambda frame: fixed_score(
            frame, kind="noisy_or",
        ),
        "presence_logit_max_w20_gated": lambda frame: fixed_score(
            frame, kind="presence_logit_max",
        ),
        "phone_w30_w20_gated": lambda frame: fixed_score(
            frame, weight=.30, phone_weight=.20,
        ),
        "learned_monotone_l2_1": lambda frame: monotone_score(frame, parameters),
    }

    overall, subgroups = [], []
    for bank, frame in banks.items():
        role = frame.attrs["role"]
        for model, scorer in models.items():
            score = scorer(frame)
            overall.append({
                "BANK": bank, "ROLE": role, "MODEL": model, "N": len(frame),
                "FILE_EER": file_eer(frame, score),
            })
            subgroups.extend(cell_rows(bank, role, frame, model, score))
    pd.DataFrame(overall).to_csv(args.output_dir / "overall.csv", index=False)
    pd.DataFrame(subgroups).to_csv(args.output_dir / "subgroups.csv", index=False)

    sweep = []
    for kind in ("max", "noisy_or", "presence_logit_max"):
        for weight in np.arange(0.0, 1.01, .1):
            for bank, frame in banks.items():
                sweep.append({
                    "BANK": bank, "ROLE": frame.attrs["role"],
                    "EVIDENCE": kind, "WEIGHT": round(float(weight), 2),
                    "FILE_EER": file_eer(
                        frame, fixed_score(frame, kind=kind, weight=float(weight)),
                    ),
                })
    pd.DataFrame(sweep).to_csv(args.output_dir / "sweep.csv", index=False)

    cv_rows = []
    labels = train.FILE_FAKE.to_numpy(int)
    parent_groups = train.MIXTURE_ID.to_numpy(str)
    fixed_models = {
        name: scorer(train) for name, scorer in models.items()
        if name != "learned_monotone_l2_1"
    }
    for seed in range(5):
        splitter = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fold, (fit_indices, test_indices) in enumerate(
            splitter.split(train, labels, parent_groups)
        ):
            fold_parameters = fit_monotone(train, fit_indices)
            fold_scores = {
                **fixed_models,
                "learned_monotone_l2_1": monotone_score(train, fold_parameters),
            }
            for model, score in fold_scores.items():
                cv_rows.append({
                    "SPLIT": "codec_parent_disjoint", "SEED": seed,
                    "FOLD": fold, "MODEL": model, "N": len(test_indices),
                    "GROUP_N": len(set(parent_groups[test_indices])),
                    "FILE_EER": official_eer(
                        labels[test_indices], score[test_indices],
                    ),
                })

    source_groups = source_connected_groups(train)
    splitter = GroupKFold(5)
    for fold, (fit_indices, test_indices) in enumerate(
        splitter.split(train, labels, source_groups)
    ):
        fold_parameters = fit_monotone(train, fit_indices)
        fold_scores = {
            **fixed_models,
            "learned_monotone_l2_1": monotone_score(train, fold_parameters),
        }
        for model, score in fold_scores.items():
            cv_rows.append({
                "SPLIT": "source_connected_disjoint", "SEED": 0,
                "FOLD": fold, "MODEL": model, "N": len(test_indices),
                "GROUP_N": len(set(source_groups[test_indices])),
                "FILE_EER": official_eer(
                    labels[test_indices], score[test_indices],
                ),
            })
    pd.DataFrame(cv_rows).to_csv(args.output_dir / "group_cv.csv", index=False)
    pd.DataFrame([{
        "INTERCEPT": parameters[0], "FILE_LOGIT_COEF": parameters[1],
        "COMPONENT_MAX_LOGIT_COEF": parameters[2], "L2": 1.0,
    }]).to_csv(args.output_dir / "learned_monotone.csv", index=False)


if __name__ == "__main__":
    main()
