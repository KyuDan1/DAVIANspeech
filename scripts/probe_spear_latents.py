#!/usr/bin/env python3
"""Probe which frozen SPEAR layers separate RR/RF/FR/FF mixtures.

Weights are fitted only on the configured train banks.  A pre-existing
development bank controls early stopping; the factorial split is read only
for final scoring.  Locked evaluation requires an explicit opt-in so routine
grid searches cannot silently tune on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate_diagnostic import official_eer  # noqa: E402
from data_guard import assert_no_locked_eval_leakage  # noqa: E402

LAYERS = 13
DIM = 1280
MODES = ("concurrent", "partial_overlap", "sequential")
JOINT_LAYERS = (1, 2, 3)


class JointMLP(torch.nn.Module):
    """Small nonlinear RR/RF/FR/FF probe over shallow SPEAR layers."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(len(JOINT_LAYERS) * DIM, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.35),
            torch.nn.Linear(hidden_dim, 4),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features[:, JOINT_LAYERS].flatten(1))


def load_embeddings(name: str, prefix="spear_") -> tuple[pd.DataFrame, np.ndarray]:
    directory = ROOT / "output" / f"{prefix}{name}"
    ids, vectors = [], []
    for path in sorted(directory.glob("shard_*.npz")):
        shard = np.load(path)
        ids.extend(shard["ids"].astype(str))
        vectors.append(shard["embeddings"])
    if not vectors:
        raise FileNotFoundError(f"No embeddings in {directory}")
    matrix = np.concatenate(vectors).reshape(-1, LAYERS, DIM)
    truth = pd.read_csv(
        ROOT / "data" / "eval" / name / "truth.csv", dtype={"ID": str}
    ).set_index("ID").loc[ids].reset_index()
    return truth, matrix


def early_stopped_binary(
    x: torch.Tensor, y: torch.Tensor, validation_x: torch.Tensor,
    validation_y: torch.Tensor, weight_decay: float,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    weight = torch.zeros(LAYERS, DIM, 2, device=x.device, requires_grad=True)
    bias = torch.zeros(LAYERS, 2, device=x.device, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [weight, bias], lr=0.03, weight_decay=weight_decay
    )
    best = (float("inf"), None, None, 0)
    for epoch in range(301):
        optimizer.zero_grad()
        logits = torch.einsum("nld,ldc->nlc", x, weight) + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y[:, None, :].expand_as(logits)
        )
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            with torch.inference_mode():
                logits = torch.einsum(
                    "nld,ldc->nlc", validation_x, weight
                ) + bias
                value = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, validation_y[:, None, :].expand_as(logits)
                ).item()
            if value < best[0]:
                best = (value, weight.detach().clone(), bias.detach().clone(), epoch)
    return best[1], best[2], best[3], best[0]


def early_stopped_four_class(
    x: torch.Tensor, y: torch.Tensor, validation_x: torch.Tensor,
    validation_y: torch.Tensor, weight_decay: float,
) -> tuple[torch.Tensor, torch.Tensor, int, float]:
    weight = torch.zeros(LAYERS, DIM, 4, device=x.device, requires_grad=True)
    bias = torch.zeros(LAYERS, 4, device=x.device, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [weight, bias], lr=0.03, weight_decay=weight_decay
    )
    best = (float("inf"), None, None, 0)
    for epoch in range(301):
        optimizer.zero_grad()
        logits = torch.einsum("nld,ldc->nlc", x, weight) + bias
        labels = y[:, None].expand(-1, LAYERS)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 4), labels.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            with torch.inference_mode():
                logits = torch.einsum(
                    "nld,ldc->nlc", validation_x, weight
                ) + bias
                labels = validation_y[:, None].expand(-1, LAYERS)
                value = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, 4), labels.reshape(-1)
                ).item()
            if value < best[0]:
                best = (value, weight.detach().clone(), bias.detach().clone(), epoch)
    return best[1], best[2], best[3], best[0]


def component_metrics(truth: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    result = {}
    for mode in MODES:
        selected = truth.MIX_MODE.eq(mode)
        result[f"{mode}_voice_eer"] = official_eer(
            truth.loc[selected, "VOICE_FAKE"], scores[selected, 0]
        )
        result[f"{mode}_music_eer"] = official_eer(
            truth.loc[selected, "MUSIC_FAKE"], scores[selected, 1]
        )
    values = list(result.values())
    return {"MEAN_EER": float(np.mean(values)), "WORST_EER": max(values), **result}


def early_stopped_mlp(
    x: torch.Tensor, y: torch.Tensor, validation_x: torch.Tensor,
    validation_y: torch.Tensor, weight_decay: float,
) -> tuple[JointMLP, int, float]:
    model = JointMLP().to(x.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=weight_decay
    )
    best_loss, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(401):
        model.train()
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            model.eval()
            with torch.inference_mode():
                value = torch.nn.functional.cross_entropy(
                    model(validation_x), validation_y
                ).item()
            if value < best_loss:
                best_loss = value
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                best_epoch = epoch
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch, best_loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", default=[
        "external_mixed_train_v1", "mixed_devvoice_train_v1",
        "mixed_fmc_music_train_v1",
    ])
    parser.add_argument("--early-stop", default="external_mixed_v1")
    parser.add_argument("--evaluation", default="factorial_eval_1200_v2")
    parser.add_argument("--eval-split", choices=["dev", "holdout", "locked"],
                        default="dev")
    parser.add_argument("--allow-locked", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--head-output", type=Path,
        help="Optional deployable NPZ path for normalization and linear heads.",
    )
    args = parser.parse_args()
    if args.eval_split == "locked" and not args.allow_locked:
        parser.error("locked evaluation requires --allow-locked")
    torch.manual_seed(20260831)

    for name in args.train:
        assert_no_locked_eval_leakage(
            ROOT / "data" / "eval" / name / "truth.csv",
            ROOT / "configs" / "data_partitions.yaml",
        )

    truths, matrices = zip(*(load_embeddings(name) for name in args.train))
    train_truth = pd.concat(truths, ignore_index=True)
    train_matrix = np.concatenate(matrices)
    stop_truth, stop_matrix = load_embeddings(args.early_stop)
    eval_truth, eval_matrix = load_embeddings(args.evaluation)
    selected = (
        eval_truth.SPLIT.eq(args.eval_split)
        & eval_truth.VOICE_PRESENT.eq(1)
        & eval_truth.MUSIC_PRESENT.eq(1)
    )
    eval_truth = eval_truth[selected].reset_index(drop=True)
    eval_matrix = eval_matrix[selected]

    mean = train_matrix.mean(axis=0, keepdims=True)
    std = train_matrix.std(axis=0, keepdims=True) + 1e-4
    device = torch.device(args.device)
    x = torch.from_numpy((train_matrix - mean) / std).to(device)
    stop_x = torch.from_numpy((stop_matrix - mean) / std).to(device)
    eval_x = torch.from_numpy((eval_matrix - mean) / std).to(device)
    y = torch.tensor(train_truth[["VOICE_FAKE", "MUSIC_FAKE"]].to_numpy(),
                     dtype=torch.float32, device=device)
    stop_y = torch.tensor(stop_truth[["VOICE_FAKE", "MUSIC_FAKE"]].to_numpy(),
                          dtype=torch.float32, device=device)

    weight, bias, epoch, validation_loss = early_stopped_binary(
        x, y, stop_x, stop_y, args.weight_decay
    )
    with torch.inference_mode():
        scores = torch.sigmoid(
            torch.einsum("nld,ldc->nlc", eval_x, weight) + bias
        ).cpu().numpy()
    layer_rows = []
    for layer in range(LAYERS):
        layer_rows.append({
            "LAYER": layer, "EARLY_STOP_EPOCH": epoch,
            "EARLY_STOP_LOSS": validation_loss,
            **component_metrics(eval_truth, scores[:, layer]),
        })

    class_y = torch.tensor(
        2 * train_truth.VOICE_FAKE.to_numpy() + train_truth.MUSIC_FAKE.to_numpy(),
        dtype=torch.long, device=device,
    )
    stop_class_y = torch.tensor(
        2 * stop_truth.VOICE_FAKE.to_numpy() + stop_truth.MUSIC_FAKE.to_numpy(),
        dtype=torch.long, device=device,
    )
    four_weight, four_bias, four_epoch, four_loss = early_stopped_four_class(
        x, class_y, stop_x, stop_class_y, args.weight_decay
    )
    with torch.inference_mode():
        four_logits = torch.einsum(
            "nld,ldc->nlc", eval_x, four_weight
        ) + four_bias
        four_probabilities = torch.softmax(four_logits, dim=-1).cpu().numpy()
        predictions = four_probabilities.argmax(axis=-1)
    target = (
        2 * eval_truth.VOICE_FAKE.to_numpy() + eval_truth.MUSIC_FAKE.to_numpy()
    ).astype(int)
    class_rows, confusion = [], {}
    for layer in range(LAYERS):
        # Classes are RR, RF, FR, FF. Marginalizing the joint posterior gives
        # component probabilities while retaining the learned interaction.
        component_scores = np.column_stack([
            four_probabilities[:, layer, 2] + four_probabilities[:, layer, 3],
            four_probabilities[:, layer, 1] + four_probabilities[:, layer, 3],
        ])
        class_rows.append({
            "LAYER": layer, "BALANCED_ACCURACY": balanced_accuracy_score(
                target, predictions[:, layer]
            ), "EARLY_STOP_EPOCH": four_epoch, "EARLY_STOP_LOSS": four_loss,
            **component_metrics(eval_truth, component_scores),
        })
        confusion[str(layer)] = confusion_matrix(
            target, predictions[:, layer], labels=[0, 1, 2, 3]
        ).tolist()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    head_output = args.head_output or args.output_dir / "linear_heads.npz"
    head_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        head_output,
        mean=mean.astype(np.float32), std=std.astype(np.float32),
        binary_weight=weight.cpu().numpy(), binary_bias=bias.cpu().numpy(),
        joint_weight=four_weight.cpu().numpy(),
        joint_bias=four_bias.cpu().numpy(),
    )
    pd.DataFrame(layer_rows).to_csv(
        args.output_dir / f"component_layers_{args.eval_split}.csv", index=False
    )
    pd.DataFrame(class_rows).to_csv(
        args.output_dir / f"four_class_layers_{args.eval_split}.csv", index=False
    )
    with (args.output_dir / f"confusion_{args.eval_split}.json").open("w") as handle:
        json.dump(confusion, handle, indent=2)

    mlp, mlp_epoch, mlp_loss = early_stopped_mlp(
        x, class_y, stop_x, stop_class_y, args.weight_decay
    )
    with torch.inference_mode():
        mlp_probabilities = torch.softmax(mlp(eval_x), dim=-1).cpu().numpy()
    mlp_predictions = mlp_probabilities.argmax(axis=-1)
    mlp_component_scores = np.column_stack([
        mlp_probabilities[:, 2] + mlp_probabilities[:, 3],
        mlp_probabilities[:, 1] + mlp_probabilities[:, 3],
    ])
    mlp_result = {
        "LAYERS": list(JOINT_LAYERS),
        "BALANCED_ACCURACY": balanced_accuracy_score(target, mlp_predictions),
        "EARLY_STOP_EPOCH": mlp_epoch,
        "EARLY_STOP_LOSS": mlp_loss,
        **component_metrics(eval_truth, mlp_component_scores),
        "CONFUSION": confusion_matrix(
            target, mlp_predictions, labels=[0, 1, 2, 3]
        ).tolist(),
    }
    with (args.output_dir / f"joint_mlp_{args.eval_split}.json").open("w") as handle:
        json.dump(mlp_result, handle, indent=2)
    print(pd.DataFrame(layer_rows).sort_values(
        ["WORST_EER", "MEAN_EER"]
    ).head(13).to_string(index=False))
    print(pd.DataFrame(class_rows).sort_values(
        "BALANCED_ACCURACY", ascending=False
    ).head(13).to_string(index=False))
    print(json.dumps(mlp_result, indent=2))


if __name__ == "__main__":
    main()
