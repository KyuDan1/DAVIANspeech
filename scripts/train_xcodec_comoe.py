#!/usr/bin/env python3
"""Train and audit a separation-free CoMoE-style X-Codec music expert."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data_guard import assert_no_locked_eval_leakage  # noqa: E402
from evaluate_diagnostic import official_eer  # noqa: E402
from train_xcodec_ngram_probe import load_tokens, truth_for  # noqa: E402
from xcodec_comoe import XCodecCoMoE  # noqa: E402


class TokenDataset(Dataset):
    def __init__(self, tokens: np.ndarray, labels: np.ndarray) -> None:
        self.tokens = tokens
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int):
        return torch.from_numpy(self.tokens[index].astype(np.int64)), self.labels[index]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generator_weights(frame: pd.DataFrame) -> np.ndarray:
    generator = frame.MUSIC_GENERATOR.fillna("unknown").astype(str)
    fake = frame.MUSIC_FAKE.eq(1)
    weights = np.ones(len(frame), dtype=np.float32)
    counts = generator[fake].value_counts()
    if len(counts):
        fake_target = float(fake.sum()) / len(counts)
        weights[fake] = generator[fake].map(lambda name: fake_target / counts[name])
    return weights / weights.mean()


@torch.inference_mode()
def predict(model, tokens: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    output = []
    for offset in range(0, len(tokens), batch_size):
        batch = torch.from_numpy(
            tokens[offset:offset + batch_size].astype(np.int64)
        ).to(device)
        output.append(torch.sigmoid(model(batch)[0]).cpu().numpy())
    return np.concatenate(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-dataset", default="temporal_mixed_train_v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=.15)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--aux-weight", type=float, default=.2)
    args = parser.parse_args()
    seed_all(args.seed)
    assert_no_locked_eval_leakage(
        ROOT / "data/eval" / args.train_dataset / "truth.csv",
        ROOT / "configs/data_partitions.yaml",
    )

    datasets, ids, tokens = load_tokens(args.tokens)
    source_rows = np.flatnonzero(datasets == args.train_dataset)
    truth = truth_for(args.train_dataset).loc[ids[source_rows]]
    train_selector = truth.SPLIT.eq("train").to_numpy()
    dev_selector = truth.SPLIT.eq("dev").to_numpy()
    train_rows, dev_rows = source_rows[train_selector], source_rows[dev_selector]
    train_frame = truth.iloc[np.flatnonzero(train_selector)]
    train_labels = train_frame.MUSIC_FAKE.astype(int).to_numpy()
    sample_weights = generator_weights(train_frame)

    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        TokenDataset(tokens[train_rows], np.arange(len(train_rows))),
        batch_size=args.batch_size, shuffle=True, generator=loader_generator,
        num_workers=0, drop_last=False,
    )
    device = torch.device(args.device)
    model = XCodecCoMoE(
        width=args.width, layers=args.layers, heads=args.heads, dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    history, best_eer, best_epoch, stale = [], 1.0, 0, 0
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "xcodec_comoe.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_tokens, local_indices in loader:
            batch_tokens = batch_tokens.to(device)
            local_indices = local_indices.long().numpy()
            labels = torch.from_numpy(train_labels[local_indices]).float().to(device)
            weights = torch.from_numpy(sample_weights[local_indices]).to(device)
            fused, lower, upper = model(batch_tokens)
            direct = F.binary_cross_entropy_with_logits(fused, labels, reduction="none")
            auxiliary = (
                F.binary_cross_entropy_with_logits(lower, labels, reduction="none")
                + F.binary_cross_entropy_with_logits(upper, labels, reduction="none")
            ) / 2
            loss = ((direct + args.aux_weight * auxiliary) * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()

        dev_probability = predict(model, tokens[dev_rows], device, args.batch_size * 2)
        dev_label = truth.iloc[np.flatnonzero(dev_selector)].MUSIC_FAKE.astype(int)
        dev_eer = official_eer(dev_label, dev_probability)
        history.append({"epoch": epoch, "loss": np.mean(losses), "dev_eer": dev_eer})
        print(f"epoch={epoch:02d} loss={np.mean(losses):.5f} dev_eer={dev_eer:.5f}", flush=True)
        if dev_eer < best_eer - 1e-9:
            best_eer, best_epoch, stale = dev_eer, epoch, 0
            torch.save({
                "state_dict": model.state_dict(),
                "config": {
                    "width": args.width, "layers": args.layers,
                    "heads": args.heads, "dropout": args.dropout,
                },
            }, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break

    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    probabilities = predict(model, tokens, device, args.batch_size * 2)
    rows, summary = [], []
    for name in sorted(set(datasets.tolist())):
        selected = np.flatnonzero(datasets == name)
        frame = truth_for(name).loc[ids[selected]]
        valid = frame.MUSIC_PRESENT.eq(1) & frame.MUSIC_FAKE.notna()
        score = probabilities[selected]
        eer = official_eer(
            frame.loc[valid, "MUSIC_FAKE"].astype(int), score[valid.to_numpy()]
        )
        summary.append({"DATASET": name, "ROWS": int(valid.sum()), "MUSIC_EER": eer})
        generator = frame.get("MUSIC_GENERATOR", frame.get("GENERATOR", "unknown"))
        if not isinstance(generator, pd.Series):
            generator = pd.Series(generator, index=frame.index)
        for item, label, present, gen, probability in zip(
            frame.index, frame.MUSIC_FAKE, frame.MUSIC_PRESENT, generator, score
        ):
            rows.append({
                "DATASET": name, "ID": item, "MUSIC_FAKE": label,
                "MUSIC_PRESENT": present, "GENERATOR": gen,
                "XCODEC_FAKE_PROB": probability,
            })
    pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output / "predictions.csv", index=False)
    result = pd.DataFrame(summary)
    result.to_csv(args.output / "summary.csv", index=False)
    metadata = {"best_epoch": best_epoch, "best_dev_eer": best_eer, **saved["config"]}
    (args.output / "selected.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
