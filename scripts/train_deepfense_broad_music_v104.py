#!/usr/bin/env python3
"""Domain-balanced adaptation of a public EAT--AASIST music detector.

The complete public EAT frontend is frozen in this first controlled stage.
Only AASIST and its binary projection are adapted.  Each source-domain/label
cell contributes the same number of draws per epoch, preventing the large
MixFake source from becoming the implicit target distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from evaluate_deepfense_fakemusiccaps_v103 import (  # noqa: E402
    SAMPLES, crop_or_repeat, exact_fbank, fixed_starts, load_detector,
    official_eer,
)
from pipeline import load_audio  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


class AudioRows(Dataset):
    def __init__(self, frame: pd.DataFrame, training: bool, seed: int):
        self.frame = frame.reset_index(drop=True)
        self.training = training
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        audio = load_audio(Path(row.filepath))
        if self.training and len(audio) > SAMPLES:
            maximum = len(audio) - SAMPLES
            key = stable_int(f"{self.seed}:{self.epoch}:{row.ID}")
            start = key % (maximum + 1)
        else:
            start = fixed_starts(len(audio))[1]
        features = exact_fbank(crop_or_repeat(audio, start))
        # Public DeepFense label map is spoof=0, bonafide=1.
        class_index = 0 if int(row.target) == 1 else 1
        return features, class_index, str(row.ID), str(row.DATASET)


class BalancedCellSampler(Sampler[int]):
    def __init__(self, frame: pd.DataFrame, draws_per_cell: int, seed: int):
        self.cells = [
            block.index.to_numpy(np.int64)
            for _, block in frame.groupby(["DATASET", "target"], sort=True)
        ]
        if len(self.cells) != 2 * frame.DATASET.nunique() or any(not len(x) for x in self.cells):
            raise ValueError("every domain must contain both labels")
        self.draws_per_cell = int(draws_per_cell)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.cells) * self.draws_per_cell

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        draws = np.concatenate([
            rng.choice(cell, self.draws_per_cell, replace=len(cell) < self.draws_per_cell)
            for cell in self.cells
        ])
        rng.shuffle(draws)
        return iter(draws.tolist())


def seed_worker(worker_id: int) -> None:
    value = torch.initial_seed() % (2 ** 32)
    np.random.seed(value)
    random.seed(value)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict, pd.DataFrame]:
    model.eval()
    rows = []
    for features, classes, ids, domains in loader:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(features.to(device))
        margins = (logits[:, 0] - logits[:, 1]).float().cpu().numpy()
        labels = (1 - classes.numpy()).astype(np.int64)
        rows.extend({
            "ID": item_id, "DATASET": domain, "label": int(label),
            "margin": float(margin),
        } for item_id, domain, label, margin in zip(ids, domains, labels, margins))
    frame = pd.DataFrame(rows)
    pooled = official_eer(frame.label.to_numpy(), frame.margin.to_numpy())
    domain_eers = {
        str(domain): official_eer(block.label.to_numpy(), block.margin.to_numpy())
        for domain, block in frame.groupby("DATASET") if block.label.nunique() == 2
    }
    metrics = {
        "pooled_eer": pooled,
        "macro_domain_eer": float(np.mean(list(domain_eers.values()))),
        "worst_domain_eer": float(np.max(list(domain_eers.values()))),
        "domain_eers": domain_eers,
    }
    metrics["selection"] = 1 - (
        .5 * metrics["pooled_eer"] + .3 * metrics["macro_domain_eer"]
        + .2 * metrics["worst_domain_eer"]
    )
    return metrics, frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-head", type=Path, default=None)
    parser.add_argument("--eat-dir", type=Path, default=ROOT / "models/eat-large-as2m-v56")
    parser.add_argument("--deepfense-repo", type=Path, default=Path("/tmp/deepfense-framework"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=104)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--draws-per-cell", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=0)
    parser.add_argument("--frontend-learning-rate", type=float, default=1e-6)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train = pd.read_csv(args.train, dtype={"ID": str})
    development = pd.read_csv(args.development, dtype={"ID": str})
    for name, frame in (("train", train), ("development", development)):
        if frame.ID.duplicated().any() or not {"ID", "filepath", "target", "DATASET"}.issubset(frame):
            raise ValueError(f"invalid {name} manifest")
    if set(train.ID) & set(development.ID):
        raise ValueError("train/development IDs overlap")

    device = torch.device(args.device)
    model = load_detector(args.eat_dir, args.initial_checkpoint, args.deepfense_repo)
    if args.initial_head is not None:
        head = torch.load(args.initial_head, map_location="cpu", weights_only=False)
        if head.get("model_type") != "deepfense_broad_music_v104":
            raise ValueError("initial head is not a v104 checkpoint")
        model.backend.load_state_dict(head["backend"], strict=True)
        model.losses.load_state_dict(head["loss"], strict=True)
    model.frontend.requires_grad_(False)
    if not 0 <= args.unfreeze_last_blocks <= len(model.frontend.model.model.blocks):
        raise ValueError("invalid number of EAT blocks to unfreeze")
    frontend_parameters: list[nn.Parameter] = []
    if args.unfreeze_last_blocks:
        for block in model.frontend.model.model.blocks[-args.unfreeze_last_blocks:]:
            block.requires_grad_(True)
            frontend_parameters.extend(block.parameters())
    model.to(device)
    parameters = list(model.backend.parameters()) + list(model.losses.parameters())
    optimizer_groups = [{"params": parameters, "lr": args.learning_rate}]
    if frontend_parameters:
        optimizer_groups.append({
            "params": frontend_parameters, "lr": args.frontend_learning_rate,
        })
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_data = AudioRows(train, training=True, seed=args.seed)
    sampler = BalancedCellSampler(train, args.draws_per_cell, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, sampler=sampler,
        # Recreate workers each epoch so ``train_data.set_epoch`` changes the
        # deterministic crop seen by worker-side Dataset copies.
        num_workers=args.workers, pin_memory=True, persistent_workers=False,
        worker_init_fn=seed_worker, generator=generator,
    )
    development_loader = DataLoader(
        AudioRows(development, training=False, seed=args.seed),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed + 1),
    )

    history, best = [], None
    started = time.monotonic()
    for epoch in range(args.epochs + 1):
        if epoch:
            model.backend.train()
            model.losses.train()
            model.frontend.eval()
            sampler.set_epoch(epoch)
            train_data.set_epoch(epoch)
            losses = []
            for features, classes, _, _ in train_loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(features.to(device))
                    loss = criterion(logits.float(), classes.to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(parameters + frontend_parameters, 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            train_loss = float(np.mean(losses))
        else:
            train_loss = None
        metrics, predictions = evaluate(model, development_loader, device)
        record = {"epoch": epoch, "train_loss": train_loss, **metrics,
                  "seconds": time.monotonic() - started}
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if best is None or metrics["selection"] > best["selection"]:
            best = {"epoch": epoch, **metrics}
            torch.save({
                "model_type": "deepfense_broad_music_v104",
                "epoch": epoch, "selection": metrics["selection"],
                "backend": model.backend.state_dict(),
                "loss": model.losses.state_dict(),
                "frontend_blocks": {
                    name: value.detach().cpu()
                    for name, value in model.frontend.model.model.blocks.state_dict().items()
                    if args.unfreeze_last_blocks and int(name.split(".", 1)[0]) >= (
                        len(model.frontend.model.model.blocks) - args.unfreeze_last_blocks
                    )
                },
                "initial_checkpoint_sha256": sha256(args.initial_checkpoint),
                "config": vars(args),
            }, args.output / "best_head.pt")
            predictions.to_csv(args.output / "best_development_predictions.csv", index=False)
        pd.DataFrame([{k: v for k, v in row.items() if k != "domain_eers"}
                      for row in history]).to_csv(args.output / "history.csv", index=False)
        (args.output / "history.json").write_text(json.dumps(history, indent=2, default=str) + "\n")
    (args.output / "completed.json").write_text(json.dumps({
        "best": best, "train_rows": len(train), "development_rows": len(development),
        "train_domains": sorted(train.DATASET.unique()),
        "balanced_draws_per_epoch": len(sampler),
        "initial_checkpoint_sha256": sha256(args.initial_checkpoint),
    }, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
