"""Enforce the repository's data-role and identity-separation contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


IDENTITY_COLUMNS = (
    "ID", "GROUP_ID", "VOICE_SOURCE_ID", "MUSIC_SOURCE_ID", "SOURCE_FILE",
    # Two-party call banks name each voice explicitly.  Treat both source and
    # group identifiers as protected identities, just like the single-source
    # columns above, so a derived conversation cannot hide eval leakage.
    "FIRST_SOURCE_ID", "SECOND_SOURCE_ID", "FIRST_GROUP", "SECOND_GROUP",
    "MUSIC_GROUP", "MUSIC_GROUP_ID", "VOICE_SPEAKER", "PAIR_GROUP",
    "PARENT_ID",
)

TRAIN_ROLES = ("train", "router_train")
PROTECTED_FROM_TRAIN_ROLES = (
    "development", "locked_eval", "ood_holdout", "stress_eval",
    "retrospective_diagnostic", "invalid_eval",
)
ROW_ONLY_PROTECTED_ROLES = ("training_validation",)
# Stress banks intentionally reuse development parents to measure paired channel
# damage.  Retrospective banks are already contaminated by prior selection.  Both
# remain protected from training, but neither can support a source-disjoint claim.
OPTIMIZATION_ROLES = ("development", "training_validation")
OPTIMIZATION_DISJOINT_ROLES = ("locked_eval", "ood_holdout")


def identity_tokens(frame: pd.DataFrame) -> set[str]:
    tokens: set[str] = set()
    for column in IDENTITY_COLUMNS:
        if column in frame:
            values = frame[column].dropna().astype(str).str.strip()
            tokens.update(values.loc[values.ne("")])
    return tokens


def _load_config(partition_config: Path) -> tuple[dict, Path]:
    config = yaml.safe_load(partition_config.read_text("utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Partition config must be a mapping of roles to paths")
    config_dir = partition_config.resolve().parent
    root = config_dir.parent if config_dir.name == "configs" else config_dir
    return config, root


def _role_paths(config: dict, root: Path, role: str) -> list[Path]:
    declared = config.get(role, []) or []
    if not isinstance(declared, list) or not all(
        isinstance(path, str) for path in declared
    ):
        raise ValueError(f"Partition role {role!r} must be a list of paths")
    return [root / path for path in declared]


def _tokens_for_path(path: Path, role: str) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"{role} truth is missing: {path}")
    return identity_tokens(pd.read_csv(path, dtype=str))


def _sample_ids_for_path(path: Path, role: str) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"{role} truth is missing: {path}")
    frame = pd.read_csv(path, dtype=str)
    if "ID" not in frame:
        raise ValueError(f"{role} truth has no ID column: {path}")
    if frame["ID"].isna().any():
        raise ValueError(f"{role} truth contains missing IDs: {path}")
    values = frame["ID"].astype(str).str.strip()
    if values.eq("").any():
        raise ValueError(f"{role} truth contains blank IDs: {path}")
    if values.duplicated().any():
        duplicates = values.loc[values.duplicated()].unique()[:5].tolist()
        raise ValueError(f"{role} truth contains duplicate IDs {duplicates}: {path}")
    return set(values)


def assert_development_eval_separation(partition_config: Path) -> None:
    """Reject identity reuse between optimization and locked/OOD holdouts.

    ``stress_eval`` and ``retrospective_diagnostic`` are deliberately excluded
    from this cross-role assertion.  They may reuse development parents, but
    are still protected from training by :func:`assert_no_locked_eval_leakage`.
    """
    config, root = _load_config(partition_config)
    optimization = [
        (role, path, _tokens_for_path(path, role))
        for role in OPTIMIZATION_ROLES
        for path in _role_paths(config, root, role)
    ]
    violations: list[str] = []
    for role in OPTIMIZATION_DISJOINT_ROLES:
        for eval_path in _role_paths(config, root, role):
            eval_tokens = _tokens_for_path(eval_path, role)
            for optimization_role, optimization_path, tokens in optimization:
                overlap = sorted(tokens & eval_tokens)
                if overlap:
                    violations.append(
                        f"{optimization_role}:"
                        f"{optimization_path.relative_to(root)} <-> "
                        f"{role}:{eval_path.relative_to(root)}: "
                        f"{len(overlap)} tokens {overlap[:5]}"
                    )
    if violations:
        detail = "; ".join(violations[:20])
        if len(violations) > 20:
            detail += f"; ... {len(violations) - 20} more path pairs"
        raise ValueError(
            "DEVELOPMENT/EVAL IDENTITY OVERLAP: " + detail
        )


def assert_no_locked_eval_leakage(
    training_truth: Path,
    partition_config: Path,
) -> None:
    """Raise if training rows reuse any protected evaluation identity."""
    assert_development_eval_separation(partition_config)
    config, root = _load_config(partition_config)
    protected_paths = [
        (role, path)
        for role in PROTECTED_FROM_TRAIN_ROLES
        for path in _role_paths(config, root, role)
    ]
    row_only_paths = [
        (role, path)
        for role in ROW_ONLY_PROTECTED_ROLES
        for path in _role_paths(config, root, role)
    ]
    if not protected_paths and not row_only_paths:
        raise ValueError("Partition config must contain protected evaluation data")

    train_tokens = identity_tokens(pd.read_csv(training_truth, dtype=str))
    train_ids = _sample_ids_for_path(training_truth, "training")
    locked_tokens: set[str] = set()
    for role, path in protected_paths:
        locked_tokens.update(_tokens_for_path(path, role))
    overlap = sorted(train_tokens & locked_tokens)
    if overlap:
        raise ValueError(
            f"TRAIN/EVAL LEAKAGE: {len(overlap)} protected identity tokens reused: "
            f"{overlap[:10]}"
        )
    validation_ids: set[str] = set()
    for role, path in row_only_paths:
        validation_ids.update(_sample_ids_for_path(path, role))
    row_overlap = sorted(train_ids & validation_ids)
    if row_overlap:
        raise ValueError(
            "TRAIN/VALIDATION ROW LEAKAGE: "
            f"{len(row_overlap)} validation IDs reused: {row_overlap[:10]}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data_partitions.yaml"))
    parser.add_argument("--training-truth", type=Path, action="append", default=[])
    args = parser.parse_args()
    paths = args.training_truth
    if not paths:
        config, root = _load_config(args.config)
        paths = [
            path
            for role in TRAIN_ROLES
            for path in _role_paths(config, root, role)
        ]
    if not paths:
        raise ValueError("No training truth files were declared")
    for path in paths:
        assert_no_locked_eval_leakage(path, args.config)
        print(f"PASS: {path}")


if __name__ == "__main__":
    main()
