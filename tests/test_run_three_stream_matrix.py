from pathlib import Path

import pytest

from scripts.run_three_stream_matrix import build_command, load_variant_specs


def write_matrix(path: Path) -> None:
    path.write_text(
        "schema_version: test\n"
        "seed: 7\n"
        "train_core: [train_a, train_b]\n"
        "fixed_training:\n"
        "  epochs: 4\n"
        "  xlsr: true\n"
        "  residual_limits: [1.0, 1.0, 1.0]\n"
        "variants:\n"
        "  base:\n"
        "    train_set: train_core\n"
        "    ranking_weight: 0.1\n"
        "    generator_aware: true\n"
        "  no_xlsr:\n"
        "    train_set: train_core\n"
        "    xlsr: false\n"
        "    seed: 9\n",
        encoding="utf-8",
    )


def test_matrix_commands_are_predeclared_and_xlsr_is_explicit(tmp_path: Path):
    matrix_path = tmp_path / "matrix.yaml"
    write_matrix(matrix_path)
    _, specs = load_variant_specs(matrix_path)
    assert [item["name"] for item in specs] == ["base", "no_xlsr"]
    cache = tmp_path / "cache"
    output = tmp_path / "output"
    config = tmp_path / "partitions.yaml"
    base = build_command("python", specs[0], cache, output / "base", config)
    ablation = build_command(
        "python", specs[1], cache, output / "no_xlsr", config,
    )
    assert base[base.index("--train-datasets") + 1:base.index("--xlsr-cache-root")] == [
        "train_a", "train_b",
    ]
    assert "--xlsr-cache-root" in base
    assert "--xlsr-cache-root" not in ablation
    assert base[base.index("--seed") + 1] == "7"
    assert ablation[ablation.index("--seed") + 1] == "9"
    assert base[base.index("--ranking-weight") + 1] == "0.1"
    assert "--generator-aware" in base
    assert "--generator-aware" not in ablation


def test_matrix_rejects_post_hoc_or_unknown_parameters(tmp_path: Path):
    matrix_path = tmp_path / "matrix.yaml"
    write_matrix(matrix_path)
    source = matrix_path.read_text("utf-8").replace(
        "    ranking_weight: 0.1\n",
        "    ranking_weight: 0.1\n    score_selected_weight: 0.25\n",
    )
    matrix_path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported parameters"):
        load_variant_specs(matrix_path)


def test_matrix_variant_subset_must_be_known_and_unique(tmp_path: Path):
    matrix_path = tmp_path / "matrix.yaml"
    write_matrix(matrix_path)
    _, specs = load_variant_specs(matrix_path, ["no_xlsr"])
    assert [item["name"] for item in specs] == ["no_xlsr"]
    with pytest.raises(ValueError, match="unknown"):
        load_variant_specs(matrix_path, ["missing"])
    with pytest.raises(ValueError, match="unique"):
        load_variant_specs(matrix_path, ["base", "base"])


def test_matrix_resolves_a_predeclared_initial_checkpoint(tmp_path: Path):
    matrix_path = tmp_path / "matrix.yaml"
    write_matrix(matrix_path)
    checkpoint = tmp_path / "warm.pt"
    checkpoint.write_bytes(b"checkpoint")
    source = matrix_path.read_text("utf-8").replace(
        "  epochs: 4\n",
        f"  epochs: 4\n  initial_checkpoint: {checkpoint}\n",
    )
    matrix_path.write_text(source, encoding="utf-8")
    _, specs = load_variant_specs(matrix_path, ["base"])
    command = build_command(
        "python", specs[0], tmp_path / "cache", tmp_path / "output",
        tmp_path / "partitions.yaml",
    )
    assert command[command.index("--initial-checkpoint") + 1] == str(checkpoint)


def test_matrix_can_freeze_an_explicit_development_subset(tmp_path: Path):
    matrix_path = tmp_path / "matrix.yaml"
    write_matrix(matrix_path)
    source = matrix_path.read_text("utf-8").replace(
        "train_core: [train_a, train_b]\n",
        "train_core: [train_a, train_b]\n"
        "development_set: development_v1\n"
        "development_v1: [dev_a, dev_b]\n",
    )
    matrix_path.write_text(source, encoding="utf-8")
    _, specs = load_variant_specs(matrix_path, ["base"])
    assert specs[0]["development_set"] == "development_v1"
    assert specs[0]["development_datasets"] == ["dev_a", "dev_b"]
    command = build_command(
        "python", specs[0], tmp_path / "cache", tmp_path / "output",
        tmp_path / "partitions.yaml",
    )
    index = command.index("--dev-datasets")
    assert command[index + 1:command.index("--xlsr-cache-root")] == [
        "dev_a", "dev_b",
    ]


def test_matrix_passes_strict_identity_exclusions(tmp_path: Path):
    matrix_path = tmp_path / "matrix.yaml"
    write_matrix(matrix_path)
    exclusions = tmp_path / "exclude.txt"
    exclusions.write_text("source-a\n", encoding="utf-8")
    source = matrix_path.read_text("utf-8").replace(
        "seed: 7\n",
        f"seed: 7\ntrain_identity_exclusions: {exclusions}\n"
        "selected_split_only: true\n",
    )
    matrix_path.write_text(source, encoding="utf-8")
    _, specs = load_variant_specs(matrix_path, ["base"])
    command = build_command(
        "python", specs[0], tmp_path / "cache", tmp_path / "output",
        tmp_path / "partitions.yaml",
    )
    assert command[command.index("--train-identity-exclusions") + 1] == str(exclusions)
    assert "--selected-split-only" in command
