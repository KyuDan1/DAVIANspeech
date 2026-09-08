import copy
import importlib.util
from pathlib import Path
import sys
import csv

import numpy as np
import pandas as pd
import pytest
import torch

from src.three_stream_anchor_residual import (
    ThreeStreamAnchorResidualHead,
    anchor_residual_loss,
    batch_auc_ranking_loss,
    channel_consistency_losses,
    group_dro_loss,
    noisy_or_logit,
    quartet_component_invariance_losses,
    quartet_ranking_losses,
    robust_anchor_residual_loss,
)
from src.three_stream_anchor_residual_inference import (
    TASK_OUTPUT_INDICES,
    apply_task_mode,
    apply_three_stream_anchor_residual,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_three_stream_anchor_residual",
    ROOT / "scripts/train_three_stream_anchor_residual.py",
)
TRAINING = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = TRAINING
SPEC.loader.exec_module(TRAINING)


def inputs(batch: int = 3, *, with_xlsr: bool = True):
    temporal = torch.randn(batch, 2, 2, 2, 4, 8)
    spectral = torch.randn(batch, 2, 2, 2, 3, 8)
    eat_mask = torch.tensor([[1, 1], [1, 0], [1, 1]], dtype=torch.bool)[:batch]
    spear = torch.randn(batch, 2, 3, 2, 2, 4)
    spear_mask = eat_mask[:, :, None].expand(-1, -1, 3).clone()
    anchor_authenticity = torch.randn(batch, 3)
    anchor_presence = torch.randn(batch, 2)
    values = [
        temporal, spectral, eat_mask, spear, spear_mask,
        anchor_authenticity, anchor_presence,
    ]
    if with_xlsr:
        embeddings = torch.randn(batch, 3, 1920)
        mask = torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=torch.bool)[:batch]
        embeddings[~mask] = 0
        values.extend((embeddings, mask))
    return values


def model(
    *, with_xlsr: bool = True, task_interaction_depth: int = 0,
) -> ThreeStreamAnchorResidualHead:
    return ThreeStreamAnchorResidualHead(
        eat_layers=2,
        eat_dimension=8,
        spear_layers=2,
        spear_stats=2,
        spear_dimension=4,
        xlsr_dimension=1920 if with_xlsr else None,
        width=12,
        heads=3,
        depth=1,
        maximum_views=2,
        maximum_time_nodes=4,
        maximum_frequency_nodes=3,
        maximum_bins=3,
        maximum_xlsr_windows=3,
        dropout=0,
        residual_limits=(0.25, 0.5, 0.75),
        task_interaction_depth=task_interaction_depth,
    )


def test_fresh_head_is_exact_anchor_identity_and_keeps_presence_object():
    values = inputs()
    anchor_authenticity, anchor_presence = values[5:7]
    authenticity, presence, residual = model().eval()(*values)
    assert torch.equal(authenticity, anchor_authenticity)
    assert presence is anchor_presence
    assert torch.count_nonzero(residual) == 0


def test_task_interaction_is_zero_adapter_at_initialization_and_trains():
    head = model(task_interaction_depth=1)
    values = inputs()
    output, presence, residual = head(*values)
    torch.testing.assert_close(output, values[5], rtol=0, atol=0)
    torch.testing.assert_close(presence, values[6], rtol=0, atol=0)
    assert torch.count_nonzero(residual) == 0
    output.sum().backward()
    assert head.residual_heads[0][-1].weight.grad is not None


def test_task_interaction_warm_start_preserves_authorized_old_head(tmp_path):
    base = model()
    with torch.no_grad():
        for residual_head in base.residual_heads:
            residual_head[-1].weight.fill_(0.05)
    target = model(task_interaction_depth=1)
    train_truth = tmp_path / "train.csv"
    dev_truth = tmp_path / "dev.csv"
    old_config = {
        "eat_layers": 2, "eat_dimension": 8,
        "spear_layers": 2, "spear_stats": 2, "spear_dimension": 4,
        "xlsr_dimension": 1920, "width": 12, "heads": 3, "depth": 1,
        "maximum_views": 2, "maximum_time_nodes": 4,
        "maximum_frequency_nodes": 3, "maximum_bins": 3,
        "maximum_xlsr_windows": 3, "dropout": 0,
        "residual_limits": (0.25, 0.5, 0.75),
    }
    target_config = {**old_config, "task_interaction_depth": 1}
    checkpoint_path = tmp_path / "old.pt"
    torch.save({
        "model_type": "three_stream_anchor_residual_component_query_v1",
        "model": base.state_dict(), "config": old_config,
        "train_truths": [str(train_truth)],
        "development_truths": [str(dev_truth)],
    }, checkpoint_path)
    TRAINING.load_initial_checkpoint(
        target, target_config, checkpoint_path,
        [TRAINING.Partition("train", train_truth, "train")],
        [TRAINING.Partition("dev", dev_truth, "development")],
    )
    values = inputs()
    with torch.no_grad():
        expected = base.eval()(*values)[0]
        actual = target.eval()(*values)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_interaction_only_training_freezes_every_preexisting_parameter():
    head = model(task_interaction_depth=1)
    selected = TRAINING.configure_trainable_parameters(head, True)
    assert selected
    for name, parameter in head.named_parameters():
        assert parameter.requires_grad == name.startswith("task_interaction_")


def test_xlsr_stream_is_optional_only_when_disabled_in_config():
    values = inputs(with_xlsr=False)
    authenticity, presence, _ = model(with_xlsr=False)(*values)
    assert torch.equal(authenticity, values[5])
    assert presence is values[6]
    with pytest.raises(ValueError, match="requires embeddings and mask"):
        model(with_xlsr=True)(*values)


def test_forward_with_latent_keeps_identity_and_exposes_task_latents():
    values = inputs()
    authenticity, presence, residual, latent = model().eval().forward_with_latent(
        *values
    )
    assert torch.equal(authenticity, values[5])
    assert presence is values[6]
    assert torch.count_nonzero(residual) == 0
    assert latent.shape == (3, 3, 72)


def test_inference_task_mode_is_a_strict_single_output_intervention():
    anchor = torch.tensor([[1.0, 2.0, 3.0]])
    full = torch.tensor([[10.0, 20.0, 30.0]])
    residual = torch.tensor([[0.5, 0.25, 1.0]])
    torch.testing.assert_close(
        apply_task_mode(anchor, full, residual, "voice"),
        torch.tensor([[1.5, 2.0, 3.0]]),
    )
    torch.testing.assert_close(
        apply_task_mode(anchor, full, residual, "music"),
        torch.tensor([[1.0, 2.25, 3.0]]),
    )
    torch.testing.assert_close(
        apply_task_mode(anchor, full, residual, "file"),
        torch.tensor([[1.0, 2.0, 3.7]]),
    )
    assert apply_task_mode(anchor, full, residual, "full") is full
    assert TASK_OUTPUT_INDICES["voice"] == (0,)
    assert TASK_OUTPUT_INDICES["music"] == (1,)
    assert TASK_OUTPUT_INDICES["file"] == (2,)


def test_residuals_are_bounded_and_file_fusion_is_fixed_70_30():
    head = model(with_xlsr=False)
    anchor = torch.tensor([
        [-2.0, 0.5, -0.75],
        [1.5, -1.0, 0.25],
    ])
    raw = torch.tensor([[100.0, -100.0, 3.0], [-4.0, 5.0, -100.0]])
    final, direct = head.apply_residuals(anchor, raw)
    residual = head.bounded_residuals(raw)
    assert torch.all(residual.abs() <= head.residual_limits)
    expected_file = (
        head.FILE_DIRECT_WEIGHT * direct[:, 2]
        + head.FILE_COMPONENT_WEIGHT * noisy_or_logit(final[:, 0], final[:, 1])
    )
    torch.testing.assert_close(final[:, 2], expected_file)
    identity, _ = head.apply_residuals(anchor, torch.zeros_like(anchor))
    assert torch.equal(identity, anchor)


def test_all_three_streams_receive_gradients_after_residual_head_is_opened():
    head = model()
    with torch.no_grad():
        for residual_head in head.residual_heads:
            residual_head[-1].weight.fill_(0.1)
    authenticity, _, residuals = head(*inputs())
    fake = torch.tensor([[0, 0, 0], [1, 0, 1], [0, 1, 1.0]])
    presence = torch.tensor([[1, 1], [1, 0], [1, 1.0]])
    loss, terms = anchor_residual_loss(
        authenticity, fake, presence, torch.ones(3), residuals
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert set(terms) == {"voice", "music", "file", "residual"}
    assert head.eat_projection.weight.grad is not None
    assert head.spear_projection.weight.grad is not None
    assert head.xlsr_projection is not None
    assert head.xlsr_projection.weight.grad is not None


def quartet_targets():
    fake = torch.tensor([
        [0, 0, 0],  # RR
        [0, 1, 1],  # RF
        [1, 0, 1],  # FR
        [1, 1, 1],  # FF
    ], dtype=torch.float32)
    presence = torch.ones(4, 2)
    quartets = torch.tensor([[0, 1, 2, 3]])
    return fake, presence, quartets


def test_official_task_weights_and_presence_masks_are_exact():
    logits = torch.tensor([
        [0.4, -0.2, 0.1],
        [-0.7, 0.3, -0.5],
    ], requires_grad=True)
    fake = torch.tensor([[1, 1, 1], [0, 0, 0.0]])
    presence = torch.tensor([[1, 0], [1, 1.0]])
    sample_weight = torch.ones(2)
    loss, terms = anchor_residual_loss(
        logits, fake, presence, sample_weight, torch.zeros_like(logits),
        residual_weight=0,
    )
    expected = 0.2 * terms["voice"] + 0.3 * terms["music"] + 0.5 * terms["file"]
    torch.testing.assert_close(loss, expected)
    # The absent Music target on row zero must not affect its task loss.
    changed = fake.clone()
    changed[0, 1] = 0
    changed_loss, _ = anchor_residual_loss(
        logits, changed, presence, sample_weight, torch.zeros_like(logits),
        residual_weight=0,
    )
    torch.testing.assert_close(loss, changed_loss)


def test_strict_identity_exclusion_filters_frame_and_every_cache_stream():
    frame = pd.DataFrame({
        "ID": ["a", "b"], "DATASET": ["d", "d"],
        "MUSIC_SOURCE_ID": ["keep", "leaked"],
    })
    block = {"x": np.asarray([[1.], [2.]]), "mask": np.asarray([True, False])}
    filtered, removed = TRAINING.filter_train_pair(
        (frame, block, {"schema": "test"}), ("leaked",),
    )
    selected_frame, selected_block, metadata = filtered
    assert removed == ["b"]
    assert selected_frame.ID.tolist() == ["a"]
    assert selected_block["x"].tolist() == [[1.]]
    assert selected_block["mask"].tolist() == [True]
    assert metadata == {"schema": "test"}


def test_quartet_ranking_is_task_specific_and_strictly_validated():
    fake, presence, quartets = quartet_targets()
    good = torch.tensor([
        [-2, -2, -2], [-2, 2, 2], [2, -2, 2], [2, 2, 2.0],
    ])
    bad = -good
    good_terms = quartet_ranking_losses(good, fake, presence, quartets)
    bad_terms = quartet_ranking_losses(bad, fake, presence, quartets)
    assert all(good_loss < bad_loss for good_loss, bad_loss in zip(good_terms, bad_terms))
    malformed = torch.tensor([[0, 2, 1, 3]])
    with pytest.raises(ValueError, match="RR, RF, FR, FF"):
        quartet_ranking_losses(good, fake, presence, malformed)


def test_quartet_component_invariance_ignores_file_and_penalizes_leakage():
    fake, presence, quartets = quartet_targets()
    # Voice is invariant to Music origin and Music is invariant to Voice
    # origin. File deliberately varies and must not enter this objective.
    disentangled = torch.tensor([
        [-2, -3, -9], [-2, 4, 9], [5, -3, 20], [5, 4, -20.0],
    ])
    leaked = disentangled.clone()
    leaked[1, 0] = 3
    leaked[2, 1] = 2
    latent = torch.zeros(4, 3, 5)
    weights = torch.tensor([0.2, 0.3, 0.5])
    good_logit, good_latent = quartet_component_invariance_losses(
        disentangled, latent, fake, presence, quartets, weights,
    )
    bad_logit, bad_latent = quartet_component_invariance_losses(
        leaked, latent, fake, presence, quartets, weights,
    )
    assert good_logit == 0
    assert good_latent == 0
    assert bad_logit > good_logit
    assert bad_latent == 0

    latent[1, 0] = 2
    _, leaked_latent = quartet_component_invariance_losses(
        disentangled, latent, fake, presence, quartets, weights,
    )
    assert leaked_latent > 0


def test_batch_auc_ranking_directly_rewards_correct_eer_ordering():
    fake, presence, _ = quartet_targets()
    weights = torch.tensor([0.2, 0.3, 0.5])
    ordered = torch.tensor([
        [-3, -3, -3], [-2, 3, 2], [3, -2, 3], [2, 2, 4.0],
    ])
    reversed_logits = -ordered
    good = batch_auc_ranking_loss(
        ordered, fake, presence, torch.ones(4), weights,
    )
    bad = batch_auc_ranking_loss(
        reversed_logits, fake, presence, torch.ones(4), weights,
    )
    assert good < bad
    # A missing component is masked exactly like the official component EER.
    masked_presence = presence.clone()
    masked_presence[:, 1] = 0
    changed_music = ordered.clone()
    changed_music[:, 1] = -changed_music[:, 1]
    reference = batch_auc_ranking_loss(
        ordered, fake, masked_presence, torch.ones(4), weights,
    )
    changed = batch_auc_ranking_loss(
        changed_music, fake, masked_presence, torch.ones(4), weights,
    )
    torch.testing.assert_close(reference, changed)


def test_channel_consistency_covers_logits_and_component_query_latents():
    fake = torch.tensor([[0, 1, 1], [0, 1, 1.0]])
    presence = torch.ones(2, 2)
    logits = torch.tensor([[0.0, 0.0, 0.0], [1.0, -1.0, 2.0]])
    latent = torch.zeros(2, 3, 4)
    latent[1] = 2
    weights = torch.tensor([0.2, 0.3, 0.5])
    logit_loss, latent_loss = channel_consistency_losses(
        logits, latent, fake, presence, torch.tensor([[0, 1]]), weights
    )
    assert logit_loss > 0
    assert latent_loss > 0
    inconsistent = fake.clone()
    inconsistent[1, 2] = 0
    with pytest.raises(ValueError, match="authenticity targets differ"):
        channel_consistency_losses(
            logits, latent, inconsistent, presence,
            torch.tensor([[0, 1]]), weights,
        )


def test_channel_consistency_uses_clean_side_as_stop_gradient_teacher():
    fake = torch.tensor([[1, 0, 1], [1, 0, 1.0]])
    presence = torch.ones(2, 2)
    logits = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, -1.0, 2.0]], requires_grad=True
    )
    latent = torch.stack((torch.zeros(3, 2), torch.ones(3, 2))).requires_grad_()
    logit_loss, latent_loss = channel_consistency_losses(
        logits, latent, fake, presence, torch.tensor([[0, 1]]),
        torch.tensor([0.2, 0.3, 0.5]),
    )
    (logit_loss + latent_loss).backward()
    assert torch.count_nonzero(logits.grad[0]) == 0
    assert torch.count_nonzero(latent.grad[0]) == 0
    assert torch.count_nonzero(logits.grad[1]) > 0
    assert torch.count_nonzero(latent.grad[1]) > 0


def test_group_dro_is_worst_group_softmax_and_missing_metadata_is_zero():
    logits = torch.tensor([[5.0, 5.0, 5.0], [-5.0, -5.0, -5.0]])
    fake = torch.ones(2, 3)
    presence = torch.ones(2, 2)
    sample_weight = torch.ones(2)
    task_weights = torch.tensor([0.2, 0.3, 0.5])
    hard_worst = group_dro_loss(
        logits, fake, presence, sample_weight, torch.tensor([0, 1]),
        task_weights, temperature=0,
    )
    expected_worst = torch.nn.functional.softplus(torch.tensor(5.0))
    torch.testing.assert_close(hard_worst, expected_worst)
    soft_worst = group_dro_loss(
        logits, fake, presence, sample_weight, torch.tensor([0, 1]),
        task_weights, temperature=0.2,
    )
    assert soft_worst < hard_worst
    assert soft_worst > torch.nn.functional.softplus(torch.tensor(-5.0))
    assert group_dro_loss(
        logits, fake, presence, sample_weight, None, task_weights, 0.2,
    ).item() == 0


def test_normalized_available_ads_keeps_single_component_domains_in_selection():
    speech_only = {
        "FILE_EER": 0.10, "VOICE_EER": 0.20, "MUSIC_EER": np.nan,
    }
    music_only = {
        "FILE_EER": 0.10, "VOICE_EER": np.nan, "MUSIC_EER": 0.30,
    }
    expected_speech = (0.5 * 0.90 + 0.2 * 0.80) / 0.7
    expected_music = (0.5 * 0.90 + 0.3 * 0.70) / 0.8
    assert TRAINING.normalized_available_ads(speech_only) == pytest.approx(
        expected_speech
    )
    assert TRAINING.normalized_available_ads(music_only) == pytest.approx(
        expected_music
    )
    assert np.isnan(TRAINING.normalized_available_ads({
        "FILE_EER": np.nan, "VOICE_EER": np.nan, "MUSIC_EER": np.nan,
    }))


def test_full_robust_loss_reports_every_auxiliary_and_backpropagates():
    fake, presence, quartets = quartet_targets()
    logits = torch.zeros(4, 3, requires_grad=True)
    residual = torch.zeros(4, 3, requires_grad=True)
    latent = torch.randn(4, 3, 5, requires_grad=True)
    total, terms = robust_anchor_residual_loss(
        logits, fake, presence, torch.ones(4), residual,
        latent=latent,
        quartets=quartets,
        channel_pairs=torch.empty((0, 2), dtype=torch.long),
        environment_ids=torch.tensor([0, 0, 1, 1]),
    )
    assert set(terms) == {
        "voice", "music", "file", "residual", "rank_voice", "rank_music",
        "rank_file", "ranking", "logit_consistency", "latent_consistency",
        "quartet_invariance", "quartet_latent_invariance", "auc_ranking",
        "group_dro",
    }
    assert torch.isfinite(total)
    total.backward()
    assert logits.grad is not None


def robustness_frame() -> pd.DataFrame:
    rows = []
    for index, (voice, music) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        rows.append({
            "ID": f"q{index}", "DATASET": "quartet", "PAIR_GROUP": "q",
            "FILE_FAKE": max(voice, music), "VOICE_FAKE": voice,
            "MUSIC_FAKE": music, "VOICE_PRESENT": 1, "MUSIC_PRESENT": 1,
            "MIX_MODE": "sequential", "CHANNEL": "clean",
            "MIXTURE_ID": np.nan, "PARENT_ID": np.nan,
        })
    for index, channel in enumerate(("clean", "g711", "opus")):
        rows.append({
            "ID": f"m{index}", "DATASET": "channel", "PAIR_GROUP": np.nan,
            "FILE_FAKE": 1, "VOICE_FAKE": 1, "MUSIC_FAKE": 0,
            "VOICE_PRESENT": 1, "MUSIC_PRESENT": 1,
            "MIX_MODE": "concurrent", "CHANNEL": channel,
            "MIXTURE_ID": "mixture", "PARENT_ID": np.nan if index == 0 else "m0",
        })
    rows.append({
        "ID": "single", "DATASET": "plain", "PAIR_GROUP": np.nan,
        "FILE_FAKE": 0, "VOICE_FAKE": np.nan, "MUSIC_FAKE": 0,
        "VOICE_PRESENT": 0, "MUSIC_PRESENT": 1,
        "MIX_MODE": np.nan, "CHANNEL": np.nan,
        "MIXTURE_ID": np.nan, "PARENT_ID": np.nan,
    })
    return pd.DataFrame(rows)


def test_group_balanced_sampler_keeps_quartets_and_channel_groups_atomic():
    frame = robustness_frame()
    quartets = TRAINING.complete_pair_group_quartets(frame)
    channel_groups, pairs = TRAINING.channel_consistency_groups(frame)
    assert quartets.tolist() == [[0, 1, 2, 3]]
    assert len(channel_groups) == 1  # PARENT_ID/MIXTURE_ID descriptions deduplicate.
    assert pairs.tolist() == [[4, 5], [4, 6]]
    batches = TRAINING.group_balanced_batches(
        frame, 8, np.random.default_rng(7), quartets, channel_groups,
        epoch_size=40,
    )
    for batch in batches:
        for group in (quartets[0], channel_groups[-1]):
            counts = [int(np.count_nonzero(batch == index)) for index in group]
            assert len(set(counts)) == 1
        TRAINING.remap_complete_groups(batch, quartets)
        TRAINING.remap_complete_groups(batch, pairs)


def test_missing_robustness_metadata_skips_only_auxiliary_groups():
    frame = robustness_frame().drop(
        columns=["PAIR_GROUP", "MIXTURE_ID", "PARENT_ID", "MIX_MODE", "CHANNEL"]
    )
    assert TRAINING.complete_pair_group_quartets(frame).shape == (0, 4)
    groups, pairs = TRAINING.channel_consistency_groups(frame)
    assert groups == []
    assert pairs.shape == (0, 2)
    assert np.all(TRAINING.environment_codes(frame) == -1)
    batches = TRAINING.group_balanced_batches(
        frame, 4, np.random.default_rng(1), np.empty((0, 4), np.int64), [],
    )
    assert batches


def test_generator_aware_environments_and_sampling_preserve_generator_identity():
    frame = robustness_frame()
    frame["VOICE_GENERATOR"] = [
        "real", "real", "tts_a", "tts_a", "tts_b", "tts_b", "tts_b", "real",
    ]
    frame["MUSIC_GENERATOR"] = [
        "real", "music_a", "real", "music_a", "real", "real", "real", "music_b",
    ]
    ordinary = TRAINING.environment_codes(frame)
    aware = TRAINING.environment_codes(frame, generator_aware=True)
    assert len(set(aware[aware >= 0])) > len(set(ordinary[ordinary >= 0]))
    metadata = TRAINING._sampling_metadata(frame, generator_aware=True)
    quartet = np.asarray([0, 1, 2, 3])
    assert TRAINING._unit_stratum(
        metadata, quartet, generator_aware=True,
    )[-1].startswith("paired:")
    signature = TRAINING._generator_signature(frame)
    assert signature.iloc[0] == "voice=real|music=real"


def test_component_split_contract_checks_both_mixed_sides_and_parents():
    train = pd.DataFrame([
        {
            "ID": "train", "VOICE_PRESENT": 1, "MUSIC_PRESENT": 1,
            "VOICE_SOURCE_ID": "voice_train", "VOICE_GENERATOR": "tts_a",
            "MUSIC_SOURCE_ID": "music_train", "MUSIC_GENERATOR": "suno",
            "PARENT_ID": "parent_train",
        },
    ])
    development = pd.DataFrame([
        {
            "ID": "dev", "VOICE_PRESENT": 1, "MUSIC_PRESENT": 1,
            "VOICE_SOURCE_ID": "voice_dev", "VOICE_GENERATOR": "tts_a",
            "MUSIC_SOURCE_ID": "music_dev", "MUSIC_GENERATOR": "udio",
            "PARENT_ID": "parent_dev",
        },
    ])
    report = TRAINING.assert_component_split_contract(train, development)
    assert report["overlap"] == {"voice": 0, "music": 0, "parent": 0}
    assert report["generator_signatures"]["policy"].startswith("domain_overlap")

    for column, value in (
        ("VOICE_SOURCE_ID", "voice_train"),
        ("MUSIC_SOURCE_ID", "music_train"),
        ("PARENT_ID", "parent_train"),
    ):
        leaked = development.copy()
        leaked.loc[0, column] = value
        with pytest.raises(ValueError, match="COMPONENT IDENTITY LEAKAGE"):
            TRAINING.assert_component_split_contract(train, leaked)


def test_component_split_contract_rejects_unidentified_mixed_component():
    complete = pd.DataFrame([
        {
            "ID": "complete", "VOICE_PRESENT": 1, "MUSIC_PRESENT": 1,
            "VOICE_SOURCE_ID": "voice", "MUSIC_SOURCE_ID": "music",
        },
    ])
    missing = complete.copy()
    missing["MUSIC_SOURCE_ID"] = ""
    with pytest.raises(ValueError, match="lack component identity"):
        TRAINING.assert_component_split_contract(missing, complete)


def test_full_numpy_cache_validation_checks_cross_stream_alignment():
    values = inputs(batch=2)
    block = {
        "temporal": values[0].numpy(),
        "spectral": values[1].numpy(),
        "eat_mask": values[2].numpy(),
        "spear": values[3].numpy(),
        "spear_mask": values[4].numpy(),
        "anchor_authenticity": values[5].numpy(),
        "anchor_presence": values[6].numpy(),
        "xlsr_embeddings": values[7].numpy(),
        "xlsr_mask": values[8].numpy(),
    }
    TRAINING.validate_numpy_block(block)
    broken = {key: value.copy() for key, value in block.items()}
    broken["spear_mask"][1, 0] = False
    with pytest.raises(ValueError, match="at least one valid SPEAR"):
        TRAINING.validate_numpy_block(broken)

    # Extractors may legitimately disagree about whether a non-primary stem
    # is usable. The two attention streams own independent masks.
    independent = {key: value.copy() for key, value in block.items()}
    independent["spear_mask"][0, 1] = False
    TRAINING.validate_numpy_block(independent)


def test_masked_xlsr_padding_cannot_change_outputs():
    head = model().eval()
    with torch.no_grad():
        for residual_head in head.residual_heads:
            residual_head[-1].weight.normal_()
    values = inputs()
    with torch.no_grad():
        reference = head(*values)[0]
        values[7][1, 1:] = 1_000
        changed = head(*values)[0]
    torch.testing.assert_close(reference[1], changed[1])


def test_alignment_and_mask_contracts_fail_loudly():
    head = model()
    values = inputs()
    values[4][1, 0] = False
    with pytest.raises(ValueError, match="at least one valid SPEAR"):
        head(*values)

    values = inputs()
    values[4][0, 1] = False
    head(*values)  # Independent nonempty EAT/SPEAR masks are supported.

    values = inputs()
    values[8][0] = torch.tensor([True, False, True])
    with pytest.raises(ValueError, match="prefix"):
        head(*values)

    values = inputs()
    values[8] = values[8].long()
    with pytest.raises(TypeError, match="torch.bool"):
        head(*values)

    values = inputs()
    values[7] = torch.randn(3, 3, 1919)
    with pytest.raises(ValueError, match="1920"):
        head(*values)


def write_xlsr(path: Path, **overrides) -> None:
    mask = np.asarray([[True, True, False], [True, False, False]])
    embeddings = np.random.default_rng(2).normal(size=(2, 3, 1920)).astype(np.float16)
    embeddings[~mask] = 0
    payload = {
        "ids": np.asarray(["b", "a"]),
        "embeddings": embeddings,
        "mask": mask,
        "starts": np.asarray([[0, 64_000, -1], [0, -1, -1]], dtype=np.int64),
        "window": np.asarray(64_000, dtype=np.int64),
        "sample_rate": np.asarray(16_000, dtype=np.int64),
    }
    payload.update(overrides)
    np.savez_compressed(path, **payload)


def test_xlsr_schema_loader_reorders_ids_and_validates_metadata(tmp_path):
    path = tmp_path / "windows.npz"
    write_xlsr(path)
    result = TRAINING.load_xlsr_window_cache(path, np.asarray(["a", "b"]))
    assert result["embeddings"].shape == (2, 3, 1920)
    assert result["mask"].tolist() == [
        [True, False, False], [True, True, False],
    ]
    assert result["window"] == 64_000
    assert result["sample_rate"] == 16_000


def test_per_dataset_xlsr_windows_are_globally_zero_padded_before_concat():
    first = {
        "xlsr_embeddings": np.ones((2, 2, 1920), dtype=np.float32),
        "xlsr_mask": np.ones((2, 2), dtype=np.bool_),
    }
    second = {
        "xlsr_embeddings": np.ones((1, 4, 1920), dtype=np.float32),
        "xlsr_mask": np.asarray([[True, True, False, False]]),
    }
    second["xlsr_embeddings"][:, 2:] = 0
    TRAINING.pad_xlsr_window_axis([first, second])
    assert first["xlsr_embeddings"].shape == (2, 4, 1920)
    assert first["xlsr_mask"].shape == (2, 4)
    assert not first["xlsr_mask"][:, 2:].any()
    assert np.count_nonzero(first["xlsr_embeddings"][:, 2:]) == 0
    assert second["xlsr_embeddings"].shape == (1, 4, 1920)


def write_spear(path: Path, *, bins: int = 3, projection_scale: float = 1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=np.asarray(["a", "b"]),
        features=np.zeros((2, 2, bins, 2, 2, 4), dtype=np.float16),
        mask=np.ones((2, 2, bins), dtype=np.bool_),
        projection=np.eye(4, dtype=np.float32) * projection_scale,
        layers=np.asarray([1, 3], dtype=np.int64),
        bins=np.asarray(bins, dtype=np.int64),
    )


def test_per_dataset_spear_cache_loader_and_union_fallback(tmp_path, monkeypatch):
    write_spear(tmp_path / "first/features.npz")
    write_spear(tmp_path / "second/features.npz")
    cache = TRAINING.load_spear_caches(tmp_path, ["first", "second"])
    assert cache["first"]["features"].shape == (2, 2, 3, 2, 2, 4)
    assert cache["__metadata__"]["bins"].item() == 3

    empty = tmp_path / "union"
    empty.mkdir()
    sentinel = {"union": {}}
    monkeypatch.setattr(TRAINING, "load_spear_archive", lambda root: sentinel)
    assert TRAINING.load_spear_caches(empty, ["first"]) is sentinel


def test_per_dataset_spear_cache_root_must_be_complete_and_consistent(tmp_path):
    write_spear(tmp_path / "first/features.npz")
    with pytest.raises(FileNotFoundError, match="incomplete"):
        TRAINING.load_spear_caches(tmp_path, ["first", "second"])
    write_spear(tmp_path / "second/features.npz", projection_scale=2.0)
    with pytest.raises(ValueError, match="metadata differs"):
        TRAINING.load_spear_caches(tmp_path, ["first", "second"])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"embeddings": np.zeros((2, 3, 1919), np.float16)}, "1920"),
        ({"mask": np.asarray([[True, False, True], [True, False, False]])}, "prefix"),
        ({"starts": np.asarray([[0, -1, -1], [0, -1, -1]])}, "mask must equal"),
        ({"starts": np.asarray([[64_000, 0, -1], [0, -1, -1]])}, "increasing"),
    ],
)
def test_xlsr_schema_loader_rejects_malformed_archives(tmp_path, overrides, message):
    path = tmp_path / "bad.npz"
    write_xlsr(path, **overrides)
    with pytest.raises((ValueError, TypeError), match=message):
        TRAINING.load_xlsr_window_cache(path, np.asarray(["b", "a"]))


@pytest.mark.parametrize("ensemble", [False, True])
def test_zero_three_stream_inference_preserves_anchor_and_presence(
    tmp_path, ensemble,
):
    head = model().eval()
    config = {
        "eat_layers": 2, "eat_dimension": 8,
        "spear_layers": 2, "spear_stats": 2, "spear_dimension": 4,
        "xlsr_dimension": 1920, "width": 12, "heads": 3, "depth": 1,
        "maximum_views": 2, "maximum_time_nodes": 4,
        "maximum_frequency_nodes": 3, "maximum_bins": 3,
        "maximum_xlsr_windows": 3, "dropout": 0,
        "residual_limits": (0.25, 0.5, 0.75),
    }
    projection = np.eye(4, dtype=np.float32)
    eat_layers = np.asarray([1, 3], dtype=np.int64)
    spear_layers = np.asarray([2, 4], dtype=np.int64)
    checkpoint = {
        "model_type": "three_stream_anchor_residual_component_query_v1",
        "model": head.state_dict(), "config": config,
        "feature_metadata": {
            "eat_projection": projection, "eat_layers": eat_layers,
            "xlsr_window": 64_000, "xlsr_sample_rate": 16_000,
        },
        "spear_projection": projection,
        "spear_layers": spear_layers,
        "spear_bins": np.asarray(3, dtype=np.int64),
    }
    checkpoint_path = tmp_path / "head.pt"
    torch.save(checkpoint, checkpoint_path)
    checkpoint_paths = checkpoint_path
    if ensemble:
        second = copy.deepcopy(checkpoint)
        second["config"]["residual_limits"] = (1.0, 1.0, 1.0)
        second["config"]["task_interaction_depth"] = 1
        second["model"] = ThreeStreamAnchorResidualHead(
            **second["config"]
        ).state_dict()
        second_path = tmp_path / "head_second.pt"
        torch.save(second, second_path)
        checkpoint_paths = [checkpoint_path, second_path]

    values = inputs()
    ids = np.asarray(["a", "b", "c"])
    eat_path, spear_path, xlsr_path = (
        tmp_path / "eat.npz", tmp_path / "spear.npz", tmp_path / "xlsr.npz"
    )
    np.savez_compressed(
        eat_path, ids=ids, temporal=values[0].numpy(),
        spectral=values[1].numpy(), view_mask=values[2].numpy(),
        projection=projection, layers=eat_layers,
    )
    spear_order = np.asarray([2, 0, 1])
    np.savez_compressed(
        spear_path, ids=ids[spear_order], features=values[3].numpy()[spear_order],
        mask=values[4].numpy()[spear_order], projection=projection,
        layers=spear_layers, bins=np.asarray(3, dtype=np.int64),
    )
    xlsr_order = np.asarray([1, 2, 0])
    np.savez_compressed(
        xlsr_path, ids=ids[xlsr_order],
        embeddings=values[7].numpy()[xlsr_order],
        mask=values[8].numpy()[xlsr_order],
        starts=np.asarray([[0, 64_000, 128_000], [0, -1, -1], [0, 64_000, -1]])[
            xlsr_order
        ],
        window=np.asarray(64_000, dtype=np.int64),
        sample_rate=np.asarray(16_000, dtype=np.int64),
    )
    submission = tmp_path / "submission.csv"
    rows = [
        {
            "ID": item, "FILE_FAKE_PROB": str(0.2 + index * 0.1),
            "VOICE_FAKE_PROB": str(0.3 + index * 0.1),
            "MUSIC_FAKE_PROB": str(0.4 + index * 0.1),
            "VOICE_PRESENT_PROB": str(0.61 + index * 0.01),
            "MUSIC_PRESENT_PROB": str(0.71 + index * 0.01),
        }
        for index, item in enumerate(ids)
    ]
    with submission.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    apply_three_stream_anchor_residual(
        submission, eat_path, spear_path, xlsr_path,
        checkpoint_paths,
        device="cpu", batch_size=2,
    )
    result = pd.read_csv(submission, dtype={"ID": str})
    expected = pd.DataFrame(rows).set_index("ID")
    actual = result.set_index("ID")
    for column in (
        "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    ):
        np.testing.assert_allclose(
            actual[column].astype(float), expected[column].astype(float),
            atol=2e-7, rtol=0,
        )


def test_partition_hook_only_accepts_declared_train_and_development(tmp_path):
    config = tmp_path / "partitions.yaml"
    config.write_text(
        "train:\n  - data/train_a/truth.csv\n"
        "development:\n  - data/dev_a/truth.csv\n"
        "locked_eval:\n  - data/locked/truth.csv\n",
        encoding="utf-8",
    )
    train = TRAINING.authorized_partitions(config, "train", ["train_a"])
    development = TRAINING.authorized_partitions(
        config, "development", ["dev_a"]
    )
    assert [item.name for item in train] == ["train_a"]
    assert [item.name for item in development] == ["dev_a"]
    with pytest.raises(ValueError, match="not allowed"):
        TRAINING.authorized_partitions(config, "locked_eval")
    with pytest.raises(ValueError, match="not declared"):
        TRAINING.authorized_partitions(config, "train", ["locked"])


def test_guard_hook_invokes_data_guard_for_every_training_truth(monkeypatch, tmp_path):
    train = [TRAINING.Partition("a", tmp_path / "a.csv", "train")]
    development = [
        TRAINING.Partition("b", tmp_path / "b.csv", "development")
    ]
    calls = []
    monkeypatch.setattr(
        TRAINING, "assert_development_eval_separation",
        lambda config: calls.append(("development", config)),
    )
    monkeypatch.setattr(
        TRAINING, "assert_no_locked_eval_leakage",
        lambda truth, config: calls.append((truth, config)),
    )
    config = tmp_path / "partitions.yaml"
    TRAINING.guard_partitions(train, development, config)
    assert calls == [("development", config), (train[0].truth_path, config)]
