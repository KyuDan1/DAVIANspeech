import csv
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.eat_large_aasist import EatLargeAASISTExpert, multitask_loss
from src.eat_large_aasist_data import (
    component_protected_tokens, crop_targets, group_split, split_plan,
)
from src.eat_large_aasist_inference import (
    PROTECTED_COLUMNS, apply_eat_large_aasist_fusion, audio_views, fbank,
    view_starts,
)


class FakePatchEmbed(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.proj = nn.Conv2d(1, width, kernel_size=2, stride=2)

    def forward(self, values):
        return self.proj(values).flatten(2).transpose(1, 2)


class FakeBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, values):
        output = self.linear(values)
        return output, None


class FakeEat(nn.Module):
    def __init__(self, width: int = 16, depth: int = 6):
        super().__init__()
        self.model = nn.Module()
        self.model.local_encoder = FakePatchEmbed(width)
        self.model.extra_tokens = nn.Parameter(torch.zeros(1, 1, width))
        self.model.pre_norm = nn.LayerNorm(width)
        self.model.pos_drop = nn.Identity()
        self.model.fixed_positional_encoder = None
        self.model.blocks = nn.ModuleList([FakeBlock(width) for _ in range(depth)])


def make_model():
    return EatLargeAASISTExpert(
        FakeEat(), capture_layers=(1, 3, 4, 5), adapter_blocks=(4, 5),
        bottleneck=4, graph_width=8, graph_hidden=4, branches=1, dropout=0,
    )


def test_forward_and_gradient_are_separation_free():
    model = make_model()
    assert all(not parameter.requires_grad for parameter in model.eat.parameters())
    features = torch.randn(3, 1, 8, 4)
    logits = model(features)
    assert logits.shape == (3, 3)
    targets = torch.tensor([[0., 0., 0.], [1., 0., 1.], [0., 1., 1.]])
    presence = torch.ones(3, 2)
    loss, terms = multitask_loss(model, logits, targets, presence)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(terms) == {"bce", "ranking", "component_or", "voice", "music", "file"}
    assert model.adapters["4"].up.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.eat.parameters())


def test_multiview_and_adaptive_view_contract():
    model = make_model().eval()
    features = torch.randn(2, 3, 1, 8, 4)
    mask = torch.tensor([[True, False, False], [True, True, True]])
    assert model(features, mask).shape == (2, 3)
    assert len(view_starts(10 * 16_000)) == 1
    assert len(view_starts(20 * 16_000)) == 2
    assert len(view_starts(40 * 16_000)) == 3
    views, valid = audio_views(np.zeros(4 * 16_000, dtype=np.float32))
    assert views.shape == (3, 1024, 128)
    assert valid.tolist() == [True, False, False]


def test_pure_torch_fbank_does_not_require_torchaudio():
    audio = np.random.default_rng(0).normal(0, .05, 163_840).astype(np.float32)
    features = fbank(audio)
    assert features.shape == (1024, 128)
    assert torch.isfinite(features).all()


def test_pure_torch_fbank_matches_previous_kaldi_features():
    import torchaudio  # reference only; v56 runtime module does not import it

    audio = np.random.default_rng(2).normal(0, .05, 163_840).astype(np.float32)
    waveform = torch.from_numpy(audio)
    waveform -= waveform.mean()
    reference = torchaudio.compliance.kaldi.fbank(
        waveform[None], htk_compat=True, sample_frequency=16_000,
        use_energy=False, window_type="hanning", num_mel_bins=128,
        dither=0.0, frame_shift=10,
    )
    reference = torch.nn.functional.pad(reference, (0, 0, 0, 1024 - len(reference)))
    reference = (reference + 4.268) / (4.569 * 2)
    actual = fbank(audio)
    assert torch.allclose(actual, reference, atol=3e-5, rtol=1e-5)

    tree = ast.parse(Path("src/eat_large_aasist_inference.py").read_text())
    imports = {
        alias.name for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "torchaudio" not in imports


def test_file_music_fusion_preserves_voice_and_cps_strings(tmp_path):
    submission = tmp_path / "submission.csv"
    columns = [
        "ID", "FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB",
        "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB",
    ]
    original = {
        "ID": "x", "FILE_FAKE_PROB": "0.2", "VOICE_FAKE_PROB": "0.123456789",
        "MUSIC_FAKE_PROB": "0.3", "VOICE_PRESENT_PROB": "0.987654321",
        "MUSIC_PRESENT_PROB": "0.777777777",
    }
    with submission.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(original)
    statistics = tmp_path / "v56.npz"
    np.savez(statistics, ids=np.asarray(["x"]), probabilities=np.asarray([[.8, .9, .85]]))
    apply_eat_large_aasist_fusion(submission, statistics)
    with submission.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert tuple(row[column] for column in PROTECTED_COLUMNS) == tuple(
        original[column] for column in PROTECTED_COLUMNS
    )
    assert float(row["FILE_FAKE_PROB"]) != float(original["FILE_FAKE_PROB"])
    assert float(row["MUSIC_FAKE_PROB"]) != float(original["MUSIC_FAKE_PROB"])


def test_crop_targets_recompute_sequential_file_label():
    row = pd.Series({
        "VOICE_PRESENT": 1, "VOICE_FAKE": 1, "VOICE_START": 0., "VOICE_END": 4.,
        "MUSIC_PRESENT": 1, "MUSIC_FAKE": 0, "MUSIC_START": 6., "MUSIC_END": 12.,
        "DURATION": 12.,
    })
    fake, presence = crop_targets(row, 6., 10.)
    assert fake.tolist() == [0., 0., 0.]
    assert presence.tolist() == [0., 1.]
    fake, presence = crop_targets(row, 0., 4.)
    assert fake.tolist() == [1., 0., 1.]
    assert presence.tolist() == [1., 0.]


def test_group_split_keeps_generator_and_identity_disjoint():
    frame = pd.DataFrame([
        {"MUSIC_PRESENT": 1, "MUSIC_FAKE": 1, "MUSIC_GENERATOR": "MusicGen_medium"},
        {"MUSIC_PRESENT": 1, "MUSIC_FAKE": 1, "MUSIC_GENERATOR": "musicgen"},
        {"MUSIC_PRESENT": 1, "MUSIC_FAKE": 1, "MUSIC_GENERATOR": "suno"},
        {"MUSIC_PRESENT": 1, "MUSIC_FAKE": 0, "MUSIC_SOURCE_ID": "real_a"},
        {"MUSIC_PRESENT": 1, "MUSIC_FAKE": 0, "MUSIC_SOURCE_ID": "real_b"},
        {"MUSIC_PRESENT": 0, "VOICE_FAKE": 1, "VOICE_GENERATOR": "f5_tts"},
    ])
    # Pick a non-empty deterministic fold for this tiny fixture.
    for fold in range(3):
        try:
            train, validation, groups = group_split(frame, held_out_fold=fold)
        except ValueError:
            continue
        assert not set(groups.iloc[train]) & set(groups.iloc[validation])
        musicgen = groups.eq("fake_music_generator:musicgen").to_numpy()
        assert set(musicgen[train]) != {True} or not musicgen[validation].any()
        break
    else:
        raise AssertionError("fixture produced no non-empty group split")


def test_mixed_hypergraph_falls_back_to_two_task_logo_views():
    rows = []
    # Crossed FF combinations intentionally join every generator into one
    # hypergraph component.  Task-specific folds must still have zero overlap.
    for voice in range(8):
        for music in range(8):
            index = len(rows)
            rows.append({
                "ID": f"x{index}", "PARENT_ID": f"p{index}",
                "VOICE_PRESENT": 1, "VOICE_FAKE": 1,
                "VOICE_GENERATOR": f"voice_{voice}",
                "VOICE_SOURCE_ID": f"v{index}",
                "MUSIC_PRESENT": 1, "MUSIC_FAKE": 1,
                "MUSIC_GENERATOR": f"music_{music}",
                "MUSIC_SOURCE_ID": f"m{index}",
            })
    frame = pd.DataFrame(rows)
    plan = split_plan(frame, maximum_atomic_fraction=.20)
    assert plan.mode == "task_specific_logo"
    assert plan.audit["music_group_overlap"] == 0
    assert plan.audit["voice_group_overlap"] == 0
    assert len(plan.train_index)
    assert len(plan.music_validation_index)
    assert len(plan.voice_validation_index)

    # Audit every available identity/generator/parent view, not only the one
    # primary group returned by the legacy balanced sampler.
    for component, validation in (
        ("MUSIC", plan.music_validation_index),
        ("VOICE", plan.voice_validation_index),
    ):
        train_tokens = set().union(*(
            component_protected_tokens(frame.iloc[index], component)
            for index in plan.train_index
        ))
        validation_tokens = set().union(*(
            component_protected_tokens(frame.iloc[index], component)
            for index in validation
        ))
        assert not train_tokens & validation_tokens
