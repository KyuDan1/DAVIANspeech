import torch

from src.common_encoder_probe import encode_windows_independently, CommonTokenHead


def test_native_encoder_never_observes_other_lengths_or_batch_members():
    calls = []

    def native(windows, lengths):
        assert len(windows) == 1
        calls.append(int(lengths[0]))
        count = int(lengths[0])
        tokens = torch.full((1, 3, count, 8), float(windows[0, 0]))
        return tokens, torch.ones(1, count, dtype=torch.bool)

    windows, lengths = torch.tensor([[1.], [2.], [3.]]), torch.tensor([2, 7, 4])
    tokens, mask = encode_windows_independently(native, windows, lengths)
    assert calls == [2, 7, 4]
    assert tokens.shape == (3, 3, 7, 8)
    assert mask.sum(-1).tolist() == [2, 7, 4]
    assert not tokens[0, :, 2:].any()
    head = CommonTokenHead(8, width=4).eval()
    alone, alone_mask = native(windows[:1], lengths[:1])
    torch.testing.assert_close(head(tokens, mask)[:1], head(alone, alone_mask))


def test_corrected_spear_config_preserves_training_settings():
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parents[1]
    original = yaml.safe_load((root / 'configs/common_encoder_probe.yaml').read_text())
    corrected = yaml.safe_load((root / 'configs/common_encoder_probe_spear_independent.yaml').read_text())
    assert corrected['encoders'].pop('spear_independent') == original['encoders']['spear']
    assert original == corrected
