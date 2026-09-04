import torch

from src.multistream_prompt_spectra import (
    DeepMultiStreamPrompts,
    MultiScaleInstantaneousFrequency,
    SignalTextureConditioner,
    warm_start_shared_backend,
)


def test_analytic_signal_and_frequency_prompt_are_finite_and_differentiable():
    module = MultiScaleInstantaneousFrequency(8)
    values = torch.randn(3, 6, 8, requires_grad=True)
    analytic = module.analytic_signal(values)
    output = module(values)
    assert analytic.shape == output.shape == values.shape
    assert torch.is_complex(analytic)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()


def test_texture_conditioner_uses_input_sequence_and_handles_short_audio():
    module = SignalTextureConditioner(8)
    prompts = torch.randn(2, 6, 8, requires_grad=True)
    regular = torch.randn(2, 12, 8, requires_grad=True)
    short = torch.randn(2, 1, 8)
    tkeo, gate = module.statistics(regular)
    output = module(prompts, tkeo, gate)
    assert output.shape == prompts.shape
    assert tkeo.shape == gate.shape == (2, 1, 8)
    assert ((gate >= 0) & (gate <= 1)).all()
    short_tkeo, short_gate = module.statistics(short)
    assert torch.isfinite(short_tkeo).all() and torch.isfinite(short_gate).all()
    output.sum().backward()
    assert prompts.grad is not None and regular.grad is not None


def test_multistream_prompts_have_expected_order_shape_and_gradients():
    prompts = DeepMultiStreamPrompts(
        layers=3, width=8, base_tokens=4,
        frequency_tokens=6, texture_tokens=2, dropout=0,
    )
    raw = torch.randn(5, 9, 8)
    tkeo, gate = prompts.texture_statistics(raw)
    first = prompts(0, 5, tkeo, gate)
    second = prompts(1, 5, tkeo, gate)
    assert prompts.count == 12
    assert first.shape == second.shape == (5, 12, 8)
    assert not torch.equal(first, second)
    first.mean().backward()
    assert prompts.base.grad is not None
    assert prompts.frequency.grad is not None
    assert prompts.texture.grad is not None


class _WarmStartTarget(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_encoder = torch.nn.Module()
        self.prompt_encoder.prompts = torch.nn.Linear(2, 2)
        self.bridge = torch.nn.Linear(2, 3)
        self.aasist = torch.nn.Linear(3, 4)
        self.task_head = torch.nn.Linear(4, 3)


def test_warm_start_transfers_shared_modules_but_never_prompt_weights():
    target = _WarmStartTarget()
    original_prompt = target.prompt_encoder.prompts.weight.detach().clone()
    source = {
        name: torch.full_like(value, 7)
        for name, value in target.state_dict().items()
    }
    source["prompt_encoder.prompts.old_wavelet"] = torch.ones(1)
    report = warm_start_shared_backend(target, source)
    assert torch.equal(target.prompt_encoder.prompts.weight, original_prompt)
    assert torch.all(target.bridge.weight == 7)
    assert torch.all(target.aasist.weight == 7)
    assert torch.all(target.task_head.weight == 7)
    assert report["tensors_loaded"] == 6
    assert report["prompt_tensors_skipped"] == 3
