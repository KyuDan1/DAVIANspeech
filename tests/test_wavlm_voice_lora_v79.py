import unittest
import pandas as pd
import torch
from torch.nn.utils import parametrize
from src.wavlm_voice_lora_v79 import LowRankDelta, known_voice_targets


class WavlmLoraTests(unittest.TestCase):
    def test_zero_delta_exact_and_only_adapter_gradients(self):
        torch.manual_seed(4)
        linear = torch.nn.Linear(6, 5).requires_grad_(False)
        original = linear.weight.detach().clone()
        adapter = LowRankDelta(5, 6, 2)
        parametrize.register_parametrization(linear, 'weight', adapter)
        self.assertTrue(torch.equal(linear.weight, original))
        linear(torch.randn(3, 6)).square().mean().backward()
        self.assertIsNone(linear.parametrizations.weight.original.grad)
        self.assertIsNotNone(adapter.b.grad)
        self.assertGreater(float(adapter.b.grad.abs().sum()), 0)
        with torch.no_grad():
            adapter.b.add_(.1)
        self.assertFalse(torch.equal(linear.weight, original))
        adapter.enabled = False
        self.assertTrue(torch.equal(linear.weight, original))

    def test_only_known_train_voice_labels(self):
        frame = pd.DataFrame(dict(VOICE_PRESENT=[1, 1, 1, 1, 0], VOICE_FAKE=[0, 0, 1, 0, None],
            MUSIC_PRESENT=[1, 1, 1, 0, 1], MUSIC_FAKE=[0, 1, 1, None, 1]))
        before = frame.copy(deep=True)
        self.assertEqual(known_voice_targets(frame).tolist(), [True, False, True, True, False])
        pd.testing.assert_frame_equal(frame, before)


if __name__ == '__main__':
    unittest.main()
