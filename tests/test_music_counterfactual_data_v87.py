import unittest
import numpy as np
from src.music_counterfactual_data_v87 import paired_mixtures, crop_or_tile


class CounterfactualTest(unittest.TestCase):
    def test_identical_foregrounds_identical_views(self):
        rng = np.random.default_rng(3)
        m, v = [rng.normal(0, .03, 64000).astype(np.float32) for _ in range(2)]
        for layout in ['concurrent', 'voice_first', 'music_first']:
            views = paired_mixtures(m, v, v, 5, layout)
            np.testing.assert_array_equal(views[0], views[1])

    def test_sequential_music_identical(self):
        rng = np.random.default_rng(4)
        m, v, f = [rng.normal(0, .1, 64000).astype(np.float32) for _ in range(3)]
        for layout, sl in [('voice_first', slice(32000, None)), ('music_first', slice(None, 32000))]:
            views = paired_mixtures(m, v, f, 10, layout)
            np.testing.assert_array_equal(views[0, sl], views[1, sl])
            self.assertLessEqual(float(np.abs(views).max()), .980001)

    def test_native_crop_not_tiled_and_deterministic(self):
        audio = np.sin(np.arange(100000) * .01).astype(np.float32)
        a, trace = crop_or_tile(audio, 64000, np.random.default_rng(9))
        b, other = crop_or_tile(audio, 64000, np.random.default_rng(9))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(trace, other)
        self.assertFalse(trace['tiled'])

    def test_reject_invalid(self):
        a = np.ones(16000, np.float32)
        with self.assertRaises(ValueError):
            paired_mixtures(a, a[:-1], a, 0, 'concurrent')
        with self.assertRaises(ValueError):
            paired_mixtures(a, a, a, 0, 'unknown')


if __name__ == '__main__':
    unittest.main()
