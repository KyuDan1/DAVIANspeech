import unittest
import numpy as np
from src.music_factorial_data_v89 import factorial_mixtures


class FactorialMixturesTest(unittest.TestCase):
    def test_identical_music_collapses_label_pairs(self):
        rng = np.random.default_rng(8)
        m, vr, vf = [rng.normal(0, .03, 64000).astype(np.float32) for _ in range(3)]
        for layout in ['concurrent', 'voice_first', 'music_first']:
            x = factorial_mixtures(m, m, vr, vf, 0, layout)
            np.testing.assert_array_equal(x[0], x[2])
            np.testing.assert_array_equal(x[1], x[3])

    def test_identical_voice_isolates_music_difference(self):
        rng = np.random.default_rng(9)
        mr, mf, v = [rng.normal(0, .03, 64000).astype(np.float32) for _ in range(3)]
        x = factorial_mixtures(mr, mf, v, v, -5, 'concurrent')
        np.testing.assert_array_equal(x[0], x[1])
        np.testing.assert_array_equal(x[2], x[3])
        np.testing.assert_allclose(x[2]-x[0], x[3]-x[1], rtol=0, atol=1e-7)

    def test_sequential_shared_segments(self):
        rng = np.random.default_rng(10)
        mr, mf, vr, vf = [rng.normal(0, .03, 64000).astype(np.float32) for _ in range(4)]
        x = factorial_mixtures(mr, mf, vr, vf, 10, 'music_first')
        np.testing.assert_array_equal(x[0, :32000], x[1, :32000])
        np.testing.assert_array_equal(x[2, :32000], x[3, :32000])
        np.testing.assert_array_equal(x[0, 32000:], x[2, 32000:])
        np.testing.assert_array_equal(x[1, 32000:], x[3, 32000:])

    def test_invalid(self):
        x = np.ones(16000, np.float32)
        with self.assertRaises(ValueError):
            factorial_mixtures(x, x[:-1], x, x, 0, 'concurrent')
        with self.assertRaises(ValueError):
            factorial_mixtures(x, x, x, x, 0, 'other')


if __name__ == '__main__':
    unittest.main()
