import unittest
import pandas as pd
from src.diagnostic_axes_compat_v78 import add_diagnostic_axes


class DiagnosticAxesTests(unittest.TestCase):
    def test_cells_and_no_truth_changes(self):
        frame = pd.DataFrame(dict(VOICE_PRESENT=[1, 1, 1, 1, 0], MUSIC_PRESENT=[1]*5,
            VOICE_FAKE=[0, 0, 1, 1, None], MUSIC_FAKE=[0, 1, 0, 1, 0],
            VOICE_GENERATOR=['x', 'x', '', '', 'x'], SECOND_GENERATOR=['x', 'y', '', '', ''],
            CHANNEL=['clean', '', '', '', ''], CODEC=['flac']*5))
        before = frame.copy(deep=True)
        actual = add_diagnostic_axes(frame)
        pd.testing.assert_frame_equal(frame, before)
        pd.testing.assert_frame_equal(actual[before.columns], before)
        self.assertEqual(actual.CELL_V57.tolist(), ['RR', 'RF', 'FR', 'FF', 'not_mixed'])
        self.assertEqual(actual.CHANNEL_V57.tolist(), ['clean', 'flac', 'flac', 'flac', 'flac'])
        self.assertEqual(actual.VOICE_GENERATOR_V57.tolist(), ['x', 'x+y', 'unknown_fake', 'unknown_fake', 'absent'])


if __name__ == '__main__':
    unittest.main()
