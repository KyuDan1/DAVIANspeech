from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_current_eer_v84 import analyze


class CurrentEerTest(unittest.TestCase):
    def test_no_unidentified_component_filled(self):
        result=analyze()
        self.assertAlmostEqual(result['voice_eer'],0.1755555555)
        self.assertAlmostEqual(result['weighted_file_music_error'],0.211)
        self.assertIsNone(result['file_eer'])
        self.assertIsNone(result['music_eer'])

    def test_matched_hypothetical_probe(self):
        # A synthetic algebra unit test, NOT an observed leaderboard score.
        result=analyze('0.6938888889')
        self.assertAlmostEqual(result['music_eer'],0.3)
        self.assertAlmostEqual(result['file_eer'],0.242)

    def test_actual_v84_official_result(self):
        result=analyze('0.699031746')
        self.assertAlmostEqual(result['music_eer'],0.317142857)
        self.assertAlmostEqual(result['file_eer'],0.2317142858)
        self.assertAlmostEqual(1-0.5*result['file_eer']-0.2*result['voice_eer']-0.3*result['music_eer'],0.7538888889)


if __name__=='__main__':
    unittest.main()
