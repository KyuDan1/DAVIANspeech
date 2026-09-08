import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_best_voice_probe_v82 import apply_voice_probe, patched_script


class VoiceProbeTest(unittest.TestCase):
    def test_preserves_other_field_strings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'submission.csv'
            original = [['ID','FILE_FAKE_PROB','VOICE_FAKE_PROB','MUSIC_FAKE_PROB',
                         'VOICE_PRESENT_PROB','MUSIC_PRESENT_PROB'],
                        ['a','0.000001','0.8','9e-1','0.98','0.1'],
                        ['b','0.31','0.0','0.123456789123','1.0','0.9']]
            with path.open('w', newline='') as handle:
                csv.writer(handle).writerows(original)
            apply_voice_probe(path)
            with path.open(newline='') as handle:
                actual = list(csv.reader(handle))
            self.assertEqual(actual[0], original[0])
            for before, after in zip(original[1:], actual[1:]):
                self.assertEqual(after[2], '0.5')
                self.assertEqual(before[:2]+before[3:], after[:2]+after[3:])

    def test_final_call_after_anchor(self):
        text = patched_script(b'import os\nfrom pathlib import Path\ndef main():\n    pass\n\n\nif __name__ == "__main__":\n    main()\n').decode()
        self.assertTrue(text.endswith('    apply_voice_probe(BASE_DIR / "output" / "submission.csv")\n'))
        compile(text, 'script.py', 'exec')


if __name__ == '__main__':
    unittest.main()
