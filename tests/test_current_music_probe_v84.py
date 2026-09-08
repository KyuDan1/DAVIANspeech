import ast
import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_current_music_probe_v84 import patch
from build_v18_music_probe import _apply_music_constant_probe, verify_prediction_pair


class MusicProbeTest(unittest.TestCase):
    def test_packaged_helper_preserves_other_fields(self):
        root=Path(__file__).resolve().parents[1]
        source=root/'reports/music_deadpass_v83/fast/output/submission.csv'
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'submission.csv'
            target.write_bytes(source.read_bytes())
            text=patch('def main():\n    pass\n\n\nif __name__ == "__main__":\n    main()\n')
            tree=ast.parse(text)
            helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_apply_music_constant_probe')
            import os
            namespace={'Path':Path,'csv':csv,'os':os}
            exec(compile(ast.Module(body=[helper],type_ignores=[]),'helper','exec'),namespace)
            namespace['_apply_music_constant_probe'](target)
            report=verify_prediction_pair(source,target)
            self.assertEqual(report['rows'],3)
            self.assertEqual(report['music_constant'],'0.5')

    def test_probe_after_main_only(self):
        text=patch('def main():\n    pass\n\n\nif __name__ == "__main__":\n    main()\n')
        self.assertTrue(text.endswith('    _apply_music_constant_probe(BASE_DIR / "output" / "submission.csv")\n'))


if __name__=='__main__':
    unittest.main()
