import ast
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_music_deadpass_v83 import CACHE, RESIDUAL, patch


class DeadPassTest(unittest.TestCase):
    def test_only_two_statements_change(self):
        original = 'def main():\n' + CACHE + RESIDUAL + '    apply_music_only(args.test_dir)\n'
        revised = patch(original)
        before, after = ast.parse(original).body[0].body, ast.parse(revised).body[0].body
        self.assertEqual(len(before),3)
        self.assertEqual(len(after),2)
        self.assertIsNone(after[0].value.value)
        self.assertEqual(ast.dump(before[-1]),ast.dump(after[-1]))

    def test_refuses_missing_marker(self):
        with self.assertRaises(ValueError):
            patch('def main():\n    pass\n')

    def test_refuses_residual_after_replacement(self):
        with self.assertRaises(ValueError):
            patch('def main():\n' + CACHE + '    apply_music_only(args.test_dir)\n' + RESIDUAL)


if __name__ == '__main__':
    unittest.main()
