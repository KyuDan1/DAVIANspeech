from pathlib import Path
import tempfile
import unittest

from src.audio_inventory_v78 import AUDIO_EXTENSIONS, find_audio_files


class AudioInventoryTests(unittest.TestCase):
    def test_same_paths_and_sort_including_uppercase_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['z.wav', 'A.FLAC', 'not_audio.csv', 'sample.mp3']:
                (root / name).touch()
            (root / 'directory.wav').mkdir()
            (root / 'link.wav').symlink_to(root / 'A.FLAC')
            (root / 'missing.wav').symlink_to(root / 'missing_target')
            expected = sorted((p for p in root.iterdir() if p.is_file()
                and p.suffix.lower() in AUDIO_EXTENSIONS), key=lambda p: p.stem)
            self.assertEqual(find_audio_files(root), expected)

    def test_duplicate_stem_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'x.wav').touch()
            (root / 'x.flac').touch()
            with self.assertRaisesRegex(ValueError, 'unique'):
                find_audio_files(root)

    def test_missing_or_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                find_audio_files(root)
            with self.assertRaises(FileNotFoundError):
                find_audio_files(root / 'absent')


if __name__ == '__main__':
    unittest.main()
