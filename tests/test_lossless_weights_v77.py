import json
import os
from pathlib import Path
import tempfile
import unittest

from src.lossless_weights_v77 import pack_file, restore_file, shuffle_bytes, unshuffle_bytes


class LosslessWeightsTests(unittest.TestCase):
    def test_byte_shuffle_all_tail_sizes(self):
        for size in range(132):
            original = os.urandom(size)
            self.assertEqual(unshuffle_bytes(shuffle_bytes(original)), original)

    def test_multiple_parts_exact_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = os.urandom(210_013)
            source = root / 'model.safetensors'
            source.write_bytes(original)
            manifest = pack_file(source, root / 'packed', workers=2, block_size=32768, member_limit=70000)
            self.assertGreater(len(manifest['parts']), 1)
            self.assertTrue(all(p['packed_bytes'] < 70000 for p in manifest['parts']))
            report = restore_file(root / 'packed', root / 'restored')
            self.assertEqual(report['restored_bytes'], len(original))
            self.assertEqual((root / 'restored').read_bytes(), original)
            with self.assertRaises(FileExistsError):
                restore_file(root / 'packed', source)
            self.assertEqual(source.read_bytes(), original)

    def test_corrupt_part_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'model'
            source.write_bytes(os.urandom(8192))
            manifest = pack_file(source, root / 'packed', workers=1)
            part = root / 'packed' / manifest['parts'][0]['name']
            part.write_bytes(part.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, 'hash, size, or order'):
                restore_file(root / 'packed', root / 'restored')

    def test_unsafe_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'model'
            source.write_bytes(b'train weights')
            manifest = pack_file(source, root / 'packed', workers=1)
            manifest['parts'][0]['name'] = '../outside.dw77'
            (root / 'packed' / 'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'unsafe'):
                restore_file(root / 'packed', root / 'restored')


if __name__ == '__main__':
    unittest.main()
