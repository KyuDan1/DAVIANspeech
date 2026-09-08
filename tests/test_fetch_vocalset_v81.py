import importlib.util
from pathlib import Path
import stat
import unittest
import zipfile

path=Path(__file__).resolve().parents[1]/'scripts/fetch_vocalset_v81.py'
spec=importlib.util.spec_from_file_location('fetch_vocalset_v81',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class VocalSetFetchTest(unittest.TestCase):
    def test_unsafe_members_rejected(self):
        for name in ['../escape.wav','/escape.wav','C:/escape.wav','a\\b.wav']:
            with self.assertRaises(ValueError):module.safe_member(zipfile.ZipInfo(name))
        link=zipfile.ZipInfo('link');link.external_attr=(stat.S_IFLNK|0o777)<<16
        with self.assertRaises(ValueError):module.safe_member(link)
        regular=zipfile.ZipInfo('male1/a.wav');regular.CRC=0
        self.assertEqual(module.safe_member(regular)['name'],'male1/a.wav')

    def test_record_and_license_pinned(self):
        record=dict(key=module.NAME,size=module.EXPECTED_SIZE,checksum='md5:'+module.EXPECTED_MD5,
                    links=dict(self=module.RECORD+'/files/'+module.NAME+'/content'))
        metadata=dict(id=1442513,metadata=dict(license=dict(id='cc-by-4.0')),files=[record])
        self.assertEqual(module.archive_record(metadata),record)
        metadata['metadata']['license']['id']='mit'
        with self.assertRaises(ValueError):module.archive_record(metadata)


if __name__=='__main__':unittest.main()
