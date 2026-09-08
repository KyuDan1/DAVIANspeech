"""Replace one ZIP entry using Info-ZIP; preserve all compressed model entries."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

from build_v18_music_probe import inspect_inventory, sha256_file
from verify_music_deadpass_v83 import BASE_SHA, patch


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / 'best_eat_music_v82.zip'
    output = root / 'best_eat_music_v83.zip'
    verified = root / 'reports/music_deadpass_v83'
    report = json.loads((verified/'report.json').read_text())
    frozen = json.loads((verified/'frozen.json').read_text())
    assert report['status'] == 'complete_deadpass_equivalence'
    assert report['all_fields_string_exact'] and report['all_five_maximum_difference'] == 0
    assert sha256_file(source) == BASE_SHA == frozen['source_archive_sha256']
    assert not output.exists()
    if shutil.which('zip') is None:
        raise RuntimeError('Info-ZIP build utility missing')
    with tempfile.TemporaryDirectory(prefix='zip-build-', dir=verified) as directory:
        temporary = Path(directory)
        entry = temporary / 'script.py'
        with zipfile.ZipFile(source) as handle:
            text = patch(handle.read('script.py').decode())
        entry.write_text(text)
        assert sha256_file(entry) == frozen['script_sha256']
        assert entry.read_bytes() == (verified/'fast/script.py').read_bytes()
        archive = temporary / 'candidate.zip'
        shutil.copyfile(source, archive)
        subprocess.run(['zip','-q',str(archive),'script.py'],cwd=temporary,check=True)
        with zipfile.ZipFile(source) as base, zipfile.ZipFile(archive) as candidate:
            inventory = inspect_inventory(candidate)
            assert set(candidate.namelist()) == set(base.namelist())
            assert set(inventory['top_level']) == {'model','script.py','requirements.txt'}
            assert not any(inventory[k] for k in ['duplicates','unsafe_paths','symlinks'])
            assert archive.stat().st_size < 10_000_000_000 and inventory['expanded_bytes'] < 32_000_000_000
            assert max(i.file_size for i in candidate.infolist()) < 4_000_000_000
            for info in base.infolist():
                if info.filename != 'script.py':
                    other = candidate.getinfo(info.filename)
                    assert (info.CRC,info.file_size,info.compress_size) == (other.CRC,other.file_size,other.compress_size)
            assert candidate.read('script.py') == entry.read_bytes()
            assert candidate.testzip() is None
        digest = sha256_file(archive)
        os.replace(archive,output)
    result = dict(status='complete_archive_validation',archive=str(output),sha256=digest,
                  bytes=output.stat().st_size,inventory=inventory,all_crc_verified=True,
                  source_archive_sha256=BASE_SHA,script_matches_full_pipeline_verified_fast_script=True,
                  only_entrypoint_changed=True,probability_change_from_v82='none on three verified mixed files',
                  official_submitted=False,L4_1200_file_time_verified=False,
                  evidence_sha256={str(p):sha256_file(p) for p in [verified/'report.json',verified/'frozen.json',verified/'fast/script.py']})
    output.with_suffix('.report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
