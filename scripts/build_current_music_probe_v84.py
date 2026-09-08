"""One-column Music diagnostic of the actually submitted v83 pipeline."""
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

from build_v18_music_probe import _apply_music_constant_probe, inspect_inventory, sha256_file

BASE_SHA = '92d5f2f3508f7b3d7748f2214a7319c5a9029d826f85779bb8ccb64f18ecf48f'


def patch(source):
    marker = '\n\nif __name__ == "__main__":\n    main()\n'
    if source.count(marker) != 1:
        raise ValueError('unexpected entrypoint')
    helper = '\n\nimport csv\n' + inspect.getsource(_apply_music_constant_probe)
    result = source.replace(marker, helper + marker + '    _apply_music_constant_probe(BASE_DIR / "output" / "submission.csv")\n')
    compile(result,'script.py','exec')
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    source = root/'best_eat_music_v83.zip'
    target = root/'current_music_probe_v84.zip'
    if target.exists():
        raise FileExistsError(target)
    assert sha256_file(source) == BASE_SHA
    with tempfile.TemporaryDirectory(prefix='music-probe-v84-',dir=root/'reports') as directory:
        stage = Path(directory)
        with zipfile.ZipFile(source) as z:
            entry = patch(z.read('script.py').decode()).encode()
        (stage/'script.py').write_bytes(entry)
        archive = stage/'candidate.zip'
        shutil.copyfile(source,archive)
        subprocess.run(['zip','-q',str(archive),'script.py'],cwd=stage,check=True)
        with zipfile.ZipFile(source) as base, zipfile.ZipFile(archive) as z:
            inventory = inspect_inventory(z)
            assert set(z.namelist()) == set(base.namelist())
            assert set(inventory['top_level']) == {'model','script.py','requirements.txt'}
            assert not any(inventory[k] for k in ['duplicates','unsafe_paths','symlinks'])
            assert inventory['expanded_bytes'] < 32_000_000_000 and archive.stat().st_size < 10_000_000_000
            assert max(i.file_size for i in z.infolist()) < 4_000_000_000
            for info in base.infolist():
                if info.filename != 'script.py':
                    other = z.getinfo(info.filename)
                    assert (info.CRC,info.file_size,info.compress_size) == (other.CRC,other.file_size,other.compress_size)
            assert z.read('script.py') == entry
            assert z.testzip() is None
        digest = sha256_file(archive)
        os.replace(archive,target)
    report = dict(status='complete_archive_validation',archive=target.name,sha256=digest,
                  base_sha256=BASE_SHA,all_crc_verified=True,inventory=inventory,
                  only_output_change='MUSIC_FAKE_PROB=0.5',purpose='diagnostic_not_performance_improvement')
    target.with_suffix('.report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
