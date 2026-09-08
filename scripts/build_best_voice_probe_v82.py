"""Make a Voice-only constant diagnostic from the SHA-pinned official best ZIP."""
import argparse
import csv
import inspect
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

from build_v18_music_probe import inspect_inventory, sha256_file, _clone_info

BASE_SHA = 'a5730fde4e0c49100ee9aca5cc373eb550bf02948517072f23afe02fd62b91d6'


def apply_voice_probe(path):
    path = Path(path)
    temporary = path.with_suffix('.voice-probe.tmp')
    with path.open(newline='') as source, temporary.open('w', newline='') as target:
        reader = csv.reader(source)
        header = next(reader)
        if header.count('VOICE_FAKE_PROB') != 1:
            raise ValueError('expected one Voice probability column')
        index = header.index('VOICE_FAKE_PROB')
        writer = csv.writer(target, lineterminator='\n')
        writer.writerow(header)
        count = 0
        for row in reader:
            if len(row) != len(header):
                raise ValueError('malformed submission row')
            row[index] = '0.5'
            writer.writerow(row)
            count += 1
        if not count:
            raise ValueError('empty predictions')
    os.replace(temporary, path)


def patched_script(original):
    text = original.decode()
    marker = '\n\nif __name__ == "__main__":\n    main()\n'
    if text.count(marker) != 1:
        raise ValueError('anchor entrypoint marker mismatch')
    text = text.replace(marker, '\n\nimport csv\n' + inspect.getsource(apply_voice_probe)
                        + marker + '    apply_voice_probe(BASE_DIR / "output" / "submission.csv")\n')
    compile(text, 'script.py', 'exec')
    return text.encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=Path('v50_v57m_challenger_v2.zip'))
    parser.add_argument('--output', type=Path, default=Path('best_voice_probe_v82.zip'))
    args = parser.parse_args()
    if args.output.exists() or len(args.output.name) > 30:
        raise ValueError('output exists or filename exceeds platform limit')
    if sha256_file(args.base) != BASE_SHA:
        raise ValueError('not the exact official best archive')
    partial = args.output.with_suffix('.zip.partial')
    if partial.exists():
        raise FileExistsError(partial)
    with zipfile.ZipFile(args.base) as source, zipfile.ZipFile(partial, 'w', allowZip64=True) as target:
        for info in source.infolist():
            clone = _clone_info(info)
            if info.filename == 'script.py':
                target.writestr(clone, patched_script(source.read(info)))
            else:
                with source.open(info) as inp, target.open(clone, 'w', force_zip64=info.file_size >= zipfile.ZIP64_LIMIT) as out:
                    shutil.copyfileobj(inp, out, 8 * 1024 * 1024)
    with zipfile.ZipFile(args.base) as source, zipfile.ZipFile(partial) as target:
        inventory = inspect_inventory(target)
        assert set(source.namelist()) == set(target.namelist())
        assert set(inventory['top_level']) == {'model', 'script.py', 'requirements.txt'}
        assert not any(inventory[k] for k in ('duplicates', 'unsafe_paths', 'symlinks'))
        assert inventory['expanded_bytes'] < 32_000_000_000
        assert partial.stat().st_size < 10_000_000_000
        assert max(i.file_size for i in target.infolist()) < 4_000_000_000
        for info in source.infolist():
            if info.filename != 'script.py':
                other = target.getinfo(info.filename)
                assert (info.CRC, info.file_size) == (other.CRC, other.file_size)
        assert target.read('script.py') == patched_script(source.read('script.py'))
        assert target.testzip() is None
    os.replace(partial, args.output)
    report = dict(inventory, archive=str(args.output), sha256=sha256_file(args.output),
                  base_sha256=BASE_SHA, crc_verified=True, changed_column='VOICE_FAKE_PROB',
                  unchanged_members_except_script=True, purpose='diagnostic_not_score_improvement')
    args.output.with_suffix('.report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
