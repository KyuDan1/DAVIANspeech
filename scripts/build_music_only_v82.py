"""Build exact-best plus independently validated original-audio EAT Music output."""
import json
import os
from pathlib import Path
import shutil
import zipfile

from build_best_voice_probe_v82 import BASE_SHA
from build_v18_music_probe import sha256_file, inspect_inventory, _clone_info


def main():
    base = Path('v50_v57m_challenger_v2.zip')
    branch = Path('reports/music_only_submission_v82/staged_v2')
    smoke = json.loads(Path('reports/music_only_submission_v82/smoke_v2/report.json').read_text())
    assert smoke['status'] == 'complete_staged_branch_smoke'
    assert smoke['other_fields_string_exact'] and smoke['max_music_difference'] <= 5e-4
    manifest = json.loads((branch.parent / 'staging.json').read_text())['files']
    for name, spec in manifest.items():
        assert sha256_file(branch / name) == spec['sha256'], name
    assert sha256_file(base) == BASE_SHA
    output = Path('best_eat_music_v82.zip')
    partial = output.with_suffix('.zip.partial')
    if output.exists() or partial.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(base) as source, zipfile.ZipFile(partial, 'w', allowZip64=True) as target:
        for info in source.infolist():
            clone = _clone_info(info)
            if info.filename == 'script.py':
                text = source.read(info).decode()
                marker = '    for path in (\n        eat_stats, spear_stats, eat_patch_graph, spear_component_bins,\n'
                assert text.count(marker) == 1
                insertion = ('    sys.path.insert(0, str(BASE_DIR / "model" / "music82"))\n'
                             '    from branch82.music_only_submission_v82 import apply_music_only\n'
                             '    apply_music_only(args.test_dir, args.output, BASE_DIR / "model" / "music82", device=args.device)\n')
                patched = text.replace(marker, insertion + marker)
                compile(patched, 'script.py', 'exec')
                target.writestr(clone, patched.encode())
            else:
                with source.open(info) as inp, target.open(clone, 'w', force_zip64=info.file_size >= zipfile.ZIP64_LIMIT) as out:
                    shutil.copyfileobj(inp, out, 8 * 1024 * 1024)
        for name in manifest:
            target.write(branch / name, 'model/music82/' + name, compress_type=zipfile.ZIP_DEFLATED, compresslevel=1)
    with zipfile.ZipFile(base) as source, zipfile.ZipFile(partial) as target:
        inventory = inspect_inventory(target)
        assert set(inventory['top_level']) == {'model', 'script.py', 'requirements.txt'}
        assert not any(inventory[k] for k in ('duplicates','unsafe_paths','symlinks'))
        assert inventory['expanded_bytes'] < 32_000_000_000
        assert partial.stat().st_size < 10_000_000_000
        assert max(i.file_size for i in target.infolist()) < 4_000_000_000
        for info in source.infolist():
            if info.filename != 'script.py':
                other = target.getinfo(info.filename)
                assert (other.CRC, other.file_size) == (info.CRC, info.file_size)
        assert target.testzip() is None
    os.replace(partial, output)
    report = dict(inventory, archive=str(output), sha256=sha256_file(output), base_sha256=BASE_SHA,
                  crc_verified=True, changed_column='MUSIC_FAKE_PROB', branch_smoke=smoke,
                  status='archive_validated_full_pipeline_smoke_pending')
    output.with_suffix('.report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
