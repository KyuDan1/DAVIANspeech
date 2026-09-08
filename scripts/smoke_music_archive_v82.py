"""Run the complete packaged entrypoint on three development audio files."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

from build_v18_music_probe import sha256_file, inspect_inventory


def main():
    root = Path(__file__).resolve().parents[1]
    archive = root / 'best_eat_music_v82.zip'
    report = json.loads(archive.with_suffix('.report.json').read_text())
    assert report['crc_verified'] and sha256_file(archive) == report['sha256']
    stage = root / 'reports/music_only_submission_v82/full_package_smoke'
    stage.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as handle:
        inventory = inspect_inventory(handle)
        assert not any(inventory[k] for k in ('duplicates','unsafe_paths','symlinks'))
        handle.extractall(stage)
    audio = stage / 'data/test'
    audio.mkdir(parents=True)
    inventory_path = root / 'reports/training_inventory_v78/development.csv'
    with inventory_path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    def matches(row, voice_fake, music_fake):
        try:
            return (float(row['VOICE_PRESENT']) == 1 and float(row['MUSIC_PRESENT']) == 1
                    and float(row['VOICE_FAKE']) == float(voice_fake)
                    and float(row['MUSIC_FAKE']) == float(music_fake))
        except (ValueError, TypeError):
            return False
    for voice_fake, music_fake in [('0.0','0.0'), ('1.0','0.0'), ('0.0','1.0')]:
        candidates = [r for r in rows if matches(r, voice_fake, music_fake)]
        if not candidates:
            raise ValueError('required development component case missing')
        selected.append(candidates[0])
    columns = ['ID','FILE_FAKE_PROB','VOICE_FAKE_PROB','MUSIC_FAKE_PROB','VOICE_PRESENT_PROB','MUSIC_PRESENT_PROB']
    with (stage / 'data/sample_submission.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for index, row in enumerate(selected):
            original = Path(row['PATH'])
            identifier = f'package_smoke_{index}'
            shutil.copy2(original, audio / (identifier + original.suffix))
            writer.writerow([identifier] + ['0.5']*5)
    started = time.monotonic()
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    # Run the exact original entrypoint on the very same three files. The new
    # package contains the unmodified original model members in the same paths.
    with zipfile.ZipFile(root / 'v50_v57m_challenger_v2.zip') as handle:
        original_script = handle.read('script.py')
    original_entry = stage / 'script_anchor.py'
    original_entry.write_bytes(original_script)
    with (stage / 'anchor_execution.log').open('w') as log:
        baseline = subprocess.run([sys.executable, str(original_entry)], cwd=stage,
                                  env=environment, stdout=log, stderr=subprocess.STDOUT)
    if baseline.returncode:
        raise RuntimeError(f'anchor smoke failed; see {stage / "anchor_execution.log"}')
    with (stage / 'output/submission.csv').open(newline='') as handle:
        anchor_predictions = list(csv.DictReader(handle))
    shutil.copy2(stage / 'output/submission.csv', stage / 'anchor_predictions.csv')
    with (stage / 'execution.log').open('w') as log:
        result = subprocess.run([sys.executable, str(stage / 'script.py')], cwd=stage,
                                env=environment, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'packaged entrypoint failed; see {stage / "execution.log"}')
    with (stage / 'output/submission.csv').open(newline='') as handle:
        predictions = list(csv.DictReader(handle))
    assert len(predictions) == 3
    assert [r['ID'] for r in predictions] == [f'package_smoke_{i}' for i in range(3)]
    for row in predictions:
        assert all(0 <= float(row[c]) <= 1 for c in columns[1:])
    preserved = [c for c in columns[1:] if c != 'MUSIC_FAKE_PROB']
    maximum = 0.0
    for original, updated in zip(anchor_predictions, predictions):
        assert original['ID'] == updated['ID']
        maximum = max(maximum, *(abs(float(original[c])-float(updated[c])) for c in preserved))
    assert maximum <= 5e-4, 'unexpected non-Music pipeline drift'
    result = dict(status='complete_full_packaged_entrypoint_smoke', rows=3,
                  archive_sha256=report['sha256'], seconds=time.monotonic()-started,
                  anchor_vs_candidate_non_music_maximum_difference=maximum,
                  scope='B200, grader-core environment; not full 1200-file L4 deadline guarantee')
    (stage / 'report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
