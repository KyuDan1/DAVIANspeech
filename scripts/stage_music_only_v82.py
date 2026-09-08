"""Stage an isolated native EAT Music branch without modifying anchor files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


SOURCES = ('common_eat_adaptation_inference.py', 'common_encoder_probe.py',
           'common_eat_adaptation.py', 'eat_music_adapter.py', 'full_coverage_wpt.py',
           'telephone_channel.py', 'eat_large_aasist_inference.py',
           'eat_timm_compat.py', 'eat_large_aasist.py', 'music_only_submission_v82.py')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('reports/music_only_submission_v82/staged'))
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    package = destination / 'branch82'
    package.mkdir()
    (package / '__init__.py').write_text('')
    for name in SOURCES:
        shutil.copy2(Path('src') / name, package / name)
    model_dir = destination / 'models/eat-large-as2m-v56'
    model_dir.mkdir(parents=True)
    for path in Path('models/eat-large-as2m-v56').iterdir():
        if path.is_file() and path.suffix in {'.py', '.json', '.safetensors', '.md'}:
            shutil.copy2(path, model_dir / path.name)
    (destination / 'run/adapted').mkdir(parents=True)
    shutil.copy2('reports/common_eat_adaptation/full/adapted/head.pt', destination / 'run/adapted/head.pt')
    shutil.copy2('reports/common_eat_adaptation/full/completed.json', destination / 'run/completed.json')
    inventory = {}
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            inventory[str(path.relative_to(destination))] = dict(bytes=path.stat().st_size, sha256=digest.hexdigest())
    report = dict(status='staged_not_validated_not_submitted', files=inventory,
                  changed_column='MUSIC_FAKE_PROB', unchanged_columns=['FILE_FAKE_PROB', 'VOICE_FAKE_PROB', 'VOICE_PRESENT_PROB', 'MUSIC_PRESENT_PROB'])
    (destination.parent / 'staging.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(status=report['status'], bytes=sum(v['bytes'] for v in inventory.values()), files=len(inventory))), flush=True)


if __name__ == '__main__':
    main()
