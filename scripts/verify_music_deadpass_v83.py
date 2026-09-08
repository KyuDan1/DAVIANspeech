"""Verify eliminating the overwritten Music residual and its extra XLS-R pass."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

BASE_SHA = '1267660ca5b4dc5168988bec6df4114ebcc81c04bd3c9f200fad38fdf5202082'
CACHE = '    args.xlsr_window_embeddings_output = xlsr_windows\n'
RESIDUAL = '''    apply_three_stream_anchor_residual(
        args.output, eat_patch_graph, spear_component_bins, xlsr_windows,
        BASE_DIR / "model" / "three-stream-v57" / "head.pt",
        device=args.device, batch_size=64, task_mode="music",
    )
'''


def patch(text):
    if text.count(CACHE) != 1 or text.count(RESIDUAL) != 1:
        raise ValueError('unexpected source; cannot prove exact two-statement removal')
    if text.index(RESIDUAL) >= text.index('    apply_music_only('):
        raise ValueError('Music replacement must follow deleted residual')
    result = text.replace(CACHE, '    args.xlsr_window_embeddings_output = None\n').replace(RESIDUAL, '')
    compile(result, 'script.py', 'exec')
    return result


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('reports/music_deadpass_v83'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    archive = root / 'best_eat_music_v82.zip'
    source = root / 'reports/music_only_submission_v82/full_package_smoke'
    assert sha(archive) == BASE_SHA
    assert json.loads((source/'report.json').read_text())['status'] == 'complete_full_packaged_entrypoint_smoke'
    with zipfile.ZipFile(archive) as handle:
        original = handle.read('script.py').decode()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    changed = patch(original)
    inputs = sorted((source/'data/test').iterdir())
    assert len(inputs) == 3
    frozen = dict(source_archive_sha256=BASE_SHA, inputs_sha256={str(p):sha(p) for p in inputs},
                  change='skip discarded Music residual and its dedicated XLS-R embedding pass',
                  script_sha256=hashlib.sha256(changed.encode()).hexdigest(),
                  predictions_tolerance=1e-8, scope='three mixed development files; NOT accuracy selection')
    (output/'frozen.json').write_text(json.dumps(frozen,indent=2))
    results = {}
    for name, text in [('reference', original), ('fast', changed)]:
        stage = output/name
        stage.mkdir()
        (stage/'model').symlink_to(source/'model',target_is_directory=True)
        (stage/'data').symlink_to(source/'data',target_is_directory=True)
        (stage/'script.py').write_text(text)
        started = time.monotonic()
        with (stage/'execution.log').open('w') as log:
            result = subprocess.run([sys.executable,str(stage/'script.py')],cwd=stage,
                 env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1'),
                 stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{name} failed; inspect {stage / "execution.log"}')
        with (stage/'output/submission.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        results[name] = dict(seconds=time.monotonic()-started,rows=rows)
    assert len(results['reference']['rows']) == len(results['fast']['rows']) == 3
    maximum = 0.0
    exact = True
    for before, after in zip(results['reference']['rows'],results['fast']['rows']):
        assert before['ID'] == after['ID']
        exact = exact and before == after
        for column in before:
            if column != 'ID':
                maximum = max(maximum,abs(float(before[column])-float(after[column])))
    assert maximum <= frozen['predictions_tolerance']
    assert all(sha(Path(p)) == digest for p,digest in frozen['inputs_sha256'].items())
    report = dict(status='complete_deadpass_equivalence',rows=3,all_five_maximum_difference=maximum,
                  all_fields_string_exact=exact,timings={k:v['seconds'] for k,v in results.items()},
                  limitation='One sequential run per arm on three files with model loading; NOT L4 throughput guarantee',
                  automatic_submission_allowed=False)
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
