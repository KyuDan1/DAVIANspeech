"""Exercise staged submission branch in isolation on 80 development files."""
import csv
import json
from pathlib import Path
import shutil
import sys
import time


def main():
    import numpy as np
    import pandas as pd
    import torch
    root = Path(__file__).resolve().parents[1]
    staging = root / 'reports/music_only_submission_v82/staged_v2'
    output = staging.parent / 'smoke_v2'
    output.mkdir(exist_ok=False)
    sys.path.insert(0, str(staging))
    from branch82.music_only_submission_v82 import apply_music_only
    reference = pd.read_csv(root / 'reports/component_composition_v73/grader_stack_v77/predictions.csv')
    inventory = pd.read_csv(root / 'reports/training_inventory_v78/development.csv', low_memory=False)
    frame = reference.merge(inventory[['DATASET', 'ID', 'PATH']], on=['DATASET','ID'], validate='one_to_one')
    assert len(frame) == len(reference) == 80
    audio_dir = output / 'audio'
    audio_dir.mkdir()
    columns = ['ID', 'FILE_FAKE_PROB', 'VOICE_FAKE_PROB', 'MUSIC_FAKE_PROB', 'VOICE_PRESENT_PROB', 'MUSIC_PRESENT_PROB']
    rows = []
    for index, row in frame.iterrows():
        path = Path(row.PATH)
        identifier = f'smoke_{index:04d}'
        shutil.copy2(path, audio_dir / (identifier + path.suffix))
        rows.append([identifier] + [repr(float(row['integrated_' + c])) for c in columns[1:]])
    path = output / 'submission.csv'
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    torch.set_num_threads(2)
    started = time.monotonic()
    apply_music_only(audio_dir, path, staging)
    with path.open(newline='') as handle:
        actual = list(csv.reader(handle))[1:]
    maximum = 0.0
    for before, after in zip(rows, actual):
        assert before[:3] + before[4:] == after[:3] + after[4:]
        maximum = max(maximum, abs(float(before[3]) - float(after[3])))
    assert len(actual) == 80 and maximum <= 5e-4
    report = dict(status='complete_staged_branch_smoke', rows=len(actual),
                  max_music_difference=maximum, other_fields_string_exact=True,
                  seconds_including_load=time.monotonic()-started,
                  peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20,
                  scope='staged Music branch only; not full anchor or L4 runtime validation')
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
