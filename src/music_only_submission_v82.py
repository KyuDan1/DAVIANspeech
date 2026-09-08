"""Replace only conditional Music probability; never gate or update other fields."""
import csv
import os
from pathlib import Path

import librosa
import numpy as np
from transformers import PreTrainedModel  # Initialize dependency discovery before EAT's local timm shim.

from .common_eat_adaptation_inference import CommonEatAdaptationPredictor


def apply_music_only(test_dir, output, root, device='cuda'):
    root, output, test_dir = Path(root), Path(output), Path(test_dir)
    predictor = CommonEatAdaptationPredictor(root / 'run/adapted/head.pt', root, device=device)
    extensions = {'.aac', '.flac', '.m4a', '.mp3', '.ogg', '.opus', '.wav', '.wma'}
    paths = {}
    for path in test_dir.iterdir():
        if path.is_file() and path.suffix.lower() in extensions:
            if path.stem in paths:
                raise ValueError(f'duplicate audio ID: {path.stem}')
            paths[path.stem] = path
    temporary = output.with_suffix('.music-v82.tmp')
    with output.open(newline='') as source, temporary.open('w', newline='') as target:
        reader = csv.reader(source)
        header = next(reader)
        if header.count('ID') != 1 or header.count('MUSIC_FAKE_PROB') != 1:
            raise ValueError('invalid prediction columns')
        id_index, music_index = header.index('ID'), header.index('MUSIC_FAKE_PROB')
        writer = csv.writer(target, lineterminator='\n')
        writer.writerow(header)
        count = 0
        for row in reader:
            if len(row) != len(header):
                raise ValueError('malformed predictions')
            audio, _ = librosa.load(paths[row[id_index]], sr=16000, mono=True, dtype=np.float32)
            probability = float(predictor(audio)[2])
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError('invalid Music probability')
            row[music_index] = repr(probability)
            writer.writerow(row)
            count += 1
        if not count:
            raise ValueError('empty predictions')
    os.replace(temporary, output)
