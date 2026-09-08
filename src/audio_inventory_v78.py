"""Metadata-efficient equivalent of pipeline.find_audio_files for future runs.

Use explicitly in a new/frozen run. Do not replace functions inside an already
running evaluation. No waveform decoding, labels, predictions or training here.
"""
import os
from pathlib import Path

AUDIO_EXTENSIONS = {'.aac', '.flac', '.m4a', '.mp3', '.ogg', '.opus', '.wav', '.wma'}


def find_audio_files(test_dir):
    test_dir = Path(test_dir)
    if not test_dir.is_dir():
        raise FileNotFoundError(f'Test directory not found: {test_dir}')
    # DirEntry can reuse readdir's file type; Path.is_file performs a separate
    # metadata query for every file, which is very expensive on the shared FS.
    with os.scandir(test_dir) as entries:
        files = sorted((test_dir / item.name for item in entries
                        if Path(item.name).suffix.lower() in AUDIO_EXTENSIONS
                        and item.is_file(follow_symlinks=True)), key=lambda p: p.stem)
    if not files:
        raise FileNotFoundError(f'No audio files found in {test_dir}')
    if len({path.stem for path in files}) != len(files):
        raise ValueError('Audio IDs must be unique')
    return files
