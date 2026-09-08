#!/usr/bin/env python3
"""Training-source-only batching benchmark; not accuracy or model selection."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch

from src.common_encoder_probe import complete_windows
from src.dense_component_inference_v71 import DenseComponentPredictor
from src.dense_component_v71 import paired_dense_logits
from src.pipeline import load_audio


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    model = DenseComponentPredictor(args.checkpoint, ROOT, allow_smoke=args.allow_smoke)
    catalog = pd.read_csv(ROOT / model.config['source_catalog'])
    windows, lengths, ids = [], [], []
    for _, row in catalog.groupby(['COMPONENT', 'LABEL']).head(4).iterrows():
        values, sizes, _ = complete_windows(load_audio(Path(row.PATH)))
        windows.append(values[:1])
        lengths.append(sizes[:1])
        ids.append(row.ID)
    windows, lengths = torch.from_numpy(np.concatenate(windows)), torch.from_numpy(np.concatenate(lengths))
    counts = [1] * len(windows)
    heads = {'first': model.head, 'second': model.head}
    results, reference = [], None
    for chunk in [2, 8, 16]:
        # Warm up each shape before timing; same fixed training waveforms.
        paired_dense_logits(model.encoder, heads, windows, lengths, counts, chunk, 5.)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        elapsed = []
        for _ in range(3):
            started = time.perf_counter()
            outputs, _, _ = paired_dense_logits(model.encoder, heads, windows, lengths, counts, chunk, 5.)
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - started)
        actual = outputs['first'].sigmoid().cpu().numpy()
        if reference is None:
            reference = actual
        difference = float(np.max(np.abs(reference - actual)))
        if difference > 5e-4:
            raise ValueError('batching changes predictions beyond fixed tolerance')
        result = dict(encoder_chunk=chunk, files=len(counts), median_seconds=float(np.median(elapsed)),
            seconds=elapsed, peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
            max_abs_probability_difference=difference)
        results.append(result)
        print(json.dumps(result), flush=True)
    args.output.mkdir(parents=True)
    (args.output / 'report.json').write_text(json.dumps(dict(scope='training-only throughput / numerical equivalence',
        gpu=torch.cuda.get_device_name(), checkpoint=str(args.checkpoint), source_ids=ids, results=results), indent=2) + '\n')


if __name__ == '__main__':
    main()
