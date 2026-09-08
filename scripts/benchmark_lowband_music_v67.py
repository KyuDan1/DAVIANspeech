#!/usr/bin/env python3
"""Synthetic 4/30/60s runtime and independence test, no dataset or score tuning."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.lowband_music_inference_v67 import load_model, predict_one_audio
from src.full_coverage_wpt import coverage_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(6)
    device = torch.device('cuda')
    started = time.perf_counter()
    model, config = load_model(args.checkpoint, device, allow_smoke=args.allow_smoke)
    load_seconds = time.perf_counter() - started
    generator = np.random.default_rng(20260905)
    audios = {seconds: (.02 * generator.standard_normal(seconds * 16000)).astype(np.float32)
              for seconds in [4, 30, 60]}
    predict_one_audio(model, audios[4], config, device)  # warm-up
    rows, first = [], {}
    for seconds in [4, 30, 60]:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(3):
            score = predict_one_audio(model, audios[seconds], config, device)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) / 3
        first[seconds] = score
        rows.append(dict(duration=seconds, windows=len(coverage_windows(audios[seconds], config['window'])[0]), seconds_per_file=elapsed,
                         peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                         peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30))
    reverse = {seconds: predict_one_audio(model, audios[seconds], config, device) for seconds in [60, 30, 4]}
    if first != reverse:
        raise RuntimeError('changing file order changed a probability')
    report = dict(gpu=torch.cuda.get_device_name(), checkpoint=str(args.checkpoint),
                  smoke=args.allow_smoke, parameters=sum(p.numel() for p in model.parameters()), load_seconds=load_seconds, files=rows,
                  reordered_file_predictions_bit_exact=True,
                  scope='Synthetic runtime only; NOT L4 runtime or detection accuracy')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
