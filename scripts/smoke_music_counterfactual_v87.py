"""Audit and exercise TRAIN-only paired-music rendering before any learning."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]


def main():
    import numpy as np
    import pandas as pd
    from src.music_counterfactual_data_v87 import MusicCounterfactualPairs
    from src.full_coverage_wpt import resolve_ffmpeg
    from train_common_encoder_probe import audit_protected_sources
    from train_dense_component_v71 import audited_catalog
    from train_paired_wpt_file_v60 import sha256
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inventory_path = ROOT / 'reports/training_inventory_v78/report.json'
    inventory = json.loads(inventory_path.read_text())
    assert inventory['status'] == 'complete_metadata_cache'
    assert all(sha256(p) == digest for p, digest in inventory['artifacts_sha256'].items())
    table = inventory['tables']['train']
    assert sha256(table['path']) == table['sha256']
    train = pd.read_csv(table['path'], low_memory=False)
    config = dict(partition_config='configs/data_partitions.yaml',
                  source_catalog='reports/dense_component_v71/source_catalog/sources.csv')
    audit = audit_protected_sources(train, ROOT / config['partition_config'])
    catalog = audited_catalog(config, train)
    ffmpeg = resolve_ffmpeg()
    artifacts = [Path(__file__), inventory_path, Path(table['path']), ffmpeg,
                 ROOT / config['partition_config'], ROOT / config['source_catalog'],
                 *[ROOT / 'src' / name for name in ['music_counterfactual_data_v87.py',
                   'dense_component_data_v71.py', 'long_component_stress.py', 'telephone_channel.py']]]
    hashes = {str(p): sha256(p) for p in artifacts}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'frozen.json').write_text(json.dumps(dict(artifacts_sha256=hashes,
        seed=20260907, draws=32, source_audit=audit, scope='TRAIN rendering only; no learning/evaluation'), indent=2))
    sampler = MusicCounterfactualPairs(catalog, 20260907, ffmpeg)
    counters = {k: Counter() for k in ['music_label', 'layout', 'channel', 'seconds']}
    tiled = 0
    with (args.output / 'traces.jsonl').open('w') as stream:
        for index in range(32):
            audio, label, trace = sampler.draw(1, index)
            repeat, repeated_label, repeated_trace = sampler.draw(1, index)
            np.testing.assert_array_equal(audio, repeat)
            assert repeated_label == label and repeated_trace == trace
            assert trace['sources'][0]['label'] == label
            assert [s['label'] for s in trace['sources'][1:]] == [0, 1]
            for key in counters:
                counters[key][str(trace[key])] += 1
            tiled += sum(s['tiled'] for s in trace['sources'])
            stream.write(json.dumps(trace) + '\n')
    assert all(sha256(p) == digest for p, digest in hashes.items())
    result = dict(status='complete_TRAIN_data_smoke', pairs=32, rendered_calls=64,
                  deterministic_bit_exact=True, counts={k: dict(v) for k, v in counters.items()},
                  tiled_sources=tiled, consumed_source_slots=96,
                  trained=False, protected_evaluation_consumed=False)
    (args.output / 'report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
