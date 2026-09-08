#!/usr/bin/env python3
"""Two fixed component combinations on completed file-local development scores."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
import yaml

from src.component_composition_v73 import compose_components
from src.evaluate_diagnostic import PREDICTION_COLUMNS, LABEL_COLUMNS, score_frame
from train_common_encoder_probe import metrics
from train_paired_wpt_file_v60 import sha256
from train_three_stream_anchor_residual import authorized_partitions
from compare_common_encoder_probe import align_truth, typed_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/component_composition_v73.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = yaml.safe_load(args.config.read_text())
    if (config['schema'] != 'component_composition_v73' or config['voice_pair_weights'] != [.5, .5]
            or config['variants'] != {'xlsr_voice_eat_music': 'xlsr', 'equal_voice_pair_eat_music': 'equal_xlsr_spear_logits'}
            or config['music_source'] != 'eat' or config['presence_source'] != 'eat'
            or config['weight_or_temperature_search'] or not config['no_automatic_submission']
            or config['probability_clip_for_logit'] != 1e-6
            or config['file_rule'] != 'noisy_or_of_presence_weighted_components'):
        raise ValueError('only the two declared fixed rules are implemented')
    frames, hashes = {}, {}
    for name, relative in config['predictions'].items():
        path = ROOT / relative
        verification_path = path.parent / 'report.json'
        report = json.loads(verification_path.read_text())
        frame = pd.read_csv(path, dtype={'ID': str, 'DATASET': str})
        if report['rows'] != len(frame) or not report['repeated_files_bit_exact'] or report['checkpoint_reselected']:
            raise ValueError('complete fixed file-local development scores required')
        if 'file_local_prediction_sha256' in report and sha256(path) != report['file_local_prediction_sha256']:
            raise ValueError('file-local CSV changed')
        for artifact, digest in report['artifacts_sha256'].items():
            if sha256(Path(artifact)) != digest:
                raise ValueError('selected source model or saved scores changed')
        frames[name] = frame
        hashes.update({str(p): sha256(p) for p in [path, verification_path]})
    matrix = yaml.safe_load((ROOT / config['data_matrix']).read_text())
    pieces = []
    for partition in authorized_partitions(ROOT / config['partition_config'], 'development', matrix[matrix['development_set']]):
        truth = pd.read_csv(partition.truth_path, dtype={'ID': str})
        truth['DATASET'] = partition.name
        pieces.append(truth)
        hashes[str(partition.truth_path)] = sha256(partition.truth_path)
    truth = pd.concat(pieces, ignore_index=True)
    for name, frame in frames.items():
        expected = frame.set_index(['DATASET', 'ID'])[LABEL_COLUMNS]
        aligned = align_truth(truth, frame)
        np.testing.assert_allclose(aligned[LABEL_COLUMNS].to_numpy(float),
            expected.loc[list(map(tuple, aligned[['DATASET', 'ID']].to_numpy()))].to_numpy(float), equal_nan=True)
        frames[name] = aligned
    args.output.mkdir(parents=True)
    frozen = dict(config=config, sources_sha256=hashes,
        code_sha256={str(p): sha256(p) for p in [Path(__file__), args.config, ROOT / 'src/component_composition_v73.py']})
    (args.output / 'frozen.json').write_text(json.dumps(frozen, indent=2) + '\n')
    # Freeze input/code provenance BEFORE calculating candidate metrics.
    xlsr, eat, spear = [frames[name][PREDICTION_COLUMNS].to_numpy(float) for name in ['xlsr', 'eat', 'spear']]
    combinations = {'xlsr_voice_eat_music': compose_components(xlsr, eat),
                    'equal_voice_pair_eat_music': compose_components(xlsr, eat, spear)}
    outputs = {'eat_parent': frames['eat'], 'xlsr_parent': frames['xlsr'], 'spear_parent': frames['spear']}
    for name, values in combinations.items():
        frame = frames['eat'].copy()
        frame[PREDICTION_COLUMNS] = values
        outputs[name] = frame
    overall, types, channels = [], [], []
    for name, frame in outputs.items():
        summary, slices = metrics(frame)
        overall.append(dict(variant=name, **summary['overall'], macro_ads=summary['macro_ads'], selection=summary['selection']))
        types.extend(dict(variant=name, **row) for row in typed_metrics(frame))
        slices.to_csv(args.output / f'{name}_slices.csv', index=False)
        codec = frame.loc[frame.DATASET.eq('codec_mixed_dev_v4')]
        channels.append(dict(variant=name, channel='all', **score_frame(codec)))
        channels.extend(dict(variant=name, channel=channel, **score_frame(block)) for channel, block in codec.groupby('CHANNEL'))
        if name in combinations:
            frame.to_csv(args.output / f'{name}_predictions.csv', index=False)
    if any(sha256(Path(path)) != digest for path, digest in hashes.items()):
        raise ValueError('source artifacts changed during composition evaluation')
    pd.DataFrame(overall).to_csv(args.output / 'overall.csv', index=False)
    pd.DataFrame(types).to_csv(args.output / 'by_type.csv', index=False)
    pd.DataFrame(channels).to_csv(args.output / 'codec_channels.csv', index=False)
    report = dict(stage='fixed component composition development diagnostic', overall=overall,
        newly_fitted_parameters=0, rows=len(truth), no_cross_file_statistics=True, automatic_submission_allowed=False,
        official_improvement_verified=False,
        limitations=['same previously used development sources; not source/generator OOD evidence',
                     'fixed noisy-OR may be poorly calibrated; no weight/temperature search performed',
                     'local total score is not an estimate of leaderboard score'])
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(overall), flush=True)


if __name__ == '__main__':
    main()
