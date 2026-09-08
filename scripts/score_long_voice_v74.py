#!/usr/bin/env python3
"""Score the frozen v74 comparison once; never adjust candidate or gates here."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
from src.evaluate_diagnostic import official_eer
from run_exact_anchor_v74 import COLUMNS, sha256, verify_inventory
from run_long_voice_v74 import RUN, frozen_run


def gate_decision(paired, gates):
    required_axes = {'ALL', 'DURATION', 'CHANNEL'}
    decisions = []
    for bank in ['uniform', 'fixed']:
        selected = paired.loc[paired.BANK.eq(bank) & paired.AXIS.isin(required_axes)]
        if (len(selected) != 7 or selected[['AXIS', 'GROUP']].duplicated().any()
                or set(selected.AXIS) != required_axes
                or len(selected[selected.AXIS.eq('ALL')]) != 1
                or set(selected.loc[selected.AXIS.eq('DURATION'), 'GROUP'].astype(str)) != {'30', '45', '60'}
                or set(selected.loc[selected.AXIS.eq('CHANNEL'), 'GROUP']) != {'clean', 'g711_ulaw', 'opus_nb_8k'}
                or not np.isfinite(selected.FILE_EER_delta.to_numpy(float)).all()):
            raise ValueError('all seven predeclared gate cells required per bank')
        for _, row in selected.iterrows():
            limit = gates['max_length_channel_regression']
            if row.AXIS == 'ALL':
                limit = (-gates['uniform_min_pooled_file_eer_improvement'] if bank == 'uniform'
                         else gates['fixed_max_pooled_regression'])
            decisions.append(dict(BANK=bank, AXIS=row.AXIS, GROUP=row.GROUP,
                delta=float(row.FILE_EER_delta), maximum_allowed_delta=limit,
                passed=bool(row.FILE_EER_delta <= limit + 1e-12)))
    return dict(passed=all(d['passed'] for d in decisions), cells=decisions)


def main():
    output = RUN / 'comparison'
    if output.exists():
        raise FileExistsError('one-shot comparison already scored')
    frozen = frozen_run()
    verify_inventory(frozen['package_attestation'])
    root_hash = sha256(RUN / 'frozen.json')
    frames, prediction_hashes = {}, {}
    for name in ['anchor', 'candidate']:
        pieces = []
        for shard in range(frozen['shards']):
            stage = RUN / name / f'shard_{shard}'
            local = json.loads((stage / 'frozen.json').read_text())
            report = json.loads((stage / 'completed.json').read_text())
            path = stage / 'output/submission.csv'
            if (local['root_frozen_sha256'] != root_hash
                    or report['frozen_sha256'] != sha256(stage / 'frozen.json')
                    or report['predictions_sha256'] != sha256(path)):
                raise ValueError('changed/mismatched prediction artifacts')
            prediction = pd.read_csv(path, dtype={'ID': str})
            if prediction.ID.duplicated().any() or set(prediction.ID) != {r['ID'] for r in local['inputs']}:
                raise ValueError('incomplete or duplicate prediction IDs')
            values = prediction[COLUMNS].to_numpy(float)
            if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
                raise ValueError('invalid probabilities')
            pieces.append(prediction[['ID', *COLUMNS]])
            prediction_hashes[str(path)] = sha256(path)
        predictions = pd.concat(pieces, ignore_index=True)
        if predictions.ID.duplicated().any() or set(predictions.ID) != {r['ID'] for r in frozen['inputs']}:
            raise ValueError('wrong full comparison population')
        frames[name] = predictions
    rows, bank_frames = [], {}
    for bank, record in frozen['banks'].items():
        truth = pd.read_csv(record['truth'], dtype={'ID': str, 'GROUP_ID': str})
        if len(truth) != record['rows'] or truth.GROUP_ID.nunique() != record['source_groups']:
            raise ValueError('bank count changed')
        for name, predictions in frames.items():
            frame = truth.merge(predictions, on='ID', validate='one_to_one')
            if len(frame) != len(truth):
                raise ValueError('bank missing predictions')
            bank_frames[bank, name] = frame
            for axes in [[], ['DURATION'], ['CHANNEL'], ['RECORDING'], ['POSITION'],
                         ['DURATION', 'CHANNEL'], ['DURATION', 'RECORDING', 'CHANNEL']]:
                groups = [('ALL', frame)] if not axes else frame.groupby(axes, dropna=False)
                for keys, part in groups:
                    if not isinstance(keys, tuple):
                        keys = (keys,)
                    rows.append(dict(BANK=bank, MODEL=name, AXIS='|'.join(axes) if axes else 'ALL',
                        GROUP='|'.join(map(str, keys)), N=len(part),
                        FILE_EER=official_eer(part.FILE_FAKE, part.FILE_FAKE_PROB),
                        VOICE_EER=official_eer(part.VOICE_FAKE, part.VOICE_FAKE_PROB)))
    table = pd.DataFrame(rows)
    keys = ['BANK', 'AXIS', 'GROUP']
    anchor = table[table.MODEL.eq('anchor')].set_index(keys)
    candidate = table[table.MODEL.eq('candidate')].set_index(keys)
    paired = anchor[['N', 'FILE_EER', 'VOICE_EER']].join(candidate[['FILE_EER', 'VOICE_EER']],
        lsuffix='_anchor', rsuffix='_candidate')
    for task in ['FILE_EER', 'VOICE_EER']:
        paired[task + '_delta'] = paired[task + '_candidate'] - paired[task + '_anchor']
    paired = paired.reset_index()
    decision = gate_decision(paired, frozen['gates'])
    bootstrap = {}
    for bank in frozen['banks']:
        anchor, candidate = [bank_frames[bank, name].set_index('ID') for name in ['anchor', 'candidate']]
        candidate = candidate.loc[anchor.index]
        groups = sorted(anchor.GROUP_ID.unique())
        group_indices = {g: np.flatnonzero(anchor.GROUP_ID.to_numpy() == g) for g in groups}
        generator = np.random.default_rng(20260905)
        deltas = []
        for _ in range(1000):
            indices = np.concatenate([group_indices[g] for g in generator.choice(groups, len(groups), replace=True)])
            real, new = anchor.iloc[indices], candidate.iloc[indices]
            deltas.append(official_eer(new.FILE_FAKE, new.FILE_FAKE_PROB)
                          - official_eer(real.FILE_FAKE, real.FILE_FAKE_PROB))
        bootstrap[bank] = dict(group_count=len(groups), repetitions=1000,
            file_eer_delta_ci95=np.quantile(deltas, [.025, .975]).tolist(),
            scope='source-group bootstrap; fixed/uniform share source groups')
    frozen_run()
    output.mkdir()
    table.to_csv(output / 'by_condition.csv', index=False)
    paired.to_csv(output / 'paired_differences.csv', index=False)
    for name, frame in frames.items():
        frame.to_csv(output / (name + '_predictions.csv'), index=False)
    report = dict(status='complete', gate=decision, bootstrap=bootstrap,
        frozen_sha256=root_hash, prediction_sha256=prediction_hashes,
        no_retuning=True, automatic_submission_allowed=False,
        scope='synthetic held-source long voice only; NOT music/CPS/natural-call/leaderboard validation')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(paired.loc[paired.AXIS.eq('ALL')].to_string(index=False), flush=True)
    print(json.dumps(dict(gate_passed=decision['passed'], bootstrap=bootstrap)), flush=True)


if __name__ == '__main__':
    main()
