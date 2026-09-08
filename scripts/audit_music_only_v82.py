"""Evaluate exactly one changed output against recomputed official-anchor scores."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import numpy as np
    import pandas as pd
    from src.evaluate_diagnostic import PREDICTION_COLUMNS, score_frame
    base_path = ROOT / 'reports/exact_anchor_v74/codec_recomputed/predictions.csv'
    base_report_path = base_path.parent / 'report.json'
    candidate_path = ROOT / 'reports/component_composition_v73/equal_voice_pair_eat_music_predictions.csv'
    out = ROOT / 'reports/music_only_submission_v82/codec_audit'
    out.mkdir(exist_ok=False)
    base_report = json.loads(base_report_path.read_text())
    assert base_report['status'] == 'complete'
    assert sha(base_path) == base_report['predictions_sha256']
    assert sha(candidate_path) == 'd13b0d9818d6d583da6c8cf4cfdc136e6e27d5af81a4e01624e8666ff015b183'
    base = pd.read_csv(base_path)
    candidate = pd.read_csv(candidate_path, low_memory=False)
    candidate = candidate[candidate.DATASET.eq('codec_mixed_dev_v4')].copy()
    keys = ['DATASET','ID']
    assert len(base) == len(candidate) == 600
    assert not base.duplicated(keys).any() and not candidate.duplicated(keys).any()
    indexed = base.set_index(keys).loc[list(map(tuple, candidate[keys].to_numpy()))]
    anchor = candidate.copy()
    anchor[PREDICTION_COLUMNS] = indexed[PREDICTION_COLUMNS].to_numpy()
    changed = anchor.copy()
    changed['MUSIC_FAKE_PROB'] = candidate['MUSIC_FAKE_PROB'].to_numpy()
    unchanged = [c for c in PREDICTION_COLUMNS if c != 'MUSIC_FAKE_PROB']
    assert np.array_equal(anchor[unchanged].to_numpy(), changed[unchanged].to_numpy())
    metrics = []
    for name, frame in [('actual_anchor', anchor), ('music_only_v82', changed)]:
        metrics.append(dict(model=name, axis='ALL', group='ALL', **score_frame(frame)))
        for column in ['CHANNEL','VOICE_FAKE','MIX_MODE']:
            if column in frame:
                for value, subset in frame.groupby(column):
                    metrics.append(dict(model=name, axis=column, group=str(value), **score_frame(subset)))
    pd.DataFrame(metrics).to_csv(out / 'metrics.csv', index=False)
    changed.to_csv(out / 'predictions.csv', index=False)
    report = dict(status='complete_development_only_audit', rows=600,
                  inputs_sha256={str(p):sha(p) for p in [base_path,base_report_path,candidate_path]},
                  changed_column='MUSIC_FAKE_PROB', other_four_outputs_exact=True,
                  metrics=metrics, official_score_prediction=False,
                  limitation='Previously exposed synthetic codec-mixed development; single-class presence makes CPS/total undefined.')
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
