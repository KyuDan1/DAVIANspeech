from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluate import equal_error_rate, evaluate


def test_equal_error_rate_for_perfect_and_reversed_scores():
    labels = np.array([0, 0, 1, 1])
    assert equal_error_rate(labels, np.array([0.1, 0.2, 0.8, 0.9])) == 0.0
    assert equal_error_rate(labels, np.array([0.9, 0.8, 0.2, 0.1])) == 1.0


def test_evaluate_matches_rows_by_id(tmp_path):
    submission = tmp_path / "submission.csv"
    truth = tmp_path / "truth.csv"
    pd.DataFrame({"ID": ["b", "a"], "FILE_FAKE_PROB": [0.9, 0.1]}).to_csv(submission, index=False)
    pd.DataFrame({"ID": ["a", "b"], "FILE_FAKE": [0, 1]}).to_csv(truth, index=False)

    result = evaluate(submission, truth)
    assert result.loc["FILE_FAKE_PROB", "auc"] == 1.0
    assert result.loc["FILE_FAKE_PROB", "eer"] == 0.0
