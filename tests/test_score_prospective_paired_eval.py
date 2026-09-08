from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.score_codec_development import write_development_report
from scripts.score_prospective_paired_eval import (
    CHANNELS,
    COMPONENT_CASES,
    LAYOUTS,
    build_provenance,
    main,
    official_eer,
    score_frozen_v47_v48,
    score_prospective_paired_eval,
    write_reports,
)


@pytest.fixture
def paired_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    truth_rows = []
    prediction_rows = []
    for layout in LAYOUTS:
        for component_case in COMPONENT_CASES:
            parent = f"{layout}_{component_case}"
            voice_fake = int(component_case[0] == "F")
            music_fake = int(component_case[1] == "F")
            file_fake = voice_fake | music_fake
            for channel in CHANNELS:
                sample_id = f"{parent}_{channel}"
                truth_rows.append({
                    "ID": sample_id,
                    "BASE_ID": parent,
                    "FILE_FAKE": file_fake,
                    "VOICE_FAKE": voice_fake,
                    "MUSIC_FAKE": music_fake,
                    "VOICE_PRESENT": 1,
                    "MUSIC_PRESENT": 1,
                    "MIX_MODE": layout,
                    "COMPONENT_CASE": component_case,
                    "CHANNEL": channel,
                })
                voice_probability = .9 if voice_fake else .1
                if channel == "g711_ulaw":
                    voice_probability = .05 if voice_fake else .95
                prediction_rows.append({
                    "ID": sample_id,
                    "FILE_FAKE_PROB": .9 if file_fake else .1,
                    "VOICE_FAKE_PROB": voice_probability,
                    "MUSIC_FAKE_PROB": .9 if music_fake else .1,
                    "VOICE_PRESENT_PROB": .9,
                    "MUSIC_PRESENT_PROB": .9,
                })
    truth = pd.DataFrame(truth_rows)
    prediction = pd.DataFrame(prediction_rows).sample(
        frac=1, random_state=7
    ).reset_index(drop=True)
    return truth, prediction


def test_official_eer_uses_full_roc_thresholds():
    assert official_eer([0, 0, 1, 1], [.1, .4, .35, .8]) == pytest.approx(.5)
    assert np.isnan(official_eer([1, 1], [.1, .9]))


def test_scores_overall_channels_cells_and_paired_changes(paired_fixture):
    truth, prediction = paired_fixture
    reports = score_prospective_paired_eval(truth, prediction)

    assert set(reports) == {
        "overall", "channels", "cell_contrasts", "paired_deltas",
        "paired_rank_flips",
    }
    overall = reports["overall"].iloc[0]
    assert overall.N == 60
    assert overall.FILE_EER == pytest.approx(0)
    assert overall.MUSIC_EER == pytest.approx(0)
    assert 0 < overall.VOICE_EER < 1
    # This 12-cell bank has no presence-negative examples.  The scorer must
    # expose that CPS/Total are undefined instead of inventing a substitute.
    assert not bool(overall.CPS_DEFINED)
    assert np.isnan(overall.CPS)
    assert np.isnan(overall.TOTAL)

    channels = reports["channels"].set_index("CHANNEL")
    assert list(reports["channels"].CHANNEL) == list(CHANNELS)
    assert channels.loc["clean", "VOICE_EER"] == pytest.approx(0)
    assert channels.loc["g711_ulaw", "VOICE_EER"] == pytest.approx(1)

    cells = reports["cell_contrasts"]
    assert len(cells) == 12
    assert set(zip(cells.MIX_MODE, cells.COMPONENT_CASE)) == {
        (layout, case) for layout in LAYOUTS for case in COMPONENT_CASES
    }
    ff = cells.loc[
        cells.MIX_MODE.eq("concurrent") & cells.COMPONENT_CASE.eq("FF")
    ].iloc[0]
    assert ff.FILE_REFERENCE_CASE == "RR"
    assert ff.VOICE_REFERENCE_CASE == "RF"
    assert ff.MUSIC_REFERENCE_CASE == "FR"

    deltas = reports["paired_deltas"]
    ranks = reports["paired_rank_flips"]
    assert len(deltas) == 12 * 4 * 3
    assert len(ranks) == 4 * 3
    g711_voice = ranks.loc[
        ranks.CHANNEL.eq("g711_ulaw") & ranks.TASK.eq("VOICE")
    ].iloc[0]
    assert g711_voice.PAIRS == 36
    assert g711_voice.CORRECT_TO_INCORRECT == 36
    assert g711_voice.INCORRECT_TO_CORRECT == 0
    assert g711_voice.RANK_STATE_FLIPS == 36


def test_scores_frozen_v47_and_v48_together(paired_fixture):
    truth, v47 = paired_fixture
    v48 = v47.copy()
    v48["FILE_FAKE_PROB"] = (
        .98 * v48.FILE_FAKE_PROB + .01
    )
    reports = score_frozen_v47_v48(truth, v47, v48)
    expected_rows = {
        "overall": 2,
        "channels": 10,
        "cell_contrasts": 24,
        "paired_deltas": 288,
        "paired_rank_flips": 24,
    }
    for name, rows in expected_rows.items():
        assert len(reports[name]) == rows
        assert set(reports[name].MODEL) == {"v47", "v48"}


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("id_mismatch", "ID mismatch"),
        ("invalid_probability", "finite and within"),
        ("invalid_label", "only 0/1"),
        ("file_or", "must equal"),
        ("missing_channel", "every channel"),
        ("cell_mismatch", "COMPONENT_CASE disagrees"),
        ("parent_metadata", "changes labels or cell"),
    ),
)
def test_rejects_invalid_inputs(paired_fixture, mutation: str, message: str):
    truth, prediction = (frame.copy() for frame in paired_fixture)
    if mutation == "id_mismatch":
        prediction.loc[0, "ID"] = "not_a_truth_id"
    elif mutation == "invalid_probability":
        prediction.loc[0, "FILE_FAKE_PROB"] = 1.01
    elif mutation == "invalid_label":
        truth.loc[0, "VOICE_FAKE"] = 2
    elif mutation == "file_or":
        truth.loc[truth.COMPONENT_CASE.eq("RR"), "FILE_FAKE"] = 1
    elif mutation == "missing_channel":
        sample_id = truth.iloc[0].ID
        truth = truth.loc[truth.ID.ne(sample_id)].copy()
        prediction = prediction.loc[prediction.ID.ne(sample_id)].copy()
    elif mutation == "cell_mismatch":
        truth.loc[0, "COMPONENT_CASE"] = "FF"
    elif mutation == "parent_metadata":
        truth.loc[0, "MIX_MODE"] = "sequential"
    with pytest.raises(ValueError, match=message):
        score_prospective_paired_eval(truth, prediction)


def test_rejects_duplicate_ids(paired_fixture):
    truth, prediction = (frame.copy() for frame in paired_fixture)
    prediction.loc[1, "ID"] = prediction.loc[0, "ID"]
    with pytest.raises(ValueError, match="duplicate"):
        score_prospective_paired_eval(truth, prediction)


def test_writes_fixed_report_set_once(tmp_path: Path, paired_fixture):
    truth, prediction = paired_fixture
    reports = score_frozen_v47_v48(truth, prediction, prediction.copy())
    truth_path = tmp_path / "truth.csv"
    v47_path = tmp_path / "v47.csv"
    v48_path = tmp_path / "v48.csv"
    truth.to_csv(truth_path, index=False)
    prediction.to_csv(v47_path, index=False)
    prediction.to_csv(v48_path, index=False)
    provenance = build_provenance(
        truth_path,
        v47_path,
        v48_path,
        truth_rows=len(truth),
        v47_rows=len(prediction),
        v48_rows=len(prediction),
    )
    output = tmp_path / "score"
    write_reports(reports, output, provenance)
    assert {path.name for path in output.iterdir()} == {
        "overall.csv", "channels.csv", "cell_contrasts.csv",
        "paired_deltas.csv", "paired_rank_flips.csv", "provenance.json",
    }
    written_provenance = json.loads(
        (output / "provenance.json").read_text(encoding="utf-8")
    )
    assert written_provenance["single_evaluation"] is True
    assert len(written_provenance["evaluation_fingerprint"]) == 64
    assert written_provenance["frozen_models"] == ["v47", "v48"]
    assert written_provenance["selection_or_sweep_supported"] is False
    assert written_provenance["eer_drop_intermediate"] is False
    assert written_provenance["inputs"]["truth"]["rows"] == 60
    assert written_provenance["inputs"]["truth"]["sha256"] == (
        hashlib.sha256(truth_path.read_bytes()).hexdigest()
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_reports(reports, output, provenance)


def test_refuses_even_an_existing_empty_output_directory(
    tmp_path: Path, paired_fixture,
):
    truth, prediction = paired_fixture
    reports = score_frozen_v47_v48(truth, prediction, prediction.copy())
    source = tmp_path / "source.csv"
    truth.to_csv(source, index=False)
    provenance = build_provenance(
        source, source, source,
        truth_rows=len(truth), v47_rows=len(truth), v48_rows=len(truth),
    )
    output = tmp_path / "already_exists"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_reports(reports, output, provenance)


def test_repeatable_development_writer_records_one_candidate(
    tmp_path: Path, paired_fixture,
):
    truth, prediction = paired_fixture
    truth_path = tmp_path / "truth.csv"
    prediction_path = tmp_path / "prediction.csv"
    truth.to_csv(truth_path, index=False)
    prediction.to_csv(prediction_path, index=False)
    output = tmp_path / "development_score"
    reports = write_development_report(truth_path, prediction_path, output)
    # The fixture deliberately reverses Voice ranking on one of five channels.
    assert reports["overall"].iloc[0].ADS == pytest.approx(0.96)
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["purpose"] == "repeatable_development_checkpoint_evaluation"
    assert provenance["threshold_or_weight_search"] is False
    assert provenance["prediction"]["rows"] == len(prediction)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_development_report(truth_path, prediction_path, output)


def test_development_writer_filters_one_dataset_from_trainer_output(
    tmp_path: Path, paired_fixture,
):
    truth, prediction = paired_fixture
    truth_path = tmp_path / "truth.csv"
    prediction_path = tmp_path / "trainer_predictions.csv"
    truth.to_csv(truth_path, index=False)
    selected = prediction.copy()
    selected.insert(0, "DATASET", "codec_mixed_dev_v4")
    unrelated = selected.copy()
    unrelated["DATASET"] = "another_development_domain"
    unrelated["ID"] = "other_" + unrelated["ID"]
    pd.concat((selected, unrelated), ignore_index=True).to_csv(
        prediction_path, index=False,
    )
    output = tmp_path / "filtered_score"
    reports = write_development_report(
        truth_path, prediction_path, output, "codec_mixed_dev_v4",
    )
    assert reports["overall"].iloc[0].ADS == pytest.approx(0.96)
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["prediction"]["rows"] == len(prediction)
    assert provenance["prediction"]["dataset_filter"] == "codec_mixed_dev_v4"
    with pytest.raises(ValueError, match="no rows"):
        write_development_report(
            truth_path, prediction_path, tmp_path / "missing_filter", "unknown",
        )


def test_cli_refuses_existing_output_before_reading_inputs(
    tmp_path: Path, monkeypatch,
):
    output = tmp_path / "already_scored"
    output.mkdir()
    monkeypatch.setattr(sys, "argv", [
        "score_prospective_paired_eval.py",
        "--truth", str(tmp_path / "missing_truth.csv"),
        "--v47-prediction", str(tmp_path / "missing_v47.csv"),
        "--v48-prediction", str(tmp_path / "missing_v48.csv"),
        "--output-dir", str(output),
    ])
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main()
