from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('gain_training_v65', ROOT / 'scripts/train_paired_wpt_file_v60.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def setup_data(monkeypatch):
    audio = np.linspace(-.5, .5, 8, dtype=np.float32)
    monkeypatch.setattr(training, 'load_audio', lambda _: audio.copy())
    monkeypatch.setattr(training, 'resolve_ffmpeg', lambda: Path('/test/ffmpeg'))
    monkeypatch.setattr(training, 'codec_pair', lambda windows, *args: windows.copy())
    return audio, pd.DataFrame([dict(PATH='/unused.wav', FILE_FAKE=1)])


def test_default_gain_preserves_v60_draw_and_next_rng_state(monkeypatch):
    audio, frame = setup_data(monkeypatch)
    config = dict(window=8, channels=['g711_ulaw'])
    np.random.seed(93)
    expected = audio * np.float32(np.random.uniform(.7, 1.3))
    np.random.randint(1)
    np.random.randint(2**30)
    next_draw = np.random.random()
    np.random.seed(93)
    clean, channel, label, index = training.Bags(frame, config, True)[0]
    np.testing.assert_array_equal(clean[0], expected)
    np.testing.assert_array_equal(channel, clean)
    assert np.random.random() == next_draw
    assert (label, index) == (1., 0)


def test_broad_gain_only_affects_training_and_same_pair(monkeypatch):
    audio, frame = setup_data(monkeypatch)
    config = dict(window=8, channels=['g711_ulaw'], gain_min=.1, gain_max=.1)
    clean, channel, label, _ = training.Bags(frame, config, True)[0]
    np.testing.assert_array_equal(clean[0], audio * np.float32(.1))
    np.testing.assert_array_equal(channel, clean)
    assert label == 1.
    clean_eval, channel_eval, _, _ = training.Bags(frame, config, False)[0]
    np.testing.assert_array_equal(clean_eval[0], audio)
    assert channel_eval is None


@pytest.mark.parametrize('low,high', [(0, 1), (-1, 1), (2, 1), (float('nan'), 1), (.1, float('inf'))])
def test_invalid_gain_is_rejected(low, high):
    with pytest.raises(ValueError, match='gain_min'):
        training.Bags(pd.DataFrame(), dict(gain_min=low, gain_max=high), False)


def test_config_is_matched_bce_except_gain_range():
    control = yaml.safe_load((ROOT / 'configs/paired_wpt_file_v60.yaml').read_text())
    candidate = yaml.safe_load((ROOT / 'configs/paired_gain_wpt_v65.yaml').read_text())
    ignored = {'schema_version', 'variants', 'decision', 'gain_min', 'gain_max'}
    assert {k: v for k, v in candidate.items() if k not in ignored} == {
        k: v for k, v in control.items() if k not in ignored}
    assert candidate['variants']['paired_gain'] == control['variants']['paired_bce']
    assert candidate['gain_min'] == .1 and candidate['gain_max'] == 1.3
    assert not candidate.get('peak_normalize', False)
