import numpy as np
import pytest

from src.component_composition_v73 import compose_components


def test_formula_and_conditional_component_outputs():
    xlsr = np.array([[.1, .8, .2, 1., 1.]])
    eat = np.array([[.2, .1, .6, 1., 1.]])
    np.testing.assert_allclose(compose_components(xlsr, eat), [[.92, .8, .6, 1., 1.]])


def test_absent_components_do_not_change_file_risk():
    source = np.ones((1, 5))
    eat = np.array([[.9, .9, .6, 0., 1.]])
    np.testing.assert_allclose(compose_components(source, eat), [[.6, 1., .6, 0., 1.]])
    eat[:, 4] = 0
    assert compose_components(source, eat)[0, 0] == 0


def test_equal_voice_logits_and_row_independence():
    xlsr = np.array([[.1, .2, .2, 1., 1.], [.2, .9, .3, 1., 1.]])
    spear = xlsr.copy()
    spear[:, 1] = 1 - xlsr[:, 1]
    result = compose_components(xlsr, xlsr, spear)
    np.testing.assert_allclose(result[:, 1], .5)
    np.testing.assert_array_equal(result[:1], compose_components(xlsr[:1], xlsr[:1], spear[:1]))
    changed = xlsr.copy()
    changed[1] = 0
    np.testing.assert_array_equal(result[:1], compose_components(changed, xlsr, spear)[:1])


@pytest.mark.parametrize('bad', [np.full((1, 5), np.nan), np.full((1, 5), 1.1), np.zeros((2, 4)), np.zeros((2, 5))])
def test_invalid_or_unaligned_input_rejected(bad):
    with pytest.raises(ValueError):
        compose_components(bad, np.zeros((1, 5)))


def test_integrated_models_receive_only_the_current_waveform():
    from src.component_composition_inference_v73 import ComponentCompositionPredictor
    model = object.__new__(ComponentCompositionPredictor)
    calls = []
    audio = np.zeros(16000, np.float32)
    def source(name, values):
        def predict(waveform):
            assert waveform is audio
            calls.append(name)
            return np.asarray(values, np.float64)
        return predict
    model.models = {'xlsr': source('xlsr', [.1, .8, .1, 1., 1.]),
                    'eat': source('eat', [.1, .1, .6, 1., 1.]),
                    'spear': source('spear', [.1, .2, .1, 1., 1.])}
    np.testing.assert_allclose(model(audio), [.8, .5, .6, 1., 1.])
    assert calls == ['xlsr', 'eat', 'spear']
