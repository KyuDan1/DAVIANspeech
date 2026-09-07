import numpy as np
import pytest

from src.multistream_prompt_inference import blend_file_experts


def logit(values):
    values = np.asarray(values, dtype=np.float64)
    return np.log(values) - np.log1p(-values)


def test_blend_file_experts_changes_only_file_in_logit_space():
    wpt = np.asarray([[.1, .2, .3], [.8, .7, .6]])
    multistream = np.asarray([[.9, .8, .7], [.2, .3, .4]])
    result = blend_file_experts(wpt, multistream, .2)
    assert np.array_equal(result[:, :2], wpt[:, :2])
    assert np.allclose(
        logit(result[:, 2]), .8 * logit(wpt[:, 2]) + .2 * logit(multistream[:, 2]),
    )


def test_blend_file_experts_validates_weight_and_shape():
    values = np.ones((2, 3)) * .5
    with pytest.raises(ValueError):
        blend_file_experts(values, values, 1.1)
    with pytest.raises(ValueError):
        blend_file_experts(values, values[:, :2], .2)
