import numpy as np
import pytest
from scripts.verify_paired_checkpoint_inference_v68 import compare_arrays


def test_matching_arrays_pass_and_bf16_tolerance_is_explicit():
    assert compare_arrays([.1, .2], [.1, .2], [.1, .2])['pass_check']
    assert compare_arrays([.1, .2], [.1001, .2], [.1001, .2])['pass_check']
    assert not compare_arrays([.1, .2], [.102, .2], [.102, .2])['pass_check']


def test_file_order_difference_fails_even_if_small():
    result = compare_arrays([.1, .2], [.1, .2], [.100000001, .2])
    assert result['saved_evaluation_close'] and not result['file_order_bit_exact']
    assert not result['pass_check']


@pytest.mark.parametrize('array', [[], [np.nan, .2], [np.inf, .2], [-.1, .2], [1.1, .2]])
def test_invalid_probabilities_rejected(array):
    with pytest.raises(ValueError):
        compare_arrays(array, [.1, .2], [.1, .2])
