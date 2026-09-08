import numpy as np
import pytest

from scripts.benchmark_gpu_colocation_v73 import assemble, shard_indices


@pytest.mark.parametrize('workers', [1, 2, 4, 8])
def test_shards_cover_each_file_once(workers):
    values = [i for rank in range(workers) for i in shard_indices(7, workers, rank)]
    assert sorted(values) == list(range(7))


def test_assembly_preserves_file_order():
    expected = np.arange(15).reshape(3, 5) / 15
    parts = [dict(indices=[1], probabilities=expected[[1]]),
             dict(indices=[0, 2], probabilities=expected[[0, 2]])]
    np.testing.assert_array_equal(assemble(parts, 3), expected)


@pytest.mark.parametrize('parts', [
    [dict(indices=[0, 0], probabilities=[[.5] * 5] * 2)],
    [dict(indices=[0, 0, 1], probabilities=[[.5] * 5] * 3)],
    [dict(indices=[0, 2], probabilities=[[.5] * 5] * 2)],
    [dict(indices=[0], probabilities=[[.5] * 5])],
    [dict(indices=[0, 1], probabilities=[[float('nan')] * 5] * 2)],
])
def test_assembly_rejects_invalid_results(parts):
    with pytest.raises(ValueError):
        assemble(parts, 2)
