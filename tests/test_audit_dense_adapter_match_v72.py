import copy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_dense_adapter_match_v72 import compare_configs


def configs():
    reference = dict(schema='dense_component_v71', seed=71, epochs=6, samples_per_epoch=4096,
                     synthetic_channels=['clean', 'g711_ulaw'], temperature=5.)
    adapted = copy.deepcopy(reference)
    adapted.update(schema='dense_component_adapter_v72', adapter_learning_rate=.0001)
    return reference, adapted


def test_only_adapter_training_change_is_accepted():
    reference, adapted = configs()
    compare_configs(reference, adapted)


@pytest.mark.parametrize('key,value', [('seed', 72), ('epochs', 8), ('temperature', 2.),
    ('samples_per_epoch', 8192), ('synthetic_channels', ['clean']), ('adapter_learning_rate', .001)])
def test_data_budget_or_unregistered_settings_cannot_change(key, value):
    reference, adapted = configs()
    adapted[key] = value
    with pytest.raises(ValueError):
        compare_configs(reference, adapted)
