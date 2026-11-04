"""A corrupt or hand-edited saved model raises ``DataError``, not a bare KeyError.

Missing state, unknown configuration keys, dangling references and malformed
nodes are caught at the load boundary and chained to the original error.
"""

import numpy as np
import pytest

import non_local_detector as nld
from non_local_detector.exceptions import DataError
from non_local_detector.model_format import _registry, register_type
from non_local_detector.model_format._codec import Decoder
from tests.model_format.conftest import algorithm_params
from tests.model_format.test_refusals import _edit_json

pytestmark = pytest.mark.integration


@pytest.fixture
def saved(sorted_recording, tmp_path):
    time, position, spike_times = sorted_recording
    detector = nld.NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm_params=algorithm_params("sorted_spikes_kde")
    ).fit(time, position, spike_times)
    path = tmp_path / "model.nld"
    detector.save_model(path)
    return path


def test_fitted_flag_without_state_is_a_data_error(saved):
    def edit(state):
        for key in list(state):
            if key != "fitted":
                del state[key]

    _edit_json(saved, "state.json", edit)

    with pytest.raises(DataError, match="marked fitted but its state has no"):
        nld.NonLocalSortedSpikesDetector.load_model(saved)


def test_missing_fitted_flag_is_a_data_error(saved):
    _edit_json(saved, "state.json", lambda state: state.pop("fitted"))

    with pytest.raises(DataError, match="fitted"):
        nld.NonLocalSortedSpikesDetector.load_model(saved)


def test_unknown_config_key_points_to_a_different_package_version(saved):
    _edit_json(saved, "config.json", lambda config: config.update(not_a_param=1))

    with pytest.raises(DataError, match="not_a_param") as info:
        nld.NonLocalSortedSpikesDetector.load_model(saved)
    assert "version" in str(info.value)


@pytest.mark.parametrize(
    "node",
    [
        {"$item": 7, "index": 0},
        {"$array": "arrays/x.npy", "jax": False},
        {"$dict": [[[1, 2], 3]]},
        {"$tuple": 5},
    ],
    ids=["dangling_item", "array_without_ref", "unhashable_dict_key", "bad_tuple"],
)
def test_malformed_nodes_are_data_errors(node):
    with pytest.raises(DataError, match="corrupt or was edited") as info:
        Decoder(lambda member: np.zeros(1)).decode(node)
    assert isinstance(info.value.__cause__, (KeyError, TypeError))


def test_a_registered_types_own_constructor_error_is_chained(sorted_recording):
    class Strict:
        def __init__(self, weight):
            self.weight = weight

    register_type(
        Strict,
        "testlab.Strict",
        to_state=lambda obj: {"weight": obj.weight},
        from_state=lambda state: Strict(**state),
    )
    try:
        node = {"$type": "testlab.Strict", "state": {"weight": 1, "extra": 2}}
        with pytest.raises(DataError, match="testlab.Strict") as info:
            Decoder(lambda member: np.zeros(1)).decode(node)
        assert isinstance(info.value.__cause__, TypeError)
        assert "extra" in str(info.value)
    finally:
        _registry._BY_NAME.pop("testlab.Strict", None)
        _registry._BY_CLASS.pop(Strict, None)
