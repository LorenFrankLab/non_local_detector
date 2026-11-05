"""A corrupt or hand-edited saved model raises ``DataError``, not a bare KeyError.

Missing state, unknown configuration keys, dangling references and malformed
nodes are caught at the load boundary and chained to the original error.
"""

import json
import re

import numpy as np
import pytest

import non_local_detector as nld
from non_local_detector.exceptions import (
    CompatibilityWarning,
    DataError,
    ValidationError,
)
from non_local_detector.model_format import _registry, register_type
from non_local_detector.model_format._codec import Decoder
from tests.model_format.conftest import algorithm_params
from tests.model_format.test_refusals import _edit_json, _rewrite

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


def test_unknown_config_key_is_ignored_naming_a_newer_version(saved):
    _edit_json(saved, "config.json", lambda config: config.update(not_a_param=1))

    with pytest.warns(CompatibilityWarning, match="not_a_param") as info:
        detector = nld.NonLocalSortedSpikesDetector.load_model(saved)
    assert "newer version" in str(info[0].message)
    assert "not_a_param" not in detector.get_params()


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
        with pytest.raises(DataError, match=re.escape("testlab.Strict")) as info:
            Decoder(lambda member: np.zeros(1)).decode(node)
        assert isinstance(info.value.__cause__, TypeError)
        assert "extra" in str(info.value)
    finally:
        _registry._BY_NAME.pop("testlab.Strict", None)
        _registry._BY_CLASS.pop(Strict, None)


def _edit_state(path, edit):
    _edit_json(path, "state.json", edit)


@pytest.mark.parametrize(
    ("document", "edit"),
    [
        (
            "state.json",
            lambda s: s["encoding_models"][0].pop("environment_name"),
        ),
        ("state.json", lambda s: s.update(encoding_models={"a": 1})),
        ("manifest.json", lambda s: s.pop("detector_type")),
        ("state.json", lambda s: s.update(fingerprints=5)),
        ("state.json", lambda s: s.update(fingerprints={})),
        ("state.json", lambda s: s["fingerprints"].pop("continuous_transitions")),
        (
            "state.json",
            lambda s: s["fingerprints"].update(environments=[]),
        ),
    ],
    ids=[
        "encoding_entry_without_name",
        "encoding_models_not_entries",
        "manifest_without_type",
        "fingerprints_scalar",
        "fingerprints_empty",
        "fingerprints_without_transitions",
        "fingerprints_wrong_environment_count",
    ],
)
def test_edited_documents_are_data_errors(saved, document, edit):
    _edit_json(saved, document, edit)

    with pytest.raises(DataError, match="corrupt or was edited"):
        nld.NonLocalSortedSpikesDetector.load_model(saved)


def test_dataframe_with_mismatched_columns_is_a_data_error():
    node = {"$dataframe": {"columns": ["a", "b"], "values": [[1]], "index": {}}}

    with pytest.raises(DataError, match="corrupt or was edited") as info:
        Decoder(lambda member: np.zeros(1), plain=True).decode(node)
    assert isinstance(info.value.__cause__, ValueError)


def test_a_package_error_inside_a_node_is_not_rewrapped():
    with pytest.raises(ValidationError, match="Unknown tag"):
        Decoder(lambda member: np.zeros(1)).decode({"$tuple": [{"$bogus": 1}]})


def test_a_user_from_state_reading_a_missing_key_names_the_type():
    class Keyed:
        def __init__(self, weight):
            self.weight = weight

    register_type(
        Keyed,
        "testlab.Keyed",
        to_state=lambda obj: {"weight": obj.weight},
        from_state=lambda state: Keyed(state["weight"]),
    )
    try:
        node = {"$type": "testlab.Keyed", "state": {}}
        with pytest.raises(DataError) as info:
            Decoder(lambda member: np.zeros(1)).decode(node)
        text = str(info.value)
        assert "testlab.Keyed" in text
        assert "from_state" in text
        assert "stored state" in text
        assert isinstance(info.value.__cause__, KeyError)
    finally:
        _registry._BY_NAME.pop("testlab.Keyed", None)
        _registry._BY_CLASS.pop(Keyed, None)


def _nested_tuples(depth):
    return json.loads('{"$tuple": [' * depth + "]}" * depth)


@pytest.mark.parametrize(
    ("document", "transform"),
    [
        # Too deep for the JSON parser.
        ("state.json", lambda data: b"[" * 100_000 + b"]" * 100_000),
        # Parsed, but too deep to decode.
        (
            "config.json",
            lambda data: json.dumps(
                {**json.loads(data), "place_bin_size": _nested_tuples(900)}
            ).encode(),
        ),
    ],
    ids=["parse", "decode"],
)
def test_deeply_nested_documents_are_data_errors(saved, document, transform):
    _rewrite(saved, document, transform)

    with pytest.raises(DataError, match=r"not valid JSON|RecursionError"):
        nld.NonLocalSortedSpikesDetector.load_model(saved)
