"""What a saved model must hold, and what it may hold that is ignored.

A newer version may add a stored name or field. An older version loads the
file without it, with one ``DataQualityWarning`` per load listing what it
ignored, from ``load_model`` and ``read_model_summary`` alike. A fitted file
must hold every name its kind of model always has, or ``DataError``; one
missing raised a bare ``AttributeError`` (or loaded and failed later)
before.
"""

import shutil

import pytest

import non_local_detector as nld
from non_local_detector.exceptions import DataError, DataQualityWarning
from non_local_detector.model_format import read_model_summary
from non_local_detector.model_format._detector import _OPTIONAL_STORED, _STORED
from tests.model_format._compare import assert_same_fitted_state
from tests.model_format.conftest import (
    CLUSTERLESS_RATE,
    SORTED_RATE,
    algorithm_params,
)
from tests.model_format.test_refusals import _edit_json

pytestmark = pytest.mark.integration

# Every stored name a fitted sorted-spikes detector without EM has.
_ALWAYS_STORED = [name for name, _ in _STORED if name not in _OPTIONAL_STORED]


@pytest.fixture(scope="module")
def sorted_file(sorted_recording, tmp_path_factory):
    time, position, spike_times = sorted_recording
    detector = nld.NonLocalSortedSpikesDetector(
        sampling_frequency=SORTED_RATE,
        sorted_spikes_algorithm_params=algorithm_params("sorted_spikes_kde"),
    ).fit(time, position, spike_times)
    path = tmp_path_factory.mktemp("schema") / "sorted.nld"
    detector.save_model(path)
    return path


@pytest.fixture
def saved(sorted_file, tmp_path):
    """A copy of the saved sorted-spikes model, for a test to edit."""
    return shutil.copy(sorted_file, tmp_path / "model.nld")


def _refused(path, match):
    with pytest.raises(DataError, match=match) as loading:
        nld.NonLocalSortedSpikesDetector.load_model(path)
    with pytest.raises(DataError, match=match):
        read_model_summary(path)
    return loading.value


def _loads_with_one_warning(path, *names, summary=True):
    """Load (and summarize) ``path``, which warns once, naming ``names``."""
    with pytest.warns(DataQualityWarning, match="newer version") as warned:
        model = nld.NonLocalSortedSpikesDetector.load_model(path)
    assert len(warned) == 1
    # A UserWarning, which Python's default filters show.
    assert issubclass(warned[0].category, UserWarning)
    for name in names:
        assert name in str(warned[0].message)
    if summary:
        with pytest.warns(DataQualityWarning, match=names[0]) as warned:
            read_model_summary(path)
        assert len(warned) == 1
    return model


@pytest.mark.parametrize(
    ("document", "edit", "where"),
    [
        ("state.json", lambda s: s.update(future_state=1), "state.json"),
        ("manifest.json", lambda m: m.update(future_entry=1), "manifest.json"),
        (
            "state.json",
            lambda s: s["environments"][0].update(future_field=1),
            "environments entries",
        ),
        (
            "state.json",
            lambda s: s["encoding_models"][0].update(future_field=1),
            "encoding_models entries",
        ),
        (
            "state.json",
            lambda s: s["fingerprints"].update(future_fingerprint=1),
            "fingerprints",
        ),
    ],
    ids=["state", "manifest", "environment", "encoding_model", "fingerprints"],
)
def test_a_name_this_version_does_not_read_is_ignored_with_a_warning(
    saved, sorted_file, document, edit, where
):
    _edit_json(saved, document, edit)

    model = _loads_with_one_warning(saved, where, "future_")

    assert_same_fitted_state(
        nld.NonLocalSortedSpikesDetector.load_model(sorted_file), model
    )


@pytest.mark.parametrize("name", ["nld.Environment", "nld.Discrete"])
def test_a_field_a_stored_object_does_not_store_is_ignored_with_a_warning(saved, name):
    def add_field(config):
        nodes = [
            config["environments"]["$tuple"][0],
            *config["continuous_transition_types"][0],
        ]
        node = next(n for n in nodes if n["$type"] == name)
        node["state"]["future_field"] = 1

    _edit_json(saved, "config.json", add_field)

    # The summary builds no objects, so it returns the field as read.
    _loads_with_one_warning(saved, "future_field", summary=False)


def test_several_ignored_names_give_one_warning(saved):
    def add(state):
        state["future_state"] = 1
        state["environments"][0]["future_field"] = 1

    _edit_json(saved, "state.json", add)
    _edit_json(saved, "config.json", lambda config: config.update(future_param=1))

    _loads_with_one_warning(saved, "future_state", "future_field", "future_param")


@pytest.mark.parametrize(
    "name",
    [
        "environments",
        "encoding_models",
        "stored_transition_blocks",
        "fingerprints",
        "discrete_state_transitions",
        *_ALWAYS_STORED,
    ],
)
def test_a_fitted_file_without_a_name_it_always_has_is_refused(saved, name):
    _edit_json(saved, "state.json", lambda state: state.pop(name))

    error = _refused(saved, "marked fitted but its state has no")

    assert repr(name) in str(error)


def test_a_clusterless_file_without_its_mark_reference_is_refused(
    clusterless_recording, tmp_path
):
    time, position, spike_times, features = clusterless_recording
    path = tmp_path / "model.nld"
    nld.ClusterlessDecoder(
        sampling_frequency=CLUSTERLESS_RATE,
        clusterless_algorithm_params=algorithm_params("clusterless_kde"),
    ).fit(time, position, spike_times, features).save_model(path)
    _edit_json(path, "state.json", lambda state: state.pop("mark_reference"))

    with pytest.raises(DataError, match="'mark_reference'"):
        nld.ClusterlessDecoder.load_model(path)
    with pytest.raises(DataError, match="'mark_reference'"):
        read_model_summary(path)


def test_an_unfitted_file_ignores_fitted_state(tmp_path):
    path = tmp_path / "model.nld"
    nld.NonLocalSortedSpikesDetector().save_model(path)
    _edit_json(path, "state.json", lambda state: state.update(initial_conditions=1))

    model = _loads_with_one_warning(path, "initial_conditions")

    assert not hasattr(model, "initial_conditions_")


def test_a_malformed_entry_in_the_summary_is_a_data_error(saved):
    """The summary decodes an environment's name instead of indexing its node."""
    _edit_json(saved, "state.json", lambda s: s["environments"][0].pop("config"))

    with pytest.raises(DataError, match="corrupt or was edited") as error:
        read_model_summary(saved)
    assert isinstance(error.value.__cause__, KeyError)
