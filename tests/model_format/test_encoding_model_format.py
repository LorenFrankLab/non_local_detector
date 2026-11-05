"""A saved encoding model holds the keys of its algorithm's type.

``load_model`` refuses an encoding model without a key its algorithm's type
requires, or naming an algorithm that is not registered, and ignores, with a
warning, a key the type does not declare (a newer version wrote it).
"""

import shutil

import numpy as np
import pytest

import non_local_detector as nld
from non_local_detector.exceptions import CompatibilityWarning, DataError
from non_local_detector.model_format import read_model_summary
from tests.model_format.conftest import SORTED_RATE, algorithm_params, decode
from tests.model_format.test_refusals import _edit_json

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def sorted_file(sorted_recording, tmp_path_factory):
    time, position, spike_times = sorted_recording
    path = tmp_path_factory.mktemp("encoding") / "model.nld"
    nld.SortedSpikesDecoder(
        sampling_frequency=SORTED_RATE,
        sorted_spikes_algorithm_params=algorithm_params("sorted_spikes_kde"),
    ).fit(time, position, spike_times).save_model(path)
    return path


@pytest.fixture
def saved(sorted_file, tmp_path):
    return shutil.copy(sorted_file, tmp_path / "model.nld")


def _edit_model(path, edit):
    _edit_json(
        path, "state.json", lambda state: edit(state["encoding_models"][0]["model"])
    )


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda model: model.pop("place_fields"), "'place_fields'"),
        (lambda model: model.update(algorithm="mylab_kde"), "'mylab_kde'"),
        (lambda model: model.pop("algorithm"), "algorithm"),
    ],
    ids=["missing_key", "unknown_algorithm", "no_algorithm"],
)
def test_an_encoding_model_off_its_type_is_refused(saved, edit, match):
    _edit_model(saved, edit)

    with pytest.raises(DataError, match=match):
        nld.SortedSpikesDecoder.load_model(saved)
    with pytest.raises(DataError, match=match):
        read_model_summary(saved)


def test_a_key_the_type_does_not_declare_is_ignored_with_a_warning(
    saved, sorted_file, sorted_recording
):
    """The key is left out of the loaded model, so predict, which takes no
    other keywords, decodes as the file without it."""
    _edit_model(saved, lambda model: model.update(future_key=1))
    original = nld.SortedSpikesDecoder.load_model(sorted_file)

    with pytest.warns(CompatibilityWarning, match="future_key"):
        loaded = nld.SortedSpikesDecoder.load_model(saved)
    with pytest.warns(CompatibilityWarning, match="future_key"):
        summary = read_model_summary(saved)

    (model,) = loaded.encoding_model_.values()
    assert "future_key" not in model
    assert "future_key" not in summary["encoding_models"][0]
    time, position, spike_times = sorted_recording
    for a, b in zip(
        decode(loaded, time, position, {"spike_times": spike_times}),
        decode(original, time, position, {"spike_times": spike_times}),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)
