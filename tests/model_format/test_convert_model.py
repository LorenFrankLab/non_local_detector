"""``convert_model`` rewrites a decodable legacy pickle in the saved format.

The pickles are the stored Spyglass artifacts of ``tests/compat_data``. A
converted model must hold the pickle's fitted state and decode exactly as the
pickle does; models that cannot decode (fitted before rates were in Hz, or
with the removed ``clusterless_kde_log``) are refused and stay pickles.
"""

import pickle

import numpy as np
import pytest

from non_local_detector.exceptions import NotFittedError, ValidationError
from non_local_detector.model_format import convert_model, is_saved_model_format
from non_local_detector.models import SortedSpikesDecoder
from tests.model_format._compare import assert_same_fitted_state
from tests.models.test_stored_artifact_compatibility import (
    COMPAT_DATA,
    LEGACY_VERSIONS,
    MODELS,
    _family,
    _recording,
    _spikes_in,
)

pytestmark = [
    pytest.mark.integration,
    # NumPy 1.x pickles name numpy.core, which NumPy 2 reads with a warning.
    pytest.mark.filterwarnings(
        "ignore:numpy.core.numeric is deprecated:DeprecationWarning"
    ),
]
DECODABLE = [(v, a) for v, a in MODELS if v not in LEGACY_VERSIONS]
UNDECODABLE = [(v, a) for v, a in MODELS if v in LEGACY_VERSIONS]


@pytest.fixture(scope="module")
def inputs():
    return pickle.loads((COMPAT_DATA / "inputs.pkl").read_bytes())


def _pickle_path(version, artifact):
    return COMPAT_DATA / version / f"{artifact}_model.pkl"


def _posterior(model, inputs, artifact):
    position, spikes = _recording(inputs, "clusterless" in artifact)
    time = inputs["time"]
    time_edges = model.calculate_time_edges([time[0], time[-1]], trim=True)
    return model.predict(
        **_spikes_in(spikes, time_edges),
        time_edges=time_edges,
        position_time=time,
        position=position,
    ).acausal_posterior.values


@pytest.mark.parametrize(("version", "artifact"), DECODABLE)
def test_converted_model_decodes_as_the_pickle(version, artifact, inputs, tmp_path):
    pickled = _family(artifact).load_model(_pickle_path(version, artifact))
    path = tmp_path / f"{artifact}.nld"

    convert_model(_pickle_path(version, artifact), path)
    converted = _family(artifact).load_model(path)

    assert is_saved_model_format(path)
    # A converted model rebuilds its continuous transitions on load, and the
    # rebuild check admits differences up to its 1e-6 relative tolerance, so
    # its decode can differ from the pickle's (which keeps its stored
    # transitions) at that level.
    np.testing.assert_allclose(
        _posterior(converted, inputs, artifact),
        _posterior(pickled, inputs, artifact),
        rtol=1e-6,
        atol=2e-6,
    )
    # Versions before environments_ kept the fitted environments in
    # ``environments`` too; a converted model keeps only their configuration
    # there.
    del pickled.environments, converted.environments
    assert_same_fitted_state(pickled, converted)


@pytest.mark.parametrize(("version", "artifact"), UNDECODABLE)
def test_model_that_cannot_decode_is_not_converted(version, artifact, tmp_path):
    path = tmp_path / "converted.nld"
    match = "clusterless_kde_log" if "kde_log" in artifact else "legacy or unknown time"

    with pytest.raises(ValidationError, match=match):
        convert_model(_pickle_path(version, artifact), path)
    assert not path.exists()


@pytest.mark.parametrize(("version", "artifact"), UNDECODABLE)
def test_a_model_that_cannot_be_read_back_is_not_saved(version, artifact, tmp_path):
    """``load_model`` refuses a file without these names, so ``save_model``
    does not write one, as ``convert_model`` does not."""
    path = tmp_path / "model.nld"
    model = _family(artifact).load_model(_pickle_path(version, artifact))

    with pytest.raises(NotFittedError, match="partly fitted"):
        model.save_model(path)
    assert not path.exists()


def test_a_saved_model_is_not_converted_again(tmp_path):
    path = tmp_path / "model.nld"
    version, artifact = DECODABLE[0]
    _family(artifact).load_model(_pickle_path(version, artifact)).save_model(path)

    with pytest.raises(ValidationError, match="already"):
        convert_model(path, tmp_path / "again.nld")


def test_an_unfitted_pickle_is_not_converted(tmp_path):
    pickle_path = tmp_path / "unfitted.pkl"
    pickle_path.write_bytes(pickle.dumps(SortedSpikesDecoder()))
    path = tmp_path / "converted.nld"

    with pytest.raises(ValidationError, match="cannot decode"):
        convert_model(pickle_path, path)
    assert not path.exists()
