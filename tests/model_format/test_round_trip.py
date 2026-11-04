"""A saved and reloaded detector equals the original and decodes the same.

Every registered likelihood algorithm with every ready-made detector class of
its family is fitted on a small simulated recording, saved with
``save_model`` and reloaded with ``load_model``. The reloaded detector must
hold the same fitted state (rebuilt structures included) and its posterior
must equal the original's exactly: on one machine the rebuild repeats the
fit's arithmetic.
"""

import numpy as np
import pytest

from non_local_detector.model_format import is_saved_model_format
from tests.model_format._compare import assert_same_fitted_state
from tests.model_format.conftest import (
    CLUSTERLESS_ALGORITHMS,
    CLUSTERLESS_CLASSES,
    SORTED_ALGORITHMS,
    SORTED_CLASSES,
    decode,
    make_detector,
    recording_args,
)

pytestmark = pytest.mark.integration

CASES = [
    (cls, algorithm) for cls in SORTED_CLASSES for algorithm in SORTED_ALGORITHMS
] + [
    (cls, algorithm)
    for cls in CLUSTERLESS_CLASSES
    for algorithm in CLUSTERLESS_ALGORITHMS
]


@pytest.mark.parametrize(
    ("cls", "algorithm"),
    CASES,
    ids=[f"{cls.__name__}-{algorithm}" for cls, algorithm in CASES],
)
def test_reloaded_detector_has_the_same_state_and_posterior(
    cls, algorithm, sorted_recording, clusterless_recording, tmp_path
):
    detector = make_detector(cls, algorithm)
    time, position, spikes, fit_kwargs = recording_args(
        detector, sorted_recording, clusterless_recording
    )
    detector.fit(time, position, **spikes, **fit_kwargs)
    path = tmp_path / "model.nld"

    detector.save_model(path)
    loaded = cls.load_model(path)

    assert is_saved_model_format(path)
    assert_same_fitted_state(detector, loaded)
    posterior, states = decode(detector, time, position, spikes)
    loaded_posterior, loaded_states = decode(loaded, time, position, spikes)
    np.testing.assert_array_equal(loaded_posterior, posterior)
    np.testing.assert_array_equal(loaded_states, states)


def test_unfitted_detector_round_trips_its_configuration(tmp_path):
    detector = make_detector(SORTED_CLASSES[0], "sorted_spikes_kde")
    path = tmp_path / "unfitted.nld"

    detector.save_model(path)
    loaded = SORTED_CLASSES[0].load_model(path)

    assert_same_fitted_state(detector, loaded)
