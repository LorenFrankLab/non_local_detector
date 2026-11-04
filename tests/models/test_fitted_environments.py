"""Fitting keeps a detector's ``environments`` parameter as configuration.

``fit`` writes the fitted copies of the environments to ``environments_`` and
never changes ``environments``, so ``get_params()`` (which Spyglass stores and
scikit-learn's ``clone`` rebuilds from) returns the configuration the detector
was constructed with, before and after fitting, predicting and EM.
"""

import numpy as np
import pytest
from sklearn.base import clone
from track_linearization import make_track_graph

from non_local_detector import (
    Environment,
    NonLocalSortedSpikesDetector,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def recording():
    time = np.arange(0.0, 2.0, 0.002)
    position = 50.0 + 40.0 * np.sin(2 * np.pi * time / 2.0)
    rng = np.random.default_rng(0)
    spike_times = [np.sort(rng.uniform(0.0, 2.0, 100)) for _ in range(3)]
    return time, position, spike_times


def _assert_unfitted_configuration(environments, originals):
    assert isinstance(environments, tuple)
    assert len(environments) == len(originals)
    for environment, original in zip(environments, originals, strict=True):
        assert environment is original
        assert not environment._is_fitted
        assert environment.place_bin_centers_ is None


def test_fit_writes_environments_and_leaves_the_parameter(recording):
    time, position, spike_times = recording
    environment = Environment(place_bin_size=5.0)
    detector = NonLocalSortedSpikesDetector(environments=environment)
    configured = detector.environments

    detector.fit(time, position, spike_times)

    assert detector.environments is configured
    _assert_unfitted_configuration(detector.environments, (environment,))
    _assert_unfitted_configuration(
        detector.get_params(deep=False)["environments"], (environment,)
    )
    (fitted,) = detector.environments_
    assert fitted is not environment
    assert fitted._is_fitted
    assert fitted.get_params() == environment.get_params()
    assert detector._get_environment_by_name("") is fitted
    assert not hasattr(clone(detector), "environments_")


# One EM iteration is enough to run the M-step.
@pytest.mark.filterwarnings("ignore:EM did not converge:UserWarning")
def test_predict_and_em_leave_the_parameter(recording):
    time, position, spike_times = recording
    environment = Environment(place_bin_size=5.0)
    detector = NonLocalSortedSpikesDetector(environments=environment)
    time_edges = detector.calculate_time_edges([time[0], time[-1]], trim=True)

    detector.estimate_parameters(
        position_time=time,
        position=position,
        spike_times=spike_times,
        time_edges=time_edges,
        max_iter=1,
    )
    fitted = detector.environments_
    detector.predict(
        spike_times=spike_times,
        time_edges=time_edges,
        position_time=time,
        position=position,
    )

    _assert_unfitted_configuration(detector.environments, (environment,))
    assert detector.environments_ is fitted


def test_predict_checks_position_against_the_fitted_environments(recording):
    """``set_params(environments=...)`` without a refit leaves the fit, so
    predict validates position against ``environments_``: a 1-D position the
    fitted open field accepts is not rejected because the new configuration
    has a track graph."""
    time, position, spike_times = recording
    detector = NonLocalSortedSpikesDetector(
        environments=Environment(place_bin_size=5.0)
    )
    detector.fit(time, position, spike_times)
    graph = make_track_graph([(0.0, 0.0), (100.0, 0.0)], [(0, 1)])
    detector.set_params(
        environments=Environment(
            track_graph=graph, edge_order=[(0, 1)], edge_spacing=0.0
        )
    )
    time_edges = detector.calculate_time_edges([time[0], time[-1]], trim=True)

    results = detector.predict(
        spike_times=spike_times,
        time_edges=time_edges,
        position_time=time,
        position=position,
    )
    assert np.isfinite(results.acausal_state_probabilities).all()
