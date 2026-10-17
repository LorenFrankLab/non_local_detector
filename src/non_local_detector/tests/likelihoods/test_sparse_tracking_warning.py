"""Encoding fits warn when tracking moves farther per sample than their bandwidth.

KDE fits build occupancy from the tracking samples while spikes are placed on
the interpolated path between them, so samples much farther apart than the
position bandwidth leave the occupancy unresolved where spikes land. Diffusion
and MRF fits bin spikes with their bracketing samples, as they do exposure, so
they have no such mismatch and do not warn.
"""

import warnings

import numpy as np
import pytest

from non_local_detector.encoding_time import (
    EncodingSupport,
    largest_tracking_step,
    warn_if_tracking_is_sparse,
)
from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_diffusion import (
    fit_clusterless_diffusion_encoding_model,
)
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
)
from non_local_detector.likelihoods.sorted_spikes_diffusion import (
    fit_sorted_spikes_diffusion_encoding_model,
)
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
)
from non_local_detector.likelihoods.sorted_spikes_mrf import (
    fit_sorted_spikes_mrf_encoding_model,
)

pytestmark = pytest.mark.unit

MESSAGE = "Tracking moves up to"


def _track(step: float, n_samples: int = 41):
    """Back-and-forth motion on [0, 40] at ``step`` units per sample, 1 Hz."""
    time = np.arange(n_samples, dtype=float)
    position = 20.0 + 20.0 * np.sin(np.cumsum(np.full(n_samples, step / 20.0)))
    return time, position[:, None]


def _fit(name, time, position, position_std):
    environment = Environment(place_bin_size=2.0, position_range=((-1.0, 41.0),))
    environment = environment.fit_place_grid(position, infer_track_interior=False)
    spikes = [time[::3] + 0.25]
    if name == "sorted_kde":
        return fit_sorted_spikes_kde_encoding_model(
            time,
            position,
            spikes,
            environment,
            position_std=position_std,
            disable_progress_bar=True,
        )
    if name == "sorted_diffusion":
        return fit_sorted_spikes_diffusion_encoding_model(
            time,
            position,
            spikes,
            environment,
            position_std=position_std,
            disable_progress_bar=True,
        )
    if name == "sorted_mrf":
        return fit_sorted_spikes_mrf_encoding_model(
            time, position, spikes, environment, disable_progress_bar=True
        )
    marks = [np.zeros((len(spikes[0]), 2))]
    fit = (
        fit_clusterless_kde_encoding_model
        if name == "clusterless_kde"
        else fit_clusterless_diffusion_encoding_model
    )
    return fit(
        time,
        position,
        spikes,
        marks,
        environment,
        position_std=position_std,
        disable_progress_bar=True,
    )


FITS = ["sorted_kde", "clusterless_kde"]


@pytest.mark.parametrize("name", FITS)
def test_fit_warns_when_tracking_steps_exceed_the_bandwidth(name):
    time, position = _track(step=6.0)
    with pytest.warns(UserWarning, match=MESSAGE):
        _fit(name, time, position, position_std=2.0)


@pytest.mark.parametrize(
    "name", ["sorted_diffusion", "sorted_mrf", "clusterless_diffusion"]
)
def test_sample_basis_fits_do_not_warn_about_sparse_tracking(name):
    time, position = _track(step=6.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fit(name, time, position, position_std=2.0)
    assert not [w for w in caught if MESSAGE in str(w.message)]


@pytest.mark.parametrize("name", FITS)
def test_fit_is_quiet_when_tracking_resolves_the_bandwidth(name):
    time, position = _track(step=1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _fit(name, time, position, position_std=2.0)
    assert not [w for w in caught if MESSAGE in str(w.message)]


def test_largest_step_ignores_jumps_between_tracking_segments():
    time = np.arange(6.0)
    position = np.array([0.0, 1.0, 2.0, 50.0, 51.0, 52.0])[:, None]
    split = EncodingSupport(
        time, position, valid_position_intervals=[[0.0, 2.0], [3.0, 5.0]]
    )
    assert largest_tracking_step(split, position) == 1.0
    joined = EncodingSupport(time, position)
    assert largest_tracking_step(joined, position) == 48.0


def test_largest_step_ignores_jumps_across_missing_samples():
    time = np.arange(5.0)
    position = np.array([0.0, 1.0, np.nan, 40.0, 41.0])[:, None]
    support = EncodingSupport(time, position)
    assert largest_tracking_step(support, np.nan_to_num(position)) == 1.0


def test_steps_are_euclidean_in_two_dimensions():
    time = np.arange(3.0)
    position = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 5.0]])
    support = EncodingSupport(time, position)
    assert largest_tracking_step(support, position) == 5.0
    with pytest.warns(UserWarning, match=MESSAGE):
        warn_if_tracking_is_sparse(support, position, np.array([4.0, 2.0]))
