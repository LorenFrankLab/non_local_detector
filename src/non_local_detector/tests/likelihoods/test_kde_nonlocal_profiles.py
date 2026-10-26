"""Non-local KDE likelihood profiles match independent float64 Gaussian references.

Each encoding model is fit on an animal running 0-10 cm at 1 cm/s, with three
encoding spikes at known positions. A Gaussian position kernel of unit width
gives the place field (sorted) or mark intensity (clusterless) as
``mean_rate * mean_i N(x; x_i, 1) / mean_t N(x; x(t), 1)``, computed here with
SciPy in float64, so each decoded profile is checked against the encoding
positions rather than against the fitted model's own arrays.
"""

import numpy as np
import pytest
from scipy.stats import norm

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.likelihoods.common import LOG_EPS, RATE_REFERENCE_SECONDS
from non_local_detector.likelihoods.mark_reference import (
    background_log_intensity,
    fit_mark_references,
)
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
    predict_sorted_spikes_kde_log_likelihood,
)

pytestmark = pytest.mark.unit

POSITION_TIME = np.linspace(0.0, 10.0, 201)
POSITION = POSITION_TIME[:, None]
POSITION_STD = 1.0
# Float32 evaluation against the float64 reference: measured at most 2.3e-6
# (sorted) and 1.4e-5 (clusterless) absolute on log likelihoods of order 1-50.
# Moving one encoding position by one bin shifts the profiles by O(1).
REFERENCE_TOLERANCE = {"rtol": 1e-5, "atol": 1e-4}


def _line(n_bins: int) -> Environment:
    """A 1 cm-bin track from 0 to ``n_bins - 1`` cm."""
    return Environment(
        environment_name="line",
        place_bin_size=1.0,
        position_range=((0.0, float(n_bins - 1)),),
    ).fit_place_grid(
        position=np.linspace(0.0, float(n_bins - 1), n_bins)[:, None],
        infer_track_interior=False,
    )


def _kde_rate(centers, encoding_positions, kernel=None):
    """Float64 KDE rate in Hz at ``centers``, shape (n_bins,).

    ``kernel`` of shape (n_encoding_spikes,) weights each encoding spike (its
    mark kernel at the decoding mark); unweighted gives the ground rate.
    """
    spatial = norm.pdf(centers[:, None], loc=encoding_positions, scale=POSITION_STD)
    if kernel is not None:
        spatial = spatial * kernel
    occupancy = norm.pdf(centers[:, None], loc=POSITION[:, 0], scale=POSITION_STD)
    mean_rate = len(encoding_positions) / (
        len(POSITION_TIME) * (POSITION_TIME[1] - POSITION_TIME[0])
    )
    return mean_rate * spatial.mean(axis=1) / occupancy.mean(axis=1)


def _interior_centers(environment: Environment) -> np.ndarray:
    return environment.place_bin_centers_[environment.is_track_interior_.ravel(), 0]


def test_sorted_kde_nonlocal_profile_matches_gaussian_reference():
    """Each row with one spike is ``log(rate * dt) - rate * dt`` at the
    reference rate, so its argmax is where the reference expects one event."""
    environment = _line(11)
    encoding_spike_times = np.array([3.0, 3.1, 2.9])
    encoding_model = fit_sorted_spikes_kde_encoding_model(
        position_time=POSITION_TIME,
        position=POSITION,
        spike_times=[encoding_spike_times],
        environment=environment,
        position_std=POSITION_STD,
        block_size=32,
        disable_progress_bar=True,
    )
    # One spike per 2 s decoding bin, including the final one.
    time_edges = np.linspace(0.0, 10.0, 6)
    log_likelihood = np.asarray(
        predict_sorted_spikes_kde_log_likelihood(
            time_edges=time_edges,
            position_time=POSITION_TIME,
            position=POSITION,
            spike_times=[(time_edges[:-1] + time_edges[1:]) / 2.0],
            environment=environment,
            marginal_models=encoding_model["marginal_models"],
            occupancy_model=encoding_model["occupancy_model"],
            occupancy=encoding_model["occupancy"],
            mean_rates=encoding_model["mean_rates"],
            place_fields=encoding_model["place_fields"],
            no_spike_part_log_likelihood=encoding_model["no_spike_part_log_likelihood"],
            is_track_interior=encoding_model["is_track_interior"],
            disable_progress_bar=True,
            is_local=False,
        )
    )

    # The animal runs at 1 cm/s from 0 cm, so each spike time is its position.
    rate = _kde_rate(_interior_centers(environment), encoding_spike_times)
    dt = np.diff(time_edges)[0]
    expected = np.log(rate * dt) - rate * dt

    assert log_likelihood.shape == (len(time_edges) - 1, expected.size)
    np.testing.assert_allclose(
        log_likelihood,
        np.broadcast_to(expected, log_likelihood.shape),
        **REFERENCE_TOLERANCE,
    )
    np.testing.assert_array_equal(log_likelihood.argmax(axis=1), expected.argmax())


def test_clusterless_kde_nonlocal_profile_matches_gaussian_reference():
    """One spike in a 10 s bin scores ``logaddexp(log(mark intensity * dt),
    log background) - ground rate * dt`` at every bin.

    The profile is bimodal: the integrated ground rate penalizes the encoding
    mode for predicting many more than one event, and far from the encoding
    spikes the No-Spike background bounds the spike term.
    """
    environment = _line(31)
    waveform_std = 0.25
    encoding_spike_times = np.array([7.0, 7.1, 6.9])
    encoding_features = np.array([[0.0, 0.0], [0.05, -0.05], [-0.05, 0.05]])
    decoding_features = np.array([[0.02, -0.01]])
    encoding_model = fit_clusterless_kde_encoding_model(
        position_time=POSITION_TIME,
        position=POSITION,
        spike_times=[encoding_spike_times],
        spike_waveform_features=[encoding_features],
        environment=environment,
        position_std=POSITION_STD,
        waveform_std=waveform_std,
        block_size=16,
        disable_progress_bar=True,
    )
    mark_reference = fit_mark_references([encoding_features], waveform_std)
    time_edges = np.array([0.0, 10.0])
    log_likelihood = np.asarray(
        predict_clusterless_kde_log_likelihood(
            time_edges=time_edges,
            position_time=POSITION_TIME,
            position=POSITION,
            spike_times=[np.array([5.0])],
            spike_waveform_features=[decoding_features],
            occupancy=encoding_model["occupancy"],
            occupancy_model=encoding_model["occupancy_model"],
            gpi_models=encoding_model["gpi_models"],
            encoding_spike_waveform_features=encoding_model[
                "encoding_spike_waveform_features"
            ],
            encoding_positions=encoding_model["encoding_positions"],
            mark_reference=mark_reference,
            environment=environment,
            mean_rates=encoding_model["mean_rates"],
            summed_ground_process_intensity=encoding_model[
                "summed_ground_process_intensity"
            ],
            position_std=encoding_model["position_std"],
            waveform_std=encoding_model["waveform_std"],
            is_local=False,
            block_size=16,
            disable_progress_bar=True,
        )
    )

    centers = _interior_centers(environment)
    mark_kernel = np.prod(
        norm.pdf(decoding_features[0], loc=encoding_features, scale=waveform_std),
        axis=1,
    )
    ground_rate = _kde_rate(centers, encoding_spike_times)
    mark_intensity = _kde_rate(centers, encoding_spike_times, mark_kernel)
    dt = time_edges[1] - time_edges[0]
    # The background is a rate per RATE_REFERENCE_SECONDS, scaled to the bin.
    log_background = background_log_intensity(
        mark_reference[0], decoding_features, LOG_EPS
    )[0] + np.log(dt / RATE_REFERENCE_SECONDS)
    expected = (
        np.logaddexp(np.log(mark_intensity * dt), log_background) - ground_rate * dt
    )

    assert log_likelihood.shape == (1, expected.size)
    np.testing.assert_allclose(log_likelihood[0], expected, **REFERENCE_TOLERANCE)
    assert log_likelihood[0].argmax() == expected.argmax()
