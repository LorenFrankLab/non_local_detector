"""EM estimates the discrete initial probabilities only.

Each state's distribution over its own bins stays as configured at fit, so a
detector after ``estimate_parameters`` starts any recording the way a fitted
one does, apart from how probable each state is at the first bin.
"""

import numpy as np
import pytest

from non_local_detector import (
    Environment,
    NonLocalSortedSpikesDetector,
    SortedSpikesDecoder,
    time_edges_from_centers,
)
from non_local_detector.core import EM_MONOTONICITY_SLACK
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data
from tests.conftest import em_stops_at_max_iter

# EM that estimates only the initial conditions, so a detector after it
# differs from the fitted one in nothing else.
IC_ONLY = {
    "estimate_initial_conditions": True,
    "estimate_discrete_transition": False,
    "estimate_encoding_model": False,
}


def _window(recording, start, stop):
    time, position, spike_times = recording
    t = time[start:stop]
    return {
        "position_time": t,
        "position": position[start:stop],
        "spike_times": [st[(st >= t[0]) & (st <= t[-1])] for st in spike_times],
    }


@pytest.fixture(scope="module")
def recording():
    """Three seconds of 12 simulated place cells at 1500 Hz, and a window.

    Returns ``(fitted, other, sampling_frequency)``: the fitted recording's
    first 4,500 samples, and 300 samples from where the animal is first more
    than 60 cm from its starting position (35 position bins of 5 cm), as
    keyword arguments of ``fit``.
    """
    _, position, spike_times, time, _, fs, _, _ = make_simulated_data(
        n_neurons=12, seed=0
    )
    position = np.asarray(position).reshape(-1, 1)
    full = (np.asarray(time), position, spike_times)
    start = int(np.argmax(np.abs(position[:, 0] - position[0, 0]) > 60.0))
    assert 0 < start < 4_200
    return _window(full, 0, 4_500), _window(full, start, start + 300), float(fs)


def _make(cls, sampling_frequency, **kwargs):
    return cls(
        environments=Environment(place_bin_size=5.0, position_range=((0.0, 175.0),)),
        infer_track_interior=False,
        sampling_frequency=sampling_frequency,
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
        **kwargs,
    )


def _edges(window):
    return time_edges_from_centers(window["position_time"])


def _assert_objective_non_decreasing(results):
    """Each EM step raises the penalized log-likelihood, up to float32 noise.

    The noise allowed is the package's monotonicity slack, 10 float32 epsilons
    relative to the objective.
    """
    objectives = np.asarray(results.attrs["em_objectives"], dtype=np.float64)
    previous = np.asarray(results.attrs["em_previous_objectives"], dtype=np.float64)
    steps = objectives[1 : len(previous) + 1] - previous
    assert steps.size > 0
    slack = EM_MONOTONICITY_SLACK * np.abs(previous)
    assert np.all(steps >= -slack), steps


def _local_probability(detector, window):
    results = detector.predict(**window, time_edges=_edges(window))
    return results.acausal_state_probabilities.sel(states="Local").values


@pytest.mark.integration
@em_stops_at_max_iter
@pytest.mark.parametrize("local_position_std", [0.0, 5.0])
def test_multi_bin_local_state_keeps_its_probability_on_another_window(
    recording, local_position_std
):
    """After EM, a window that starts with the animal in another bin starts
    in the Local state, as it does after ``fit``.

    The Local state's uniform initial distribution is what its position
    kernel, scaled to sum to the number of bins, is built against. An initial
    distribution peaked at the fitted recording's first position gave Local
    probability 0 (delta kernel) or about 1e-17 (5 cm kernel) at the
    window's first bin, and below 0.5 for its first 17 or so bins.
    """
    fitted, other, fs = recording
    fit = _make(
        NonLocalSortedSpikesDetector, fs, local_position_std=local_position_std
    ).fit(**fitted)
    em = _make(NonLocalSortedSpikesDetector, fs, local_position_std=local_position_std)
    results = em.estimate_parameters(
        **fitted, time_edges=_edges(fitted), **IC_ONLY, max_iter=3
    )
    _assert_objective_non_decreasing(results)

    p_fit = _local_probability(fit, other)
    p_em = _local_probability(em, other)
    assert p_fit[0] > 0.99
    assert p_em[0] > 0.99, p_em[:5]
    # Only the discrete initial probabilities differ between the two.
    np.testing.assert_allclose(p_em[:50], p_fit[:50], atol=0.01)


@pytest.mark.integration
@em_stops_at_max_iter
@pytest.mark.parametrize("local_position_std", [0.0, 5.0])
def test_em_gain_in_marginal_likelihood_is_that_of_the_discrete_update(
    recording, local_position_std
):
    """One initial-condition M-step raises the marginal log-likelihood by
    log(sum_s g_s**2 / p_s), with p the configured and g the estimated
    discrete initial probabilities.

    With each state's distribution over its bins held fixed, the marginal
    likelihood is sum_s p_s E_s, where E_s is the evidence of the recording
    started in state s, and the first bin's smoothed state probability is
    g_s = p_s E_s / sum_r p_r E_r. Replacing p by g multiplies the marginal
    likelihood by sum_s g_s E_s / sum_r p_r E_r = sum_s g_s**2 / p_s.
    Estimating the multi-bin Local state's distribution over its bins as well
    added about log(35), the log of its number of bins, on top. EM starts
    from equal discrete probabilities, so the gain (about 0.01 nats) is well
    above float32 noise.
    """
    fitted, _, fs = recording
    em = _make(
        NonLocalSortedSpikesDetector,
        fs,
        local_position_std=local_position_std,
        discrete_initial_conditions=np.full(4, 0.25),
    )
    results = em.estimate_parameters(
        **fitted, time_edges=_edges(fitted), **IC_ONLY, max_iter=1
    )
    _assert_objective_non_decreasing(results)

    configured = np.asarray(em.discrete_initial_conditions, dtype=np.float64)
    estimated = np.asarray(em.discrete_initial_conditions_, dtype=np.float64)
    expected_gain = np.log(np.sum(estimated**2 / configured))
    marginal = np.asarray(results.attrs["marginal_log_likelihoods"], dtype=np.float64)
    assert marginal.shape == (2,)
    # Float32 noise on the marginal log-likelihood, about 2e-4 nats here; the
    # gain matched within 5e-6 on macOS arm64.
    tolerance = EM_MONOTONICITY_SLACK * abs(marginal[0])
    assert expected_gain > 10 * tolerance
    np.testing.assert_allclose(marginal[1] - marginal[0], expected_gain, atol=tolerance)


@pytest.mark.integration
@em_stops_at_max_iter
def test_decoder_after_em_decodes_another_window_as_after_fit(recording):
    """A one-state decoder's only initial condition EM estimates is its
    discrete one, which is 1, so after EM it decodes any window as the
    fitted decoder does.

    Its distribution over position bins used to become the fitted
    recording's first-bin posterior, which the decoder then carried to every
    window it decoded: on this one the posteriors differed by a summed 1.8
    at the first bin.
    """
    fitted, other, fs = recording
    fit = _make(SortedSpikesDecoder, fs).fit(**fitted)
    em = _make(SortedSpikesDecoder, fs)
    results = em.estimate_parameters(
        **fitted, time_edges=_edges(fitted), **IC_ONLY, max_iter=3
    )
    _assert_objective_non_decreasing(results)

    expected = fit.predict(**other, time_edges=_edges(other)).acausal_posterior
    actual = em.predict(**other, time_edges=_edges(other)).acausal_posterior
    np.testing.assert_allclose(actual.values, expected.values, atol=1e-6)


@pytest.mark.integration
@pytest.mark.parametrize(
    "representation",
    [
        {"transition_representation": "dense"},
        {"transition_representation": "dense", "n_chunks": 3},
        {"transition_representation": "structured", "chunk_size": 700},
    ],
    ids=["dense", "dense-row-chunks", "structured"],
)
def test_every_em_path_keeps_the_configured_state_distributions(
    recording, representation
):
    """Default EM, on each E-step driver, leaves ``continuous_initial_conditions_``
    as ``fit`` set it and weights it by the estimated discrete probabilities.
    """
    fitted, _, fs = recording
    configured = (
        _make(NonLocalSortedSpikesDetector, fs, local_position_std=0.0)
        .fit(**fitted)
        .continuous_initial_conditions_
    )
    em = _make(NonLocalSortedSpikesDetector, fs, local_position_std=0.0)
    results = em.estimate_parameters(
        **fitted, time_edges=_edges(fitted), max_iter=20, **representation
    )
    _assert_objective_non_decreasing(results)

    np.testing.assert_array_equal(em.continuous_initial_conditions_, configured)
    discrete = np.asarray(em.discrete_initial_conditions_, dtype=np.float64)
    np.testing.assert_allclose(discrete.sum(), 1.0, rtol=1e-6)
    np.testing.assert_allclose(
        em.initial_conditions_,
        configured * discrete[em.state_ind_],
        rtol=1e-6,
        atol=0.0,
    )
    initial = em.initial_conditions_[em.is_track_interior_state_bins_]
    assert np.all(np.isfinite(em.initial_conditions_))
    np.testing.assert_allclose(initial.sum(dtype=np.float64), 1.0, rtol=1e-6)
