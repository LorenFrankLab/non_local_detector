"""EM leaves the initial conditions as configured.

One recording holds one observation of how a recording starts. The
initial distribution that maximizes its likelihood puts all its mass on the
fitted recording's first state, so a detector estimated that way started
every later decode there. ``estimate_parameters`` estimates the discrete
transitions and the encoding model; ``discrete_initial_conditions_``,
``continuous_initial_conditions_`` and ``initial_conditions_`` are what
``fit`` builds from the configuration.
"""

import warnings

import numpy as np
import pytest

from non_local_detector import (
    Environment,
    NonLocalSortedSpikesDetector,
    SortedSpikesDecoder,
    time_edges_from_centers,
)
from non_local_detector.exceptions import CompatibilityWarning
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data
from tests.conftest import em_stops_at_max_iter

# EM that estimates nothing, so a detector after it must equal the fitted one.
NOTHING_ESTIMATED = {
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


def _assert_initial_conditions_as_fitted(em, fit):
    """The three initial-condition attributes of ``em`` are those ``fit`` built."""
    np.testing.assert_array_equal(
        em.discrete_initial_conditions_, np.asarray(em.discrete_initial_conditions)
    )
    np.testing.assert_array_equal(
        em.continuous_initial_conditions_, fit.continuous_initial_conditions_
    )
    np.testing.assert_array_equal(em.initial_conditions_, fit.initial_conditions_)


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
def test_every_em_path_leaves_the_initial_conditions_as_fit_set_them(
    recording, representation
):
    """Default EM, on each E-step driver, changes none of the initial
    conditions: the discrete probabilities are the configured ones, and each
    state's distribution over its bins and the joint vector are bit for bit
    what ``fit`` built.

    EM used to replace the discrete probabilities with the first bin's
    smoothed state probabilities: after these 20 iterations they were
    ``[1, 2.9e-9, 0, 0]`` against a configured ``[0.99, 0.005, 0.0025,
    0.0025]``.
    """
    fitted, _, fs = recording
    fit = _make(NonLocalSortedSpikesDetector, fs, local_position_std=0.0).fit(**fitted)
    em = _make(NonLocalSortedSpikesDetector, fs, local_position_std=0.0)
    em.estimate_parameters(
        **fitted, time_edges=_edges(fitted), max_iter=20, **representation
    )

    _assert_initial_conditions_as_fitted(em, fit)
    initial = em.initial_conditions_[em.is_track_interior_state_bins_]
    np.testing.assert_allclose(initial.sum(dtype=np.float64), 1.0, rtol=1e-6)


@pytest.mark.integration
def test_non_local_states_stay_reachable_at_the_start_of_a_decode(recording):
    """After a long EM every state can still start a decode, so the posterior
    given a non-local state is defined from the first bin of any window.

    The estimated discrete initial probabilities reached exactly zero in
    float32 for both non-local states here (EM then took 14 iterations to
    converge; it takes 9 now), and the conditional non-local posterior was
    NaN in the first row of every decode.
    """
    fitted, other, fs = recording
    em = _make(NonLocalSortedSpikesDetector, fs)
    em.estimate_parameters(
        **fitted, time_edges=_edges(fitted), max_iter=20, tolerance=1e-12
    )
    assert em.n_iter_ > 5

    assert np.all(em.discrete_initial_conditions_ > 0)
    for window in (fitted, other):
        results = em.predict(**window, time_edges=_edges(window))
        conditional = em.get_conditional_non_local_posterior(results).values
        assert np.all(np.isfinite(conditional))


@pytest.mark.integration
@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (NonLocalSortedSpikesDetector, {"local_position_std": 0.0}),
        (NonLocalSortedSpikesDetector, {"local_position_std": 5.0}),
        (SortedSpikesDecoder, {}),
    ],
    ids=["delta-local", "multi-bin-local", "decoder"],
)
def test_detector_after_em_decodes_another_window_as_after_fit(recording, cls, kwargs):
    """With nothing else estimated, a detector after EM is the fitted
    detector: it decodes a window that starts elsewhere on the track
    identically.

    The estimated initial conditions were carried to every window decoded
    afterwards. The Local probability at this window's first 50 bins differed
    from the fitted detector's by up to 0.01, and before each state's
    distribution over its bins was held fixed, a decoder's posterior differed
    by up to 0.23 in a position bin.
    """
    fitted, other, fs = recording
    fit = _make(cls, fs, **kwargs).fit(**fitted)
    em = _make(cls, fs, **kwargs)
    em.estimate_parameters(**fitted, time_edges=_edges(fitted), **NOTHING_ESTIMATED)

    _assert_initial_conditions_as_fitted(em, fit)
    expected = fit.predict(**other, time_edges=_edges(other)).acausal_posterior
    actual = em.predict(**other, time_edges=_edges(other)).acausal_posterior
    np.testing.assert_allclose(actual.values, expected.values, rtol=0, atol=1e-6)


@pytest.mark.integration
@em_stops_at_max_iter
@pytest.mark.parametrize("value", [True, False])
def test_estimate_initial_conditions_warns_and_changes_nothing(recording, value):
    """``estimate_initial_conditions`` is still accepted, because stored
    decoding parameters may pass it, but it has no effect: either value gives
    one ``CompatibilityWarning``, attributed to the caller, and the result of
    leaving it out.
    """
    fitted, _, fs = recording
    reference = _make(NonLocalSortedSpikesDetector, fs)
    expected = reference.estimate_parameters(
        **fitted, time_edges=_edges(fitted), max_iter=2
    )

    detector = _make(NonLocalSortedSpikesDetector, fs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings(
            "ignore",
            message="EM did not converge after max_iter=",
            category=UserWarning,
        )
        results = detector.estimate_parameters(
            **fitted,
            time_edges=_edges(fitted),
            max_iter=2,
            estimate_initial_conditions=value,
        )

    assert [warning.category for warning in caught] == [CompatibilityWarning]
    assert "estimate_initial_conditions" in str(caught[0].message)
    assert caught[0].filename == __file__
    _assert_initial_conditions_as_fitted(detector, reference)
    np.testing.assert_array_equal(
        results.acausal_posterior.values, expected.acausal_posterior.values
    )
    np.testing.assert_array_equal(
        detector.discrete_state_transitions_, reference.discrete_state_transitions_
    )
