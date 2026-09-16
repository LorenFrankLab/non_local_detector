"""Edge cases of chunked prediction through the public ``predict`` path.

``test_chunk_boundary_spikes.py`` establishes the headline regression: an
uncached chunked ``predict`` must equal the unchunked one when spikes fall in
the gap between adjacent chunks. This module pins the boundary cases of that
guarantee, all through the public detector API with ``cache_likelihood=False``
so the likelihood is genuinely recomputed per chunk:

* spikes exactly on a chunk-start timestamp, on ``time[0]`` and on ``time[-1]``;
* ragged (``n_time`` not divisible by ``n_chunks``) and single-row chunks;
* chunks with no spikes at all, and units with no spikes at all;
* ``is_missing`` runs straddling chunk boundaries;
* unsorted decoding spike input;
* a likelihood with legitimate ``-inf`` entries (``local_position_std=0``
  delta kernel): the non-finite mask must be identical, not merely "finite".

Both chunk drivers are covered: the stationary
``chunked_filter_smoother`` and, for the boundary-spike and ``is_missing``
cases, ``chunked_filter_smoother_covariate_dependent``.

Endpoint convention under test is the current one (Phase 6a migrates it):
in-range is ``time[0] <= t <= time[-1]`` and a spike at ``time[-1]`` lands in
row ``n_time - 2``.
"""

import numpy as np
import pytest

from non_local_detector import (
    DiscreteNonStationaryDiagonal,
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
)
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

PARITY_KWARGS = {"rtol": 1e-5, "atol": 1e-6}
# 202 rows is not divisible by 5, 6 or 7, so every chunk split below has a
# ragged final chunk (item 5) as well as uneven boundaries.
N_DECODE = 202
CHUNK_COUNTS = (5, 6, 7)
N_SINGLETON = 41


def chunk_start_rows(n_time: int, n_chunks: int) -> list[int]:
    """First row of each chunk after the first, mirroring the core's split."""
    return [int(chunk[0]) for chunk in np.array_split(np.arange(n_time), n_chunks)[1:]]


def boundary_gap_times(time: np.ndarray, n_chunks: int) -> np.ndarray:
    """Times strictly inside the gap a chunk-local likelihood would drop."""
    return np.array(
        [
            0.5 * (time[row - 1] + time[row])
            for row in chunk_start_rows(len(time), n_chunks)
        ]
    )


def boundary_timestamp_times(time: np.ndarray, n_chunks: int) -> np.ndarray:
    """Chunk-start timestamps themselves, plus both timeline endpoints (item 1)."""
    rows = chunk_start_rows(len(time), n_chunks)
    return np.array([time[0], *[time[row] for row in rows], time[-1]])


def extra_spike_times(time: np.ndarray) -> np.ndarray:
    """Every boundary time of interest for all chunk counts under test."""
    extras = [time[0], time[-1]]
    for n_chunks in CHUNK_COUNTS:
        extras.extend(boundary_gap_times(time, n_chunks))
        extras.extend(boundary_timestamp_times(time, n_chunks))
    return np.unique(np.asarray(extras))


def merge_spikes(spike_times: np.ndarray, extra: np.ndarray) -> np.ndarray:
    """Merge ``extra`` into one unit's spike times, keeping ascending order."""
    return np.sort(np.concatenate([np.asarray(spike_times), np.asarray(extra)]))


def merge_spikes_with_features(
    spike_times: np.ndarray, features: np.ndarray, extra: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Merge ``extra`` spikes into one electrode, keeping times/features aligned."""
    spike_times = np.asarray(spike_times)
    features = np.asarray(features)
    extra_features = np.repeat(
        features.mean(axis=0, keepdims=True), len(extra), axis=0
    ).astype(features.dtype)
    order = np.argsort(np.concatenate([spike_times, extra]), kind="stable")
    return (
        np.concatenate([spike_times, extra])[order],
        np.concatenate([features, extra_features])[order],
    )


COMPARED_VARIABLES = (
    "acausal_posterior",
    "acausal_state_probabilities",
    "causal_posterior",
    "causal_state_probabilities",
)


def assert_results_match(reference, chunked, check_log_likelihood: bool = True):
    """Posteriors (causal and acausal), state probabilities and evidence agree.

    The causal outputs and the log likelihood only appear in the dataset when
    the caller passes ``return_outputs``; their **presence is asserted** rather
    than skipped, because silently comparing a subset would make a test pass
    while covering less than it claims. Every caller therefore has to request
    them, which is what ``sorted_predict_kwargs`` / ``clusterless_predict_kwargs``
    do by default.

    ``equal_nan=False``: a NaN posterior or evidence is never legitimate here, so
    an all-NaN reference matching an all-NaN chunked result must not pass.
    """
    for name in COMPARED_VARIABLES:
        assert name in reference, (
            f"{name} missing from the reference dataset -- pass "
            'return_outputs="all" so this comparison is not silently skipped'
        )
        assert name in chunked, f"{name} missing from the chunked dataset"
        np.testing.assert_allclose(
            chunked[name].to_numpy(),
            reference[name].to_numpy(),
            **PARITY_KWARGS,
            equal_nan=False,
            err_msg=name,
        )
    np.testing.assert_allclose(
        chunked.attrs["marginal_log_likelihoods"],
        reference.attrs["marginal_log_likelihoods"],
        **PARITY_KWARGS,
        equal_nan=False,
        err_msg="marginal_log_likelihoods",
    )
    if check_log_likelihood:
        assert "log_likelihood" in reference, (
            'log_likelihood missing -- pass return_outputs="all"'
        )
        assert "log_likelihood" in chunked
        assert_log_likelihood_matches(
            reference.log_likelihood.to_numpy(), chunked.log_likelihood.to_numpy()
        )


def assert_log_likelihood_matches(reference: np.ndarray, chunked: np.ndarray) -> None:
    """Compare log likelihoods without demanding finiteness.

    Phase 2 legitimately produces ``-inf`` entries (a zero-probability state
    bin), so the non-finite *mask* is compared exactly and the finite values
    numerically. A blanket finiteness assertion would either fail on correct
    output or hide a mask that moved.
    """
    assert chunked.shape == reference.shape
    np.testing.assert_array_equal(
        np.isfinite(chunked), np.isfinite(reference), err_msg="non-finite mask moved"
    )
    np.testing.assert_array_equal(
        np.isneginf(chunked), np.isneginf(reference), err_msg="-inf mask moved"
    )
    finite = np.isfinite(reference)
    np.testing.assert_allclose(
        chunked[finite], reference[finite], **PARITY_KWARGS, err_msg="finite values"
    )


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(scope="module")
def sorted_simulation():
    """Simulated sorted-spikes recording, shared by every sorted fixture."""
    (
        speed,
        position,
        spike_times,
        time,
        _event_times,
        _sampling_frequency,
        is_event,
        _place_fields,
    ) = make_simulated_data(n_neurons=5)
    return {
        "speed": speed,
        "position": position,
        "spike_times": spike_times,
        "time": time,
        "is_event": is_event,
    }


def fit_sorted_detector(sim, **detector_kwargs):
    """Fit a non-local sorted-spikes detector on the simulated recording."""
    return NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={"position_std": 6.0, "block_size": int(2**12)},
        **detector_kwargs,
    ).fit(
        sim["time"],
        sim["position"],
        sim["spike_times"],
        is_training=~sim["is_event"],
    )


@pytest.fixture(scope="module")
def sorted_setup(sorted_simulation):
    """Fitted sorted detector plus a decode window carrying every boundary spike."""
    sim = sorted_simulation
    detector = fit_sorted_detector(sim)

    decode = slice(10_000, 10_000 + N_DECODE)
    decode_time = sim["time"][decode]
    extras = extra_spike_times(decode_time)
    assert len(extras) > len(CHUNK_COUNTS)

    return {
        "detector": detector,
        "time": decode_time,
        "position": sim["position"][decode],
        "spike_times": [
            merge_spikes(unit_times, extras) for unit_times in sim["spike_times"]
        ],
        "clean_spike_times": [np.asarray(s) for s in sim["spike_times"]],
    }


@pytest.fixture(scope="module")
def clusterless_setup():
    """Fitted clusterless detector plus a decode window with boundary spikes."""
    sim = make_simulated_run_data(n_tetrodes=2, n_runs=2)
    n_encode = int(0.7 * len(sim.position_time))
    encode_end = sim.position_time[n_encode]

    detector = NonLocalClusterlessDetector(
        clusterless_algorithm="clusterless_kde",
        clusterless_algorithm_params={
            "position_std": 6.0,
            "waveform_std": 24.0,
            "block_size": 100,
        },
    ).fit(
        sim.position_time[:n_encode],
        sim.position[:n_encode],
        [times[times <= encode_end] for times in sim.spike_times],
        [
            features[times <= encode_end]
            for times, features in zip(
                sim.spike_times, sim.spike_waveform_features, strict=True
            )
        ],
    )

    decode = slice(n_encode, n_encode + N_DECODE)
    decode_time = sim.position_time[decode]
    extras = extra_spike_times(decode_time)

    spike_times = []
    features = []
    for unit_times, unit_features in zip(
        sim.spike_times, sim.spike_waveform_features, strict=True
    ):
        merged_times, merged_features = merge_spikes_with_features(
            unit_times, unit_features, extras
        )
        spike_times.append(merged_times)
        features.append(merged_features)

    return {
        "detector": detector,
        "time": decode_time,
        "position": sim.position[decode],
        "spike_times": spike_times,
        "spike_waveform_features": features,
    }


@pytest.fixture(scope="module")
def covariate_setup(sorted_simulation):
    """Fitted covariate-dependent detector (time-varying discrete transitions)."""
    sim = sorted_simulation
    detector = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={"position_std": 6.0, "block_size": int(2**12)},
        discrete_transition_type=DiscreteNonStationaryDiagonal(
            diagonal_values=np.full((4,), 0.98), formula="1 + speed"
        ),
    ).fit(
        sim["time"],
        sim["position"],
        sim["spike_times"],
        is_training=~sim["is_event"],
        discrete_transition_covariate_data={"speed": sim["speed"]},
    )
    # Guard the premise: a 3-D transition array is what routes _predict to the
    # covariate-dependent chunk driver.
    assert detector.discrete_state_transitions_.ndim == 3

    decode = slice(10_000, 10_000 + N_DECODE)
    decode_time = sim["time"][decode]
    extras = extra_spike_times(decode_time)

    return {
        "detector": detector,
        "time": decode_time,
        "position": sim["position"][decode],
        "speed": sim["speed"][decode],
        "spike_times": [
            merge_spikes(unit_times, extras) for unit_times in sim["spike_times"]
        ],
    }


@pytest.fixture(scope="module")
def delta_kernel_setup(sorted_simulation):
    """A detector whose likelihood has legitimate ``-inf`` entries (item 9).

    ``local_position_std=0`` makes the Local state's position kernel a delta:
    ``log(n_bins)`` at the animal's bin and ``-inf`` at every other bin. Those
    ``-inf``s are correct output, and a row request must reproduce them in the
    same places.
    """
    sim = sorted_simulation
    detector = fit_sorted_detector(sim, local_position_std=0.0)

    decode = slice(10_000, 10_000 + N_SINGLETON)
    decode_time = sim["time"][decode]
    extras = extra_spike_times(decode_time)

    return {
        "detector": detector,
        "time": decode_time,
        "position": sim["position"][decode],
        "spike_times": [
            merge_spikes(unit_times, extras) for unit_times in sim["spike_times"]
        ],
    }


def sorted_predict_kwargs(setup, **overrides):
    """Predict arguments for a sorted-spikes setup.

    ``return_outputs="all"`` is the default so that ``assert_results_match``
    really compares the causal posteriors and the log likelihood; without it the
    dataset carries only the acausal outputs and the comparison would cover less
    than it claims.
    """
    kwargs = {
        "spike_times": setup["spike_times"],
        "time": setup["time"],
        "position": setup["position"],
        "position_time": setup["time"],
        "return_outputs": "all",
    }
    kwargs.update(overrides)
    return kwargs


def clusterless_predict_kwargs(setup, **overrides):
    """Predict arguments for a clusterless setup (see ``sorted_predict_kwargs``)."""
    kwargs = {
        "spike_times": setup["spike_times"],
        "spike_waveform_features": setup["spike_waveform_features"],
        "time": setup["time"],
        "position": setup["position"],
        "position_time": setup["time"],
        "return_outputs": "all",
    }
    kwargs.update(overrides)
    return kwargs


# ==============================================================================
# Items 1, 2, 5: boundary timestamps, in-gap spikes, ragged/uneven chunks
# ==============================================================================


@pytest.mark.integration
@pytest.mark.parametrize("n_chunks", CHUNK_COUNTS)
def test_sorted_ragged_chunks_match_unchunked(sorted_setup, n_chunks):
    """Ragged, uneven chunk splits must not move a spike (sorted spikes).

    The decode window carries spikes exactly on every chunk-start timestamp, on
    both timeline endpoints, and strictly inside every inter-chunk gap.
    """
    detector = sorted_setup["detector"]
    kwargs = sorted_predict_kwargs(sorted_setup)
    assert len(sorted_setup["time"]) % n_chunks != 0

    reference = detector.predict(**kwargs, n_chunks=1)
    chunked = detector.predict(**kwargs, n_chunks=n_chunks, cache_likelihood=False)

    assert_results_match(reference, chunked)


@pytest.mark.integration
@pytest.mark.parametrize("n_chunks", CHUNK_COUNTS)
def test_clusterless_ragged_chunks_match_unchunked(clusterless_setup, n_chunks):
    """Ragged, uneven chunk splits must not move a spike (clusterless)."""
    detector = clusterless_setup["detector"]
    kwargs = clusterless_predict_kwargs(clusterless_setup)
    assert len(clusterless_setup["time"]) % n_chunks != 0

    reference = detector.predict(**kwargs, n_chunks=1)
    chunked = detector.predict(**kwargs, n_chunks=n_chunks, cache_likelihood=False)

    assert_results_match(reference, chunked)


@pytest.mark.integration
def test_spikes_only_on_chunk_boundaries_match_unchunked(sorted_setup):
    """Item 1 in isolation: the ONLY spikes are on boundary timestamps.

    With every other spike removed the boundary spikes carry all the evidence,
    so an off-by-one row assignment cannot be masked by neighbouring spikes.
    """
    detector = sorted_setup["detector"]
    time = sorted_setup["time"]
    n_chunks = 7
    only_boundary = boundary_timestamp_times(time, n_chunks)
    spike_times = [only_boundary.copy() for _ in sorted_setup["clean_spike_times"]]

    kwargs = {
        "spike_times": spike_times,
        "time": time,
        "position": sorted_setup["position"],
        "position_time": time,
        "return_outputs": "all",
    }
    reference = detector.predict(**kwargs, n_chunks=1)
    chunked = detector.predict(**kwargs, n_chunks=n_chunks, cache_likelihood=False)

    # Premise: those spikes really are counted (a silent drop on both sides
    # would make this test vacuous).
    assert reference.log_likelihood.to_numpy().shape[0] == len(time)
    assert_results_match(reference, chunked)


# ==============================================================================
# Item 4: singleton chunks (n_chunks == n_time)
# ==============================================================================


@pytest.mark.integration
def test_singleton_chunks_match_unchunked(delta_kernel_setup):
    """One row per chunk (``n_chunks == n_time``) must still match exactly."""
    detector = delta_kernel_setup["detector"]
    kwargs = sorted_predict_kwargs(delta_kernel_setup)
    n_time = len(delta_kernel_setup["time"])

    reference = detector.predict(**kwargs, n_chunks=1)
    chunked = detector.predict(**kwargs, n_chunks=n_time, cache_likelihood=False)

    assert chunked.log_likelihood.to_numpy().shape[0] == n_time
    assert_results_match(reference, chunked)


# ==============================================================================
# Item 3: spike-free chunks and spike-free units
# ==============================================================================


@pytest.mark.integration
@pytest.mark.parametrize("emptiness", ["one_unit_empty", "all_units_empty", "gap"])
def test_spike_free_chunks_match_unchunked(sorted_setup, emptiness):
    """Chunks (or whole units) with no spikes must not change the result.

    ``gap`` deletes every spike in the middle third of the window, so at least
    one chunk selects zero spikes on every unit.
    """
    detector = sorted_setup["detector"]
    time = sorted_setup["time"]
    n_chunks = 6
    spike_times = [np.asarray(s) for s in sorted_setup["spike_times"]]

    if emptiness == "one_unit_empty":
        spike_times[0] = np.array([])
    elif emptiness == "all_units_empty":
        spike_times = [np.array([]) for _ in spike_times]
    else:
        lo, hi = time[len(time) // 3], time[2 * len(time) // 3]
        spike_times = [s[(s < lo) | (s > hi)] for s in spike_times]
        # Premise: a whole chunk really is spike-free.
        chunk_bounds = np.array_split(np.arange(len(time)), n_chunks)
        empty_chunks = [
            chunk
            for chunk in chunk_bounds
            if not any(
                np.any((s >= time[chunk[0]]) & (s <= time[chunk[-1]]))
                for s in spike_times
            )
        ]
        assert empty_chunks, "no spike-free chunk in this configuration"

    kwargs = {
        "spike_times": spike_times,
        "time": time,
        "position": sorted_setup["position"],
        "position_time": time,
        "return_outputs": "all",
    }
    reference = detector.predict(**kwargs, n_chunks=1)
    chunked = detector.predict(**kwargs, n_chunks=n_chunks, cache_likelihood=False)

    assert_results_match(reference, chunked)


@pytest.mark.integration
def test_clusterless_spike_free_chunks_match_unchunked(clusterless_setup):
    """Clusterless: a chunk with zero spikes on every electrode.

    This is the ``segment_sum`` over zero selected rows and the zero-row
    waveform-feature selection, which the sorted-spikes backends do not reach.
    """
    detector = clusterless_setup["detector"]
    time = clusterless_setup["time"]
    n_chunks = 6
    lo, hi = time[len(time) // 3], time[2 * len(time) // 3]

    spike_times = []
    features = []
    for unit_times, unit_features in zip(
        clusterless_setup["spike_times"],
        clusterless_setup["spike_waveform_features"],
        strict=True,
    ):
        keep = (unit_times < lo) | (unit_times > hi)
        spike_times.append(unit_times[keep])
        features.append(unit_features[keep])

    chunk_bounds = np.array_split(np.arange(len(time)), n_chunks)
    assert any(
        not any(
            np.any((s >= time[chunk[0]]) & (s <= time[chunk[-1]])) for s in spike_times
        )
        for chunk in chunk_bounds
    )

    kwargs = {
        "spike_times": spike_times,
        "spike_waveform_features": features,
        "time": time,
        "position": clusterless_setup["position"],
        "position_time": time,
        "return_outputs": "all",
    }
    reference = detector.predict(**kwargs, n_chunks=1)
    chunked = detector.predict(**kwargs, n_chunks=n_chunks, cache_likelihood=False)

    assert_results_match(reference, chunked)


# ==============================================================================
# Item 7: unsorted decoding spike input through the public path
# ==============================================================================


@pytest.mark.integration
def test_unsorted_spike_input_matches_unchunked(clusterless_setup):
    """Permuting each electrode's (spike, waveform-feature) pairs changes nothing.

    The permutation keeps every feature row with its own spike, so the observed
    data is unchanged: both the unchunked and the chunked result must equal the
    ascending-input result.
    """
    detector = clusterless_setup["detector"]
    rng = np.random.default_rng(31)
    n_chunks = 6

    shuffled_times = []
    shuffled_features = []
    for unit_times, unit_features in zip(
        clusterless_setup["spike_times"],
        clusterless_setup["spike_waveform_features"],
        strict=True,
    ):
        order = rng.permutation(len(unit_times))
        shuffled_times.append(unit_times[order])
        shuffled_features.append(unit_features[order])
        assert not np.all(np.diff(shuffled_times[-1]) >= 0)

    ascending = detector.predict(
        **clusterless_predict_kwargs(clusterless_setup), n_chunks=1
    )
    shuffled_kwargs = clusterless_predict_kwargs(
        clusterless_setup,
        spike_times=shuffled_times,
        spike_waveform_features=shuffled_features,
    )
    shuffled_reference = detector.predict(**shuffled_kwargs, n_chunks=1)
    shuffled_chunked = detector.predict(
        **shuffled_kwargs, n_chunks=n_chunks, cache_likelihood=False
    )

    assert_results_match(ascending, shuffled_reference)
    assert_results_match(shuffled_reference, shuffled_chunked)


# ==============================================================================
# Item 8: is_missing straddling chunk boundaries
# ==============================================================================


def missing_mask_straddling_boundaries(n_time: int, n_chunks: int) -> np.ndarray:
    """Missing-data runs that each span a chunk boundary, plus both endpoints."""
    is_missing = np.zeros(n_time, dtype=bool)
    for row in chunk_start_rows(n_time, n_chunks):
        is_missing[max(row - 2, 0) : row + 3] = True
    is_missing[0] = True
    is_missing[-1] = True
    return is_missing


@pytest.mark.integration
@pytest.mark.parametrize("n_chunks", CHUNK_COUNTS)
def test_is_missing_straddling_boundaries_matches_unchunked(sorted_setup, n_chunks):
    """Item 8: missing-data runs crossing chunk boundaries, chunked == unchunked."""
    detector = sorted_setup["detector"]
    time = sorted_setup["time"]
    is_missing = missing_mask_straddling_boundaries(len(time), n_chunks)
    # Premise: the mask really does straddle boundaries (a mask aligned to the
    # chunk edges would not exercise the core's per-chunk slicing).
    for row in chunk_start_rows(len(time), n_chunks):
        assert is_missing[row - 1] and is_missing[row]

    kwargs = sorted_predict_kwargs(sorted_setup, is_missing=is_missing)
    reference = detector.predict(**kwargs, n_chunks=1)
    chunked = detector.predict(**kwargs, n_chunks=n_chunks, cache_likelihood=False)

    # Missing rows carry no spike evidence: their likelihood is exactly zero.
    reference_ll = reference.log_likelihood.to_numpy()
    np.testing.assert_array_equal(reference_ll[is_missing], 0.0)
    assert_results_match(reference, chunked)


@pytest.mark.integration
def test_clusterless_is_missing_straddling_boundaries_matches_unchunked(
    clusterless_setup,
):
    """Item 8 for the clusterless assembly path."""
    detector = clusterless_setup["detector"]
    time = clusterless_setup["time"]
    n_chunks = 6
    is_missing = missing_mask_straddling_boundaries(len(time), n_chunks)

    kwargs = clusterless_predict_kwargs(clusterless_setup, is_missing=is_missing)
    reference = detector.predict(**kwargs, n_chunks=1)
    chunked = detector.predict(**kwargs, n_chunks=n_chunks, cache_likelihood=False)

    assert_results_match(reference, chunked)


# ==============================================================================
# Item 10: the covariate-dependent chunk driver (items 2 and 8)
# ==============================================================================


@pytest.mark.integration
@pytest.mark.parametrize("n_chunks", CHUNK_COUNTS)
def test_covariate_dependent_ragged_chunks_match_unchunked(covariate_setup, n_chunks):
    """Item 2 and 5 on the covariate-dependent driver."""
    detector = covariate_setup["detector"]
    kwargs = {
        "spike_times": covariate_setup["spike_times"],
        "time": covariate_setup["time"],
        "position": covariate_setup["position"],
        "position_time": covariate_setup["time"],
        "discrete_transition_covariate_data": {"speed": covariate_setup["speed"]},
        "return_outputs": "all",
    }
    reference = detector.predict(**kwargs, n_chunks=1)
    chunked = detector.predict(**kwargs, n_chunks=n_chunks, cache_likelihood=False)

    assert_results_match(reference, chunked)


@pytest.mark.integration
def test_covariate_dependent_is_missing_matches_unchunked(covariate_setup):
    """Item 8 on the covariate-dependent driver."""
    detector = covariate_setup["detector"]
    time = covariate_setup["time"]
    n_chunks = 7
    is_missing = missing_mask_straddling_boundaries(len(time), n_chunks)

    kwargs = {
        "spike_times": covariate_setup["spike_times"],
        "time": time,
        "position": covariate_setup["position"],
        "position_time": time,
        "is_missing": is_missing,
        "discrete_transition_covariate_data": {"speed": covariate_setup["speed"]},
        "return_outputs": "all",
    }
    reference = detector.predict(**kwargs, n_chunks=1)
    chunked = detector.predict(**kwargs, n_chunks=n_chunks, cache_likelihood=False)

    assert_results_match(reference, chunked)


# ==============================================================================
# Item 9: legitimate -inf entries must be preserved, not "made finite"
# ==============================================================================


@pytest.mark.integration
def test_delta_local_kernel_has_nonfinite_rows(delta_kernel_setup):
    """Guard the premise of the next test: this likelihood really has ``-inf``."""
    setup = delta_kernel_setup
    log_likelihood = np.asarray(
        setup["detector"].compute_log_likelihood(
            setup["time"], setup["time"], setup["position"], setup["spike_times"]
        )
    )
    assert np.any(np.isneginf(log_likelihood))
    assert not np.any(np.isnan(log_likelihood))
    # Every row has some -inf (the Local state's non-animal bins) and some
    # finite entries, so neither a blanket finiteness nor a blanket -inf
    # assertion would be meaningful.
    assert np.all(np.any(np.isneginf(log_likelihood), axis=1))
    assert np.all(np.any(np.isfinite(log_likelihood), axis=1))


@pytest.mark.integration
@pytest.mark.parametrize("n_chunks", [3, 6, N_SINGLETON])
def test_nonfinite_mask_survives_chunking(delta_kernel_setup, n_chunks):
    """Item 9: the ``-inf`` mask and the finite values are both reproduced."""
    setup = delta_kernel_setup
    kwargs = sorted_predict_kwargs(setup)

    reference = setup["detector"].predict(**kwargs, n_chunks=1)
    chunked = setup["detector"].predict(
        **kwargs, n_chunks=n_chunks, cache_likelihood=False
    )

    # Premise, asserted on the array actually compared below (the delta-kernel
    # premise test checks ``compute_log_likelihood``; this is the ``predict``
    # output, which travels through the chunk accumulation path).
    reference_log_likelihood = reference.log_likelihood.to_numpy()
    assert np.any(np.isneginf(reference_log_likelihood))
    assert not np.any(np.isnan(reference_log_likelihood))

    assert_log_likelihood_matches(
        reference_log_likelihood, chunked.log_likelihood.to_numpy()
    )
    assert_results_match(reference, chunked)


@pytest.mark.integration
def test_nonfinite_mask_survives_row_requests(delta_kernel_setup):
    """The same at the model layer: every row range reproduces the ``-inf`` mask."""
    setup = delta_kernel_setup
    detector = setup["detector"]
    args = (setup["time"], setup["time"], setup["position"], setup["spike_times"])
    full = np.asarray(detector.compute_log_likelihood(*args))

    for partition in (
        [slice(0, 7), slice(7, 20), slice(20, N_SINGLETON)],
        [slice(row, row + 1) for row in range(N_SINGLETON)],
    ):
        tiled = np.concatenate(
            [
                np.asarray(detector.compute_log_likelihood(*args, row_slice=row_slice))
                for row_slice in partition
            ]
        )
        assert_log_likelihood_matches(full, tiled)
