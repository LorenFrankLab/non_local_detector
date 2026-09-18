"""Edge cases of the ``row_slice`` row-request interface.

``test_row_slice_parity.py`` establishes the headline property for every
registered backend: a requested row range equals the full-time result sliced,
and a row partition tiles the full-time result. This module pins the *boundary*
cases of that interface, which is where a chunked driver actually breaks:

* spikes exactly on a timestamp -- including ``time[0]`` and ``time[-1]``;
* spikes strictly between adjacent timestamps under *irregular* partitions;
* chunks with no spikes at all (``segment_sum`` over zero selected rows);
* singleton row ranges (``n_chunks == n_time``) and ragged final chunks;
* empty row requests (``slice(a, a)``) and rejected non-contiguous requests;
* unsorted decoding spikes, where the waveform features must stay paired with
  their own spike.

Endpoint convention under test: a spike is in range iff
``time[0] <= t <= time[-1]``, it lands in row ``np.digitize(t, time[1:-1])``,
and therefore a spike at ``time[-1]`` lands in row ``n_time - 2`` and the final
row of a multi-row timeline is always empty.
"""

import os
import subprocess
import sys
import tracemalloc
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.ops
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import _CLUSTERLESS_ALGORITHMS, common
from non_local_detector.likelihoods.common import (
    _SpikeTimeOrder,
    get_spikecount_per_time_bin,
    resolve_row_slice,
    select_spike_rows,
    select_spikes_in_rows,
    sum_spikes_into_rows,
)
from non_local_detector.likelihoods.no_spike import predict_no_spike_log_likelihood
from non_local_detector.tests.likelihoods.conftest import (
    ALGORITHMS,
    EXACT,
    fit_registered_backends,
    parity_kwargs,
)

# A relative tolerance is only a meaningful guard if the values it is applied to
# have a sane magnitude. Losing or misplacing one decoding spike changes an
# affected row by O(1) at the very least (measured on this fixture: 23-46 on
# every backend), so the absolute error the tolerance admits,
# ``rtol * max|LL| + atol``, must stay far below 1. Capping ``max|LL|`` at 1e4
# admits at most 1e-2 -- two orders of magnitude below the smallest real change
# a regression could produce. Measured max |finite log likelihood| over every
# spike case and both ``is_local`` values on this fixture:
#
#   sorted_spikes_mrf 69.5 | sorted_spikes_diffusion 111.8 | sorted_spikes_kde 113.7
#   sorted_spikes_glm 116.8 | clusterless_diffusion 136.6 | clusterless_kde 138.4
#   clusterless_kde_log 138.4 | clusterless_gmm 763.7
#
# ``clusterless_gmm`` reaches 1.1e8 with its DEFAULT mixture sizes on this
# fixture's 40 encoding spikes (64 joint components over 40 samples is a
# degenerate fit), which would let the relative tolerance admit ~100 absolute
# and hide an ordinary misplaced spike. ``BACKEND_FIT_PARAMS`` below fits it
# sanely instead; ``assert_fixture_scale`` fails loudly if that ever drifts.
MAX_ABS_LOG_LIKELIHOOD = 1e4

# ``clusterless_gmm``'s defaults (64 joint / 32 GPI / 32 occupancy components)
# over-parameterize this deliberately small fixture. See MAX_ABS_LOG_LIKELIHOOD.
BACKEND_FIT_PARAMS: dict[str, dict] = {
    "clusterless_gmm": {
        "gmm_components_joint": 16,
        "gmm_components_gpi": 8,
        "gmm_components_occupancy": 8,
        "gmm_reg_covar": 1e-4,
    }
}


def assert_fixture_scale(log_likelihood: np.ndarray, algorithm: str) -> None:
    """Fail if the fixture's magnitudes make the tolerance meaningless.

    See ``MAX_ABS_LOG_LIKELIHOOD``: a degenerate fit inflates the log
    likelihoods until the relative tolerance admits more absolute error than a
    lost spike would produce, which would silently blunt every parity assertion
    for that backend.
    """
    finite = log_likelihood[np.isfinite(log_likelihood)]
    if finite.size == 0:
        return
    scale = float(np.abs(finite).max())
    assert scale < MAX_ABS_LOG_LIKELIHOOD, (
        f"{algorithm}: fixture log likelihoods reach {scale:.3g}; the parity "
        f"tolerance would admit {parity_kwargs(algorithm)['rtol'] * scale:.3g} "
        "absolute, which is not far enough below the O(1) change a lost or "
        "misplaced spike causes. Refit this backend with saner parameters."
    )


# Per-row (singleton) requests recompile the jitted kernels once per distinct
# row count, so the exhaustive one-row-at-a-time sweep uses one representative
# backend per likelihood family instead of all eight.
REPRESENTATIVE_ALGORITHMS = [
    "sorted_spikes_kde",
    "sorted_spikes_glm",
    "clusterless_kde",
    "clusterless_gmm",
]

N_TIME = 13


def spike_placements(time: np.ndarray) -> dict[str, np.ndarray]:
    """Decoding spike placements that hit every boundary case of ``time``.

    ``on_timestamps`` puts a spike on every timestamp, including both endpoints;
    ``between_timestamps`` puts one strictly inside every bin (the original
    chunk-boundary defect); ``mixed`` is both; ``duplicates`` repeats ``mixed``
    three times, since identical spike times must be counted once each, not
    lost or duplicated by the searchsorted range.
    """
    midpoints = 0.5 * (time[:-1] + time[1:])
    mixed = np.sort(np.concatenate([time, midpoints]))
    return {
        "on_timestamps": time.copy(),
        "between_timestamps": midpoints,
        "mixed": mixed,
        "duplicates": np.sort(np.repeat(mixed, 3)),
    }


def row_partitions(n_time: int) -> list[list[slice]]:
    """Contiguous row partitions of ``range(n_time)`` that tile it exactly.

    Includes the even ``np.array_split`` partitions the core actually uses
    (ragged final chunks included), plus hand-picked irregular partitions whose
    chunk lengths differ, and the singleton partition.
    """
    partitions = []
    for n_chunks in (1, 2, 3, 5, 7, n_time):
        chunks = np.array_split(np.arange(n_time), n_chunks)
        partitions.append(
            [slice(int(c[0]), int(c[-1]) + 1) for c in chunks if len(c) > 0]
        )
    # Irregular: chunk lengths 1, 4, 2, rest.
    for cuts in ([1, 5, 7], [2, 3, 11], [6]):
        bounds = [0, *cuts, n_time]
        partitions.append(
            [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:], strict=True) if a < b]
        )
    return partitions


# ==============================================================================
# Helper-level edge cases (``select_spikes_in_rows`` / ``resolve_row_slice``)
# ==============================================================================


@pytest.mark.unit
def test_endpoint_convention_is_unchanged():
    """Pin the in-range-inclusive convention the row request derives from.

    A row request must reproduce it exactly, so it is asserted here as the
    reference the parity tests compare to.
    """
    time = np.arange(6.0)

    # time[0] lands in row 0; time[-1] lands in row n_time - 2 (not the last row).
    np.testing.assert_array_equal(
        get_spikecount_per_time_bin(np.array([time[0]]), time), [1, 0, 0, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        get_spikecount_per_time_bin(np.array([time[-1]]), time), [0, 0, 0, 0, 1, 0]
    )
    # Out-of-range spikes are dropped entirely.
    np.testing.assert_array_equal(
        get_spikecount_per_time_bin(np.array([-0.5, 5.5]), time), np.zeros(6, dtype=int)
    )
    # The final row of a multi-row timeline can never own a spike.
    all_edges_and_midpoints = np.sort(
        np.concatenate([time, 0.5 * (time[:-1] + time[1:])])
    )
    assert get_spikecount_per_time_bin(all_edges_and_midpoints, time)[-1] == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "spike_kind", ["on_timestamps", "between_timestamps", "mixed", "duplicates"]
)
def test_every_row_partition_tiles_the_full_counts(spike_kind):
    """Row-partition counts must tile the full counts for every spike placement.

    ``on_timestamps`` is item 1 (spikes exactly on a chunk-start timestamp and on
    both endpoints); ``between_timestamps`` is the original defect; the
    partitions include ragged and irregular chunk sizes and the singleton split.
    """
    time = np.linspace(0.0, 1.0, N_TIME)
    spikes = spike_placements(time)[spike_kind]

    full = get_spikecount_per_time_bin(spikes, time)
    assert full.sum() == np.sum((spikes >= time[0]) & (spikes <= time[-1]))

    for partition in row_partitions(N_TIME):
        tiled = np.concatenate(
            [get_spikecount_per_time_bin(spikes, time, row_slice=s) for s in partition]
        )
        np.testing.assert_array_equal(tiled, full, err_msg=f"partition={partition}")


@pytest.mark.unit
@pytest.mark.parametrize(
    "spike_kind", ["on_timestamps", "between_timestamps", "mixed", "duplicates"]
)
def test_selection_carries_its_own_row_count(spike_kind):
    """A selection knows how many rows it was made for, so reductions cannot drift.

    ``bin_ind`` holds LOCAL rows of the requested range; the segment count
    that pairs with it is a property of the same request. Callers that derive
    it separately can pair one chunk's selection with another chunk's row
    count and silently drop spikes, so the selection records it and the
    reduction helper reads it from there. Covers the empty early return (rows
    past the last owning row) and the ascending and unsorted selection paths.
    """
    time = np.linspace(0.0, 1.0, N_TIME)
    spikes = spike_placements(time)[spike_kind]
    full = get_spikecount_per_time_bin(spikes, time)

    for partition in row_partitions(N_TIME):
        for rows in partition:
            selection = select_spikes_in_rows(spikes, time, rows.start, rows.stop)
            assert selection.n_rows == rows.stop - rows.start
            row_sums = sum_spikes_into_rows(
                jnp.ones(selection.bin_ind.shape[0]), selection
            )
            assert row_sums.shape == (selection.n_rows,)
            np.testing.assert_array_equal(np.asarray(row_sums), full[rows])

    # Requests entirely past the last owning row still report their row count.
    selection = select_spikes_in_rows(spikes, time, N_TIME - 1, N_TIME)
    assert selection.n_rows == 1
    assert selection.bin_ind.shape == (0,)
    assert sum_spikes_into_rows(jnp.zeros(0), selection).shape == (1,)


@pytest.mark.unit
@pytest.mark.parametrize("n_time", [1, 2, 3, 6, 17])
def test_singleton_row_requests_tile_the_full_counts(n_time):
    """``n_chunks == n_time``: every row requested on its own must still tile."""
    time = np.linspace(0.0, 1.0, n_time)
    rng = np.random.default_rng(0)
    spikes = np.sort(
        np.concatenate([time, rng.uniform(-0.1, 1.1, 20), np.array([time[-1]])])
    )

    full = get_spikecount_per_time_bin(spikes, time)
    singleton = np.concatenate(
        [
            get_spikecount_per_time_bin(spikes, time, row_slice=slice(row, row + 1))
            for row in range(n_time)
        ]
    )
    np.testing.assert_array_equal(singleton, full)
    for row in range(n_time):
        assert get_spikecount_per_time_bin(
            spikes, time, row_slice=slice(row, row + 1)
        ).shape == (1,)


@pytest.mark.unit
def test_empty_row_request_returns_zero_rows():
    """Documented behaviour of ``slice(a, a)``: a 0-row result, never an error."""
    time = np.arange(6.0)
    spikes = np.array([0.5, 2.5, 4.5])

    for a in range(len(time) + 1):
        assert resolve_row_slice(slice(a, a), len(time)) == (a, a)
        counts = get_spikecount_per_time_bin(spikes, time, row_slice=slice(a, a))
        assert counts.shape == (0,)
        selection = select_spikes_in_rows(spikes, time, a, a)
        indexer, bin_ind = selection.indexer, selection.bin_ind
        assert spikes[indexer].size == 0
        assert bin_ind.shape == (0,)

    # An inverted request is normalized to empty rather than a negative length.
    assert resolve_row_slice(slice(4, 2), len(time)) == (4, 4)
    assert get_spikecount_per_time_bin(spikes, time, row_slice=slice(4, 2)).shape == (
        0,
    )


@pytest.mark.unit
def test_non_contiguous_row_request_is_rejected():
    """A strided row request is not a chunk; it must fail loudly."""
    with pytest.raises(ValidationError, match="contiguous"):
        resolve_row_slice(slice(0, 6, 2), 6)
    with pytest.raises(ValidationError, match="contiguous"):
        get_spikecount_per_time_bin(np.array([0.5]), np.arange(6.0), slice(0, 6, 2))


@pytest.mark.unit
def test_empty_spike_input_selects_nothing():
    """Item 3 at the helper: no spikes at all still yields the requested rows."""
    time = np.arange(6.0)
    for empty in (np.array([]), np.zeros((0,), dtype=float)):
        selection = select_spikes_in_rows(empty, time, 1, 4)
        indexer, bin_ind = selection.indexer, selection.bin_ind
        assert empty[indexer].size == 0
        assert bin_ind.shape == (0,)
        np.testing.assert_array_equal(
            get_spikecount_per_time_bin(empty, time, row_slice=slice(1, 4)), [0, 0, 0]
        )


@pytest.mark.unit
@pytest.mark.parametrize("spike_kind", ["on_timestamps", "between_timestamps", "mixed"])
def test_chunk_local_binning_is_detectably_wrong(spike_kind):
    """The edge cases above must be able to fail: show the legacy call differs.

    The historical chunked call handed the backend ``time[chunk]``, which bins
    the spikes against the chunk instead of the recording. For each spike
    placement and each partition shape used above, that produces *different*
    counts from the full timeline -- spikes strictly between chunks are dropped,
    and spikes on a chunk-start timestamp are moved one row earlier. Without
    this assertion a row-request test could pass vacuously.
    """
    time = np.linspace(0.0, 1.0, N_TIME)
    spikes = spike_placements(time)[spike_kind]

    full = get_spikecount_per_time_bin(spikes, time)
    for partition in row_partitions(N_TIME):
        if len(partition) == 1:
            continue  # a single chunk IS the full timeline
        chunk_local = np.concatenate(
            [
                get_spikecount_per_time_bin(spikes, time[s.start : s.stop])
                for s in partition
            ]
        )
        row_aware = np.concatenate(
            [get_spikecount_per_time_bin(spikes, time, row_slice=s) for s in partition]
        )
        assert not np.array_equal(chunk_local, full), (
            f"chunk-local binning is indistinguishable here: {partition}"
        )
        np.testing.assert_array_equal(row_aware, full)


@pytest.mark.unit
@pytest.mark.parametrize("shuffled", [False, True])
def test_selected_features_stay_paired_with_their_spike(shuffled):
    """Item 7: the one indexer must carry per-spike features with their spike.

    Each spike's "waveform feature" is its own index, so a feature that moved to
    a different spike is detectable, not merely a value that looks plausible.
    """
    time = np.linspace(0.0, 1.0, N_TIME)
    rng = np.random.default_rng(7)
    spike_times = np.sort(
        np.concatenate([0.5 * (time[:-1] + time[1:]), rng.uniform(0.0, 1.0, 9)])
    )
    if shuffled:
        spike_times = spike_times[rng.permutation(len(spike_times))]
    spike_ids = np.arange(len(spike_times))
    features = spike_ids[:, None].astype(float) * 1000.0 + 0.5

    in_range = (spike_times >= time[0]) & (spike_times <= time[-1])
    seen: list[np.ndarray] = []
    for partition in row_partitions(N_TIME):
        partition_ids: list[np.ndarray] = []
        for row_slice in partition:
            row_start, row_stop = resolve_row_slice(row_slice, len(time))
            selection = select_spikes_in_rows(spike_times, time, row_start, row_stop)
            indexer, bin_ind = selection.indexer, selection.bin_ind
            selected_ids = spike_ids[indexer]
            selected_features = features[indexer]

            # The feature rows are exactly the selected spikes' own features.
            np.testing.assert_array_equal(
                selected_features, features[selected_ids], strict=True
            )
            assert selected_features.shape == (len(selected_ids), 1)
            # Every selected spike's local row is the global row minus row_start.
            expected_rows = np.digitize(spike_times[selected_ids], time[1:-1])
            np.testing.assert_array_equal(bin_ind, expected_rows - row_start)
            assert np.all(bin_ind >= 0) and np.all(bin_ind < row_stop - row_start)
            partition_ids.append(selected_ids)

        # A partition of the rows partitions the in-range spikes: each owned
        # exactly once, none invented.
        owned = np.concatenate(partition_ids) if partition_ids else np.zeros(0, int)
        np.testing.assert_array_equal(np.sort(owned), spike_ids[in_range])
        seen.append(owned)

    assert any(len(o) for o in seen)


# ==============================================================================
# Backend-level edge cases
# ==============================================================================


@pytest.fixture(scope="module")
def edge_case_data():
    """Encoding/decoding data whose decoding spikes hit every boundary case."""
    rng = np.random.default_rng(11)
    position_time = np.linspace(0.0, 4.0, 400)
    position = (50.0 + 40.0 * np.sin(2 * np.pi * position_time / 4.0))[:, None]
    environment = Environment(
        environment_name="line",
        place_bin_size=10.0,
        position_range=((0.0, 100.0),),
    ).fit_place_grid(position=position, infer_track_interior=False)

    time = np.linspace(0.0, 4.0, N_TIME)
    placements = spike_placements(time)
    on_timestamps = placements["on_timestamps"]
    between = placements["between_timestamps"]

    # Unit 1 has encoding spikes; unit 2 has none, which reaches the zero-rate
    # fast paths of every clusterless backend.
    encoding_spike_times = [np.sort(rng.uniform(0.0, 4.0, 40)), np.array([])]
    encoding_features = [rng.standard_normal((40, 2)) * 5.0 + 20.0, np.zeros((0, 2))]

    def features_for(spike_times):
        return rng.standard_normal((len(spike_times), 2)) * 5.0 + 20.0

    # "empty" has no spikes at all.
    #
    # "gap" leaves rows 5, 6, 9 and 10 with no spikes on ANY electrode, chosen so
    # that BOTH partitions used by the ragged/irregular test contain a fully
    # spike-free chunk: the even 13/5 split's [9, 11) and the irregular split's
    # [5, 7). ``assert_some_chunk_is_spike_free`` re-derives and enforces that,
    # so this arithmetic cannot silently drift.
    spike_free_rows = {5, 6, 9, 10}
    gap = np.array([t for row, t in enumerate(between) if row not in spike_free_rows])
    spike_cases = {
        "on_timestamps": [on_timestamps, on_timestamps.copy()],
        "between_timestamps": [between, between.copy()],
        "mixed": [placements["mixed"]] * 2,
        "spike_free_gap": [gap, gap.copy()],
        "one_unit_empty": [between, np.array([])],
        "all_units_empty": [np.array([]), np.array([])],
    }

    return {
        "position_time": position_time,
        "position": position,
        "environment": environment,
        "time": time,
        "encoding_spike_times": encoding_spike_times,
        "encoding_features": encoding_features,
        "spike_cases": {
            name: (
                spikes,
                [features_for(s) for s in spikes],
            )
            for name, spikes in spike_cases.items()
        },
    }


@pytest.fixture(scope="module")
def fitted_backends(edge_case_data):
    """Fit every registered backend once on the shared encoding data."""
    return {
        name: (predict_func, encoding_model, is_clusterless)
        for name, predict_func, encoding_model, is_clusterless in (
            fit_registered_backends(edge_case_data, BACKEND_FIT_PARAMS)
        )
    }


def call_backend(fitted_backends, edge_case_data, algorithm, case, time=None, **kwargs):
    """Evaluate one backend on one decoding-spike case.

    ``time`` defaults to the full decoding timeline. Passing a sub-range instead
    reproduces the historical chunk-local call, which is what
    ``chunk_local_result`` uses to show these assertions are not vacuous.
    """
    predict_func, encoding_model, is_clusterless = fitted_backends[algorithm]
    spike_times, features = edge_case_data["spike_cases"][case]
    args = [
        edge_case_data["position_time"],
        edge_case_data["position"],
        spike_times,
    ]
    if is_clusterless:
        args.append(features)
    if time is None:
        time = edge_case_data["time"]
    return np.asarray(predict_func(time, *args, **encoding_model, **kwargs))


@pytest.mark.unit
@pytest.mark.parametrize("algorithm", sorted(_CLUSTERLESS_ALGORITHMS))
@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("ascending", [False, True])
@pytest.mark.parametrize("extra_rows", [-1, 1])
def test_backend_rejects_unpaired_feature_rows(
    fitted_backends, edge_case_data, algorithm, is_local, ascending, extra_rows
):
    """A prefix chunk must reject a mismatch elsewhere in the paired arrays."""
    predict, encoding, _ = fitted_backends[algorithm]
    spikes, features = edge_case_data["spike_cases"]["between_timestamps"]
    spikes = [s.copy() for s in spikes]
    features = [f.copy() for f in features]
    if not ascending:
        spikes[0] = spikes[0][::-1]
        features[0] = features[0][::-1]
    features[0] = (
        features[0][:-1]
        if extra_rows < 0
        else np.concatenate([features[0], features[0][-1:]])
    )
    with pytest.raises(ValidationError):
        predict(
            edge_case_data["time"],
            edge_case_data["position_time"],
            edge_case_data["position"],
            spikes,
            features,
            **encoding,
            is_local=is_local,
            row_slice=slice(0, 2),
        )


def assert_some_chunk_is_spike_free(edge_case_data, case, partitions) -> None:
    """At least one chunk of every partition must select zero spikes everywhere.

    That is the premise of the zero-row ``segment_sum`` / zero-row waveform
    feature selection this case exists to exercise; without the check the
    fixture could drift and the case would quietly become a duplicate of the
    ordinary boundary-spike ones.
    """
    time = edge_case_data["time"]
    spike_times, _ = edge_case_data["spike_cases"][case]
    for partition in partitions:
        spike_free = [
            row_slice
            for row_slice in partition
            if all(
                len(
                    select_spikes_in_rows(
                        unit_times, time, *resolve_row_slice(row_slice, len(time))
                    ).bin_ind
                )
                == 0
                for unit_times in spike_times
            )
        ]
        assert spike_free, f"no spike-free chunk in partition {partition}"


def chunk_local_result(fitted_backends, edge_case_data, algorithm, case, partition):
    """The legacy per-chunk call: each chunk gets only its own slice of ``time``."""
    full_time = edge_case_data["time"]
    return np.concatenate(
        [
            call_backend(
                fitted_backends,
                edge_case_data,
                algorithm,
                case,
                time=full_time[row_slice],
            )
            for row_slice in partition
        ]
    )


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("is_local", [False, True])
def test_zero_row_request_returns_zero_rows(
    fitted_backends, edge_case_data, algorithm, is_local
):
    """Documented behaviour of an empty row request at every backend: 0 rows."""
    full = call_backend(
        fitted_backends, edge_case_data, algorithm, "mixed", is_local=is_local
    )
    for row_slice in (slice(0, 0), slice(5, 5), slice(N_TIME, N_TIME)):
        rows = call_backend(
            fitted_backends,
            edge_case_data,
            algorithm,
            "mixed",
            is_local=is_local,
            row_slice=row_slice,
        )
        assert rows.shape == (0, full.shape[1])


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize(
    "case",
    ["on_timestamps", "between_timestamps", "spike_free_gap", "one_unit_empty"],
)
def test_irregular_and_ragged_partitions_tile_the_full_result(
    fitted_backends, edge_case_data, algorithm, case
):
    """Items 1-3 and 5: boundary spikes, spike-free chunks, ragged/uneven chunks.

    ``spike_free_gap`` requests a chunk whose selected-spike count is zero on
    every electrode, i.e. ``segment_sum`` over zero rows.
    """
    full = call_backend(fitted_backends, edge_case_data, algorithm, case)
    assert_fixture_scale(full, algorithm)
    # Even (ragged: 13 rows / 5 chunks) and irregular (chunk lengths 1, 4, 2, 6).
    partitions = [
        [
            slice(int(c[0]), int(c[-1]) + 1)
            for c in np.array_split(np.arange(N_TIME), 5)
        ],
        [slice(0, 1), slice(1, 5), slice(5, 7), slice(7, N_TIME)],
    ]
    if case == "spike_free_gap":
        assert_some_chunk_is_spike_free(edge_case_data, case, partitions)

    for partition in partitions:
        tiled = np.concatenate(
            [
                call_backend(
                    fitted_backends,
                    edge_case_data,
                    algorithm,
                    case,
                    row_slice=row_slice,
                )
                for row_slice in partition
            ]
        )
        assert tiled.shape == full.shape
        np.testing.assert_allclose(
            tiled,
            full,
            **parity_kwargs(algorithm),
            err_msg=f"{algorithm} {case} {partition}",
        )

        # Guard the guard: the legacy chunk-local call must FAIL the same
        # assertion, otherwise this case could not detect the defect it exists
        # for. Only the boundary-spike cases move a spike; the spike-free cases
        # are shape/allocation tests and are exempt.
        if case in ("on_timestamps", "between_timestamps"):
            chunk_local = chunk_local_result(
                fitted_backends, edge_case_data, algorithm, case, partition
            )
            assert not np.allclose(chunk_local, full, **parity_kwargs(algorithm)), (
                f"{algorithm} {case} {partition}: chunk-local binning is "
                "indistinguishable from global binning here, so this case "
                "cannot detect a lost or misplaced spike."
            )


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_all_units_empty_matches_full_result(
    fitted_backends, edge_case_data, algorithm
):
    """Item 3, hardest form: not one spike on any electrode, chunked or not."""
    full = call_backend(fitted_backends, edge_case_data, algorithm, "all_units_empty")
    assert np.all(np.isfinite(full))
    assert_fixture_scale(full, algorithm)
    # With no observed spikes every row of a backend's result is the same
    # (ground-process-only) row, so a wrong row count would still "look right";
    # the shape assertions below are what catch that.
    for partition in ([slice(0, 6), slice(6, N_TIME)], [slice(0, 1), slice(1, N_TIME)]):
        tiled = np.concatenate(
            [
                call_backend(
                    fitted_backends,
                    edge_case_data,
                    algorithm,
                    "all_units_empty",
                    row_slice=row_slice,
                )
                for row_slice in partition
            ]
        )
        assert tiled.shape == full.shape
        np.testing.assert_allclose(tiled, full, **parity_kwargs(algorithm))


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", REPRESENTATIVE_ALGORITHMS)
def test_singleton_row_requests_tile_the_full_result(
    fitted_backends, edge_case_data, algorithm
):
    """Item 4: ``n_chunks == n_time``, one row per call."""
    full = call_backend(fitted_backends, edge_case_data, algorithm, "mixed")
    assert_fixture_scale(full, algorithm)
    tiled = np.concatenate(
        [
            call_backend(
                fitted_backends,
                edge_case_data,
                algorithm,
                "mixed",
                row_slice=slice(row, row + 1),
            )
            for row in range(N_TIME)
        ]
    )
    assert tiled.shape == full.shape
    np.testing.assert_allclose(tiled, full, **parity_kwargs(algorithm))


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("prepared", [False, True])
def test_shuffled_spike_order_row_slice_parity(
    fitted_backends, edge_case_data, algorithm, is_local, prepared, monkeypatch
):
    """Item 7 at the backend: unsorted decoding input, features still paired.

    The spikes are permuted together with their waveform features, so the set of
    (time, feature) pairs is unchanged: a correct backend must return the same
    full-time likelihood as the sorted input *and* the same row ranges.

    CPU scatter implementations may ignore an incorrect sorted-indices hint.
    Check the JAX contract at the call boundary as well as the numerical result,
    so this catches invalid optimization promises without requiring a GPU.
    Both the fitted and zero-rate electrodes pass through this check.
    """
    original_segment_sum = jax.ops.segment_sum

    def checked_segment_sum(values, segment_ids, **kwargs):
        if kwargs.get("indices_are_sorted", False):
            ids = np.asarray(segment_ids)
            assert np.all(ids[1:] >= ids[:-1]), (
                "JAX was promised sorted segment IDs for unsorted spikes"
            )
        return original_segment_sum(values, segment_ids, **kwargs)

    monkeypatch.setattr(jax.ops, "segment_sum", checked_segment_sum)
    order_cache = _SpikeTimeOrder() if prepared else None
    check_order = common._spikes_are_ascending
    order_checks = []

    def tracked_order_check(spikes):
        order_checks.append(id(spikes))
        return check_order(spikes)

    monkeypatch.setattr(common, "_spikes_are_ascending", tracked_order_check)
    predict_func, encoding_model, is_clusterless = fitted_backends[algorithm]
    time = edge_case_data["time"]
    spike_times, features = edge_case_data["spike_cases"]["mixed"]

    rng = np.random.default_rng(23)
    shuffled_times = []
    shuffled_features = []
    for unit_times, unit_features in zip(spike_times, features, strict=True):
        order = rng.permutation(len(unit_times))
        shuffled_times.append(unit_times[order])
        shuffled_features.append(unit_features[order])
        assert not np.all(np.diff(shuffled_times[-1]) >= 0)

    def predict(spikes, feats, **kwargs):
        args = [edge_case_data["position_time"], edge_case_data["position"], spikes]
        if is_clusterless:
            args.append(feats)
        return np.asarray(
            predict_func(
                time,
                *args,
                **encoding_model,
                is_local=is_local,
                _spike_time_order=order_cache,
                **kwargs,
            )
        )

    sorted_full = predict(spike_times, features)
    assert_fixture_scale(sorted_full, algorithm)
    shuffled_full = predict(shuffled_times, shuffled_features)
    np.testing.assert_allclose(shuffled_full, sorted_full, **parity_kwargs(algorithm))
    checks_after_full = len(order_checks)

    row_slice = slice(3, 10)
    rows = predict(shuffled_times, shuffled_features, row_slice=row_slice)
    np.testing.assert_allclose(
        rows, shuffled_full[row_slice], **parity_kwargs(algorithm)
    )
    if prepared:
        assert checks_after_full > 0
        assert len(order_checks) == checks_after_full


@pytest.mark.unit
@pytest.mark.parametrize(
    "case", ["on_timestamps", "between_timestamps", "all_units_empty"]
)
def test_no_spike_model_edge_cases(edge_case_data, case):
    """The no-spike model shares the helper, so it shares the edge cases."""
    time = edge_case_data["time"]
    spike_times, _ = edge_case_data["spike_cases"][case]

    full = np.asarray(predict_no_spike_log_likelihood(time, spike_times))
    assert full.shape == (N_TIME, 1)

    assert np.asarray(
        predict_no_spike_log_likelihood(time, spike_times, row_slice=slice(4, 4))
    ).shape == (0, 1)
    for partition in row_partitions(N_TIME):
        tiled = np.concatenate(
            [
                np.asarray(
                    predict_no_spike_log_likelihood(
                        time, spike_times, row_slice=row_slice
                    )
                )
                for row_slice in partition
            ]
        )
        np.testing.assert_allclose(tiled, full, **EXACT)


# ---------------------------------------------------------------------------
# Allocation: a fixed row request must not scale with TOTAL decoding spikes
# ---------------------------------------------------------------------------

# The row request below always selects the same two decoding spikes; everything
# else is added far outside the requested rows. A backend that converts the
# whole waveform array (or the whole spike-time array) before selecting pays for
# every spike in the recording on every chunk call, which is what this pins.
ALLOC_N_TIME = 200
ALLOC_ROWS = slice(0, 5)
ALLOC_N_FEATURES = 4
ALLOC_SPIKE_COUNTS = (200, 6400)
# Threshold: a quarter of one float32 copy of the largest feature array. A
# backend that converts the full array allocates ~one such copy per call (the
# measured pre-fix growth was 77-90 KiB of a 100 KiB float32 copy); a backend
# that selects first allocates a fixed few KiB regardless of the total. A
# quarter therefore sits an order of magnitude above the post-fix noise and
# well below the defect, so it fails loudly without being flaky.
ALLOC_GROWTH_FRACTION = 0.25

# The local paths also take a recording-length ``position`` array, which they
# only ever interpolate on the host. Converting it to a device array (e.g. for
# its dtype) costs one float32 copy of the whole recording per chunk call.
# ``scipy.interpolate.interpn`` inside ``get_position_at_time`` allocates its own
# ~1 byte/sample temporary, which is shared by every backend and is not a device
# buffer, so the bound below is half of one float32 copy: comfortably above that
# shared floor (~0.24 of a copy) and far below a whole extra copy.
ALLOC_POSITION_SAMPLES = (2_000, 400_000)
POSITION_GROWTH_FRACTION = 0.5


@pytest.fixture(scope="module")
def allocation_data():
    """Encoding data plus decoding sets that differ only OUTSIDE the request."""
    rng = np.random.default_rng(11)
    position_time = np.linspace(0.0, 20.0, 2000)
    position = (50.0 + 40.0 * np.sin(2 * np.pi * position_time / 20.0))[:, None]
    environment = Environment(
        environment_name="line",
        place_bin_size=10.0,
        position_range=((0.0, 100.0),),
    ).fit_place_grid(position=position, infer_track_interior=False)
    time = np.linspace(0.0, 20.0, ALLOC_N_TIME)

    encoding_spike_times = [np.sort(rng.uniform(0.0, 20.0, 300))]
    encoding_features = [rng.standard_normal((300, ALLOC_N_FEATURES)) * 5.0 + 20.0]

    decoding = {}
    for n_total in ALLOC_SPIKE_COUNTS:
        inside = np.array([time[1] + 1e-3, time[3] + 1e-3])
        outside = np.sort(rng.uniform(time[50], time[-1], n_total))
        spike_times = np.concatenate([inside, outside])
        # float64 on purpose (numpy's default, and what most feature pipelines
        # hand in): on the CPU backend ``jnp.asarray`` of a float32 array is
        # zero-copy, so a full-array conversion of float32 features allocates
        # nothing ``tracemalloc`` can see and would silently disarm this test.
        # Same reason as scripts/benchmark_chunk_likelihood_memory.py's sweep.
        features = (
            rng.standard_normal((spike_times.shape[0], ALLOC_N_FEATURES)) * 5.0 + 20.0
        )
        decoding[n_total] = ([spike_times], [features])

    # The request must select the same two spikes in every set.
    for n_total, (spike_times, _) in decoding.items():
        selection = select_spikes_in_rows(spike_times[0], time, 0, 5)
        _, bin_ind = selection.indexer, selection.bin_ind
        assert bin_ind.shape[0] == 2, (n_total, bin_ind)

    return {
        "position_time": position_time,
        "position": position,
        "environment": environment,
        "time": time,
        "encoding_spike_times": encoding_spike_times,
        "encoding_features": encoding_features,
        "decoding": decoding,
    }


@pytest.fixture(scope="module")
def allocation_backends(allocation_data):
    """Fit the clusterless backends whose per-chunk allocation this pins."""
    fitted = {}
    for name in sorted(_CLUSTERLESS_ALGORITHMS):
        fit_func, predict_func = _CLUSTERLESS_ALGORITHMS[name]
        fitted[name] = (
            predict_func,
            fit_func(
                position_time=allocation_data["position_time"],
                position=allocation_data["position"],
                spike_times=allocation_data["encoding_spike_times"],
                spike_waveform_features=allocation_data["encoding_features"],
                environment=allocation_data["environment"],
                **BACKEND_FIT_PARAMS.get(name, {}),
            ),
        )
    return fitted


def peak_host_bytes(predict_func, args, call_kwargs) -> int:
    """Host peak of ONE call, after a warm-up call to exclude tracing."""
    predict_func(*args, **call_kwargs)  # warm up: compile/trace once
    tracemalloc.start()
    predict_func(*args, **call_kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def peak_bytes_for_spike_total(predict_func, encoding_model, data, n_total, is_local):
    """Host peak of one ``ALLOC_ROWS`` call with ``n_total`` decoding spikes."""
    spike_times, features = data["decoding"][n_total]
    args = (
        data["time"],
        data["position_time"],
        data["position"],
        spike_times,
        features,
    )
    call_kwargs = dict(**encoding_model, is_local=is_local, row_slice=ALLOC_ROWS)
    return peak_host_bytes(predict_func, args, call_kwargs)


@pytest.mark.integration
@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("algorithm", ["clusterless_diffusion", "clusterless_gmm"])
def test_fixed_row_request_does_not_allocate_per_total_spike(
    allocation_backends, allocation_data, algorithm, is_local
):
    """The same row request must cost the same whatever else is in the recording.

    Both backends select their spikes with ``select_spikes_in_rows`` and then
    evaluate only those. If the waveform features (or spike times) are converted
    to a device array *before* that selection, the call pays a full-array copy
    per chunk, so chunked prediction scales with the recording rather than with
    the request.
    """
    predict_func, encoding_model = allocation_backends[algorithm]
    small, large = ALLOC_SPIKE_COUNTS

    peak_small = peak_bytes_for_spike_total(
        predict_func, encoding_model, allocation_data, small, is_local
    )
    peak_large = peak_bytes_for_spike_total(
        predict_func, encoding_model, allocation_data, large, is_local
    )

    one_float32_copy = (large + 2) * ALLOC_N_FEATURES * 4
    allowed_growth = ALLOC_GROWTH_FRACTION * one_float32_copy
    growth = peak_large - peak_small
    assert growth < allowed_growth, (
        f"{algorithm} is_local={is_local}: host peak grew {growth / 1024:.1f} KiB "
        f"when total decoding spikes went {small} -> {large} at a fixed "
        f"{ALLOC_ROWS} request (allowed {allowed_growth / 1024:.1f} KiB, one "
        f"float32 copy of the feature array is {one_float32_copy / 1024:.1f} KiB)"
    )


def position_of_length(n_samples: int, duration_s: float = 20.0):
    """A position record of the requested length over the same time span."""
    position_time = np.linspace(0.0, duration_s, n_samples)
    position = (50.0 + 40.0 * np.sin(2 * np.pi * position_time / duration_s))[:, None]
    return position_time, position


def peak_bytes_for_position_length(
    predict_func, encoding_model, data, n_samples, is_local
):
    """Host peak of one ``ALLOC_ROWS`` call with a position record of this length."""
    spike_times, features = data["decoding"][ALLOC_SPIKE_COUNTS[0]]
    position_time, position = position_of_length(n_samples)
    args = (data["time"], position_time, position, spike_times, features)
    call_kwargs = dict(**encoding_model, is_local=is_local, row_slice=ALLOC_ROWS)
    return peak_host_bytes(predict_func, args, call_kwargs)


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", sorted(_CLUSTERLESS_ALGORITHMS))
def test_fixed_row_request_does_not_allocate_per_position_sample(
    allocation_backends, allocation_data, algorithm
):
    """A fixed row request must not cost a copy of the whole position record.

    The local paths interpolate position on the host at the requested rows (and
    at the selected spikes' times), so nothing recording-length should be
    converted to a device array -- not even to read its dtype.
    """
    predict_func, encoding_model = allocation_backends[algorithm]
    small, large = ALLOC_POSITION_SAMPLES

    peak_small = peak_bytes_for_position_length(
        predict_func, encoding_model, allocation_data, small, True
    )
    peak_large = peak_bytes_for_position_length(
        predict_func, encoding_model, allocation_data, large, True
    )

    one_float32_copy = large * 4
    allowed_growth = POSITION_GROWTH_FRACTION * one_float32_copy
    growth = peak_large - peak_small
    assert growth < allowed_growth, (
        f"{algorithm} is_local=True: host peak grew {growth / 1024:.1f} KiB when "
        f"the position record went {small:,} -> {large:,} samples at a fixed "
        f"{ALLOC_ROWS} request (allowed {allowed_growth / 1024:.1f} KiB, one "
        f"float32 copy of the position record is {one_float32_copy / 1024:.1f} KiB)"
    )


@pytest.mark.integration
@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("algorithm", sorted(_CLUSTERLESS_ALGORITHMS))
def test_row_slice_accepts_jax_array_inputs(
    allocation_backends, allocation_data, algorithm, is_local
):
    """A row request must select the same spikes from ``jax.Array`` inputs.

    Every other test in this module hands the backends NumPy arrays, so the
    ``jax.Array`` branch of each selection (``features[indexer]`` on a device
    array, ``np.asarray(times)`` on one) is otherwise untested. The decoding
    arrays are float32 here so the two input types carry bit-identical values
    and any difference is the selection, not a dtype truncation.
    """
    predict_func, encoding_model = allocation_backends[algorithm]
    spike_times, features = allocation_data["decoding"][ALLOC_SPIKE_COUNTS[0]]
    host_times = [np.asarray(t, dtype=np.float32) for t in spike_times]
    host_features = [np.asarray(f, dtype=np.float32) for f in features]
    device_times = [jnp.asarray(t) for t in host_times]
    device_features = [jnp.asarray(f) for f in host_features]

    common = (allocation_data["time"], allocation_data["position_time"])
    call_kwargs = dict(**encoding_model, is_local=is_local, row_slice=ALLOC_ROWS)
    from_host = np.asarray(
        predict_func(
            *common,
            allocation_data["position"],
            host_times,
            host_features,
            **call_kwargs,
        )
    )
    from_device = np.asarray(
        predict_func(
            *common,
            allocation_data["position"],
            device_times,
            device_features,
            **call_kwargs,
        )
    )

    assert from_device.shape == from_host.shape
    np.testing.assert_allclose(from_device, from_host, **parity_kwargs(algorithm))

    # ``jax.Array._npy_value`` caches the array's full host value the first time
    # a conversion has to COPY, and keeps it for the array's lifetime, so it is an
    # exact proxy for "this call materialized the whole recording-length array on
    # the host and retained it". On a single CPU device the conversion is a
    # zero-copy view and the cache is never populated (measured), so this
    # assertion is a guard -- it bites where the transfer must copy, i.e. a GPU
    # install -- and NOT the failing-then-passing evidence for the defect. That
    # evidence is the 2-device subprocess check below, where the gather is real.
    #
    # Only the 2-D FEATURES are asserted on: ``select_spikes_in_rows`` reads the
    # whole 1-D spike-time array for its ascending-order check (done once per
    # prediction on the detector path), so the spike times are materialized.
    for array in device_features:
        if not hasattr(array, "_npy_value"):
            pytest.skip(
                "jax.Array has no _npy_value cache in this JAX version, so "
                "the retained-host-copy proxy is unavailable"
            )
        assert array._npy_value is None, (
            f"{algorithm} is_local={is_local}: the call materialized the whole "
            f"decoding feature array on the host (its cached host value is set), "
            f"so a chunk call costs the whole recording instead of the request"
        )


# Exit codes of _sharded_jax_input_check.py (kept in sync with that module).
CHECK_EXIT_OK = 0
CHECK_EXIT_ENVIRONMENT = 2
# The child prints this before exiting with CHECK_EXIT_ENVIRONMENT. CPython
# itself exits 2 when it cannot open the script, with nothing on stdout.
CHECK_SKIP_MARKER = "SKIP EnvironmentUnavailable"


@pytest.mark.integration
@pytest.mark.parametrize("axis_type", ["Auto", "Explicit"])
def test_sharded_jax_inputs_do_not_materialize_the_recording(axis_type):
    """A chunk call on SHARDED ``jax.Array`` inputs must not gather the inputs.

    Runs in a subprocess because ``XLA_FLAGS`` has to be set before JAX is
    imported and this process has already imported it. The child checks every
    clusterless backend and both ``is_local`` values, on both mesh axis types:
    under ``Auto`` the selection stays on device (nothing recording-length
    reaches the host); under ``Explicit`` JAX refuses a gather whose output
    sharding it cannot infer, so the documented fallback gathers first — it must
    still produce a result rather than raising.
    """
    script = Path(__file__).with_name("_sharded_jax_input_check.py")
    result = subprocess.run(
        [sys.executable, str(script), "2", axis_type],
        capture_output=True,
        text=True,
        timeout=900,
    )
    # The child uses distinct exit codes so an environment that cannot run the
    # check (fewer than two devices, an older JAX) skips, while a failed
    # assertion fails. The skip also requires the child's marker: a missing
    # script or a setup bug exits 2 too, and must fail rather than skip forever.
    if result.returncode == CHECK_EXIT_ENVIRONMENT and result.stdout.startswith(
        CHECK_SKIP_MARKER
    ):
        pytest.skip(f"sharded check cannot run here: {result.stdout.strip()[-300:]}")
    assert result.returncode == CHECK_EXIT_OK, (
        f"exit code {result.returncode}\n{result.stdout}\n{result.stderr[-2000:]}"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("inject", "expected"),
    [("setup", CHECK_EXIT_ENVIRONMENT), ("setup-bug", 1), ("run", 1)],
    ids=["unavailable-is-a-skip", "setup-bug-is-a-failure", "run-error-is-a-failure"],
)
def test_sharded_check_maps_only_setup_errors_to_a_skip(inject, expected):
    """Only ``EnvironmentUnavailable`` from setup may become an environment skip.

    Any other exception -- a ``RuntimeError`` while building the mesh, fitting,
    predicting or asserting -- must reach the parent as a non-skip, non-zero
    exit (the uncaught exception's 1), so a broken backend or a broken check
    cannot hide behind ``pytest.skip``. Uses the child's
    ``NLD_SHARDED_CHECK_INJECT`` hook so no real fit or predict runs.

    Independent of device availability: the child is asked for a ONE-device
    mesh on the CPU platform, so its device-count check passes on any machine
    (a single GPU, ``JAX_NUM_CPU_DEVICES=1``) and the injected error is what
    decides the exit code. The two-device behaviour is the other test's job.
    """
    script = Path(__file__).with_name("_sharded_jax_input_check.py")
    result = subprocess.run(
        [sys.executable, str(script), "1", "Auto"],
        capture_output=True,
        text=True,
        timeout=300,
        env={
            **os.environ,
            "NLD_SHARDED_CHECK_INJECT": inject,
            "JAX_PLATFORMS": "cpu",
        },
    )
    assert result.returncode == expected, (
        f"exit code {result.returncode}\n{result.stdout}\n{result.stderr[-2000:]}"
    )
    assert "injected" in (result.stdout + result.stderr)
    if expected == CHECK_EXIT_ENVIRONMENT:
        assert result.stdout.startswith(CHECK_SKIP_MARKER)
    else:
        assert "SKIP" not in result.stdout


@pytest.mark.unit
@pytest.mark.parametrize("as_jax", [False, True])
@pytest.mark.parametrize("indexer_kind", ["mask", "slice"])
def test_select_spike_rows_rejects_a_mismatched_per_spike_array(as_jax, indexer_kind):
    """Fewer feature rows than spike times must raise, not NaN-fill.

    ``jnp.take`` defaults to ``mode="fill"``, which returns NaN for an
    out-of-range row instead of raising the way NumPy does -- a silent-NaN path
    into the likelihood if a caller's waveform features and spike times ever
    disagree in length.
    """
    features = np.arange(8, dtype=np.float32).reshape(4, 2)
    if as_jax:
        features = jnp.asarray(features)

    times = np.arange(6.0)
    if indexer_kind == "mask":
        times = times[::-1]
    selection = select_spikes_in_rows(times, np.arange(7.0), 2, 6)

    with pytest.raises(ValidationError, match="per-spike array"):
        select_spike_rows(features, selection)


@pytest.mark.unit
@pytest.mark.parametrize("as_jax", [False, True])
def test_select_spike_rows_selects_the_named_rows(as_jax):
    """The helper is a selection, whatever the input type or indexer."""
    features = np.arange(10, dtype=np.float32).reshape(5, 2)
    if as_jax:
        features = jnp.asarray(features)

    selection = select_spikes_in_rows(
        np.array([0.5, 4.5, 5.5, 1.5, 6.5]), np.arange(8.0), 0, 2
    )
    np.testing.assert_array_equal(
        np.asarray(select_spike_rows(features, selection)), [[0.0, 1.0], [6.0, 7.0]]
    )
    selection = select_spikes_in_rows(np.arange(5.0), np.arange(6.0), 1, 3)
    np.testing.assert_array_equal(
        np.asarray(select_spike_rows(features, selection)), [[2.0, 3.0], [4.0, 5.0]]
    )
    selection = select_spikes_in_rows(np.arange(5.0), np.arange(6.0), 2, 2)
    assert np.asarray(select_spike_rows(features, selection)).shape == (0, 2)


# JAX releases without explicit sharding (``AxisType``) have no
# ``ShardingTypeError`` either, and the fallback correctly catches nothing there.
requires_explicit_sharding = pytest.mark.skipif(
    not hasattr(jax.sharding, "AxisType"),
    reason="this JAX predates explicit sharding",
)


@pytest.mark.unit
@requires_explicit_sharding
def test_sharding_error_type_is_known_on_this_jax():
    """On a JAX with explicit sharding, the fallback's exception tuple is set.

    ``ShardingTypeError`` is imported from a private JAX path; if a release
    moves it the ``ImportError`` guard leaves an empty tuple and every
    explicitly-sharded gather raises instead of falling back.
    """
    assert common._JAX_SHARDING_ERRORS
    assert all(issubclass(t, Exception) for t in common._JAX_SHARDING_ERRORS)


@pytest.mark.unit
@requires_explicit_sharding
@pytest.mark.parametrize("indexer_kind", ["mask", "slice"])
def test_select_spike_rows_falls_back_to_a_host_copy_on_sharding_errors(
    monkeypatch, indexer_kind
):
    """A sharding refusal on device selects the same rows on the host.

    The subprocess check exercises this on a real ``Explicit`` mesh but skips
    on single-device machines; this pins the fallback in-process by raising
    the sharding error type from the device-side selection.
    """
    features = jnp.arange(10, dtype=jnp.float32).reshape(5, 2)
    error = common._JAX_SHARDING_ERRORS[0]("simulated sharding refusal")

    def fail_selection(*args, **kwargs):
        raise error

    if indexer_kind == "mask":
        selection = select_spikes_in_rows(
            np.array([0.5, 4.5, 5.5, 1.5, 6.5]), np.arange(8.0), 0, 2
        )
        expected = [[0.0, 1.0], [6.0, 7.0]]
        monkeypatch.setattr(jnp, "take", fail_selection)
    else:
        selection = select_spikes_in_rows(np.arange(5.0), np.arange(6.0), 1, 3)
        expected = [[2.0, 3.0], [4.0, 5.0]]
        monkeypatch.setattr(type(features), "__getitem__", fail_selection)

    selected = select_spike_rows(features, selection)

    assert isinstance(selected, np.ndarray)
    np.testing.assert_array_equal(selected, expected)


@pytest.mark.unit
@pytest.mark.parametrize("indexer_kind", ["mask", "slice"])
@pytest.mark.parametrize(
    "error_type", [RuntimeError, ValueError, TypeError, MemoryError]
)
def test_select_spike_rows_propagates_unexpected_errors(
    monkeypatch, indexer_kind, error_type
):
    """Only unsupported sharding may trigger a recording-sized host copy."""
    features = jnp.arange(10, dtype=jnp.float32).reshape(5, 2)
    error = error_type("unexpected selection failure")
    host_copies = []
    asarray = np.asarray

    def tracked_asarray(array, *args, **kwargs):
        if array is features:
            host_copies.append(array)
        return asarray(array, *args, **kwargs)

    def fail_selection(*args, **kwargs):
        raise error

    if indexer_kind == "mask":
        selection = select_spikes_in_rows(
            np.array([0.5, 4.5, 5.5, 1.5, 6.5]), np.arange(8.0), 0, 2
        )
        monkeypatch.setattr(jnp, "take", fail_selection)
    else:
        selection = select_spikes_in_rows(np.arange(5.0), np.arange(6.0), 1, 3)
        monkeypatch.setattr(type(features), "__getitem__", fail_selection)
    monkeypatch.setattr(np, "asarray", tracked_asarray)

    with pytest.raises(error_type, match="unexpected selection failure") as exc:
        select_spike_rows(features, selection)

    assert exc.value is error
    assert not host_copies
