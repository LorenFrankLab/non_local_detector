"""Padded, compiled clusterless KDE kernels match the per-electrode formulation."""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods import clusterless_kde
from non_local_detector.likelihoods.clusterless_kde import (
    block_estimate_log_joint_mark_intensity,
    fit_clusterless_kde_encoding_model,
    kde_distance,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.likelihoods.common import (
    RATE_EPS_HZ,
    RATE_REFERENCE_SECONDS,
    SpikeSelection,
    _padded_sample_count,
    _padded_spike_count,
    _SpikeTimeOrder,
    as_std_array,
    block_kde,
    decode_bin_centers,
    get_position_at_time,
    log_bin_duration_evidence,
    safe_log,
    select_spike_rows,
    select_spikes_in_rows,
    spike_row_ids,
    sum_spikes_into_rows,
)

pytestmark = pytest.mark.unit


def test_padded_spike_counts_are_few_and_cover_each_count():
    sizes = {_padded_spike_count(n, 100) for n in range(0, 5001)}
    for n in range(0, 5001):
        size = _padded_spike_count(n, 100)
        assert size >= max(n, 16)
        assert size <= 100 or size % 100 == 0
    assert sizes == {16, 32, 64, 100, 200, 400, 800, 1600, 3200, 6400}


def test_padded_sample_counts_add_at_most_a_quarter():
    for n in range(17, 20000):
        size = _padded_sample_count(n)
        assert n <= size <= 1.25 * n
    assert _padded_sample_count(0) == _padded_sample_count(16) == 16
    assert len({_padded_sample_count(n) for n in range(1025, 2049)}) == 4


def test_spike_row_ids_normalize_invalid_rows_and_pad_with_dropped_row():
    selection = SpikeSelection(
        indexer=slice(0, 4),
        bin_ind=np.array([-3, 0, 2, 9], dtype=np.int64),
        indices_are_sorted=True,
        n_spikes=4,
        n_rows=5,
    )
    np.testing.assert_array_equal(spike_row_ids(selection), [-1, 0, 2, 5])
    np.testing.assert_array_equal(spike_row_ids(selection, 7), [-1, 0, 2, 5, 5, 5, 5])


def test_spike_time_order_memo_reuses_builds_per_source_identity():
    order = _SpikeTimeOrder()
    first, second = np.zeros(3), np.zeros(3)
    calls = []

    def build():
        calls.append(1)
        return object()

    value = order.memo("padded", (first,), build)
    assert order.memo("padded", (first,), build) is value
    assert order.memo("padded", (second,), build) is not value
    assert order.memo("other", (first,), build) is not value
    assert len(calls) == 3


@pytest.fixture(scope="module")
def heterogeneous_recording():
    """Three electrodes with different encoding counts and mark dimensions."""
    rng = np.random.default_rng(11)
    env = Environment(place_bin_size=4.0, position_range=((0.0, 40.0), (0.0, 40.0)))
    position_time = np.arange(0, 30.0, 1 / 30)
    position = np.column_stack(
        (20 + 15 * np.sin(position_time / 3), 20 + 15 * np.cos(position_time / 5))
    )
    env = env.fit_place_grid(position, infer_track_interior=False)
    rates, dims = (3.0, 12.0, 25.0), (2, 4, 4)
    encoding_times = [np.sort(rng.uniform(0, 30, int(30 * r))) for r in rates]
    encoding_marks = [
        rng.normal(0, 30, (len(t), d))
        for t, d in zip(encoding_times, dims, strict=True)
    ]
    encoding = fit_clusterless_kde_encoding_model(
        position_time=position_time,
        position=position,
        spike_times=encoding_times,
        spike_waveform_features=encoding_marks,
        environment=env,
        position_std=3.0,
        waveform_std=24.0,
        block_size=16,
        disable_progress_bar=True,
    )
    decode_times = [np.sort(rng.uniform(0, 4, int(4 * r))) for r in rates]
    decode_marks = [
        rng.normal(0, 30, (len(t), d)) for t, d in zip(decode_times, dims, strict=True)
    ]
    edges = np.arange(0, 4.0 + 1e-9, 0.002)
    return env, position_time, position, encoding, decode_times, decode_marks, edges


def predict(recording, is_local, row_slice):
    env, position_time, position, encoding, times, marks, edges = recording
    return predict_clusterless_kde_log_likelihood(
        position_time=position_time,
        position=position,
        spike_times=times,
        spike_waveform_features=marks,
        occupancy=encoding["occupancy"],
        occupancy_model=encoding["occupancy_model"],
        gpi_models=encoding["gpi_models"],
        encoding_spike_waveform_features=encoding["encoding_spike_waveform_features"],
        encoding_positions=encoding["encoding_positions"],
        environment=env,
        mean_rates=np.asarray(encoding["mean_rates"]),
        summed_ground_process_intensity=encoding["summed_ground_process_intensity"],
        position_std=encoding["position_std"],
        waveform_std=encoding["waveform_std"],
        is_local=is_local,
        block_size=encoding["block_size"],
        disable_progress_bar=True,
        encoding_weights=encoding["encoding_weights"],
        row_slice=row_slice,
        time_edges=edges,
    )


def reference(recording, is_local, row_slice):
    """The per-electrode formulation from the public building blocks."""
    env, position_time, position, encoding, times, marks, edges = recording
    start, stop = row_slice.start, row_slice.stop
    durations = jnp.asarray(np.diff(edges)[start:stop])
    block_size = encoding["block_size"]
    electrodes = zip(
        encoding["encoding_spike_waveform_features"],
        encoding["encoding_positions"],
        encoding["encoding_weights"],
        encoding["mean_rates"],
        encoding["gpi_models"],
        marks,
        times,
        strict=True,
    )
    if not is_local:
        interior = env.place_bin_centers_[env.is_track_interior_.ravel()]
        total = -durations[:, None] * encoding["summed_ground_process_intensity"]
        for enc_marks, enc_pos, weights, rate, _, dec_marks, dec_times in electrodes:
            selection = select_spikes_in_rows(dec_times, start, stop, time_edges=edges)
            total = total + sum_spikes_into_rows(
                block_estimate_log_joint_mark_intensity(
                    select_spike_rows(dec_marks, selection),
                    enc_marks,
                    as_std_array(encoding["waveform_std"], enc_marks.shape[1]),
                    encoding["occupancy"],
                    rate * RATE_REFERENCE_SECONDS,
                    kde_distance(interior, enc_pos, encoding["position_std"]),
                    block_size,
                    encoding_weights=weights,
                ),
                selection,
            )
    else:
        occupancy_model = encoding["occupancy_model"]
        positions = get_position_at_time(
            position_time, position, decode_bin_centers(edges, start, stop), env
        )
        occupancy = occupancy_model.predict(positions)
        total, expected = jnp.zeros(stop - start), jnp.zeros(stop - start)
        for enc_marks, enc_pos, weights, rate, gpi, dec_marks, dec_times in electrodes:
            selection = select_spikes_in_rows(dec_times, start, stop, time_edges=edges)
            spike_positions = get_position_at_time(
                position_time,
                position,
                select_spike_rows(dec_times, selection),
                env,
            )
            marginal = block_kde(
                jnp.concatenate(
                    (spike_positions, select_spike_rows(dec_marks, selection)), axis=1
                ),
                jnp.concatenate((enc_pos, enc_marks), axis=1),
                jnp.concatenate(
                    (
                        encoding["position_std"],
                        as_std_array(encoding["waveform_std"], enc_marks.shape[1]),
                    )
                ),
                block_size,
                weights,
            )
            at_spikes = occupancy_model.predict(spike_positions)
            total = total + sum_spikes_into_rows(
                safe_log(
                    rate
                    * RATE_REFERENCE_SECONDS
                    * jnp.where(
                        at_spikes > 0,
                        marginal / jnp.where(at_spikes > 0, at_spikes, 1),
                        0,
                    )
                ),
                selection,
            )
            expected = expected + rate * jnp.where(
                occupancy > 0,
                gpi.predict(positions) / jnp.where(occupancy > 0, occupancy, 1),
                0,
            )
        total = (total - durations * jnp.clip(expected, min=RATE_EPS_HZ))[:, None]
    return (
        total
        + log_bin_duration_evidence(
            times,
            edges,
            row_slice,
            None,
            intensity_time_scale=RATE_REFERENCE_SECONDS,
        )[:, None]
    )


@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("rows", [(0, 2000), (300, 333), (1990, 2000), (5, 6)])
def test_padded_kernels_match_the_per_electrode_formulation(
    heterogeneous_recording, is_local, rows
):
    row_slice = slice(*rows)
    actual = predict(heterogeneous_recording, is_local, row_slice)
    expected = reference(heterogeneous_recording, is_local, row_slice)
    assert actual.shape == expected.shape
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)


def test_electrodes_with_nearby_encoding_sizes_share_compiled_kernels():
    rng = np.random.default_rng(12)
    env = Environment(place_bin_size=5.0, position_range=((0.0, 30.0), (0.0, 30.0)))
    position_time = np.arange(0, 20.0, 1 / 30)
    position = np.column_stack(
        (15 + 10 * np.sin(position_time), 15 + 10 * np.cos(position_time))
    )
    env = env.fit_place_grid(position, infer_track_interior=False)
    # 200 and 210 encoding spikes share one padded size; 40 and 50 decoding
    # spikes share another.
    encoding_times = [np.sort(rng.uniform(0, 20, n)) for n in (200, 210)]
    encoding = fit_clusterless_kde_encoding_model(
        position_time=position_time,
        position=position,
        spike_times=encoding_times,
        spike_waveform_features=[
            rng.normal(0, 30, (len(t), 4)) for t in encoding_times
        ],
        environment=env,
        disable_progress_bar=True,
    )
    assert _padded_sample_count(200) == _padded_sample_count(210)
    decode_times = [np.sort(rng.uniform(0, 2, n)) for n in (40, 50)]
    assert _padded_spike_count(40, 100) == _padded_spike_count(50, 100)
    recording = (
        env,
        position_time,
        position,
        encoding,
        decode_times,
        [rng.normal(0, 30, (len(t), 4)) for t in decode_times],
        np.arange(0, 2.0 + 1e-9, 0.002),
    )
    kernels = (
        clusterless_kde._add_electrode_mark_intensities,
        clusterless_kde._add_electrode_local_terms,
    )
    for kernel in kernels:
        kernel.clear_cache()
    for is_local in (False, True):
        predict(recording, is_local, None)
    assert [kernel._cache_size() for kernel in kernels] == [1, 1]
