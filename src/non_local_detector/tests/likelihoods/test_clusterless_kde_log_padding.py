"""Padded, compiled log-space clusterless KDE kernels match the per-electrode formulation."""

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods import clusterless_kde_log
from non_local_detector.likelihoods.clusterless_kde_log import (
    _log_joint_from_log_marginal,
    block_estimate_log_joint_mark_intensity,
    fit_clusterless_kde_encoding_model,
    log_kde_distance,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.likelihoods.common import (
    LOG_EPS,
    RATE_EPS_HZ,
    RATE_REFERENCE_SECONDS,
    _padded_sample_count,
    _padded_spike_count,
    as_std_array,
    block_log_kde,
    decode_bin_centers,
    get_position_at_time,
    log_bin_duration_evidence,
    select_spike_rows,
    select_spikes_in_rows,
    sum_spikes_into_rows,
)

pytestmark = pytest.mark.unit


@contextmanager
def precision_mode(x64):
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", x64)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def make_recording(x64, waveform_dims=(2, 4, 4)):
    """Electrodes with different encoding counts and mark dimensions."""
    rng = np.random.default_rng(31)
    env = Environment(place_bin_size=4.0, position_range=((0.0, 40.0), (0.0, 40.0)))
    position_time = np.arange(0, 30.0, 1 / 30)
    position = np.column_stack(
        (20 + 15 * np.sin(position_time / 3), 20 + 15 * np.cos(position_time / 5))
    )
    env = env.fit_place_grid(position, infer_track_interior=False)
    rates = (3.0, 12.0, 25.0)
    encoding_times = [np.sort(rng.uniform(0, 30, int(30 * r))) for r in rates]
    encoding_marks = [
        rng.normal(0, 30, (len(t), d))
        for t, d in zip(encoding_times, waveform_dims, strict=True)
    ]
    decode_times = [np.sort(rng.uniform(0, 4, int(4 * r))) for r in rates]
    decode_marks = [
        rng.normal(0, 30, (len(t), d))
        for t, d in zip(decode_times, waveform_dims, strict=True)
    ]
    edges = np.arange(0, 4.0 + 1e-9, 0.002)
    with precision_mode(x64):
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
    return env, position_time, position, encoding, decode_times, decode_marks, edges


@pytest.fixture(scope="module", params=[False, True], ids=["float32", "float64"])
def recording(request):
    return request.param, make_recording(request.param)


def predict(recording, is_local, row_slice):
    env, position_time, position, encoding, times, marks, edges = recording
    return predict_clusterless_kde_log_likelihood(
        position_time,
        position,
        times,
        marks,
        time_edges=edges,
        **encoding,
        is_local=is_local,
        row_slice=row_slice,
    )


def reference(recording, is_local, row_slice):
    """The per-electrode formulation from the public building blocks."""
    env, position_time, position, encoding, times, marks, edges = recording
    start, stop = row_slice.start, row_slice.stop
    durations = jnp.asarray(np.diff(edges)[start:stop])
    block_size = encoding["block_size"]
    position_std = jnp.asarray(encoding["position_std"])
    weights = encoding["encoding_weights"] or [None] * len(times)
    electrodes = zip(
        encoding["encoding_spike_waveform_features"],
        encoding["encoding_positions"],
        weights,
        encoding["mean_rates"],
        encoding["gpi_models"],
        marks,
        times,
        strict=True,
    )
    if not is_local:
        bins = env.place_bin_centers_[env.is_track_interior_.ravel()]
        total = -durations[:, None] * encoding["summed_ground_process_intensity"]
        for enc_marks, enc_pos, w, rate, _, dec_marks, dec_times in electrodes:
            selection = select_spikes_in_rows(dec_times, start, stop, time_edges=edges)
            total = total + sum_spikes_into_rows(
                block_estimate_log_joint_mark_intensity(
                    select_spike_rows(dec_marks, selection),
                    enc_marks,
                    as_std_array(encoding["waveform_std"], enc_marks.shape[1]),
                    encoding["occupancy"],
                    rate * RATE_REFERENCE_SECONDS,
                    log_kde_distance(bins, enc_pos, std=position_std),
                    block_size=block_size,
                    encoding_weights=w,
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
        for enc_marks, enc_pos, w, rate, gpi, dec_marks, dec_times in electrodes:
            selection = select_spikes_in_rows(dec_times, start, stop, time_edges=edges)
            spike_positions = get_position_at_time(
                position_time, position, select_spike_rows(dec_times, selection), env
            )
            log_marginal = block_log_kde(
                jnp.concatenate(
                    (spike_positions, select_spike_rows(dec_marks, selection)), axis=1
                ),
                jnp.concatenate((enc_pos, enc_marks), axis=1),
                jnp.concatenate(
                    (
                        position_std,
                        as_std_array(encoding["waveform_std"], enc_marks.shape[1]),
                    )
                ),
                block_size,
                w,
            )
            contribution = jnp.maximum(
                _log_joint_from_log_marginal(
                    log_marginal[None, :],
                    rate * RATE_REFERENCE_SECONDS,
                    occupancy_model.predict(spike_positions),
                )[0],
                LOG_EPS,
            )
            total = total + sum_spikes_into_rows(contribution, selection)
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
def test_padded_kernels_match_the_per_electrode_formulation(recording, is_local, rows):
    x64, recording = recording
    with precision_mode(x64):
        row_slice = slice(*rows)
        actual = predict(recording, is_local, row_slice)
        expected = reference(recording, is_local, row_slice)
        assert actual.shape == expected.shape
        assert actual.dtype == (jnp.float64 if x64 else jnp.float32)
        assert np.all(np.isfinite(actual))
        tolerance = (
            {"rtol": 1e-10, "atol": 1e-10} if x64 else {"rtol": 1e-5, "atol": 1e-4}
        )
        np.testing.assert_allclose(actual, expected, **tolerance)


def test_high_dimensional_marks_use_the_logsumexp_path():
    """Above eight waveform dimensions the kernels take the logsumexp branch."""
    recording = make_recording(False, waveform_dims=(9, 9, 10))
    for is_local in (False, True):
        actual = predict(recording, is_local, slice(0, 2000))
        expected = reference(recording, is_local, slice(0, 2000))
        assert np.all(np.isfinite(actual))
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)


def test_electrodes_with_nearby_sizes_share_compiled_kernels():
    rng = np.random.default_rng(32)
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
    assert _padded_spike_count(40, 100) == _padded_spike_count(50, 100)
    decode_times = [np.sort(rng.uniform(0, 2, n)) for n in (40, 50)]
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
        clusterless_kde_log._add_electrode_log_mark_intensities,
        clusterless_kde_log._add_electrode_local_terms,
    )
    for kernel in kernels:
        kernel.clear_cache()
    for is_local in (False, True):
        predict(recording, is_local, None)
    assert [kernel._cache_size() for kernel in kernels] == [1, 1]
