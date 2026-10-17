"""Padded, compiled clusterless GMM kernels match the per-electrode formulation."""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods import clusterless_gmm
from non_local_detector.likelihoods.clusterless_gmm import (
    _ground_process_intensity,
    fit_clusterless_gmm_encoding_model,
    mark_reference_scale,
    predict_clusterless_gmm_log_likelihood,
)
from non_local_detector.likelihoods.common import (
    LOG_RATE_EPS_HZ,
    RATE_EPS_HZ,
    _padded_spike_count,
    decode_bin_centers,
    get_position_at_time,
    log_bin_duration_evidence,
    safe_log,
    select_spike_rows,
    select_spikes_in_rows,
    sum_spikes_into_rows,
)
from non_local_detector.likelihoods.mark_reference import (
    background_log_intensity,
)
from non_local_detector.tests.conftest import precision_mode
from non_local_detector.tests.likelihoods.conftest import training_mark_references

pytestmark = pytest.mark.unit

GMM_COMPONENTS = {
    "gmm_components_occupancy": 4,
    "gmm_components_joint": 8,
}


def fit_reference(position_time, times, marks):
    """Mark references from the training spikes, as a detector fits them."""
    return training_mark_references(position_time, times, marks, mark_reference_scale())


def make_recording(x64):
    """Electrodes with different spike rates and mark dimensions, one unfitted.

    The last electrode has no encoding spikes, so its joint model is ``None``.
    """
    rng = np.random.default_rng(21)
    env = Environment(place_bin_size=4.0, position_range=((0.0, 40.0), (0.0, 40.0)))
    position_time = np.arange(0, 60.0, 1 / 30)
    position = np.column_stack(
        (20 + 15 * np.sin(position_time / 3), 20 + 15 * np.cos(position_time / 5))
    )
    env = env.fit_place_grid(position, infer_track_interior=False)
    rates, dims = (3.0, 12.0, 25.0, 0.0), (2, 4, 4, 4)
    encoding_times = [np.sort(rng.uniform(0, 60, int(60 * r))) for r in rates]
    encoding_marks = [
        rng.normal(0, 30, (len(t), d))
        for t, d in zip(encoding_times, dims, strict=True)
    ]
    decode_times = [np.sort(rng.uniform(0, 4, int(4 * max(r, 5.0)))) for r in rates]
    decode_marks = [
        rng.normal(0, 30, (len(t), d)) for t, d in zip(decode_times, dims, strict=True)
    ]
    edges = np.arange(0, 4.0 + 1e-9, 0.002)
    with precision_mode(x64):
        encoding = fit_clusterless_gmm_encoding_model(
            position_time=position_time,
            position=position,
            spike_times=encoding_times,
            spike_waveform_features=encoding_marks,
            environment=env,
            disable_progress_bar=True,
            **GMM_COMPONENTS,
        )
    mark_reference = fit_reference(position_time, encoding_times, encoding_marks)
    return (
        env,
        position_time,
        position,
        encoding,
        mark_reference,
        decode_times,
        decode_marks,
        edges,
    )


@pytest.fixture(scope="module", params=[False, True], ids=["float32", "float64"])
def recording(request):
    return request.param, make_recording(request.param)


def predict(recording, is_local, row_slice, **kwargs):
    env, position_time, position, encoding, mark_reference, times, marks, edges = (
        recording
    )
    return predict_clusterless_gmm_log_likelihood(
        position_time,
        position,
        times,
        marks,
        time_edges=edges,
        **{**encoding, **kwargs},
        mark_reference=mark_reference,
        is_local=is_local,
        row_slice=row_slice,
    )


def reference(recording, is_local, row_slice):
    """Each spike scored eagerly with ``score_samples``, as the loop did before."""
    env, position_time, position, encoding, mark_reference, times, marks, edges = (
        recording
    )
    start, stop = row_slice.start, row_slice.stop
    durations = jnp.asarray(np.diff(edges)[start:stop])
    electrodes = zip(
        times,
        marks,
        encoding["joint_models"],
        encoding["gpi_models"],
        encoding["mean_rates"],
        mark_reference,
        strict=True,
    )

    def background(features, selection, reference):
        return jnp.asarray(
            background_log_intensity(
                reference,
                np.asarray(select_spike_rows(features, selection)),
                LOG_RATE_EPS_HZ,
            )
        )

    if not is_local:
        bins = jnp.asarray(encoding["interior_place_bin_centers"])
        total = -durations[:, None] * encoding["summed_ground_process_intensity"]
        for spike_times, features, joint, _, rate, reference in electrodes:
            selection = select_spikes_in_rows(
                spike_times, start, stop, time_edges=edges
            )
            n_spikes = selection.bin_ind.shape[0]
            if n_spikes == 0:
                continue
            log_background = background(features, selection, reference)
            if joint is None:
                total = total + sum_spikes_into_rows(log_background, selection)[:, None]
                continue
            log_density = jnp.stack(
                [
                    joint.score_samples(
                        jnp.concatenate(
                            [bins, jnp.repeat(feature[None], len(bins), axis=0)],
                            axis=1,
                        )
                    )
                    for feature in jnp.asarray(select_spike_rows(features, selection))
                ]
            )
            total = total + sum_spikes_into_rows(
                jnp.logaddexp(
                    safe_log(rate, eps=RATE_EPS_HZ)
                    + (log_density - encoding["log_occupancy"]),
                    log_background[:, None],
                ),
                selection,
            )
    else:
        occupancy_model = encoding["occupancy_model"]
        positions = get_position_at_time(
            position_time, position, decode_bin_centers(edges, start, stop), env
        )
        log_occupancy = occupancy_model.score_samples(positions)
        total, expected = jnp.zeros(stop - start), jnp.zeros(stop - start)
        for spike_times, features, joint, gpi, rate, reference in electrodes:
            selection = select_spikes_in_rows(
                spike_times, start, stop, time_edges=edges
            )
            n_spikes = selection.bin_ind.shape[0]
            if joint is None:
                if n_spikes:
                    total = total + sum_spikes_into_rows(
                        background(features, selection, reference), selection
                    )
                continue
            if n_spikes:
                spike_positions = get_position_at_time(
                    position_time,
                    position,
                    select_spike_rows(spike_times, selection),
                    env,
                )
                terms = jnp.logaddexp(
                    safe_log(rate, eps=RATE_EPS_HZ)
                    + (
                        joint.score_samples(
                            jnp.concatenate(
                                [
                                    spike_positions,
                                    jnp.asarray(select_spike_rows(features, selection)),
                                ],
                                axis=1,
                            )
                        )
                        - occupancy_model.score_samples(spike_positions)
                    ),
                    background(features, selection, reference),
                )
                total = total + sum_spikes_into_rows(terms, selection)
            expected = expected + _ground_process_intensity(
                rate, gpi.score_samples(positions), log_occupancy
            )
        n_electrodes = len(encoding["mean_rates"])
        total = (total - durations * (expected + n_electrodes * RATE_EPS_HZ))[:, None]
    return total + log_bin_duration_evidence(times, edges, row_slice, None)[:, None]


@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("rows", [(0, 2000), (300, 333), (5, 6)])
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


def test_non_local_block_size_does_not_change_the_result(recording):
    x64, recording = recording
    if x64:
        pytest.skip("float32 covers the block loop")
    whole = predict(recording, False, slice(0, 2000), spike_block_size=10_000)
    blocked = predict(recording, False, slice(0, 2000), spike_block_size=7)
    np.testing.assert_allclose(blocked, whole, rtol=1e-6, atol=1e-5)


def test_fine_padded_spike_counts_add_at_most_a_quarter():
    for n in range(17, 1001):
        assert n <= _padded_spike_count(n, 1000, fine=True) <= 1.25 * n
    assert _padded_spike_count(0, 1000, fine=True) == 16
    assert _padded_spike_count(16, 1000, fine=True) == 16


def test_electrodes_with_nearby_spike_counts_share_compiled_kernels():
    rng = np.random.default_rng(22)
    env = Environment(place_bin_size=5.0, position_range=((0.0, 30.0), (0.0, 30.0)))
    position_time = np.arange(0, 30.0, 1 / 30)
    position = np.column_stack(
        (15 + 10 * np.sin(position_time), 15 + 10 * np.cos(position_time))
    )
    env = env.fit_place_grid(position, infer_track_interior=False)
    encoding_times = [np.sort(rng.uniform(0, 30, n)) for n in (200, 210)]
    encoding_marks = [rng.normal(0, 30, (len(t), 4)) for t in encoding_times]
    encoding = fit_clusterless_gmm_encoding_model(
        position_time=position_time,
        position=position,
        spike_times=encoding_times,
        spike_waveform_features=encoding_marks,
        environment=env,
        disable_progress_bar=True,
        **GMM_COMPONENTS,
    )
    # 41 and 47 decoding spikes share one padded size in both kernels.
    assert _padded_spike_count(41, 1000, fine=True) == _padded_spike_count(
        47, 1000, fine=True
    )
    assert _padded_spike_count(41, 1000) == _padded_spike_count(47, 1000)
    decode_times = [np.sort(rng.uniform(0, 2, n)) for n in (41, 47)]
    recording = (
        env,
        position_time,
        position,
        encoding,
        fit_reference(position_time, encoding_times, encoding_marks),
        decode_times,
        [rng.normal(0, 30, (len(t), 4)) for t in decode_times],
        np.arange(0, 2.0 + 1e-9, 0.002),
    )
    kernels = (
        clusterless_gmm._add_electrode_gmm_intensities,
        clusterless_gmm._add_electrode_gmm_spike_terms,
        clusterless_gmm._add_gmm_expected_counts,
    )
    for kernel in kernels:
        kernel.clear_cache()
    for is_local in (False, True):
        predict(recording, is_local, None)
    assert [kernel._cache_size() for kernel in kernels] == [1, 1, 1]
