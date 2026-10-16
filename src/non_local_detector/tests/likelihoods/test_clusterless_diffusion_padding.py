"""Padded, compiled clusterless diffusion kernels share executables across sizes."""

import numpy as np
import pytest

from non_local_detector import time_edges_from_centers
from non_local_detector.environment import Environment
from non_local_detector.likelihoods import clusterless_diffusion
from non_local_detector.likelihoods.clusterless_diffusion import (
    _dense_kernel_fits,
    _electrode_layout,
    _source_tile_size,
    fit_clusterless_diffusion_encoding_model,
    predict_clusterless_diffusion_log_likelihood,
)
from non_local_detector.likelihoods.common import (
    _padded_spike_count,
)
from non_local_detector.tests.likelihoods.conftest import training_mark_references

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def nearby_sizes_recording():
    """Two electrodes whose encoding and decoding counts differ but pad alike."""
    rng = np.random.default_rng(12)
    position_time = np.arange(0, 20.0, 1 / 30)
    position = (50 + 45 * np.sin(position_time / 3))[:, None]
    env = Environment(place_bin_size=2.0, position_range=((0.0, 100.0),))
    env = env.fit_place_grid(position, infer_track_interior=False)
    encoding_times = [np.sort(rng.uniform(0, 20, n)) for n in (200, 210)]
    encoding_features = [rng.normal(0, 30, (len(t), 4)) for t in encoding_times]
    encoding = fit_clusterless_diffusion_encoding_model(
        position_time,
        position,
        encoding_times,
        encoding_features,
        env,
        position_std=6.0,
        waveform_std=24.0,
        block_size=16,
        disable_progress_bar=True,
    )
    decode_times = [np.sort(rng.uniform(0, 2, n)) for n in (40, 45)]
    decode_features = [rng.normal(0, 30, (len(t), 4)) for t in decode_times]
    time_edges = time_edges_from_centers(np.arange(0, 2.0 + 1e-9, 0.02))
    mark_reference = training_mark_references(
        position_time, encoding_times, encoding_features, 24.0, position=position
    )
    return (
        position_time,
        position,
        encoding,
        mark_reference,
        decode_times,
        decode_features,
        time_edges,
    )


def test_electrodes_with_nearby_sizes_share_compiled_kernels(nearby_sizes_recording):
    (
        position_time,
        position,
        encoding,
        mark_reference,
        decode_times,
        decode_features,
        time_edges,
    ) = nearby_sizes_recording
    n_bins = encoding["occupancy"].shape[0]
    tile_budget = _source_tile_size(
        encoding["memory_budget"], n_bins, encoding["resolved_rank"]
    )
    dense = _dense_kernel_fits(n_bins, encoding["memory_budget"])
    layouts = [
        _electrode_layout(bins, marks, weights, n_bins, tile_budget, dense)
        for bins, marks, weights in zip(
            encoding["encoding_bin_indices"],
            encoding["encoding_marks"],
            encoding["encoding_weights"],
            strict=True,
        )
    ]
    # Different encoding and source counts, padded to the same shapes.
    assert [layout[0].shape for layout in layouts] == [layouts[0][0].shape] * 2
    assert [layout[2].shape for layout in layouts] == [layouts[0][2].shape] * 2
    assert [layout[4] for layout in layouts] == [layouts[0][4]] * 2
    assert _padded_spike_count(40, 16) == _padded_spike_count(45, 16)

    kernels = (
        clusterless_diffusion._add_electrode_diffusion_intensities,
        clusterless_diffusion._add_electrode_local_diffusion_terms,
    )
    for kernel in kernels:
        kernel.clear_cache()
    for is_local in (False, True):
        predict_clusterless_diffusion_log_likelihood(
            position_time,
            position,
            decode_times,
            decode_features,
            time_edges=time_edges,
            is_local=is_local,
            mark_reference=mark_reference,
            **encoding,
        )
    assert [kernel._cache_size() for kernel in kernels] == [1, 1]


@pytest.mark.parametrize("is_local", [False, True])
def test_a_smaller_memory_budget_only_changes_the_tiling(
    nearby_sizes_recording, is_local
):
    """Streamed kernel tiles and smaller spike blocks give the same likelihood
    as the default budget's dense kernel."""
    (
        position_time,
        position,
        encoding,
        mark_reference,
        decode_times,
        decode_features,
        time_edges,
    ) = nearby_sizes_recording
    n_bins = encoding["occupancy"].shape[0]
    small_budget = 2 * 4 * 4 * (3 * n_bins + encoding["resolved_rank"])
    assert _source_tile_size(small_budget, n_bins, encoding["resolved_rank"]) == 4
    assert _dense_kernel_fits(n_bins, encoding["memory_budget"])
    assert not _dense_kernel_fits(n_bins, small_budget)
    default, small = (
        np.asarray(
            predict_clusterless_diffusion_log_likelihood(
                position_time,
                position,
                decode_times,
                decode_features,
                time_edges=time_edges,
                is_local=is_local,
                mark_reference=mark_reference,
                **{**encoding, "memory_budget": budget},
            )
        )
        for budget in (encoding["memory_budget"], small_budget)
    )
    assert np.all(np.isfinite(default))
    np.testing.assert_allclose(small, default, rtol=1e-5, atol=1e-4)
