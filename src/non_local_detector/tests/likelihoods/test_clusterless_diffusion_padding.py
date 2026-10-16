"""Padded, compiled clusterless diffusion kernels share executables across sizes."""

import numpy as np
import pytest

from non_local_detector import time_edges_from_centers
from non_local_detector.environment import Environment
from non_local_detector.likelihoods import clusterless_diffusion
from non_local_detector.likelihoods.clusterless_diffusion import (
    _effective_block,
    fit_clusterless_diffusion_encoding_model,
    predict_clusterless_diffusion_log_likelihood,
)
from non_local_detector.likelihoods.common import (
    _padded_sample_count,
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
    rank = encoding["resolved_rank"]
    # A budget at which the two electrodes' raw block sizes differ (20 and 19
    # decoding spikes) but round down to the same power of two.
    memory_budget = 2 * 20 * 4 * (200 + rank + 4 * n_bins)
    assert [
        memory_budget // (2 * 4 * (n_enc + rank + 4 * n_bins)) for n_enc in (200, 210)
    ] == [20, 19]
    assert [
        _effective_block(memory_budget, n_enc, rank, n_bins, 10_000)
        for n_enc in (200, 210)
    ] == [16, 16]
    assert _padded_sample_count(200) == _padded_sample_count(210)
    assert _padded_spike_count(40, 16) == _padded_spike_count(45, 16)

    kernels = (
        clusterless_diffusion._add_electrode_diffusion_intensities,
        clusterless_diffusion._add_electrode_local_diffusion_terms,
    )
    for kernel in kernels:
        kernel.clear_cache()
    for is_local in (False, True):
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
            for budget in (encoding["memory_budget"], memory_budget)
        )
        # The same 16-spike blocks, so the same arithmetic.
        np.testing.assert_array_equal(small, default)
        assert np.all(np.isfinite(default))
    # One executable per kernel: the default budget's 16-spike blocks (the
    # model's block_size) and the small budget's rounded blocks coincide.
    assert [kernel._cache_size() for kernel in kernels] == [1, 1]
