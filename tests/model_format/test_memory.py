"""Saving and loading stream arrays: no second copy of a large array.

A load decompresses each member straight into its array, and a save writes
each array into its member from views of its memory, so the peak of new
allocations stays near the array's own size on load and far below it on
save.
"""

import tracemalloc

import numpy as np
import pytest

import non_local_detector as nld
from tests.model_format.conftest import algorithm_params

pytestmark = pytest.mark.integration

N_VALUES = 1_000_000  # 8 MB of float64


def _peak(function):
    tracemalloc.start()
    try:
        result = function()
        return result, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_a_large_array_is_saved_and_loaded_without_a_second_copy(
    sorted_recording, tmp_path
):
    time, position, spike_times = sorted_recording
    detector = nld.SortedSpikesDecoder(
        sorted_spikes_algorithm_params=algorithm_params("sorted_spikes_kde")
    ).fit(time, position, spike_times)
    # Random values barely compress, like encoding spike marks.
    detector.log_likelihood_ = np.random.default_rng(0).random(N_VALUES)
    nbytes = detector.log_likelihood_.nbytes
    path = tmp_path / "model.nld"

    _, save_peak = _peak(lambda: detector.save_model(path))
    loaded, load_peak = _peak(lambda: nld.SortedSpikesDecoder.load_model(path))

    np.testing.assert_array_equal(loaded.log_likelihood_, detector.log_likelihood_)
    assert save_peak < 0.5 * nbytes
    assert load_peak < 1.25 * nbytes + 2 * 2**20
