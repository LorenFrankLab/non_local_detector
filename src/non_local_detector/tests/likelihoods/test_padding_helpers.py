"""Padded spike, sample and occupancy counts and the spike-row helpers that
let the compiled clusterless kernels share compilations."""

import numpy as np
import pytest

from non_local_detector.likelihoods.common import (
    SpikeSelection,
    _padded_occupancy_count,
    _padded_sample_count,
    _padded_spike_count,
    _SpikeTimeOrder,
    spike_row_ids,
)

pytestmark = pytest.mark.unit


def test_padded_spike_counts_are_few_and_cover_each_count():
    sizes = {_padded_spike_count(n, 100) for n in range(0, 5001)}
    for n in range(0, 5001):
        size = _padded_spike_count(n, 100)
        assert size >= max(n, 16)
        assert size <= 100 or size % 100 == 0
        if n > 100:
            # At most a quarter more whole blocks than the request needs.
            assert size // 100 <= 1.25 * -(-n // 100)
    assert sizes == {16, 32, 64, 100} | {
        100 * k
        for k in (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56)
    }


def test_padded_sample_counts_add_at_most_a_quarter():
    for n in range(17, 20000):
        size = _padded_sample_count(n)
        assert n <= size <= 1.25 * n
    assert _padded_sample_count(0) == _padded_sample_count(16) == 16
    assert len({_padded_sample_count(n) for n in range(1025, 2049)}) == 4


def test_padded_occupancy_counts_add_at_most_one_sixty_fourth():
    for n in range(1, 70000, 7):
        size = _padded_occupancy_count(n)
        assert n <= size <= n + n / 64
    assert len({_padded_occupancy_count(n) for n in range(4097, 8193)}) == 64


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
