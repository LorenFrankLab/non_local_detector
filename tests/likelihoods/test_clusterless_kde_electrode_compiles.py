"""Clusterless KDE marked kernels compile a few times per decode, however many electrodes."""

import jax
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.likelihoods.common import (
    SpikeTimeOrder,
    padded_sample_count,
)
from non_local_detector.likelihoods.mark_reference import fit_mark_references
from tests.likelihoods.conftest import count_compilations

pytestmark = pytest.mark.unit

ELECTRODE_COUNTS = (8, 40)


def recording(n_electrodes):
    """A 2-D arena sweep and electrodes of geometrically spread spike counts.

    Every electrode has a different number of encoding and decoding spikes,
    spanning more than seven octaves, and every fifth one has two waveform
    features instead of four. Each fires within 15 cm of its own center.
    """
    rng = np.random.default_rng(n_electrodes)
    time = np.arange(0.0, 60.0, 0.01)
    position = np.column_stack(
        (50 + 45 * np.sin(time / 3.1), 50 + 45 * np.cos(time / 4.7))
    )
    environment = Environment(
        place_bin_size=5.0, position_range=((0.0, 100.0), (0.0, 100.0))
    ).fit_place_grid(position, infer_track_interior=False)
    counts = np.unique(np.geomspace(17, 3_000, n_electrodes).astype(int))
    assert counts.size == n_electrodes
    centers = position[rng.integers(0, len(time), n_electrodes)]
    spike_times, marks = [], []
    for electrode, (center, n) in enumerate(zip(centers, counts, strict=True)):
        in_field = time[np.linalg.norm(position - center, axis=1) < 15.0]
        spike_times.append(np.sort(rng.choice(in_field, n) + rng.uniform(0, 0.01, n)))
        n_features = 2 if electrode % 5 == 4 else 4
        marks.append(rng.normal(0.0, 30.0, (n, n_features)))
    model = fit_clusterless_kde_encoding_model(
        time,
        position,
        spike_times,
        marks,
        environment=environment,
        position_std=4.0,
        waveform_std=24.0,
        disable_progress_bar=True,
    )
    model["mark_reference"] = fit_mark_references(marks, kernel_scale=24.0)
    return time, position, spike_times, marks, model


def marked_kernel_compiles(compiled):
    """The marked-intensity kernels among the compiled programs."""
    return [
        name
        for name in compiled
        if "log_mark_intensities" in name or "local_terms" in name
    ]


def predict(data, is_local, row_slice=None, spike_time_order=None):
    time, position, spike_times, marks, model = data
    return predict_clusterless_kde_log_likelihood(
        time,
        position,
        spike_times,
        marks,
        time_edges=np.arange(0.0, 60.0 + 1e-9, 0.01),
        is_local=is_local,
        row_slice=row_slice,
        _spike_time_order=spike_time_order,
        **model,
    )


def test_recording_electrodes_span_many_padded_sizes():
    _, _, spike_times, marks, _ = recording(ELECTRODE_COUNTS[-1])
    assert len({padded_sample_count(len(times)) for times in spike_times}) >= 20
    assert {m.shape[1] for m in marks} == {2, 4}


@pytest.mark.parametrize("is_local", [False, True], ids=["non-local", "local"])
def test_marked_kernels_compile_independent_of_electrode_count(is_local):
    """A decode compiles its marked kernels the same few times for 8 and 40
    electrodes of different spike, encoding and feature counts."""
    counts = []
    for n_electrodes in ELECTRODE_COUNTS:
        data = recording(n_electrodes)
        jax.clear_caches()
        with count_compilations() as compiled:
            predict(data, is_local)
        counts.append(len(marked_kernel_compiles(compiled)))
    assert counts[0] == counts[1] <= 2, counts


@pytest.mark.parametrize("is_local", [False, True], ids=["non-local", "local"])
def test_row_chunks_and_repeated_decodes_reuse_marked_kernels(is_local):
    """Twelve row chunks of 40 electrodes, whose spike counts per chunk vary,
    compile the marked kernels a few times in all; decoding them again
    compiles nothing. The chunks share one prediction's preparation, as a
    detector's chunks do."""
    data = recording(ELECTRODE_COUNTS[-1])
    chunks = [slice(start, start + 500) for start in range(0, 6_000, 500)]
    jax.clear_caches()
    spike_time_order = SpikeTimeOrder()
    with count_compilations() as compiled:
        for rows in chunks:
            predict(data, is_local, rows, spike_time_order)
    # Measured 4 on CPU (block sizes and padded block counts, both powers
    # of four, vary with each chunk's spikes); main compiled 72 and 88. The
    # bound leaves room for a different data mix.
    assert 1 <= len(marked_kernel_compiles(compiled)) <= 6, compiled
    with count_compilations() as compiled:
        for rows in chunks:
            predict(data, is_local, rows, spike_time_order)
    assert marked_kernel_compiles(compiled) == []


@pytest.mark.parametrize("is_local", [False, True], ids=["non-local", "local"])
def test_a_sparse_chunk_does_not_change_the_next_chunks_block_size(
    is_local, monkeypatch
):
    """A dense chunk decoded after a 3-row chunk, sharing the prediction's
    preparation, takes the block size it takes on its own and gives bitwise
    the same values: the choice depends on the chunk, not on the chunks
    decoded before it."""
    from non_local_detector.likelihoods import clusterless_kde

    chosen = []
    group_sizes = clusterless_kde._group_sizes
    monkeypatch.setattr(
        clusterless_kde,
        "_group_sizes",
        lambda *args: chosen.append(group_sizes(*args)[0]) or group_sizes(*args),
    )
    data = recording(ELECTRODE_COUNTS[-1])
    sparse, dense = slice(0, 3), slice(3, 6_000)
    alone = np.asarray(predict(data, is_local, dense, SpikeTimeOrder()))
    (rows_alone,) = chosen
    chosen.clear()
    spike_time_order = SpikeTimeOrder()
    predict(data, is_local, sparse, spike_time_order)
    after = np.asarray(predict(data, is_local, dense, spike_time_order))
    rows_sparse, rows_after = chosen
    assert rows_sparse < rows_alone
    assert rows_after == rows_alone
    np.testing.assert_array_equal(after, alone)
