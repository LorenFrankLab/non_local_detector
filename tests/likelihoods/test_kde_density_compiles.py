"""KDE density kernels compile a few times per fit or decode, however many models."""

import jax
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.likelihoods.common import padded_sample_count
from non_local_detector.likelihoods.mark_reference import fit_mark_references
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
)
from tests.likelihoods.conftest import count_compilations

pytestmark = pytest.mark.unit

UNIT_COUNTS = (8, 40)


def recording(n_units):
    """A 2-D arena sweep and place cells with geometrically spread spike counts.

    Every unit has a different spike count, and the counts span more than
    seven octaves, so they fall in many padded sizes. Each unit fires within
    15 cm of its own center, so bins across the arena are far from all of
    its spikes and need the log-space evaluation.
    """
    rng = np.random.default_rng(n_units)
    time = np.arange(0.0, 60.0, 0.01)
    position = np.column_stack(
        (50 + 45 * np.sin(time / 3.1), 50 + 45 * np.cos(time / 4.7))
    )
    environment = Environment(
        place_bin_size=5.0, position_range=((0.0, 100.0), (0.0, 100.0))
    ).fit_place_grid(position, infer_track_interior=False)
    counts = np.unique(np.geomspace(17, 3_000, n_units).astype(int))
    assert counts.size == n_units
    centers = position[rng.integers(0, len(time), n_units)]
    spike_times = []
    for center, n in zip(centers, counts, strict=True):
        in_field = time[np.linalg.norm(position - center, axis=1) < 15.0]
        spike_times.append(np.sort(rng.choice(in_field, n) + rng.uniform(0, 0.01, n)))
    return environment, time, position, spike_times


def kde_density_compiles(compiled):
    """The KDE density kernels among the compiled programs."""
    return [name for name in compiled if "kde_log_densit" in name]


def test_recording_spike_counts_span_many_padded_sizes():
    _, _, _, spike_times = recording(UNIT_COUNTS[-1])
    assert len({padded_sample_count(len(times)) for times in spike_times}) >= 20


def test_sorted_kde_fit_compiles_density_kernels_independent_of_unit_count():
    counts = []
    for n_units in UNIT_COUNTS:
        environment, time, position, spike_times = recording(n_units)
        jax.clear_caches()
        with count_compilations() as compiled:
            fit_sorted_spikes_kde_encoding_model(
                time,
                position,
                spike_times,
                environment,
                position_std=4.0,
                disable_progress_bar=True,
            )
        counts.append(len(kde_density_compiles(compiled)))
    assert counts[0] == counts[1] <= 3, counts


def test_clusterless_kde_local_predict_compiles_density_kernels_independent_of_electrode_count():
    counts = []
    for n_electrodes in UNIT_COUNTS:
        environment, time, position, spike_times = recording(n_electrodes)
        rng = np.random.default_rng(n_electrodes)
        marks = [rng.normal(0.0, 30.0, (len(times), 2)) for times in spike_times]
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
        decode = time < 5.0
        jax.clear_caches()
        with count_compilations() as compiled:
            predict_clusterless_kde_log_likelihood(
                time[decode],
                position[decode],
                [times[times < 5.0] for times in spike_times],
                [m[times < 5.0] for m, times in zip(marks, spike_times, strict=True)],
                time_edges=np.arange(0.0, 5.0 + 1e-9, 0.01),
                is_local=True,
                **model,
            )
        counts.append(len(kde_density_compiles(compiled)))
    assert counts[0] == counts[1] <= 4, counts


def test_clusterless_kde_local_chunks_share_tiled_density_kernels():
    """Row chunks with different flagged points and models reuse the tiled
    density kernels: their log-space repair pads the flagged tiles (four
    sizes per octave) and points (powers of two) to few sizes."""
    environment, time, position, spike_times = recording(UNIT_COUNTS[1])
    rng = np.random.default_rng(UNIT_COUNTS[1])
    marks = [rng.normal(0.0, 30.0, (len(times), 2)) for times in spike_times]
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
    edges = np.arange(0.0, 60.0 + 1e-9, 0.01)
    compiles = []
    jax.clear_caches()
    for start in range(0, 6_000, 500):
        with count_compilations() as compiled:
            predict_clusterless_kde_log_likelihood(
                time,
                position,
                spike_times,
                marks,
                time_edges=edges,
                is_local=True,
                row_slice=slice(start, start + 500),
                **model,
            )
        compiles.append([name for name in compiled if "_tiled_" in name])
    # The first chunk compiles the scaled pass; the 11 later chunks compile
    # only a few more repair sizes (8 when each flagged set had its own).
    assert compiles[0]
    assert sum(len(names) for names in compiles[1:]) <= 5, compiles
