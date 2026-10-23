"""Decoding a stream of fixed-length chunks reuses compiled kernels.

Real-time decoding calls the likelihood once per short chunk, and the number
of spikes on each electrode changes from chunk to chunk. After one warm-up
chunk with a spike on every electrode, later chunks of the same length must
not compile anything: a compile costs far more than a whole chunk's decode.
"""

import numpy as np
import pytest

from non_local_detector import NonLocalClusterlessDetector
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data
from non_local_detector.tests.conftest import count_compiles

BIN_SECONDS = 0.005
N_CHUNK_BINS = 10


@pytest.fixture(scope="module")
def run():
    sim = make_simulated_run_data(n_runs=4, seed=0)
    cut = sim.position_time[int(0.7 * len(sim.position_time))]
    return sim, cut


def _fit(run, algorithm):
    sim, cut = run
    is_train = sim.position_time < cut
    return NonLocalClusterlessDetector(
        clusterless_algorithm=algorithm,
        clusterless_algorithm_params={"disable_progress_bar": True},
    ).fit(
        sim.position_time[is_train],
        sim.position[is_train],
        [times[times < cut] for times in sim.spike_times],
        [
            features[times < cut]
            for times, features in zip(
                sim.spike_times, sim.spike_waveform_features, strict=True
            )
        ],
    )


def _chunk_log_likelihood(detector, sim, start, ensure_spike_on_every_electrode):
    edges = start + BIN_SECONDS * np.arange(N_CHUNK_BINS + 1)
    near = (sim.position_time >= edges[0] - 0.05) & (
        sim.position_time <= edges[-1] + 0.05
    )
    spike_times, features = [], []
    for times, marks in zip(sim.spike_times, sim.spike_waveform_features, strict=True):
        in_chunk = (times >= edges[0]) & (times < edges[-1])
        chunk_times, chunk_marks = times[in_chunk], marks[in_chunk]
        if ensure_spike_on_every_electrode and chunk_times.size == 0:
            chunk_times, chunk_marks = np.array([edges[0]]), marks[:1]
        spike_times.append(chunk_times)
        features.append(chunk_marks)
    return np.asarray(
        detector.compute_log_likelihood(
            sim.position_time[near],
            sim.position[near],
            spike_times,
            features,
            time_edges=edges,
        )
    )


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", ["clusterless_kde", "clusterless_mark_glm"])
def test_fixed_length_chunks_compile_nothing_after_warm_up(run, algorithm):
    sim, cut = run
    detector = _fit(run, algorithm)
    rng = np.random.default_rng(0)
    starts = rng.uniform(cut, sim.position_time[-1] - 1.0, 30)
    _chunk_log_likelihood(
        detector, sim, starts[0], ensure_spike_on_every_electrode=True
    )

    with count_compiles() as compiled:
        spike_counts = set()
        for start in starts[1:]:
            log_likelihood = _chunk_log_likelihood(
                detector, sim, start, ensure_spike_on_every_electrode=False
            )
            assert np.all(np.isfinite(log_likelihood))
            spike_counts.add(
                sum(
                    int(((times >= start) & (times < start + 0.05)).sum())
                    for times in sim.spike_times
                )
            )
    assert len(spike_counts) > 3, "the chunks must vary in spike count"
    assert compiled == []
