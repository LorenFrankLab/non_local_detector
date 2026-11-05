"""What the public detector methods check about their data, door by door.

Every public method takes its data through one input step, so the same bad
input gets the same answer at each door that reads it. A warning fires once
per public call, whatever the chunks or EM iterations behind it, and never on
clean data.
"""

import warnings

import networkx as nx
import numpy as np
import pytest

from non_local_detector import (
    DataQualityWarning,
    Environment,
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
    SortedSpikesDecoder,
    time_edges_from_centers,
)

pytestmark = pytest.mark.unit

RATE = 50.0
DURATION = 60.0


def _recording(start=0.0, seed=0):
    """A sweep over a 100 cm track: ``(time, position, spikes, marks)``."""
    rng = np.random.default_rng(seed)
    time = start + np.arange(int(DURATION * RATE)) / RATE
    position = 50 + 50 * np.sin(2 * np.pi * (time - start) / 20)
    spikes, marks = [], []
    for center in np.linspace(5, 95, 6):
        rate = 25 * np.exp(-0.5 * ((position - center) / 8) ** 2) + 0.2
        counts = rng.poisson(rate / RATE)
        times = np.repeat(time, counts) + rng.uniform(
            -0.5 / RATE, 0.5 / RATE, counts.sum()
        )
        spikes.append(np.sort(times))
        marks.append(rng.normal(center, 3.0, (times.size, 2)))
    return time, position[:, None], spikes, marks


def _environment():
    return Environment(place_bin_size=5.0, position_range=[(0.0, 100.0)])


def _track_environment():
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(100.0, 0.0))
    graph.add_edge(0, 1, distance=100.0, edge_id=0)
    return Environment(
        place_bin_size=5.0, track_graph=graph, edge_order=[(0, 1)], edge_spacing=0
    )


def _sorted(cls=NonLocalSortedSpikesDetector, environments=None):
    return cls(
        environments=_environment() if environments is None else environments,
        sampling_frequency=RATE,
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    )


def _clusterless(cls=NonLocalClusterlessDetector, environments=None):
    return cls(
        environments=_environment() if environments is None else environments,
        sampling_frequency=RATE,
        clusterless_algorithm_params={"disable_progress_bar": True},
    )


def _spike_data(family, spikes, marks):
    return (spikes,) if family == "sorted" else (spikes, marks)


def _detector(family, **kwargs):
    return _sorted(**kwargs) if family == "sorted" else _clusterless(**kwargs)


def _quality_warnings(function, match):
    """Run ``function``; return it and its DataQualityWarnings naming ``match``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = function()
    return result, [
        str(warning.message)
        for warning in caught
        if issubclass(warning.category, DataQualityWarning)
        and match in str(warning.message)
    ]


def _no_quality_warnings(function):
    """Run ``function`` with every DataQualityWarning an error."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataQualityWarning)
        return function()


FAMILIES = ["sorted", "clusterless"]


@pytest.fixture(scope="module")
def recording():
    return _recording()


@pytest.fixture(scope="module")
def fitted(recording):
    """A fitted detector of each family, on the clean recording."""
    time, position, spikes, marks = recording
    return {
        family: _detector(family).fit(
            time, position, *_spike_data(family, spikes, marks)
        )
        for family in FAMILIES
    }


@pytest.mark.parametrize("family", FAMILIES)
def test_clean_data_gives_no_data_quality_warning_at_any_door(family, recording):
    time, position, spikes, marks = recording
    spike_data = _spike_data(family, spikes, marks)
    edges = time_edges_from_centers(time)
    detector = _detector(family)

    def every_door():
        detector.fit(time, position, *spike_data)
        detector.fit_encoding_model(time, position, *spike_data)
        detector.predict(
            *spike_data, time_edges=edges, position=position, position_time=time
        )
        detector.most_likely_sequence(
            *spike_data, time_edges=edges, position=position, position_time=time
        )
        detector.compute_log_likelihood(time, position, *spike_data, time_edges=edges)
        detector.estimate_parameters(
            time, position, *spike_data, time_edges=edges, max_iter=1
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # EM stops at max_iter
        warnings.simplefilter("error", DataQualityWarning)
        every_door()


class TestLowPrecisionSpikeTimes:
    """A timestamp type too coarse for the bins merges spikes onto few values."""

    MESSAGE = "resolve"

    @pytest.fixture(scope="class")
    def unix_clock(self):
        return _recording(start=1.7e9)

    def test_decoding_float32_times_on_a_large_clock_warns_once(self, unix_clock):
        time, position, spikes, _ = unix_clock
        detector = _sorted(SortedSpikesDecoder).fit(time, position, spikes)
        edges = time_edges_from_centers(time)
        coarse = [times.astype(np.float32) for times in spikes]

        _, messages = _quality_warnings(
            lambda: detector.predict(coarse, time_edges=edges), self.MESSAGE
        )

        assert len(messages) == 1
        assert "float32" in messages[0]

    def test_float32_times_collapse_as_the_warning_says(self, unix_clock):
        """The reason for the check: nearly every distinct time is lost."""
        spikes = unix_clock[2]
        coarse = np.concatenate([times.astype(np.float32) for times in spikes])
        assert len(np.unique(np.concatenate(spikes))) > 1_000
        assert len(np.unique(coarse)) <= 2

    def test_fitting_float32_times_warns_once(self, unix_clock):
        time, position, spikes, _ = unix_clock
        coarse = [times.astype(np.float32) for times in spikes]

        _, messages = _quality_warnings(
            lambda: _sorted(SortedSpikesDecoder).fit(time, position, coarse),
            self.MESSAGE,
        )

        assert len(messages) == 1

    def test_estimate_parameters_warns_once(self, unix_clock):
        time, position, spikes, _ = unix_clock
        coarse = [times.astype(np.float32) for times in spikes]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _, messages = _quality_warnings(
                lambda: _sorted(SortedSpikesDecoder).estimate_parameters(
                    time,
                    position,
                    coarse,
                    time_edges=time_edges_from_centers(time),
                    max_iter=1,
                ),
                self.MESSAGE,
            )

        assert len(messages) == 1

    def test_float32_times_that_resolve_the_bins_are_fine(self, recording, fitted):
        """Near 60 s float32 resolves 4 microseconds, far under a 20 ms bin."""
        time, position, spikes, _ = recording
        _no_quality_warnings(
            lambda: fitted["sorted"].predict(
                [times.astype(np.float32) for times in spikes],
                time_edges=time_edges_from_centers(time),
                position=position,
                position_time=time,
            )
        )

    def test_float64_on_a_large_clock_is_fine(self, unix_clock):
        time, position, spikes, _ = unix_clock
        _no_quality_warnings(
            lambda: _sorted(SortedSpikesDecoder).fit(time, position, spikes)
        )
