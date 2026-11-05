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
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import _SORTED_SPIKES_ALGORITHMS
from tests.models._helpers import ALGORITHMS, _decoder

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


class TestPositionTimeInOtherUnits:
    """position_time in ms against spikes in s: all spikes overlap its start."""

    MESSAGE = "of the position_time span"

    @pytest.mark.parametrize("family", FAMILIES)
    def test_fit_warns_once(self, family, recording):
        time, position, spikes, marks = recording
        detector = _detector(family)

        _, messages = _quality_warnings(
            lambda: detector.fit(
                time * 1000, position, *_spike_data(family, spikes, marks)
            ),
            self.MESSAGE,
        )

        assert len(messages) == 1

    def test_spikes_of_a_small_part_of_the_tracking_are_fine(self, recording):
        """Under 1% is a unit mismatch; 5% is one epoch of a longer session."""
        time, position, spikes, _ = recording
        epoch = [times[times < 0.05 * time[-1]] for times in spikes]
        _no_quality_warnings(lambda: _sorted().fit(time, position, epoch))

    def test_a_single_spike_has_no_span_to_compare(self, recording):
        time, position, _, _ = recording
        _no_quality_warnings(lambda: _sorted().fit(time, position, [np.array([10.0])]))


class TestFitEncodingModelChecksPosition:
    """fit_encoding_model reads position the way fit does."""

    def test_one_dimensional_position_on_a_track_graph_raises(self, recording):
        time, position, spikes, _ = recording
        on_track = np.column_stack((position[:, 0], np.zeros(len(time))))
        detector = _sorted(SortedSpikesDecoder, environments=_track_environment()).fit(
            time, on_track, spikes
        )

        with pytest.raises(ValidationError, match="track_graph"):
            detector.fit_encoding_model(time, position, spikes)

    @pytest.mark.parametrize("family", FAMILIES)
    def test_positions_off_the_grid_warn_once(self, family, recording, fitted):
        time, position, spikes, marks = recording
        detector = fitted[family].copy()

        _, messages = _quality_warnings(
            lambda: detector.fit_encoding_model(
                time, position * 10, *_spike_data(family, spikes, marks)
            ),
            "outside the fitted grid",
        )

        assert len(messages) == 1

    def test_fit_warns_about_the_grid_once(self, recording):
        time, position, spikes, _ = recording
        grid = Environment(place_bin_size=5.0, position_range=[(0.0, 50.0)])

        _, messages = _quality_warnings(
            lambda: _sorted(environments=grid).fit(time, position, spikes),
            "outside the fitted grid",
        )

        assert len(messages) == 1

    @pytest.mark.parametrize("family", FAMILIES)
    def test_position_with_other_columns_than_the_grid_raises(
        self, family, recording, fitted
    ):
        time, position, spikes, marks = recording

        with pytest.raises(ValidationError, match="column"):
            fitted[family].copy().fit_encoding_model(
                time,
                np.column_stack((position, position)),
                *_spike_data(family, spikes, marks),
            )


class TestNoSpikesInsideThePositionSpan:
    """No spike inside the span means rate floors, at every algorithm."""

    MESSAGE = "lies inside the position_time span"

    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_every_algorithm_warns_once(self, algorithm, recording):
        time, position, spikes, marks = recording
        keep = time < 20.0
        late = [times > 30.0 for times in spikes]
        family = "sorted" if algorithm in _SORTED_SPIKES_ALGORITHMS else "clusterless"
        data = _spike_data(
            family,
            [times[mask] for times, mask in zip(spikes, late, strict=True)],
            [features[mask] for features, mask in zip(marks, late, strict=True)],
        )
        decoder = _decoder(algorithm, sampling_frequency=RATE)

        _, messages = _quality_warnings(
            lambda: decoder.fit(time[keep], position[keep], *data), self.MESSAGE
        )

        assert len(messages) == 1


class TestEncodingTimeRangePastTheTracking:
    """A range past the samples extends support: the end samples are held."""

    MESSAGE = "past the position samples"

    @pytest.mark.parametrize("family", FAMILIES)
    def test_fit_warns_once(self, family, recording):
        time, position, spikes, marks = recording
        keep = (time >= 20.0) & (time < 40.0)
        detector = _detector(family)

        _, messages = _quality_warnings(
            lambda: detector.fit(
                time[keep],
                position[keep],
                *_spike_data(family, spikes, marks),
                encoding_time_range=[time[0], time[-1]],
            ),
            self.MESSAGE,
        )

        assert len(messages) == 1

    def test_range_within_one_sample_of_the_tracking_is_fine(self, recording):
        time, position, spikes, _ = recording
        interval = 1.0 / RATE
        _no_quality_warnings(
            lambda: _sorted().fit(
                time,
                position,
                spikes,
                encoding_time_range=[time[0] - interval, time[-1] + interval],
            )
        )

    def test_range_inside_the_tracking_is_fine(self, recording):
        time, position, spikes, _ = recording
        _no_quality_warnings(
            lambda: _sorted().fit(
                time, position, spikes, encoding_time_range=[10.0, 50.0]
            )
        )

    def test_declared_tracking_intervals_do_not_extend_to_the_range(self, recording):
        """Intervals, not the range, then bound the support."""
        time, position, spikes, _ = recording
        keep = (time >= 20.0) & (time < 40.0)
        tracked = [[time[keep][0], time[keep][-1]]]
        _no_quality_warnings(
            lambda: _sorted().fit(
                time[keep],
                position[keep],
                spikes,
                encoding_time_range=[time[0], time[-1]],
                valid_position_intervals=tracked,
            )
        )
