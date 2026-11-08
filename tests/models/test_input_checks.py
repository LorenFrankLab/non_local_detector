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
from non_local_detector.exceptions import DataError, ValidationError
from non_local_detector.likelihoods import _SORTED_SPIKES_ALGORITHMS
from non_local_detector.models._input_validation import _missing_bins, _sample_labels
from tests.conftest import spread_over_sample_bins
from tests.models._helpers import ALGORITHMS, _decoder, record_refit_weights

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
        times = spread_over_sample_bins(rng, time, counts, RATE)
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
    @classmethod
    def unix_clock(cls):
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

    def test_they_skip_the_range_checks_they_would_distort(self, unix_clock):
        time, position, spikes, _ = unix_clock
        detector = _sorted(SortedSpikesDecoder).fit(time, position, spikes)
        coarse = [times.astype(np.float32) for times in spikes]
        edges = time_edges_from_centers(time + 1_000.0)

        _, messages = _quality_warnings(
            lambda: detector.predict(coarse, time_edges=edges), ""
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


class TestEstimateParametersDecodeClock:
    def test_spikes_far_from_the_decode_edges_warn_once(self, recording):
        time, position, spikes, _ = recording
        edges = time_edges_from_centers(time + 1e5)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _, messages = _quality_warnings(
                lambda: _sorted(SortedSpikesDecoder).estimate_parameters(
                    time, position, spikes, time_edges=edges, max_iter=1
                ),
                "decode time_edges",
            )

        assert len(messages) == 1


class TestWarningsFirePerPublicCall:
    @pytest.mark.parametrize("family", FAMILIES)
    def test_em_refits_do_not_repeat_the_encoding_warnings(
        self, family, recording, monkeypatch
    ):
        time, position, spikes, marks = recording
        grid = Environment(place_bin_size=5.0, position_range=[(0.0, 50.0)])
        detector = _detector(family, environments=grid)
        refit_weights = record_refit_weights(monkeypatch, detector)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _, messages = _quality_warnings(
                lambda: detector.estimate_parameters(
                    time,
                    position,
                    *_spike_data(family, spikes, marks),
                    time_edges=time_edges_from_centers(time),
                    max_iter=3,
                    estimate_encoding_model=True,
                    min_encoding_local_mass=0.0,
                    min_encoding_local_ess=0.0,
                ),
                "outside the fitted grid",
            )

        assert len(refit_weights) == 3
        assert len(messages) == 1


def _steady_state(transitions):
    """Stationary distribution of ``transitions``, shape (n_states,)."""
    eigenvalues, eigenvectors = np.linalg.eig(transitions.T)
    steady = np.real(eigenvectors[:, np.argmax(np.real(eigenvalues))])
    return steady / steady.sum()


def _gap_recording():
    """Clean tracking, with its rows from 40 to 80 s dropped.

    A 120 s sweep, ``(time, position, spikes, marks, tracked, intervals)``:
    ``time`` covers the recording, and ``tracked`` selects the samples kept.
    """
    rng = np.random.default_rng(0)
    time = np.arange(0.0, 120.0, 1.0 / RATE)
    position = (50 + 50 * np.sin(2 * np.pi * time / 20))[:, None]
    spikes, marks = [], []
    for center in np.linspace(0, 100, 20):
        rate = 30 * np.exp(-0.5 * ((position[:, 0] - center) / 8) ** 2) + 0.2
        counts = rng.poisson(rate / RATE)
        times = spread_over_sample_bins(rng, time, counts, RATE)
        order = np.argsort(times)
        spikes.append(times[order])
        marks.append(rng.normal(center, 3.0, (times.size, 2)))
    tracked = (time < 40) | (time >= 80)
    intervals = np.array(
        [
            [time[tracked][0], time[time < 40][-1]],
            [time[time >= 80][0], time[tracked][-1]],
        ]
    )
    return time, position, spikes, marks, tracked, intervals


class TestValidPositionIntervals:
    """Decoding knows what fitting knows about tracking.

    Interpolating position across a gap the fit refused puts the animal on an
    invented path, and spikes then disagree with it: the gap decoded as replay.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def gap(cls):
        return _gap_recording()

    @pytest.fixture(scope="class")
    @classmethod
    def detectors(cls, gap):
        """A detector of each family fitted on the tracked samples."""
        time, position, spikes, marks, tracked, intervals = gap
        return {
            family: _detector(family).fit(
                time[tracked],
                position[tracked],
                *_spike_data(family, spikes, marks),
                valid_position_intervals=intervals,
            )
            for family in FAMILIES
        }

    @pytest.fixture(scope="class")
    @classmethod
    def in_gap(cls, gap):
        """Bins whose centers lie more than a bin inside the gap."""
        time = gap[0]
        return (time > 40 + 1 / RATE) & (time < 80 - 1 / RATE)

    def _predict(self, family, detectors, gap, **kwargs):
        time, position, spikes, marks, tracked, _ = gap
        return detectors[family].predict(
            *_spike_data(family, spikes, marks),
            time_edges=time_edges_from_centers(time),
            position=position[tracked],
            position_time=time[tracked],
            **kwargs,
        )

    def test_omitting_them_interpolates_and_warns_once(self, detectors, gap, in_gap):
        results, messages = _quality_warnings(
            lambda: self._predict("sorted", detectors, gap),
            "valid_position_intervals",
        )

        assert len(messages) == 1
        assert "40 s" in messages[0]
        # Warned, not decoded as missing: the false replay this warns about.
        assert not results.is_missing.values[in_gap].any()
        non_local = results.acausal_state_probabilities.sel(
            states="Non-Local Continuous"
        ).values
        assert non_local[in_gap].mean() > 0.5

    @pytest.mark.parametrize("family", FAMILIES)
    def test_passing_them_marks_the_gap_missing(self, family, detectors, gap, in_gap):
        intervals = gap[-1]

        results = _no_quality_warnings(
            lambda: self._predict(
                family, detectors, gap, valid_position_intervals=intervals
            )
        )

        assert results.is_missing.values[in_gap].all()
        assert not results.is_missing.values[gap[4]].any()
        assert np.isfinite(results.acausal_posterior.values).all()

    def test_no_warning_when_every_bin_reaching_the_gap_is_missing(
        self, detectors, gap
    ):
        """The caller already masked them, as a multi-epoch pipeline does."""
        time = gap[0]
        reaches_gap = (time > 40 - 2 / RATE) & (time < 80 + 2 / RATE)

        _no_quality_warnings(
            lambda: self._predict("sorted", detectors, gap, is_missing=reaches_gap)
        )

    def test_no_warning_when_the_decode_window_avoids_the_gap(self, detectors, gap):
        time, position, spikes, _, tracked, _ = gap
        window = time < 30

        _no_quality_warnings(
            lambda: detectors["sorted"].predict(
                [unit[unit < 30 - 1 / RATE] for unit in spikes],
                time_edges=time_edges_from_centers(time[window]),
                position=position[tracked],
                position_time=time[tracked],
            )
        )

    @pytest.mark.parametrize("change", ["jitter", "dropped_frame"])
    def test_no_warning_for_a_jittery_clock_or_a_dropped_frame(
        self, recording, fitted, change
    ):
        """Interpolating across these invents no path."""
        time, position, spikes, _ = recording
        if change == "jitter":
            jitter = np.random.default_rng(0).uniform(-0.1, 0.1, time.size)
            position_time, positions = time + jitter / RATE, position
        else:
            kept = np.arange(time.size) != time.size // 2
            position_time, positions = time[kept], position[kept]

        centers = time[2:-2]  # a jittered clock's end bins count as outside it
        edges = time_edges_from_centers(centers)

        _no_quality_warnings(
            lambda: fitted["sorted"].predict(
                [unit[(unit >= edges[0]) & (unit < edges[-1])] for unit in spikes],
                time_edges=edges,
                position=positions,
                position_time=position_time,
            )
        )

    def test_passing_them_removes_the_false_replay(self, detectors, gap, in_gap):
        """Without evidence in the gap, the states relax to the transitions'
        steady state instead of locking onto Non-Local Continuous."""
        results = self._predict(
            "sorted", detectors, gap, valid_position_intervals=gap[-1]
        )
        probabilities = results.acausal_state_probabilities
        non_local = probabilities.sel(states="Non-Local Continuous").values
        steady = _steady_state(detectors["sorted"].discrete_state_transitions_)
        index = list(probabilities.states.values).index("Non-Local Continuous")

        assert non_local[in_gap].mean() < 2 * steady[index]

    def test_all_decoding_methods_agree(self, detectors, gap, in_gap):
        time, position, spikes, _, tracked, intervals = gap
        detector = detectors["sorted"]
        edges = time_edges_from_centers(time)
        decode = {
            "time_edges": edges,
            "position": position[tracked],
            "position_time": time[tracked],
            "valid_position_intervals": intervals,
        }
        dense = _no_quality_warnings(lambda: detector.predict(spikes, **decode))

        checkpointed = _no_quality_warnings(
            lambda: detector.predict(
                spikes,
                **decode,
                inference_mode="checkpointed",
                output_mode="compact",
                chunk_size=512,
            )
        )
        np.testing.assert_array_equal(checkpointed.is_missing, dense.is_missing)
        np.testing.assert_allclose(
            checkpointed.acausal_state_probabilities,
            dense.acausal_state_probabilities,
            atol=1e-4,
        )

        chunked = _no_quality_warnings(
            lambda: detector.predict(spikes, **decode, n_chunks=3)
        )
        np.testing.assert_array_equal(chunked.is_missing, dense.is_missing)

        sequence = _no_quality_warnings(
            lambda: detector.most_likely_sequence(spikes, **decode)
        )
        assert len(sequence) == len(time)

        rows = slice(2_000, 2_100)  # inside the gap
        requested = detector.compute_log_likelihood(
            time[tracked],
            position[tracked],
            spikes,
            row_slice=rows,
            time_edges=edges,
            valid_position_intervals=intervals,
        )
        assert requested.shape[0] == 100
        assert (np.asarray(requested) == 0.0).all()

    @pytest.mark.parametrize("family", FAMILIES)
    def test_compute_log_likelihood_reads_them(self, family, detectors, gap, in_gap):
        time, position, spikes, marks, tracked, intervals = gap

        log_likelihood = _no_quality_warnings(
            lambda: detectors[family].compute_log_likelihood(
                time[tracked],
                position[tracked],
                *_spike_data(family, spikes, marks),
                time_edges=time_edges_from_centers(time),
                valid_position_intervals=intervals,
            )
        )

        assert (np.asarray(log_likelihood)[in_gap] == 0.0).all()
        assert not (np.asarray(log_likelihood)[~in_gap] == 0.0).all()

    @pytest.mark.parametrize("family", FAMILIES)
    def test_most_likely_sequence_reads_them(self, family, detectors, gap, in_gap):
        time, position, spikes, marks, tracked, intervals = gap
        decode = {
            "time_edges": time_edges_from_centers(time),
            "position": position[tracked],
            "position_time": time[tracked],
        }
        data = _spike_data(family, spikes, marks)

        declared = _no_quality_warnings(
            lambda: detectors[family].most_likely_sequence(
                *data, **decode, valid_position_intervals=intervals
            )
        )
        undeclared, _ = _quality_warnings(
            lambda: detectors[family].most_likely_sequence(*data, **decode), "gaps"
        )

        assert not declared[in_gap].equals(undeclared[in_gap])

    @pytest.mark.parametrize(("samples", "warns"), [(4, False), (5, True)])
    def test_a_gap_starts_at_five_samples(self, detectors, gap, samples, warns):
        time, position, spikes, *_ = gap
        window = time < 30
        kept = window.copy()
        kept[799 : 799 + samples - 1] = False  # five samples round below 5 * median

        _, messages = _quality_warnings(
            lambda: detectors["sorted"].predict(
                spikes,
                time_edges=time_edges_from_centers(time[window]),
                position=position[kept],
                position_time=time[kept],
            ),
            "gaps",
        )

        assert len(messages) == int(warns)

    @pytest.mark.parametrize(
        "intervals",
        [[[0.0, 119.98]], [[0.0, 45.0], [80.0, 119.98]]],
        ids=["over the gap", "into the gap"],
    )
    def test_an_interval_reaching_a_gap_warns(self, detectors, gap, intervals):
        """Session bounds are not tracking: those bins read an invented path."""
        results, messages = _quality_warnings(
            lambda: self._predict(
                "sorted", detectors, gap, valid_position_intervals=intervals
            ),
            "gaps",
        )

        assert len(messages) == 1
        assert not results.is_missing.values[gap[0] < 45].any()

    def test_estimate_parameters_warns_of_an_interval_over_a_gap(self, gap):
        """Without one, its internal fit refuses the gap instead."""
        time, position, spikes, _, tracked, _ = gap

        _, messages = _quality_warnings(
            lambda: _sorted().estimate_parameters(
                time[tracked],
                position[tracked],
                spikes,
                time_edges=time_edges_from_centers(time),
                valid_position_intervals=[[time[0], time[-1]]],
                max_iter=1,
            ),
            "gaps",
        )

        assert len(messages) == 1

    def test_estimate_parameters_leaves_an_undeclared_gap_to_its_fit(self, gap):
        """The fit refuses it, so a warning first would only repeat that."""
        time, position, spikes, _, tracked, _ = gap

        with pytest.raises(DataError, match="valid_position_intervals"):
            _no_quality_warnings(
                lambda: _sorted().estimate_parameters(
                    time[tracked],
                    position[tracked],
                    spikes,
                    time_edges=time_edges_from_centers(time),
                    max_iter=1,
                )
            )

    def test_an_interval_without_samples_warns(self, detectors, gap):
        """Spikes there would be scored on a path between distant samples."""
        intervals = gap[-1]

        _, messages = _quality_warnings(
            lambda: self._predict(
                "sorted",
                detectors,
                gap,
                valid_position_intervals=[intervals[0], [50.0, 60.0], intervals[1]],
            ),
            "gaps",
        )

        assert len(messages) == 1

    def test_an_interval_warns_only_of_bins_that_read_it(self, detectors, gap):
        """Bins the caller marked or does not request read no position."""
        time, position, spikes, _, tracked, intervals = gap
        empty = [intervals[0], [50.0, 60.0], intervals[1]]
        reaches_it = (time > 50 - 2 / RATE) & (time < 60 + 2 / RATE)

        def evaluate(**kwargs):
            return detectors["sorted"].compute_log_likelihood(
                time[tracked],
                position[tracked],
                spikes,
                time_edges=time_edges_from_centers(time),
                valid_position_intervals=empty,
                **kwargs,
            )

        _no_quality_warnings(lambda: evaluate(is_missing=reaches_it))
        _no_quality_warnings(lambda: evaluate(row_slice=slice(0, 100)))
        _, messages = _quality_warnings(
            lambda: evaluate(row_slice=slice(2_600, 2_700)), "gaps"
        )
        assert len(messages) == 1

    def test_an_interval_over_nan_position_is_already_missing(self, detectors, gap):
        """NaN rows mark those bins, so the warning's claim would be false."""
        time, position, spikes, _, tracked, intervals = gap
        with_nan = position.copy()
        with_nan[~tracked] = np.nan

        results = _no_quality_warnings(
            lambda: detectors["sorted"].predict(
                spikes,
                time_edges=time_edges_from_centers(time),
                position=with_nan,
                position_time=time,
                valid_position_intervals=[intervals[0], [50.0, 60.0], intervals[1]],
            )
        )

        assert results.is_missing.values[(time >= 50) & (time < 60)].all()

    def test_an_interval_outside_the_decode_window_may_lack_samples(
        self, detectors, gap
    ):
        """Position passed for the window need not cover the whole session."""
        time, position, spikes, *_ = gap
        window = time < 30

        _no_quality_warnings(
            lambda: detectors["sorted"].predict(
                spikes,
                time_edges=time_edges_from_centers(time[window]),
                position=position[window],
                position_time=time[window],
                valid_position_intervals=[[0.0, 40.0], [50.0, 60.0]],
            )
        )

    def test_compute_log_likelihood_warns_only_of_gaps_it_reads(self, detectors, gap):
        time, position, spikes, _, tracked, _ = gap
        reaches_gap = (time > 40 - 2 / RATE) & (time < 80 + 2 / RATE)

        def evaluate(**kwargs):
            return detectors["sorted"].compute_log_likelihood(
                time[tracked],
                position[tracked],
                spikes,
                time_edges=time_edges_from_centers(time),
                **kwargs,
            )

        _no_quality_warnings(lambda: evaluate(is_missing=reaches_gap))
        _no_quality_warnings(lambda: evaluate(row_slice=slice(0, 100)))
        _no_quality_warnings(
            lambda: evaluate(
                row_slice=slice(1_000, 3_000), is_missing=reaches_gap[1_000:3_000]
            )
        )
        _, messages = _quality_warnings(
            lambda: evaluate(row_slice=slice(2_000, 2_100)), "gaps"
        )
        assert len(messages) == 1

    @pytest.mark.parametrize(
        ("door", "options"),
        [
            ("most_likely_sequence", {}),
            ("compute_log_likelihood", {}),
            ("predict", {"n_chunks": 3}),
        ],
    )
    def test_every_door_warns_about_undeclared_gaps_once(
        self, door, options, detectors, gap
    ):
        time, position, spikes, _, tracked, _ = gap

        _, messages = _quality_warnings(
            lambda: _decode_at(
                detectors["sorted"],
                door,
                (spikes,),
                time[tracked],
                position[tracked],
                time_edges=time_edges_from_centers(time),
                **options,
            ),
            "valid_position_intervals",
        )

        assert len(messages) == 1

    def test_estimate_parameters_masks_its_e_steps(self, gap, in_gap):
        time, position, spikes, _, tracked, intervals = gap
        detector = _sorted()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # EM stops at max_iter
            results = _no_quality_warnings(
                lambda: detector.estimate_parameters(
                    time[tracked],
                    position[tracked],
                    spikes,
                    time_edges=time_edges_from_centers(time),
                    valid_position_intervals=intervals,
                    max_iter=2,
                    estimate_encoding_model=True,
                    min_encoding_local_mass=0.0,
                    min_encoding_local_ess=0.0,
                )
            )

        assert results.is_missing.values[in_gap].all()
        assert not results.is_missing.values[tracked].any()
        # Without evidence the E-step's states relax to the steady state; with
        # the invented path they lock onto Non-Local Continuous.
        probabilities = results.acausal_state_probabilities
        steady = _steady_state(detector.discrete_state_transitions_)
        index = list(probabilities.states.values).index("Non-Local Continuous")
        assert (
            probabilities.sel(states="Non-Local Continuous").values[in_gap].mean()
            < 2 * steady[index]
        )

    def test_an_interval_over_all_the_tracking_changes_nothing(self, recording, fitted):
        """Bins at the ends reach half a sample past it, like the tracked
        period itself: declaring the truth must not cost them."""
        time, position, spikes, _ = recording
        decode = {
            "time_edges": time_edges_from_centers(time),
            "position": position,
            "position_time": time,
        }
        plain = fitted["sorted"].predict(spikes, **decode)

        declared = fitted["sorted"].predict(
            spikes, **decode, valid_position_intervals=[[time[0], time[-1]]]
        )

        assert not declared.is_missing.values.any()
        np.testing.assert_array_equal(
            declared.acausal_posterior, plain.acausal_posterior
        )

    def test_a_model_that_reads_no_position_ignores_them(self, recording):
        """It decodes without position, so tracking cannot make a bin missing."""
        time, position, spikes, _ = recording
        decoder = _sorted(SortedSpikesDecoder).fit(time, position, spikes)
        edges = time_edges_from_centers(time)
        plain = decoder.predict(spikes, time_edges=edges)

        declared = _no_quality_warnings(
            lambda: decoder.predict(
                spikes, time_edges=edges, valid_position_intervals=[[10.0, 20.0]]
            )
        )
        irregular = _no_quality_warnings(
            lambda: decoder.predict(
                spikes,
                time_edges=edges,
                position=position[::3][np.arange(len(time[::3])) != 5],
                position_time=time[::3][np.arange(len(time[::3])) != 5],
            )
        )

        assert not declared.is_missing.values.any()
        assert not irregular.is_missing.values.any()
        np.testing.assert_array_equal(
            declared.acausal_posterior, plain.acausal_posterior
        )

    @pytest.mark.parametrize(
        ("intervals", "match"),
        [
            ([1.0, 2.0], "shape"),
            ([[0.0, 5.0], [4.0, 9.0]], "non-overlapping"),
        ],
    )
    def test_decoding_refuses_what_fitting_refuses(self, intervals, match, recording):
        time, position, spikes, _ = recording
        detector = _sorted(SortedSpikesDecoder)
        with pytest.raises(ValidationError, match=match):
            detector.fit(time, position, spikes, valid_position_intervals=intervals)
        detector.fit(time, position, spikes)
        with pytest.raises(ValidationError, match=match):
            detector.predict(
                spikes,
                time_edges=time_edges_from_centers(time),
                valid_position_intervals=intervals,
            )


@pytest.mark.parametrize(
    ("edges", "expected"),
    [
        # Bins centered on the samples either side of the gap stay observed.
        (np.arange(0.0, 13.0) - 0.5, [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]),
        # A bin straddling the gap's edge by more than half a sample is masked
        # whole: its center is inside the interval.
        (np.array([1.0, 4.0, 5.0, 6.0]), [1, 1, 1]),
        (np.array([0.0, 3.0, 6.0, 9.0, 11.0]), [0, 1, 1, 0]),
    ],
)
def test_bins_outside_intervals_are_masked_whole(edges, expected):
    position_time = np.array([0.0, 1, 2, 3, 8, 9, 10, 11])
    position = position_time[:, None]

    is_missing = _missing_bins(
        edges,
        None,
        position_time,
        position,
        valid_position_intervals=np.array([[0.0, 3.0], [8.0, 11.0]]),
    )

    np.testing.assert_array_equal(is_missing, np.asarray(expected, dtype=bool))


def test_touching_intervals_leave_no_gap():
    position_time = np.arange(11.0)

    is_missing = _missing_bins(
        time_edges_from_centers(position_time),
        None,
        position_time,
        position_time[:, None],
        valid_position_intervals=np.array([[0.0, 5.0], [5.0, 10.0]]),
    )

    assert is_missing is None


@pytest.mark.parametrize("labels", [[0, "A"], (0, "A")], ids=["list", "tuple"])
def test_mixed_labels_keep_their_types(labels):
    """A plain sequence must not turn group 0 into the string "0"."""
    read = _sample_labels(
        labels,
        2,
        "encoding_group_labels",
        kinds="iuUSO",
        contents="integer or string encoding groups",
        compared_with="each observation model's encoding_group",
    )

    assert np.isin(read, 0).tolist() == [True, False]
    assert np.isin(read, "A").tolist() == [False, True]


def test_uniform_labels_keep_their_dtype():
    read = _sample_labels(
        [0, 1],
        2,
        "encoding_group_labels",
        kinds="iuUSO",
        contents="integer or string encoding groups",
        compared_with="each observation model's encoding_group",
    )

    assert read.dtype.kind == "i"


def _decode_at(detector, door, spike_data, position_time, position, **kwargs):
    """Call a decoding method, whose arguments differ in order."""
    if door == "compute_log_likelihood":
        return detector.compute_log_likelihood(
            position_time, position, *spike_data, **kwargs
        )
    return getattr(detector, door)(
        *spike_data, position_time=position_time, position=position, **kwargs
    )


class TestNothingToDecode:
    """A flat posterior from a model that reads position must not pass as a decode."""

    MESSAGE = "Every one of the"
    DOORS = ("predict", "most_likely_sequence", "compute_log_likelihood")

    @pytest.mark.parametrize("door", DOORS)
    def test_intervals_missing_the_decoded_period_warn_once(
        self, door, fitted, recording
    ):
        """Position covers the period, and the intervals are another period's."""
        time, position, spikes, _ = recording

        _, messages = _quality_warnings(
            lambda: _decode_at(
                fitted["sorted"],
                door,
                (spikes,),
                time,
                position,
                time_edges=time_edges_from_centers(time[1_500:]),
                valid_position_intervals=[[0.0, 29.0]],
            ),
            self.MESSAGE,
        )

        assert len(messages) == 1

    @pytest.mark.parametrize("door", DOORS)
    def test_position_without_finite_samples_warns_once(self, door, fitted, recording):
        time, position, spikes, _ = recording

        _, messages = _quality_warnings(
            lambda: _decode_at(
                fitted["sorted"],
                door,
                (spikes,),
                time,
                np.full_like(position, np.nan),
                time_edges=time_edges_from_centers(time),
            ),
            self.MESSAGE,
        )

        assert len(messages) == 1

    def test_a_caller_marking_every_bin_missing_means_it(self, fitted, recording):
        time, position, spikes, _ = recording

        _, messages = _quality_warnings(
            lambda: fitted["sorted"].predict(
                spikes,
                time_edges=time_edges_from_centers(time),
                position=position,
                position_time=time,
                is_missing=np.ones(time.size, dtype=bool),
            ),
            self.MESSAGE,
        )

        assert not messages

    def test_a_caller_mask_finishing_the_job_still_warns(self, fitted, recording):
        """Tracking left half the bins, and the caller marked those."""
        time, position, spikes, _ = recording
        half = time < 30
        with_nan = position.copy()
        with_nan[half] = np.nan

        _, messages = _quality_warnings(
            lambda: fitted["sorted"].compute_log_likelihood(
                time,
                with_nan,
                spikes,
                time_edges=time_edges_from_centers(time),
                is_missing=~half,
            ),
            self.MESSAGE,
        )

        assert len(messages) == 1

    def test_position_of_another_period_warns_only_of_that(self, fitted, recording):
        """The outside-tracking warning has already told the caller."""
        time, position, spikes, _ = recording
        before = time < 20

        _, messages = _quality_warnings(
            lambda: fitted["sorted"].predict(
                spikes,
                time_edges=time_edges_from_centers(time[~before]),
                position=position[before],
                position_time=time[before],
            ),
            "decode bins",
        )

        assert len(messages) == 1
        assert self.MESSAGE not in messages[0]

    def test_estimate_parameters_names_the_intervals(self, recording):
        time, position, spikes, _ = recording

        with pytest.raises(DataError, match="valid_position_intervals"):
            _sorted().estimate_parameters(
                time,
                position,
                spikes,
                time_edges=time_edges_from_centers(time),
                valid_position_intervals=[[100.0, 200.0]],
                max_iter=1,
            )


class TestDecodingFromSpikesAlone:
    """Fitting on one period and decoding another, as during sleep."""

    @pytest.fixture(scope="class")
    @classmethod
    def sleep(cls, recording):
        """A decoder fitted on the first half, and spikes much later.

        ``(decoder, sleep_spikes, sleep_edges, run_intervals)``
        """
        time, position, spikes, _ = recording
        is_run = time < 30
        decoder = _sorted(SortedSpikesDecoder).fit(
            time[is_run],
            position[is_run],
            [times[times < time[is_run][-1]] for times in spikes],
        )
        edges = time_edges_from_centers(time[~is_run] + 1_000.0)
        sleep_spikes = [
            shifted[(shifted >= edges[0]) & (shifted < edges[-1])]
            for shifted in (times + 1_000.0 for times in spikes)
        ]
        return decoder, sleep_spikes, edges, [[time[0], time[is_run][-1]]]

    @pytest.mark.parametrize("intervals_of_the_run", [False, True])
    def test_a_model_without_local_states_needs_no_position(
        self, sleep, intervals_of_the_run
    ):
        decoder, sleep_spikes, edges, run_intervals = sleep
        kwargs = (
            {"valid_position_intervals": run_intervals} if intervals_of_the_run else {}
        )

        results = _no_quality_warnings(
            lambda: decoder.predict(sleep_spikes, time_edges=edges, **kwargs)
        )

        assert not results.is_missing.values.any()
        posterior = results.acausal_posterior
        assert np.isfinite(posterior.values).all()
        assert posterior.std("time").max() > 0.01

    def test_a_local_state_needs_position(self, recording, fitted):
        time, _, spikes, _ = recording

        with pytest.raises(ValidationError, match="position"):
            fitted["sorted"].predict(spikes, time_edges=time_edges_from_centers(time))
