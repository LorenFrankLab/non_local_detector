"""Tests for frozen discrete transition rows.

Covers:
- ``_normalize_frozen_discrete_transition_rows`` helper (validation +
  normalization of input forms)
- Constructor plumbing in ``_DetectorBase`` and model subclasses
- Every model defaults to freezing its No-Spike rows, found from the
  observation models rather than by position
- End-to-end: running EM with a frozen row leaves that row byte-for-byte
  unchanged while other rows update
"""

import warnings

import numpy as np
import pytest

from non_local_detector import (
    ClusterlessDecoder,
    ContFragClusterlessClassifier,
    ContFragSortedSpikesClassifier,
    DiscreteNonStationaryDiagonal,
    DiscreteStationaryCustom,
    MultiEnvironmentClusterlessClassifier,
    MultiEnvironmentSortedSpikesClassifier,
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
    NoSpikeContFragClusterlessClassifier,
    NoSpikeContFragSortedSpikesClassifier,
    SortedSpikesDecoder,
    time_edges_from_centers,
)
from non_local_detector.models.base import (
    _normalize_frozen_discrete_transition_rows,
)
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

# Four states with the No-Spike state second, as in the non-local detector.
NO_SPIKE_1 = np.array([False, True, False, False])

ALL_MODEL_CLASSES = [
    SortedSpikesDecoder,
    ClusterlessDecoder,
    ContFragSortedSpikesClassifier,
    ContFragClusterlessClassifier,
    MultiEnvironmentSortedSpikesClassifier,
    MultiEnvironmentClusterlessClassifier,
    NoSpikeContFragSortedSpikesClassifier,
    NoSpikeContFragClusterlessClassifier,
    NonLocalSortedSpikesDetector,
    NonLocalClusterlessDetector,
]


@pytest.fixture(scope="module")
def em_recording():
    """First 10,000 samples of five simulated place cells (seed 42).

    Returns ``(speed, position, spike_times, time, is_event)``; spike trains
    are cut to the same window. Long enough for three EM iterations to move
    every unfrozen transition row.
    """
    speed, position, spike_times, time, _, _, is_event, _ = make_simulated_data(
        seed=42, n_neurons=5
    )
    n = 10_000
    return (
        speed[:n],
        position[:n],
        [st[st <= time[n - 1]] for st in spike_times],
        time[:n],
        is_event[:n],
    )


@pytest.mark.unit
class TestNormalizeFrozenRowsHelper:
    """Unit tests for the ``_normalize_frozen_discrete_transition_rows`` helper."""

    def test_none_returns_none(self):
        assert _normalize_frozen_discrete_transition_rows(None, NO_SPIKE_1) is None

    def test_empty_list_returns_none(self):
        assert _normalize_frozen_discrete_transition_rows([], NO_SPIKE_1) is None

    def test_empty_tuple_returns_none(self):
        assert _normalize_frozen_discrete_transition_rows((), NO_SPIKE_1) is None

    def test_single_index(self):
        mask = _normalize_frozen_discrete_transition_rows([1], NO_SPIKE_1)
        np.testing.assert_array_equal(mask, [False, True, False, False])

    def test_multiple_indices(self):
        mask = _normalize_frozen_discrete_transition_rows([0, 2], NO_SPIKE_1)
        np.testing.assert_array_equal(mask, [True, False, True, False])

    def test_tuple_indices(self):
        mask = _normalize_frozen_discrete_transition_rows((1, 3), NO_SPIKE_1)
        np.testing.assert_array_equal(mask, [False, True, False, True])

    def test_bool_mask_passthrough(self):
        input_mask = np.array([False, True, False, False])
        mask = _normalize_frozen_discrete_transition_rows(input_mask, NO_SPIKE_1)
        np.testing.assert_array_equal(mask, input_mask)
        # Must be a copy (mutating the input should not affect the returned mask)
        input_mask[0] = True
        assert mask[0] == np.False_

    def test_bool_mask_wrong_shape_raises(self):
        with pytest.raises(ValueError, match=r"boolean mask must have shape"):
            _normalize_frozen_discrete_transition_rows(
                np.array([True, False]), NO_SPIKE_1
            )

    def test_index_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"indices must be in"):
            _normalize_frozen_discrete_transition_rows([0, 5], NO_SPIKE_1)

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match=r"indices must be in"):
            _normalize_frozen_discrete_transition_rows([-1], NO_SPIKE_1)

    def test_non_integer_dtype_raises(self):
        with pytest.raises(ValueError, match=r"integer indices or"):
            _normalize_frozen_discrete_transition_rows(np.array([1.5]), NO_SPIKE_1)

    def test_no_spike_role_selects_no_spike_rows(self):
        mask = _normalize_frozen_discrete_transition_rows(
            "no_spike", np.array([True, False, False, True])
        )
        np.testing.assert_array_equal(mask, [True, False, False, True])

    def test_no_spike_role_without_no_spike_states_returns_none(self):
        assert (
            _normalize_frozen_discrete_transition_rows("no_spike", np.zeros(3, bool))
            is None
        )

    def test_unknown_role_raises(self):
        with pytest.raises(ValueError, match=r"'no_spike'"):
            _normalize_frozen_discrete_transition_rows("local", NO_SPIKE_1)

    def test_2d_integer_raises(self):
        with pytest.raises(ValueError, match=r"index array must be 1D"):
            _normalize_frozen_discrete_transition_rows(np.array([[0, 1]]), NO_SPIKE_1)


@pytest.mark.unit
class TestDetectorConstructorPlumbing:
    """Constructor-level behavior: defaults, storage, validation."""

    @pytest.mark.parametrize("model_cls", ALL_MODEL_CLASSES)
    def test_default_freezes_no_spike_states(self, model_cls):
        """Every model freezes exactly its No-Spike rows by default."""
        model = model_cls()
        assert model.frozen_discrete_transition_rows == "no_spike"
        is_no_spike = np.array([obs.is_no_spike for obs in model.observation_models])
        mask = model._frozen_discrete_transition_rows_mask_
        if is_no_spike.any():
            np.testing.assert_array_equal(mask, is_no_spike)
        else:
            assert mask is None

    def test_default_follows_reordered_no_spike_state(self):
        """The No-Spike row is frozen wherever it is listed, not at index 1."""
        model = NoSpikeContFragSortedSpikesClassifier()
        order = [1, 2, 0]
        defaults = model.discrete_transition_type.values
        reordered = NoSpikeContFragSortedSpikesClassifier(
            discrete_initial_conditions=model.discrete_initial_conditions[order],
            discrete_transition_type=DiscreteStationaryCustom(
                values=defaults[np.ix_(order, order)]
            ),
            discrete_transition_stickiness=model.discrete_transition_stickiness[order],
            continuous_transition_types=[
                [model.continuous_transition_types[i][j] for j in order] for i in order
            ],
            observation_models=[model.observation_models[i] for i in order],
            state_names=[model.state_names[i] for i in order],
        )
        np.testing.assert_array_equal(
            reordered._frozen_discrete_transition_rows_mask_, [False, False, True]
        )

    def test_non_local_explicit_none_opts_out(self):
        """Passing None to a NonLocal model disables the default freeze."""
        model = NonLocalSortedSpikesDetector(frozen_discrete_transition_rows=None)
        assert model.frozen_discrete_transition_rows is None
        assert model._frozen_discrete_transition_rows_mask_ is None

    def test_non_local_custom_frozen_rows(self):
        """User can override the default freeze list."""
        model = NonLocalSortedSpikesDetector(
            frozen_discrete_transition_rows=[0, 1],
        )
        np.testing.assert_array_equal(
            model._frozen_discrete_transition_rows_mask_,
            [True, True, False, False],
        )

    def test_explicit_none_opts_out_without_no_spike_role(self):
        """None disables freezing on a model with a No-Spike state."""
        model = NoSpikeContFragSortedSpikesClassifier(
            frozen_discrete_transition_rows=None
        )
        assert model._frozen_discrete_transition_rows_mask_ is None

    def test_constructor_rejects_bad_input(self):
        """Invalid frozen_discrete_transition_rows raises at construction."""
        with pytest.raises(ValueError, match=r"indices must be in"):
            NonLocalSortedSpikesDetector(frozen_discrete_transition_rows=[7])


@pytest.mark.unit
class TestFrozenMaskNotSerialized:
    """The derived row mask must stay out of ``vars()`` / ``get_params()``.

    It is computed lazily via a property, so serialization round-trips that
    rebuild a model from its public parameters (e.g. Spyglass's
    ``DecodingParameters``, which stores ``vars(model)`` and reconstructs with
    ``Detector(**params)``) do not pass the derived attribute back into
    ``__init__`` as an unexpected keyword argument.
    """

    @pytest.mark.parametrize(
        "make",
        [
            NonLocalSortedSpikesDetector,
            NonLocalClusterlessDetector,
            ContFragSortedSpikesClassifier,
        ],
    )
    def test_mask_absent_from_vars_and_get_params(self, make):
        model = make()
        assert "_frozen_discrete_transition_rows_mask_" not in vars(model)
        assert "_frozen_discrete_transition_rows_mask_" not in model.get_params(
            deep=False
        )

    def test_vars_roundtrip_reconstructs_via_base_detector(self):
        """A ContFrag model's ``vars()`` must reconstruct through the base
        detector (the Spyglass pattern): the derived mask is not a keyword.
        """
        from non_local_detector.models.base import SortedSpikesDetector

        model = ContFragSortedSpikesClassifier()
        rebuilt = SortedSpikesDetector(**vars(model))
        assert isinstance(rebuilt, SortedSpikesDetector)


@pytest.mark.integration
class TestFrozenRowEndToEnd:
    """Full EM integration: frozen rows must be byte-exact after fitting."""

    @staticmethod
    def _initial_transition_row(detector, row_idx):
        """Build the initial transition matrix from the detector's
        transition type and return the requested row."""
        initial_matrix, _, _ = detector.discrete_transition_type.make_state_transition(
            None
        )
        return initial_matrix[row_idx].copy()

    def test_no_spike_row_unchanged_after_em(self, em_recording):
        """Running EM on a NonLocalSortedSpikesDetector with the default
        frozen No-Spike row must leave that row byte-for-byte unchanged,
        while at least one other row should have been updated."""
        _, position, spike_times, time, is_event = em_recording

        detector = NonLocalSortedSpikesDetector(
            sorted_spikes_algorithm="sorted_spikes_kde",
        )
        initial_no_spike_row = self._initial_transition_row(detector, 1)
        initial_other_row = self._initial_transition_row(detector, 0)

        detector.estimate_parameters(
            position_time=time,
            position=position,
            spike_times=spike_times,
            time_edges=time_edges_from_centers(time),
            is_training=~is_event,
            max_iter=3,
            estimate_encoding_model=False,
        )

        # Frozen row: byte-exact equality
        np.testing.assert_array_equal(
            detector.discrete_state_transitions_[1],
            initial_no_spike_row,
            err_msg="Frozen No-Spike row was modified during EM",
        )
        # Sanity check: another row should have changed (otherwise the
        # restore test is vacuous — EM may have just not moved anything)
        final_other_row = detector.discrete_state_transitions_[0]
        assert not np.array_equal(final_other_row, initial_other_row), (
            "Expected Local (row 0) to update during EM — if it didn't, "
            "the frozen-row test is not meaningfully exercising the restore"
        )

    @pytest.mark.parametrize("transition_representation", ["dense", "structured"])
    def test_covariate_transition_keeps_frozen_row(
        self, em_recording, transition_representation
    ):
        """With covariate-dependent transitions, a frozen row keeps its
        initial coefficients, so it stays constant over time, while the other
        rows are estimated."""
        speed, position, spike_times, time, is_event = em_recording
        detector = NonLocalSortedSpikesDetector(
            discrete_transition_type=DiscreteNonStationaryDiagonal(
                np.array([0.99, 0.98, 0.96, 0.96]), formula="1 + speed"
            ),
        )
        initial_transitions, initial_coefficients, _ = (
            detector.discrete_transition_type.make_state_transition({"speed": speed})
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*Row freezing.*")
            detector.estimate_parameters(
                position_time=time,
                position=position,
                spike_times=spike_times,
                time_edges=time_edges_from_centers(time),
                is_training=~is_event,
                max_iter=3,
                estimate_encoding_model=False,
                discrete_transition_covariate_data={"speed": speed},
                transition_representation=transition_representation,
            )

        np.testing.assert_array_equal(
            detector.discrete_transition_coefficients_[:, 1],
            initial_coefficients[:, 1],
        )
        np.testing.assert_allclose(
            detector.discrete_state_transitions_[:, 1],
            np.broadcast_to(initial_transitions[0, 1], (len(speed), 4)),
            rtol=1e-6,
        )
        assert not np.allclose(
            detector.discrete_transition_coefficients_[:, 0],
            initial_coefficients[:, 0],
        )

    def test_no_spike_cont_frag_freezes_no_spike_row_by_default(self, em_recording):
        """The no-spike/continuous/fragmented classifier keeps its No-Spike
        row (index 0) fixed through EM by default, while other rows move."""
        _, position, spike_times, time, is_event = em_recording
        classifier = NoSpikeContFragSortedSpikesClassifier()
        initial_no_spike_row = self._initial_transition_row(classifier, 0)
        initial_other_row = self._initial_transition_row(classifier, 1)

        classifier.estimate_parameters(
            position_time=time,
            position=position,
            spike_times=spike_times,
            time_edges=time_edges_from_centers(time),
            is_training=~is_event,
            max_iter=3,
            estimate_encoding_model=False,
        )

        np.testing.assert_array_equal(
            classifier.discrete_state_transitions_[0], initial_no_spike_row
        )
        assert not np.array_equal(
            classifier.discrete_state_transitions_[1], initial_other_row
        )

    def test_opt_out_allows_no_spike_row_to_update(self, em_recording):
        """With frozen_discrete_transition_rows=None, every row is free to
        update during EM. This confirms the opt-out actually disables the
        restore (and is not silently equivalent to freezing)."""
        _, position, spike_times, time, is_event = em_recording

        detector = NonLocalSortedSpikesDetector(
            sorted_spikes_algorithm="sorted_spikes_kde",
            frozen_discrete_transition_rows=None,
        )
        initial_no_spike_row = self._initial_transition_row(detector, 1)

        detector.estimate_parameters(
            position_time=time,
            position=position,
            spike_times=spike_times,
            time_edges=time_edges_from_centers(time),
            is_training=~is_event,
            max_iter=3,
            estimate_encoding_model=False,
        )

        final_no_spike_row = detector.discrete_state_transitions_[1]
        assert not np.array_equal(final_no_spike_row, initial_no_spike_row), (
            "Expected No-Spike row to update when opt-out is set; "
            "if this fails, freezing and opt-out cannot be distinguished"
        )


@pytest.mark.integration
def test_clusterless_classifier_frozen_row_unchanged_after_em(clusterless_sim):
    """A two-state clusterless classifier (no default freeze) keeps an
    explicitly frozen row at its initial value through EM, while the same
    row moves when nothing is frozen."""
    fit_args = {
        "position_time": clusterless_sim.position_time,
        "position": clusterless_sim.position,
        "spike_times": clusterless_sim.spike_times,
        "spike_waveform_features": clusterless_sim.spike_waveform_features,
        "time_edges": time_edges_from_centers(clusterless_sim.position_time),
        "max_iter": 3,
        "estimate_encoding_model": False,
    }
    final_row = {}
    for label, frozen_rows in (("frozen", [0]), ("free", None)):
        classifier = ContFragClusterlessClassifier(
            frozen_discrete_transition_rows=frozen_rows
        )
        initial_row = TestFrozenRowEndToEnd._initial_transition_row(classifier, 0)
        classifier.estimate_parameters(**fit_args)
        final_row[label] = classifier.discrete_state_transitions_[0]

    np.testing.assert_array_equal(final_row["frozen"], initial_row)
    assert not np.array_equal(final_row["free"], initial_row), (
        "Row 0 did not move without freezing, so the frozen check is vacuous"
    )
