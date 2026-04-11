"""Tests for frozen discrete transition rows.

Covers:
- ``_normalize_frozen_discrete_transition_rows`` helper (validation +
  normalization of input forms)
- Constructor plumbing in ``_DetectorBase`` and model subclasses
- NonLocal models default to freezing the No-Spike row (index 1)
- End-to-end: running EM with a frozen row leaves that row byte-for-byte
  unchanged while other rows update
"""

import numpy as np
import pytest

from non_local_detector import (
    ClusterlessDecoder,
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
)
from non_local_detector.models.base import (
    _normalize_frozen_discrete_transition_rows,
)
from non_local_detector.models.cont_frag_model import ContFragSortedSpikesClassifier
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data


@pytest.mark.unit
class TestNormalizeFrozenRowsHelper:
    """Unit tests for the ``_normalize_frozen_discrete_transition_rows`` helper."""

    def test_none_returns_none(self):
        assert _normalize_frozen_discrete_transition_rows(None, 4) is None

    def test_empty_list_returns_none(self):
        assert _normalize_frozen_discrete_transition_rows([], 4) is None

    def test_empty_tuple_returns_none(self):
        assert _normalize_frozen_discrete_transition_rows((), 4) is None

    def test_single_index(self):
        mask = _normalize_frozen_discrete_transition_rows([1], 4)
        np.testing.assert_array_equal(mask, [False, True, False, False])

    def test_multiple_indices(self):
        mask = _normalize_frozen_discrete_transition_rows([0, 2], 4)
        np.testing.assert_array_equal(mask, [True, False, True, False])

    def test_tuple_indices(self):
        mask = _normalize_frozen_discrete_transition_rows((1, 3), 4)
        np.testing.assert_array_equal(mask, [False, True, False, True])

    def test_bool_mask_passthrough(self):
        input_mask = np.array([False, True, False, False])
        mask = _normalize_frozen_discrete_transition_rows(input_mask, 4)
        np.testing.assert_array_equal(mask, input_mask)
        # Must be a copy (mutating the input should not affect the returned mask)
        input_mask[0] = True
        assert mask[0] == np.False_

    def test_bool_mask_wrong_shape_raises(self):
        with pytest.raises(ValueError, match=r"boolean mask must have shape"):
            _normalize_frozen_discrete_transition_rows(np.array([True, False]), 4)

    def test_index_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"indices must be in"):
            _normalize_frozen_discrete_transition_rows([0, 5], 4)

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match=r"indices must be in"):
            _normalize_frozen_discrete_transition_rows([-1], 4)

    def test_non_integer_dtype_raises(self):
        with pytest.raises(ValueError, match=r"integer indices or"):
            _normalize_frozen_discrete_transition_rows(np.array([1.5]), 4)

    def test_2d_integer_raises(self):
        with pytest.raises(ValueError, match=r"index array must be 1D"):
            _normalize_frozen_discrete_transition_rows(np.array([[0, 1]]), 4)


@pytest.mark.unit
class TestDetectorConstructorPlumbing:
    """Constructor-level behavior: defaults, storage, validation."""

    def test_non_local_sorted_spikes_default_freezes_no_spike(self):
        """NonLocalSortedSpikesDetector freezes row 1 (No-Spike) by default."""
        model = NonLocalSortedSpikesDetector()
        assert model.frozen_discrete_transition_rows == (1,)
        np.testing.assert_array_equal(
            model._frozen_discrete_transition_rows_mask_,
            [False, True, False, False],
        )

    def test_non_local_clusterless_default_freezes_no_spike(self):
        """NonLocalClusterlessDetector freezes row 1 (No-Spike) by default."""
        model = NonLocalClusterlessDetector()
        assert model.frozen_discrete_transition_rows == (1,)
        np.testing.assert_array_equal(
            model._frozen_discrete_transition_rows_mask_,
            [False, True, False, False],
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

    def test_non_nonlocal_model_no_default_freeze(self):
        """Non-NonLocal models default to None (no frozen rows)."""
        model = ContFragSortedSpikesClassifier()
        assert model.frozen_discrete_transition_rows is None
        assert model._frozen_discrete_transition_rows_mask_ is None

    def test_constructor_rejects_bad_input(self):
        """Invalid frozen_discrete_transition_rows raises at construction."""
        with pytest.raises(ValueError, match=r"indices must be in"):
            NonLocalSortedSpikesDetector(frozen_discrete_transition_rows=[7])


@pytest.mark.integration
@pytest.mark.slow
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

    def test_no_spike_row_unchanged_after_em(self):
        """Running EM on a NonLocalSortedSpikesDetector with the default
        frozen No-Spike row must leave that row byte-for-byte unchanged,
        while at least one other row should have been updated."""
        (
            _speed,
            position,
            spike_times,
            time,
            _event_times,
            _sampling_freq,
            is_event,
            _,
        ) = make_simulated_data(seed=42, n_neurons=5)

        detector = NonLocalSortedSpikesDetector(
            sorted_spikes_algorithm="sorted_spikes_kde",
        )
        initial_no_spike_row = self._initial_transition_row(detector, 1)
        initial_other_row = self._initial_transition_row(detector, 0)

        detector.estimate_parameters(
            position_time=time,
            position=position,
            spike_times=spike_times,
            time=time,
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

    def test_opt_out_allows_no_spike_row_to_update(self):
        """With frozen_discrete_transition_rows=None, every row is free to
        update during EM. This confirms the opt-out actually disables the
        restore (and is not silently equivalent to freezing)."""
        (
            _speed,
            position,
            spike_times,
            time,
            _event_times,
            _sampling_freq,
            is_event,
            _,
        ) = make_simulated_data(seed=42, n_neurons=5)

        detector = NonLocalSortedSpikesDetector(
            sorted_spikes_algorithm="sorted_spikes_kde",
            frozen_discrete_transition_rows=None,
        )
        initial_no_spike_row = self._initial_transition_row(detector, 1)

        detector.estimate_parameters(
            position_time=time,
            position=position,
            spike_times=spike_times,
            time=time,
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
@pytest.mark.slow
class TestFrozenRowsDecoderSmoke:
    """A non-NonLocal model (ClusterlessDecoder, 1 state) accepts the
    parameter and runs to completion with a frozen row."""

    def test_clusterless_decoder_freeze_only_row(self):
        """Freezing the single row of a 1-state decoder should leave it
        exactly at its initial value after EM."""
        from non_local_detector.simulate.clusterless_simulation import (
            make_simulated_run_data,
        )

        sim = make_simulated_run_data(
            n_tetrodes=2,
            place_field_means=np.arange(0, 80, 20),
            n_runs=3,
            seed=42,
        )

        decoder = ClusterlessDecoder(frozen_discrete_transition_rows=[0])
        decoder.initialize_environments(position=sim.position)
        decoder.initialize_state_index()
        decoder.initialize_initial_conditions()
        decoder.initialize_discrete_state_transition()
        initial_row = decoder.discrete_state_transitions_[0].copy()

        decoder.estimate_parameters(
            position_time=sim.position_time,
            position=sim.position,
            spike_times=sim.spike_times,
            spike_waveform_features=sim.spike_waveform_features,
            time=sim.position_time,
            max_iter=3,
            estimate_encoding_model=False,
        )

        np.testing.assert_array_equal(
            decoder.discrete_state_transitions_[0],
            initial_row,
        )
