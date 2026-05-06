"""Tests for ``analysis.place_fields`` helpers."""

from __future__ import annotations

import numpy as np
import pytest

from non_local_detector.analysis.place_fields import (
    extract_per_cell_place_fields,
    extract_state_aligned_place_fields,
)
from non_local_detector.tests._simulated_detectors import N_NEURONS, FittedDetector


def _n_pos(detector) -> int:
    """Return ``n_position_bins`` for a single-environment detector."""
    return int(detector.environments[0].place_bin_centers_.shape[0])


@pytest.mark.unit
class TestExtractPerCellPlaceFields:
    """``extract_per_cell_place_fields`` returns ``(n_cells, n_pos)``."""

    def test_nl_detector(self, nl_fitted: FittedDetector) -> None:
        place_fields = extract_per_cell_place_fields(nl_fitted.detector)
        assert place_fields.shape == (N_NEURONS, _n_pos(nl_fitted.detector))

    def test_cf_detector(self, cf_fitted: FittedDetector) -> None:
        place_fields = extract_per_cell_place_fields(cf_fitted.detector)
        assert place_fields.shape == (N_NEURONS, _n_pos(cf_fitted.detector))

    def test_nsf_detector(self, nsf_fitted: FittedDetector) -> None:
        place_fields = extract_per_cell_place_fields(nsf_fitted.detector)
        assert place_fields.shape == (N_NEURONS, _n_pos(nsf_fitted.detector))

    def test_dec_detector(self, dec_fitted: FittedDetector) -> None:
        place_fields = extract_per_cell_place_fields(dec_fitted.detector)
        assert place_fields.shape == (N_NEURONS, _n_pos(dec_fitted.detector))

    def test_multi_entry_detector_raises_with_keys_named(
        self, nl_fitted: FittedDetector
    ) -> None:
        """A detector with multiple ``encoding_model_`` keys → ValueError."""
        from copy import copy

        # Hand-construct a multi-entry encoding_model_ on a shallow copy
        # so we don't mutate the session-scoped fixture.
        detector = copy(nl_fitted.detector)
        first_key = list(nl_fitted.detector.encoding_model_.keys())[0]
        first_entry = nl_fitted.detector.encoding_model_[first_key]
        # Two distinct keys: original + a synthetic "env_b" entry.
        detector.encoding_model_ = {
            first_key: first_entry,
            ("env_b", 0): first_entry,
        }
        with pytest.raises(ValueError) as exc_info:
            extract_per_cell_place_fields(detector)
        message = str(exc_info.value)
        # Both offending keys must appear in the message.
        assert str(first_key) in message
        assert str(("env_b", 0)) in message


@pytest.mark.unit
class TestExtractStateAlignedPlaceFields:
    """``extract_state_aligned_place_fields`` rectangular-only contract."""

    def test_dec_detector_shape(self, dec_fitted: FittedDetector) -> None:
        """Single-state Decoder → ``(n_cells, n_pos)``."""
        place_fields = extract_state_aligned_place_fields(dec_fitted.detector)
        assert place_fields.shape == (N_NEURONS, _n_pos(dec_fitted.detector))

    def test_cf_detector_shape_and_horizontal_copies(
        self, cf_fitted: FittedDetector
    ) -> None:
        """Two-state ContFrag → ``(n_cells, 2 * n_pos)`` (shared encoding group).

        ContFrag's two ``ObservationModel`` entries reuse one encoding
        group, so the result is two horizontal copies of the per-cell
        output.
        """
        n_pos = _n_pos(cf_fitted.detector)
        per_cell = extract_per_cell_place_fields(cf_fitted.detector)
        place_fields = extract_state_aligned_place_fields(cf_fitted.detector)
        assert place_fields.shape == (N_NEURONS, 2 * n_pos)
        # Both halves are exactly the per-cell output.
        np.testing.assert_array_equal(place_fields[:, :n_pos], per_cell)
        np.testing.assert_array_equal(place_fields[:, n_pos:], per_cell)

    def test_nsf_detector_raises(self, nsf_fitted: FittedDetector) -> None:
        """``NoSpikeContFrag`` (No-Spike singleton) → ValueError."""
        with pytest.raises(ValueError) as exc_info:
            extract_state_aligned_place_fields(nsf_fitted.detector)
        message = str(exc_info.value)
        # Error names the bin_sizes_ and points to the per-cell helper.
        assert "bin_sizes_" in message
        assert "extract_per_cell_place_fields" in message
        # Message contains the actual offending bin_sizes list.
        assert str(np.asarray(nsf_fitted.detector.bin_sizes_).tolist()) in message

    def test_nl_detector_local_std_raises(self, nl_fitted: FittedDetector) -> None:
        """``NonLocal(local_position_std=1.0)`` (No-Spike singleton) → ValueError."""
        with pytest.raises(ValueError) as exc_info:
            extract_state_aligned_place_fields(nl_fitted.detector)
        message = str(exc_info.value)
        assert "bin_sizes_" in message
        assert "extract_per_cell_place_fields" in message
        # bin_sizes_ for NL with local_position_std=1.0 is [n_pos, 1, n_pos, n_pos].
        assert str(np.asarray(nl_fitted.detector.bin_sizes_).tolist()) in message

    @pytest.mark.slow
    def test_nl_detector_singleton_local_raises(
        self, nl_singleton_fitted: FittedDetector
    ) -> None:
        """``NonLocal(local_position_std=None)`` (both Local + No-Spike
        singleton, ``bin_sizes_=[1, 1, n_pos, n_pos]``) → ValueError.

        Slow because the singleton-Local EM is substantially slower
        than the continuous-Gaussian Local variant.
        """
        with pytest.raises(ValueError) as exc_info:
            extract_state_aligned_place_fields(nl_singleton_fitted.detector)
        message = str(exc_info.value)
        assert "bin_sizes_" in message
        assert "extract_per_cell_place_fields" in message
        bin_sizes = np.asarray(nl_singleton_fitted.detector.bin_sizes_)
        # Verify the schema is what we expected (both Local + No-Spike singleton).
        assert int(bin_sizes[0]) == 1
        assert int(bin_sizes[1]) == 1
        assert str(bin_sizes.tolist()) in message
