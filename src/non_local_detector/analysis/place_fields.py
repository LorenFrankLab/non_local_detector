"""Detector place-field extraction helpers.

These helpers consolidate the schema-aware logic for pulling place fields
out of a fitted detector. Two helpers because there are two distinct
shapes consumers want:

- ``extract_per_cell_place_fields`` — ``(n_cells, n_position_bins)`` for
  per-cell display (slice-panel rows, raster place-field-peak sort,
  static plot raster sort).
- ``extract_state_aligned_place_fields`` —
  ``(n_cells, n_states * n_position_bins)`` for callers aligning against
  the ``predictive_posterior``'s ``state_bins`` axis (rectangular
  detectors only).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from non_local_detector.models.base import _DetectorBase


def extract_per_cell_place_fields(detector: _DetectorBase) -> np.ndarray:
    """Return per-cell place fields with shape ``(n_cells, n_position_bins)``.

    For per-cell display use cases — slice-panel per-cell row plots,
    raster place-field-peak sort, the static plot's raster sort. State
    doesn't matter for display; for v1's single-encoding-group detectors
    all states share the same place fields.

    v1 restricts to a single ``encoding_model_`` entry (single
    environment, single encoding group) so "which state's fields do we
    show" never has to be answered. Multi-environment / multi-group
    detectors are on the v3+ roadmap.

    Parameters
    ----------
    detector : _DetectorBase
        Fitted detector with an ``encoding_model_`` attribute.

    Returns
    -------
    np.ndarray, shape (n_cells, n_position_bins)
        Per-cell place fields from the detector's single encoding-model
        entry.

    Raises
    ------
    ValueError
        If the detector has more than one ``encoding_model_`` entry.
    """
    keys = list(detector.encoding_model_.keys())
    if len(keys) != 1:
        raise ValueError(
            "v1 viewer per-cell display requires exactly one "
            f"encoding-model entry. Found {len(keys)} entries with keys "
            f"{keys}. Multi-environment / multi-group support is on the "
            "v3+ roadmap; consider re-fitting as a single-group model "
            "or wait for v3."
        )
    return detector.encoding_model_[keys[0]]["place_fields"]


def extract_state_aligned_place_fields(detector: _DetectorBase) -> np.ndarray:
    """Return state-bin-aligned place fields for **rectangular** detectors only.

    Returns place fields concatenated along ``axis=1`` so the result
    aligns with the rectangular portion of the
    ``predictive_posterior``'s ``state_bins`` axis: shape
    ``(n_cells, n_states * n_position_bins)``.

    Rectangular detectors only. Defined as: every entry in
    ``detector.bin_sizes_`` equals ``n_position_bins`` (no singleton
    ``Local`` / ``No-Spike`` states). This covers
    ``SortedSpikesDecoder`` (single state, ``n_pos`` bins) and
    ``ContFragSortedSpikesClassifier`` (two states, both spatial). It
    does **not** cover ``NonLocalSortedSpikesDetector`` regardless of
    ``local_position_std`` (``No-Spike`` is always a singleton) nor
    ``NoSpikeContFragSortedSpikesClassifier``.

    Iterates ``detector.observation_models`` and looks up each state's
    ``encoding_model_`` entry by its
    ``(environment_name, encoding_group)`` key.

    Parameters
    ----------
    detector : _DetectorBase
        Fitted detector. Must be rectangular: every entry in
        ``detector.bin_sizes_`` must equal the same ``n_position_bins
        > 1``.

    Returns
    -------
    np.ndarray, shape (n_cells, n_states * n_position_bins)

    Raises
    ------
    ValueError
        If the detector is non-rectangular (any
        ``bin_sizes_[i] != n_position_bins``, or any singleton state).
    """
    bin_sizes = np.asarray(detector.bin_sizes_)
    if (
        bin_sizes.size == 0
        or not np.all(bin_sizes == bin_sizes[0])
        or bin_sizes[0] == 1
    ):
        raise ValueError(
            "extract_state_aligned_place_fields requires a rectangular "
            "detector (every state has the same n_position_bins worth of "
            "bins, with n_position_bins > 1). Got "
            f"bin_sizes_={bin_sizes.tolist()}. This typically means the "
            "detector has singleton states like `Local` or `No-Spike` "
            "(e.g. NonLocalSortedSpikesDetector or "
            "NoSpikeContFragSortedSpikesClassifier). For per-cell display "
            "use extract_per_cell_place_fields. For state-aligned place "
            "fields on non-rectangular detectors, build a custom layout "
            "from detector.state_ind_ + bin_sizes_."
        )
    return np.concatenate(
        [
            detector.encoding_model_[(obs.environment_name, obs.encoding_group)][
                "place_fields"
            ]
            for obs in detector.observation_models
        ],
        axis=1,
    )
