"""
non_local_detector.io.validation_extras
=======================================

*Structural* validation is already handled by

    • bundle.DecoderBatch.__post_init__()
    • core.validation.validate_sources()

This module adds *semantic* or experiment-specific checks that often catch
real-world issues (unit scaling, duplicate IDs, unsorted timestamps, etc.).
Nothing here is required by the core pipeline; it is convenience only.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np

from non_local_detector.bundle import DecoderBatch, RecordingBundle, TimeSeries


# --------------------------------------------------------------------------- #
#  helpers                                                                    #
# --------------------------------------------------------------------------- #
def _warn_or_raise(msg: str, strict: bool):
    if strict:
        raise ValueError(msg)
    warnings.warn(msg, RuntimeWarning)


def _check_times_sorted(times: np.ndarray, name: str, strict: bool):
    if not (np.diff(times) > 0).all():
        _warn_or_raise(f"{name} not strictly increasing", strict)


# --------------------------------------------------------------------------- #
#  1. Recording-level validation                                              #
# --------------------------------------------------------------------------- #
def sanity_check_recording(rec: RecordingBundle, *, strict: bool = False):
    """
    Fast, optional QC for a freshly loaded RecordingBundle.

    Parameters
    ----------
    strict
        If True, raise ValueError; else issue RuntimeWarnings.
    """

    # ---- spike-time integrity ----------------------------------------
    if rec.spike_times_s:
        for idx, times in enumerate(rec.spike_times_s):
            _check_times_sorted(times, f"spike_times_s[{idx}]", strict)

    # ---- waveform consistency ---------------------------------------
    if rec.spike_waveforms is not None:
        if len(rec.spike_times_s or []) != len(rec.spike_waveforms):
            _warn_or_raise(
                "Mismatch len(spike_times_s) vs len(spike_waveforms)", strict
            )

    # ---- signal sanity ----------------------------------------------
    for key, ts in rec.signals.items():
        if not isinstance(ts, TimeSeries):
            _warn_or_raise(f"{key!r} not a TimeSeries instance", strict)

        # mandatory sampling_rate_hz
        if getattr(ts, "sampling_rate_hz", 0) <= 0:
            _warn_or_raise(f"signal {key!r} missing or bad sampling_rate_hz", strict)

        if ts.data.ndim < 1:
            _warn_or_raise(f"signal {key!r} has ndim < 1", strict)

        # optional: unit scaling guard (µV vs V)
        if key.lower().startswith("lfp") and ts.data.dtype.kind == "f":
            if np.nanmax(np.abs(ts.data)) < 1e-3:  # >1 mV in Volts → suspicious
                _warn_or_raise(
                    f"{key!r}: max={ts.data.max():.3g} — looks like Volts not µV",
                    strict,
                )

    # ---- duplicate unit IDs (if user stores them) -------------------
    unit_ids = getattr(rec, "unit_ids", None)
    if unit_ids is not None and len(unit_ids) != len(set(unit_ids)):
        _warn_or_raise("duplicate unit_ids found", strict)


# --------------------------------------------------------------------------- #
#  2. DecoderBatch-level validation                                           #
# --------------------------------------------------------------------------- #
def sanity_check_decoder_batch(batch: DecoderBatch, *, strict: bool = False):
    """
    Post-preprocessing QC — spot NaNs, infs, unexpected dtypes *after*
    alignment but *before* passing to the detector.
    """

    for key, arr in batch.signals.items():
        if np.issubdtype(arr.dtype, np.floating):
            if np.isinf(arr).any():
                _warn_or_raise(f"{key!r} contains ±inf", strict)
            nan_pct = np.isnan(arr).mean() * 100
            if nan_pct > 10:  # arbitrary threshold
                _warn_or_raise(f"{key!r}: {nan_pct:.1f}% NaNs", strict)

        if arr.dtype.kind in "ui" and (arr < 0).any():
            _warn_or_raise(f"{key!r} has negative values but is unsigned", strict)

    # Check bin_edges strictly increasing
    if batch.bin_edges_s is not None:
        _check_times_sorted(batch.bin_edges_s, "bin_edges_s", strict)
