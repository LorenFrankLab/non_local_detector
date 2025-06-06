"""
non_local_detector.io.preprocessing
-----------------------------------

Utilities that convert *raw* `RecordingBundle` objects into decoder-aligned
`DecoderBatch` objects.

The single public helper below:

    to_decoder_batch(rec, bin_width_s, signals_to_use, ...)

handles

* spike-count binning (`"hist"` or `"center"`),
* up-sampling / down-sampling of continuous, categorical-integer, and
  boolean signals,
* gap-filling policy per dtype,
* automatic bin-edge generation.

Typical usage
-------------
>>> from non_local_detector.io.preprocessing import to_decoder_batch
>>> batch = to_decoder_batch(
...     rec,
...     bin_width_s      = 0.004,                     # 4 ms bins
...     signals_to_use   = ("counts", "lfp", "pos_xy"),
...     float_fill       = "ffill",
... )
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from non_local_detector.bundle import DecoderBatch, RecordingBundle, TimeSeries


# --------------------------------------------------------------------------- #
#  Helper functions                                                           #
# --------------------------------------------------------------------------- #
def _mode1d(arr: np.ndarray):
    """Return the modal value of a 1-D array (break ties by first seen)."""
    vals, counts = np.unique(arr, return_counts=True)
    return vals[np.argmax(counts)]


def _compute_bin_edges(start_s: float, stop_s: float, bin_w: float) -> np.ndarray:
    n_bins = int(np.ceil((stop_s - start_s) / bin_w))
    return start_s + np.arange(n_bins + 1) * bin_w


def _fill_float_linear(x: np.ndarray, mask_missing: np.ndarray):
    """Linear interpolate NaNs along axis-0 in-place."""
    if not mask_missing.any():
        return x
    t = np.arange(len(x))
    good = ~mask_missing
    x[mask_missing] = np.interp(t[mask_missing], t[good], x[good])
    return x


# --------------------------------------------------------------------------- #
#  Public conversion function                                                 #
# --------------------------------------------------------------------------- #
def to_decoder_batch(
    rec: RecordingBundle,
    bin_width_s: float,
    *,
    signals_to_use: Sequence[str] = ("counts",),
    # spike-count rule
    count_method: str = "hist",  # "hist" | "center"
    # aggregation when *multiple* native samples fall in a decoder bin
    float_downsample: str = "mean",  # "mean" | "median"
    # gap-fill policies when *no* sample falls in a bin
    float_fill: str = "ffill",  # "ffill" | "linear"
    int_fill: str = "ffill",  # "ffill" | "pad_zero"
    bool_fill: str = "or",  # "or" | "ffill"
    start_s: float | None = None,
    stop_s: float | None = None,
) -> DecoderBatch:
    """
    Convert a native-rate recording to a decoder-aligned batch.

    Parameters
    ----------
    signals_to_use
        Iterable of keys to keep from ``rec.signals``.  The pseudo-key
        ``"counts"`` triggers spike binning.
    count_method
        *hist*   - spikes on a bin edge go to the **lower** bin (np.histogram).
        *center* - subtract ½ bin to align to bin centres (common in KDE).
    float_downsample
        Aggregation for float signals when several native samples land in
        one decoder bin.
    float_fill / int_fill / bool_fill
        What to do when a decoder bin receives **no** native samples.
        float_fill : "ffill" | "linear"
        int_fill   : "ffill" | "pad_zero"
        bool_fill  : "or"   | "ffill"
    """
    if count_method not in {"hist", "center"}:
        raise ValueError("count_method must be 'hist' or 'center'")

    # ------------------------------------------------------------------ #
    #  1. Determine global start/stop                                    #
    # ------------------------------------------------------------------ #
    candidates = []
    if rec.spike_times_s:
        candidates += [st[0] for st in rec.spike_times_s if st.size]
        candidates += [st[-1] for st in rec.spike_times_s if st.size]
    for ts in rec.signals.values():
        if ts.data.size:
            candidates.append(ts.start_s)
            candidates.append(ts.start_s + len(ts.data) / ts.sampling_rate_hz)

    if not candidates:
        raise ValueError("RecordingBundle appears empty.")

    start_s = min(candidates) if start_s is None else start_s
    stop_s = max(candidates) if stop_s is None else stop_s
    edges_s = _compute_bin_edges(start_s, stop_s, bin_width_s)
    n_bins = len(edges_s) - 1

    # ------------------------------------------------------------------ #
    #  2. Build signals dict                                             #
    # ------------------------------------------------------------------ #
    signals: dict[str, np.ndarray] = {}

    # 2a) Spike counts
    if "counts" in signals_to_use:
        if rec.spike_times_s is None:
            raise ValueError("'counts' requested but spike_times_s is None")

        binned = []
        shift = -bin_width_s / 2 if count_method == "center" else 0.0
        for times in rec.spike_times_s:
            counts = np.histogram(times + shift, bins=edges_s)[0]
            binned.append(counts)
        signals["counts"] = np.stack(binned, axis=1)  # (n_bins, n_cells)

    # 2b) Continuous / categorical signals
    for key in signals_to_use:
        if key == "counts":
            continue
        if key not in rec.signals:
            raise KeyError(f"{key!r} not found in recording.signals")

        ts: TimeSeries = rec.signals[key]
        sample_times = ts.start_s + np.arange(len(ts.data)) / ts.sampling_rate_hz
        bin_index = np.floor((sample_times - start_s) / bin_width_s).astype(int)

        # Prepare output array
        out = np.zeros((n_bins, *ts.data.shape[1:]), dtype=ts.data.dtype)

        # Aggregate where native samples exist
        for b in range(n_bins):
            sel = ts.data[bin_index == b]
            if sel.size == 0:
                continue
            if np.issubdtype(ts.data.dtype, np.bool_):
                out[b] = sel.any()  # logical OR
            elif ts.data.dtype.kind in "ui":
                out[b] = _mode1d(sel)
            else:
                if float_downsample == "mean":
                    out[b] = sel.mean(axis=0)
                else:  # median
                    out[b] = np.median(sel, axis=0)

        # Gap-fill
        missing = (
            np.all(out == 0, axis=tuple(range(1, out.ndim)))
            if out.ndim > 1
            else (out == 0)
        )
        if np.issubdtype(out.dtype, np.bool_):
            if bool_fill == "ffill":
                for b in range(1, n_bins):
                    if missing[b]:
                        out[b] = out[b - 1]
            # 'or' policy already leaves gaps False
        elif out.dtype.kind in "ui":
            if int_fill == "ffill":
                for b in range(1, n_bins):
                    if missing[b]:
                        out[b] = out[b - 1]
            # pad_zero already satisfied
        else:  # float
            if float_fill == "ffill":
                for b in range(1, n_bins):
                    if missing[b]:
                        out[b] = out[b - 1]
            else:  # linear
                if out.ndim == 1:
                    _fill_float_linear(out, missing)
                else:
                    for ch in range(out.shape[1]):
                        _fill_float_linear(out[:, ch], missing)

        signals[key] = out

    # ------------------------------------------------------------------ #
    #  3. Assemble DecoderBatch                                          #
    # ------------------------------------------------------------------ #
    return DecoderBatch(
        signals=signals,
        bin_edges_s=edges_s,
        spike_times_s=rec.spike_times_s,
        spike_waveforms=rec.spike_waveforms,
    )
