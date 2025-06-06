"""
non_local_detector.io.preprocessing
-----------------------------------

Convert a native-rate RecordingBundle into a DecoderBatch aligned to a
uniform decoder grid.

Public
------
to_decoder_batch(...)
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

from non_local_detector.bundle import DecoderBatch, RecordingBundle, TimeSeries

__all__ = ["to_decoder_batch"]


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def _mode(arr: np.ndarray):
    """Majority vote for 1-D integer or boolean array."""
    vals, counts = np.unique(arr, return_counts=True)
    return vals[np.argmax(counts)]


def _compute_edges(start: float, stop: float, width: float) -> np.ndarray:
    n = int(np.ceil((stop - start) / width))
    return start + np.arange(n + 1) * width


def _fill_linear(vec: np.ndarray, mask: np.ndarray):
    """Linear-interpolate NaNs in 1-D vec in-place."""
    if not mask.any():
        return
    t = np.arange(len(vec))
    vec[mask] = np.interp(t[mask], t[~mask], vec[~mask])


# --------------------------------------------------------------------------- #
#  Main conversion                                                             #
# --------------------------------------------------------------------------- #
def to_decoder_batch(
    rec: RecordingBundle,
    bin_width_s: float,
    *,
    signals_to_use: Sequence[str],
    count_method: str = "hist",  # "hist" | "center"
    float_downsample: str = "mean",  # "mean" | "median"
    float_fill: str = "ffill",  # "ffill" | "linear"
    int_fill: str = "ffill",  # "ffill" | "pad_zero"
    bool_fill: str = "or",  # "or" | "ffill"
    start_s: float | None = None,
    stop_s: float | None = None,
    nan_policy: str = "warn",  # "raise" | "warn" | "ignore"
):
    """
    Parameters
    ----------
    nan_policy
        *"raise"*  - error on any NaNs in output signals.
        *"warn"*   - emit warnings (default).
        *"ignore"* - leave NaNs silently.
    """
    # ------------------------------ sanity ---------------------------------
    if count_method not in {"hist", "center"}:
        raise ValueError("count_method must be 'hist' or 'center'")
    if nan_policy not in {"raise", "warn", "ignore"}:
        raise ValueError("nan_policy must be 'raise'|'warn'|'ignore'")

    # ------------------ determine global time window -----------------------
    candidates: list[float] = []
    if rec.spike_times_s:
        candidates += [st[0] for st in rec.spike_times_s if st.size]
        candidates += [st[-1] for st in rec.spike_times_s if st.size]
    for ts in rec.signals.values():
        if ts.data.size:
            candidates.append(ts.start_s)
            candidates.append(ts.start_s + len(ts.data) / ts.sampling_rate_hz)
    if not candidates:
        raise ValueError("RecordingBundle empty.")

    start_s = min(candidates) if start_s is None else start_s
    stop_s = max(candidates) if stop_s is None else stop_s
    edges_s = _compute_edges(start_s, stop_s, bin_width_s)
    n_bins = len(edges_s) - 1

    # ------------------ build signals dict ---------------------------------
    signals: dict[str, np.ndarray] = {}

    # 1) spike counts
    if "counts" in signals_to_use:
        if rec.spike_times_s is None:
            raise ValueError("'counts' requested but spike_times_s is None")
        shift = -bin_width_s / 2 if count_method == "center" else 0.0
        counts = [
            np.histogram(times + shift, bins=edges_s)[0] for times in rec.spike_times_s
        ]
        signals["counts"] = np.stack(counts, axis=1)  # (n_bins, n_cells)

    # 2) generic numeric / bool signals
    for key in signals_to_use:
        if key == "counts":
            continue
        if key not in rec.signals:
            raise KeyError(f"{key!r} not in recording.signals")

        ts: TimeSeries = rec.signals[key]
        sample_ts = ts.start_s + np.arange(len(ts.data)) / ts.sampling_rate_hz
        bin_idx = np.floor((sample_ts - start_s) / bin_width_s).astype(int)

        out = np.full((n_bins, *ts.data.shape[1:]), np.nan, dtype=ts.data.dtype)

        # aggregate where samples exist
        for b in range(n_bins):
            sel = ts.data[bin_idx == b]
            if sel.size == 0:
                continue
            if np.issubdtype(ts.data.dtype, np.bool_):
                out[b] = sel.any()
            elif ts.data.dtype.kind in "ui":
                out[b] = _mode(sel)
            else:  # float
                agg = np.nanmean if float_downsample == "mean" else np.nanmedian
                out[b] = agg(sel, axis=0)

        # detect unaligned stream
        if np.isnan(out).all():
            msg = f"All samples of signal {key!r} fall outside [{start_s}, {stop_s}]"
            if nan_policy == "raise":
                raise ValueError(msg)
            warnings.warn(msg, RuntimeWarning)
            continue

        # gap-fill NaNs
        missing = (
            np.isnan(out).all(axis=tuple(range(1, out.ndim)))
            if out.ndim > 1
            else np.isnan(out)
        )
        if np.issubdtype(out.dtype, np.bool_):
            if bool_fill == "ffill":
                for b in range(1, n_bins):
                    if missing[b]:
                        out[b] = out[b - 1]
        elif out.dtype.kind in "ui":
            if int_fill == "ffill":
                for b in range(1, n_bins):
                    if missing[b]:
                        out[b] = out[b - 1]
            # pad_zero is default (NaN cast to 0 for int/uint)
            out[np.isnan(out)] = 0
        else:  # float
            if float_fill == "ffill":
                for b in range(1, n_bins):
                    if missing[b]:
                        out[b] = out[b - 1]
            else:  # linear
                if out.ndim == 1:
                    _fill_linear(out, missing)
                else:
                    for ch in range(out.shape[1]):
                        _fill_linear(out[:, ch], missing)

        signals[key] = out

    # ------------------ NaN policy enforcement -----------------------------
    if nan_policy in {"raise", "warn"}:
        for k, v in signals.items():
            if np.isnan(v).any():
                msg = f"NaNs remain in signal {k!r} after gap-fill"
                if nan_policy == "raise":
                    raise ValueError(msg)
                warnings.warn(msg, RuntimeWarning)

    # ------------------ build DecoderBatch ---------------------------------
    return DecoderBatch(
        signals=signals,
        bin_edges_s=edges_s,
        spike_times_s=rec.spike_times_s,
        spike_waveforms=rec.spike_waveforms,
    )
