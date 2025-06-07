"""
non_local_detector/io/preprocessing.py
--------------------------------------

Convert a native-rate RecordingBundle into a DecoderBatch aligned to a
uniform decoder grid, using scikit-learn's encoders for any string/object
arrays.  All other behavior (binning, gap-fill, NaN policy) remains the same.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from non_local_detector.bundle import DecoderBatch, RecordingBundle, TimeSeries

__all__ = ["to_decoder_batch", "df_to_recording_bundle"]


def _mode(arr: np.ndarray) -> int | float | bool:
    """
    Return the majority element of a 1-D array.
    In case of tie, picks the element that appears first.
    """
    from collections import Counter

    # Find the unique elements and their first appearance index
    unique_elements, first_indices = np.unique(arr, return_index=True)
    # Count the occurrences of each unique element
    counts = Counter(arr)

    # Find the maximum count
    max_count = 0
    for element in unique_elements:
        if counts[element] > max_count:
            max_count = counts[element]

    # Find all elements with the maximum count
    candidates = [
        element for element in unique_elements if counts[element] == max_count
    ]

    if len(candidates) == 1:
        return candidates[0]

    # If there's a tie, find the candidate that appeared first in the original array
    first_seen_index = np.inf
    result = None
    for candidate in candidates:
        # get the index of the first occurrence of the candidate
        current_index = np.where(arr == candidate)[0][0]
        if current_index < first_seen_index:
            first_seen_index = current_index
            result = candidate

    return result


def _compute_edges(start: float, stop: float, width: float) -> np.ndarray:
    """Uniform bin edges from start to stop with given width."""
    n = int(np.ceil((stop - start) / width))
    return start + np.arange(n + 1) * width


def _fill_linear(vec: np.ndarray, mask: np.ndarray):
    """In-place linear interpolation of NaNs in a 1-D array."""
    if not mask.any():
        return
    t = np.arange(len(vec))
    good = ~mask
    vec[mask] = np.interp(t[mask], t[good], vec[good])


def to_decoder_batch(
    rec: RecordingBundle,
    bin_width_s: float,
    *,
    signals_to_use: Sequence[str],
    count_method: str = "hist",  # "hist" | "center"
    float_downsample: str = "mean",  # "mean" | "median"
    float_fill: str = "ffill",  # "ffill" | "linear"
    int_fill: str = "ffill",  # "ffill" | "pad_zero"
    bool_fill: str = "ffill",  # "ffill"
    start_s: float | None = None,
    stop_s: float | None = None,
    nan_policy: str = "warn",  # "raise" | "warn" | "ignore"
    one_hot_categories: bool = False,
) -> DecoderBatch:
    """
    Bin a RecordingBundle into a DecoderBatch on a uniform grid.

    Samples exactly at stop_s are included in the last bin.
    Missing ints are -1, missing bools are False.
    Categorical encoders saved in batch._categorical_maps.
    """
    # Validate policies
    if count_method not in {"hist", "center"}:
        raise ValueError("count_method must be 'hist' or 'center'")
    if nan_policy not in {"raise", "warn", "ignore"}:
        raise ValueError("nan_policy must be 'raise', 'warn', or 'ignore'")

    # 1) determine overall time window
    times: list[float] = []
    if rec.spike_times_s:
        for st in rec.spike_times_s:
            if st.times_s.size:
                times.extend([st.times_s[0], st.times_s[-1]])
    for ts in rec.signals.values():
        n = ts.data.shape[0]
        if n:
            times.extend([ts.start_s, ts.start_s + n / ts.sampling_rate_hz])

    if not times:
        raise ValueError("RecordingBundle is empty")

    start = min(times) if start_s is None else start_s
    stop = max(times) if stop_s is None else stop_s

    edges_s = _compute_edges(start, stop, bin_width_s)
    n_bins = len(edges_s) - 1

    signals: Dict[str, np.ndarray] = {}
    categorical_labels: Dict[str, list[str]] = {}
    onehot_encoders: Dict[str, OneHotEncoder] = {}

    # 2) spike counts
    if "counts" in signals_to_use:
        if rec.spike_times_s is None:
            raise ValueError("Requested 'counts' but no spike_times_s")
        shift = -bin_width_s / 2 if count_method == "center" else 0.0
        counts = []
        for st in rec.spike_times_s:
            shifted = st.times_s + shift
            oob = (shifted < start) | (shifted > stop)
            if oob.any():
                unit_id = getattr(st, "unit_id", "Unknown")
                warnings.warn(
                    f"Unit {unit_id}: {oob.sum()} spikes outside "
                    f"[{start:.3f}, {stop:.3f}] dropped",
                    RuntimeWarning,
                )
            c, _ = np.histogram(shifted, bins=edges_s)
            counts.append(c)
        signals["counts"] = np.stack(counts, axis=1)

    # 3) other signals
    for key in signals_to_use:
        if key == "counts":
            continue
        if key not in rec.signals:
            raise KeyError(f"Signal '{key}' not in RecordingBundle.signals")
        ts = rec.signals[key]
        raw = ts.data
        tcs = ts.start_s + np.arange(raw.shape[0]) / ts.sampling_rate_hz

        # bin indices, allow exactly stop → last bin
        idx = np.floor((tcs - start) / bin_width_s).astype(int)
        valid = (idx >= 0) & (idx <= n_bins)
        if not valid.any():
            msg = f"All samples of '{key}' outside [{start:.3f}, {stop:.3f}]"
            if nan_policy == "raise":
                raise ValueError(msg)
            warnings.warn(msg, RuntimeWarning)
            continue
        idx = np.where(idx == n_bins, n_bins - 1, idx)

        # group sample indices by bin
        groups: list[list[int]] = [[] for _ in range(n_bins)]
        for i, b in enumerate(idx):
            if 0 <= b < n_bins:
                groups[b].append(i)

        kind = raw.dtype.kind
        # categorical?
        if kind in ("U", "O"):
            flat = raw.reshape(-1, 1)
            if one_hot_categories:
                enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                codes = enc.fit_transform(flat)  # shape (N, n_cat)
                labels = enc.categories_[0].tolist()
                out = np.full((n_bins, len(labels)), np.nan, dtype=float)
                for b, idxs in enumerate(groups):
                    if not idxs:
                        continue
                    sel = codes[idxs]
                    out[b] = sel.any(axis=0)
                signals[key] = out
                categorical_labels[key] = labels
                onehot_encoders[key] = enc
            else:
                enc = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                codes = enc.fit_transform(flat).astype(int).ravel()
                labels = enc.categories_[0].tolist()
                out = np.full((n_bins,), -1, dtype=int)
                for b, idxs in enumerate(groups):
                    if not idxs:
                        continue
                    out[b] = _mode(codes[idxs])
                if int_fill == "ffill":
                    for b in range(1, n_bins):
                        if out[b] == -1:
                            out[b] = out[b - 1]
                elif int_fill == "pad_zero":
                    out[out == -1] = 0

                signals[key] = out
                categorical_labels[key] = labels
            continue

        # float
        if kind == "f":
            out = np.full((n_bins, *raw.shape[1:]), np.nan, dtype=raw.dtype)
            for b, idxs in enumerate(groups):
                if not idxs:
                    continue
                sel = raw[idxs]
                agg = np.nanmean if float_downsample == "mean" else np.nanmedian
                out[b] = agg(sel, axis=0)
            # fill
            miss = (
                np.isnan(out).all(axis=tuple(range(1, out.ndim)))
                if out.ndim > 1
                else np.isnan(out)
            )
            if float_fill == "ffill":
                for b in range(1, n_bins):
                    if miss[b]:
                        out[b] = out[b - 1]
            else:  # linear
                if out.ndim == 1:
                    _fill_linear(out, miss)
                else:
                    for ch in range(out.shape[1]):
                        _fill_linear(out[:, ch], miss)
            signals[key] = out

        # integer
        elif kind in "ui":
            out = np.full((n_bins, *raw.shape[1:]), -1, dtype=int)
            for b, idxs in enumerate(groups):
                if not idxs:
                    continue
                out[b] = _mode(raw[idxs])
            if int_fill == "ffill":
                for b in range(1, n_bins):
                    if out[b] == -1:
                        out[b] = out[b - 1]
            else:  # pad_zero
                out[out == -1] = 0
            signals[key] = out

        # boolean
        elif kind == "b":
            out = np.full((n_bins, *raw.shape[1:]), False, dtype=bool)
            for b, idxs in enumerate(groups):
                if not idxs:
                    continue
                out[b] = raw[idxs].any(axis=0)
            if bool_fill == "ffill":
                for b in range(1, n_bins):
                    if not out[b]:
                        out[b] = out[b - 1]
            signals[key] = out

        else:
            raise TypeError(f"Unsupported dtype '{raw.dtype}' for signal '{key}'")

    # 4) enforce nan_policy on floats
    if nan_policy in ("raise", "warn"):
        for k, v in signals.items():
            if np.issubdtype(v.dtype, np.floating) and np.isnan(v).any():
                msg = f"NaNs remain in '{k}' after fill"
                if nan_policy == "raise":
                    raise ValueError(msg)
                warnings.warn(msg, RuntimeWarning)

    # 5) build batch and attach categorical maps
    batch = DecoderBatch(
        signals=signals,
        bin_edges_s=edges_s,
        spike_times_s=rec.spike_times_s,
        spike_waveforms=rec.spike_waveforms,
    )
    for k, labels in categorical_labels.items():
        if one_hot_categories:
            batch._categorical_maps[k] = onehot_encoders[k]
        else:
            batch._categorical_maps[k] = labels

    return batch


def df_to_recording_bundle(
    df: pd.DataFrame,
    *,
    time_column: str | None = None,
    sampling_rate_hz: float | None = None,
    signals_to_include: list[str] | None = None,
) -> RecordingBundle:
    """
    Convert a uniformly-sampled DataFrame into a RecordingBundle.
    Index may be datetime or numeric seconds.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    if time_column is not None:
        df = df.set_index(time_column)

    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        t0 = idx[0]
        times = (idx - t0).total_seconds().astype(float)
    else:
        times = idx.to_numpy().astype(float)

    if sampling_rate_hz is None:
        deltas = np.diff(times)
        med = np.median(deltas)
        if not np.allclose(deltas, med, rtol=1e-3, atol=1e-4):
            raise ValueError("Index not uniform; supply sampling_rate_hz")
        sampling_rate_hz = 1.0 / med

    signals: dict[str, TimeSeries] = {}
    keys = signals_to_include or list(df.columns)
    for key in keys:
        arr = df[key].to_numpy()
        signals[key] = TimeSeries(
            data=arr, sampling_rate_hz=sampling_rate_hz, start_s=times[0]
        )

    return RecordingBundle(spike_times_s=None, spike_waveforms=None, signals=signals)
