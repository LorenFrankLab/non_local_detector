"""
non_local_detector/io/preprocessing.py
--------------------------------------

Convert a native-rate RecordingBundle into a DecoderBatch aligned to a
uniform decoder grid, using scikit-learn's encoders for any string/object
arrays.  All other behavior (binning, gap-fill, NaN policy) remains the same.
"""

from __future__ import annotations

import warnings
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from non_local_detector.bundle import DecoderBatch, RecordingBundle, TimeSeries

__all__ = ["to_decoder_batch", "df_to_recording_bundle"]


# --------------------------------------------------------------------------- #
#  Helper functions                                                           #
# --------------------------------------------------------------------------- #
def _mode(arr: np.ndarray):
    """Return the majority vote of a 1-D array (break ties by first seen)."""
    vals, counts = np.unique(arr, return_counts=True)
    return vals[np.argmax(counts)]


def _compute_edges(start: float, stop: float, width: float) -> np.ndarray:
    """Build uniform bin edges from start to stop with given width."""
    n = int(np.ceil((stop - start) / width))
    return start + np.arange(n + 1) * width


def _fill_linear(vec: np.ndarray, mask: np.ndarray):
    """Linear-interpolate NaNs in 1-D vec (in-place)."""
    if not mask.any():
        return
    t = np.arange(len(vec))
    good = ~mask
    vec[mask] = np.interp(t[mask], t[good], vec[good])


# --------------------------------------------------------------------------- #
#  Main conversion functions                                                  #
# --------------------------------------------------------------------------- #
def to_decoder_batch(
    rec: "RecordingBundle",
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
    one_hot_categories: bool = False,
):
    """
    Convert a native-rate RecordingBundle → DecoderBatch aligned to uniform bins.

    Parameters
    ----------
    rec : RecordingBundle
        The raw data (spikes, waveforms, signals at native rates).

    bin_width_s : float
        Desired decoder bin-width in seconds.

    signals_to_use : Sequence[str]
        Keys to extract from rec.signals.  "counts" triggers spike binning.

    count_method : str = "hist" | "center"
        How to assign spikes exactly on a bin edge.

    float_downsample, float_fill, int_fill, bool_fill : str
        How to aggregate or fill gaps for numeric/bool streams.

    start_s, stop_s : float | None
        If provided, force the overall time window. Otherwise inferred
        from rec.spike_times_s and rec.signals.

    nan_policy : str = "raise" | "warn" | "ignore"
        How to treat NaNs in the binned arrays.

    one_hot_categories : bool
        If False (default), use OrdinalEncoder → one integer per category.
        If True, use OneHotEncoder → expand to N binary columns.

    Returns
    -------
    batch : DecoderBatch
        Holds aligned `signals: Dict[str, np.ndarray]`, plus `bin_edges_s`,
        and raw `spike_times_s`, `spike_waveforms`.
    """
    # ------------------------------ sanity ---------------------------------
    if count_method not in {"hist", "center"}:
        raise ValueError("count_method must be 'hist' or 'center'")
    if nan_policy not in {"raise", "warn", "ignore"}:
        raise ValueError("nan_policy must be 'raise', 'warn', or 'ignore'")

    # ------------------ determine global time window -----------------------
    candidates: list[float] = []
    if rec.spike_times_s:
        candidates += [st.times_s[0] for st in rec.spike_times_s if st.times_s.size]
        candidates += [st.times_s[-1] for st in rec.spike_times_s if st.times_s.size]
    for ts in rec.signals.values():
        if ts.data.size:
            candidates.append(ts.start_s)
            candidates.append(ts.start_s + len(ts.data) / ts.sampling_rate_hz)
    if not candidates:
        raise ValueError("RecordingBundle appears empty.")

    start_s = min(candidates) if start_s is None else start_s
    stop_s = max(candidates) if stop_s is None else stop_s
    edges_s = _compute_edges(start_s, stop_s, bin_width_s)
    n_bins = len(edges_s) - 1

    # ------------------ build signals dict ---------------------------------
    signals: Dict[str, np.ndarray] = {}
    categorical_maps: Dict[str, list[str]] = {}  # key → label list (for ordinal)
    onehot_maps: Dict[str, OneHotEncoder] = {}  # key → fitted OneHotEncoder

    # 1) Spike counts ("counts")
    if "counts" in signals_to_use:
        if rec.spike_times_s is None:
            raise ValueError("'counts' requested but spike_times_s is None")
        shift = -bin_width_s / 2 if count_method == "center" else 0.0
        counts_list = [
            np.histogram(st.times_s + shift, bins=edges_s)[0]
            for st in rec.spike_times_s
        ]
        signals["counts"] = np.stack(counts_list, axis=1)  # shape (n_bins, n_units)

    # 2) Generic signals: numeric, bool, or str/object
    for key in signals_to_use:
        if key == "counts":
            continue
        if key not in rec.signals:
            raise KeyError(f"{key!r} not found in RecordingBundle.signals")

        ts = rec.signals[key]
        raw = ts.data
        sample_times = ts.start_s + np.arange(len(raw)) / ts.sampling_rate_hz
        bin_idx = np.floor((sample_times - start_s) / bin_width_s).astype(int)

        # Determine if any raw sample actually falls into [0, n_bins-1]
        valid_bins_mask = (bin_idx >= 0) & (bin_idx < n_bins)
        if not valid_bins_mask.any():
            # Fully unaligned → warn/raise and skip
            msg = f"All samples of signal {key!r} fall outside [{start_s}, {stop_s}]"
            if nan_policy == "raise":
                raise ValueError(msg)
            warnings.warn(msg, RuntimeWarning)
            continue

        # 2a) If string/object dtype → fit encoder on raw samples
        if raw.dtype.kind in ("U", "O"):
            flattened = raw.reshape(-1, 1)  # 2D for sklearn
            if one_hot_categories:
                # Modern sklearn: use sparse_output=False
                enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                codes = enc.fit_transform(flattened)  # shape (N_samples, n_categories)
                onehot_maps[key] = enc
                labels = enc.categories_[0].tolist()
                categorical_maps[key] = labels
                data_for_binning = codes
                dtype_kind = "f"  # each column is 0/1 float
            else:
                enc = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                codes = enc.fit_transform(flattened).astype(int).flatten()
                labels = enc.categories_[0].tolist()
                categorical_maps[key] = labels
                data_for_binning = codes
                dtype_kind = "ui"
        else:
            data_for_binning = raw
            dtype_kind = raw.dtype.kind

        # 2b) Initialize output array
        if raw.dtype.kind in ("U", "O") and one_hot_categories:
            n_cat = len(categorical_maps[key])
            out = np.full((n_bins, n_cat), np.nan, dtype=float)
        else:
            if dtype_kind == "f":
                out = np.full((n_bins, *raw.shape[1:]), np.nan, dtype=raw.dtype)
            elif dtype_kind in "ui":
                # Use integer dtype explicitly, not raw.dtype (which might be string)
                out = np.zeros((n_bins, *raw.shape[1:]), dtype=int)
            elif dtype_kind == "b":
                out = np.zeros((n_bins, *raw.shape[1:]), dtype=raw.dtype)
            else:
                raise TypeError(f"Unsupported dtype '{raw.dtype}' for signal '{key}'")

        # 2c) Populate each decoder bin
        if raw.dtype.kind in ("U", "O") and one_hot_categories:
            # Bin each category‐column separately
            for cat_idx in range(out.shape[1]):
                for b in range(n_bins):
                    sel = data_for_binning[bin_idx == b, cat_idx]
                    if sel.size == 0:
                        continue
                    out[b, cat_idx] = bool(sel.any())
        else:
            for b in range(n_bins):
                sel = data_for_binning[bin_idx == b]
                if sel.size == 0:
                    continue
                if dtype_kind == "b":
                    out[b] = sel.any()
                elif dtype_kind in "ui":
                    out[b] = _mode(sel)
                else:  # float
                    agg = np.nanmean if float_downsample == "mean" else np.nanmedian
                    out[b] = agg(sel, axis=0)

        # 2e) Gap-fill missing bins
        if out.dtype.kind == "f":
            missing = (
                np.isnan(out).all(axis=tuple(range(1, out.ndim)))
                if out.ndim > 1
                else np.isnan(out)
            )
            if float_fill == "ffill":
                for b in range(1, n_bins):
                    if missing[b]:
                        out[b] = out[b - 1]
            else:  # "linear"
                if out.ndim == 1:
                    _fill_linear(out, missing)
                else:
                    for ch in range(out.shape[1]):
                        _fill_linear(out[:, ch], missing)
        elif out.dtype.kind in "ui":
            seen_zero = np.isin(0, data_for_binning)
            missing = np.zeros(n_bins, dtype=bool)
            if not seen_zero:
                missing = out == 0
            if int_fill == "ffill":
                for b in range(1, n_bins):
                    if missing[b]:
                        out[b] = out[b - 1]
            else:  # "pad_zero"
                out[missing] = 0
        else:  # bool
            seen_true = np.isin(True, data_for_binning)
            missing = np.zeros(n_bins, dtype=bool)
            if not seen_true:
                missing = ~out
            if bool_fill == "ffill":
                for b in range(1, n_bins):
                    if missing[b]:
                        out[b] = out[b - 1]

        signals[key] = out

    # ------------------ enforce NaN policy on floats ----------------------
    if nan_policy in ("raise", "warn"):
        for k, v in signals.items():
            if np.issubdtype(v.dtype, np.floating) and np.isnan(v).any():
                msg = f"NaNs remain in signal {k!r} after gap-fill"
                if nan_policy == "raise":
                    raise ValueError(msg)
                warnings.warn(msg, RuntimeWarning)

    # ------------------ Build and return DecoderBatch -----------------------
    batch = DecoderBatch(
        signals=signals,
        bin_edges_s=edges_s,
        spike_times_s=rec.spike_times_s,
        spike_waveforms=rec.spike_waveforms,
    )

    # Attach categorical maps for user reference (private)
    if categorical_maps:
        setattr(batch, "_categorical_maps", {})
        for key in categorical_maps:
            if one_hot_categories:
                getattr(batch, "_categorical_maps")[key] = onehot_maps[key]
            else:
                getattr(batch, "_categorical_maps")[key] = categorical_maps[key]

    return batch


def df_to_recording_bundle(
    df: pd.DataFrame,
    *,
    time_column: str | None = None,
    sampling_rate_hz: float | None = None,
    signals_to_include: list[str] | None = None,
) -> "RecordingBundle":
    """
    Convert a pandas DataFrame of uniformly sampled signals
    into a RecordingBundle.signals dict.

    Parameters
    ----------
    df : pd.DataFrame
      Must have either a DatetimeIndex or a numeric index in seconds (float).
      Each column is one signal channel.

    time_column : str | None
      If the DataFrame has a dedicated “time” column rather than an index,
      set this name; it will be moved into the index.

    sampling_rate_hz : float | None
      If provided, overrides any inferred rate.

    signals_to_include : list[str] | None
      Subset of columns to include; if None, take all columns.

    Returns
    -------
    RecordingBundle
    """

    if df.shape[0] == 0:
        raise ValueError("Input DataFrame is empty; cannot convert to RecordingBundle.")

    # 1) Move a column into a numeric index if requested
    if time_column is not None:
        df = df.set_index(time_column)

    # 2) Ensure index is numeric (float seconds) or Datetime
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex):
        # Convert to seconds since first timestamp
        t0 = idx[0]
        times_s = (idx - t0).total_seconds().astype(float)
    else:
        # Assume numeric already in seconds
        times_s = idx.to_numpy().astype(float)

    # 3) Infer sampling_rate if not given
    if sampling_rate_hz is None:
        deltas = np.diff(times_s)
        median_dt = np.nanmedian(deltas)
        if not np.allclose(deltas, median_dt, rtol=1e-3, atol=1e-4):
            raise ValueError(
                "Index is not uniformly sampled; provide sampling_rate_hz explicitly."
            )
        sampling_rate_hz = 1.0 / median_dt

    # 4) Build a TimeSeries for each column
    signals: dict[str, TimeSeries] = {}
    keys = signals_to_include if signals_to_include is not None else list(df.columns)
    for key in keys:
        arr = df[key].to_numpy()
        ts = TimeSeries(data=arr, sampling_rate_hz=sampling_rate_hz, start_s=times_s[0])
        signals[key] = ts

    return RecordingBundle(spike_times_s=None, spike_waveforms=None, signals=signals)
