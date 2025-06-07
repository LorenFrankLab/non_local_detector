# non_local_detector/io/preprocessing.py
# --------------------------------------

"""
Convert a native-rate RecordingBundle into a DecoderBatch aligned to a
uniform decoder grid, using scikit-learn's encoders for any string/object
arrays. All other behavior (binning, gap-fill, NaN policy) remains the same.
"""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from non_local_detector.bundle import DecoderBatch, RecordingBundle, TimeSeries

__all__ = ["to_decoder_batch", "df_to_recording_bundle"]


def _simple_mode(series: pd.Series) -> Any:
    """
    Return the most frequent element in a series.
    Ties are broken by returning the first element among the winners
    when sorted. This provides deterministic behavior.
    """
    if series.empty:
        return np.nan
    # The value_counts method returns a Series with counts of unique values,
    # sorted in descending order. The index of this series contains the
    # unique values themselves. By taking the first element of the index,
    # we get the most frequent value.
    return series.value_counts().index[0]


def _compute_edges(start: float, stop: float, width: float) -> np.ndarray:
    """
    Compute uniform bin edges from start to stop with a given width.

    Parameters
    ----------
    start : float
        The starting time for the bin edges.
    stop : float
        The ending time for the bin edges. Samples at this time are included.
    width : float
        The width of each bin in seconds.

    Returns
    -------
    np.ndarray
        A 1D array of bin edge times.
    """
    # np.ceil ensures that the final edge is at or after the stop time.
    n_bins = int(np.ceil((stop - start) / width))
    return start + np.arange(n_bins + 1) * width


def _determine_time_window(
    rec: RecordingBundle, start_s: float | None, stop_s: float | None
) -> Tuple[float, float]:
    """
    Determine the overall start and stop time for binning from the recording.

    Parameters
    ----------
    rec : RecordingBundle
        The recording data.
    start_s : float or None
        User-defined start time in seconds. If None, it's inferred from data.
    stop_s : float or None
        User-defined stop time in seconds. If None, it's inferred from data.

    Returns
    -------
    Tuple[float, float]
        The final start and stop times for the analysis window.

    Raises
    ------
    ValueError
        If the RecordingBundle is empty and times cannot be inferred.
    """
    all_times: list[float] = []
    if rec.spike_times_s:
        for st in rec.spike_times_s:
            if st.times_s.size > 0:
                all_times.extend([st.times_s[0], st.times_s[-1]])
    for ts in rec.signals.values():
        if ts.data.shape[0] > 0:
            all_times.extend(
                [ts.start_s, ts.start_s + ts.data.shape[0] / ts.sampling_rate_hz]
            )

    if not all_times:
        raise ValueError("Cannot determine time window from empty RecordingBundle.")

    start = min(all_times) if start_s is None else start_s
    stop = max(all_times) if stop_s is None else stop_s
    return start, stop


def _bin_spike_counts(
    rec: RecordingBundle,
    edges_s: np.ndarray,
    count_method: str,
    start: float,
    stop: float,
) -> np.ndarray:
    """
    Bin spike times into spike counts per time bin.

    Parameters
    ----------
    rec : RecordingBundle
        The recording data containing spike times.
    edges_s : np.ndarray, shape (n_time + 1,)
        The time bin edges.
    count_method : str
        Method for counting spikes ("hist" or "center").
    start : float
        The start time of the analysis window.
    stop : float
        The end time of the analysis window.

    Returns
    -------
    np.ndarray
        Spike counts array of shape (n_time, n_neurons).
    """
    if rec.spike_times_s is None:
        raise ValueError("Requested 'counts' but no spike_times_s in RecordingBundle")

    # A shift is applied for the "center" method to align spike times
    # to the center of bins before histogramming.
    shift = -(edges_s[1] - edges_s[0]) / 2 if count_method == "center" else 0.0
    counts_list = []
    for st in rec.spike_times_s:
        shifted_times = st.times_s + shift
        # Check for spikes outside the time window and warn the user.
        oob = (shifted_times < start) | (shifted_times > stop)
        if oob.any():
            unit_id = getattr(st, "unit_id", "Unknown")
            warnings.warn(
                f"Unit {unit_id}: {oob.sum()} spikes outside "
                f"[{start:.3f}, {stop:.3f}] dropped",
                RuntimeWarning,
            )
        # np.histogram is a highly efficient way to bin the spike times.
        c, _ = np.histogram(shifted_times, bins=edges_s)
        counts_list.append(c)
    return np.stack(counts_list, axis=1)


def _process_other_signals(
    rec: RecordingBundle,
    signals_to_use: Sequence[str],
    edges_s: np.ndarray,
    policies: Dict[str, str],
) -> Dict[str, np.ndarray]:
    """
    Bin and aggregate all signals other than spike counts and categoricals.
    This function leverages pandas for efficient processing.

    Parameters
    ----------
    rec : RecordingBundle
        The recording data containing signals.
    signals_to_use : Sequence[str]
        A list of signal keys to process.
    edges_s : np.ndarray
        The time bin edges.
    policies : Dict[str, str]
        A dictionary containing policies for downsampling and filling.

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary of processed signal arrays.
    """
    binned_signals = {}
    bin_centers = (edges_s[:-1] + edges_s[1:]) / 2.0
    n_bins = len(bin_centers)

    for key in signals_to_use:
        if key not in rec.signals or key == "counts":
            continue

        ts = rec.signals[key]
        kind = ts.data.dtype.kind

        # Skip categorical signals, as they are handled separately.
        if kind in ("U", "O"):
            continue

        # Create a pandas DataFrame for the signal to leverage pandas'
        # powerful time-series manipulation tools.
        sample_times = ts.start_s + np.arange(ts.data.shape[0]) / ts.sampling_rate_hz

        # FIX: Check for out-of-bounds signals and warn
        if ts.data.size > 0 and (
            sample_times[-1] < edges_s[0] or sample_times[0] > edges_s[-1]
        ):
            msg = (
                f"All samples of '{key}' outside [{edges_s[0]:.3f}, {edges_s[-1]:.3f}]"
            )
            warnings.warn(msg, RuntimeWarning)
            continue

        df = pd.DataFrame(ts.data, index=sample_times)

        # Use pd.cut to assign each sample to a time bin.
        df["bin"] = pd.cut(df.index, bins=edges_s, labels=False, right=False)
        df = df.dropna(subset=["bin"])  # Drop samples outside the main time window
        df["bin"] = df["bin"].astype(int)

        # Define aggregation logic based on data type.
        if kind == "f":
            agg_func = "mean" if policies["float_downsample"] == "mean" else "median"
            fill_method = "ffill" if policies["float_fill"] == "ffill" else "linear"
        elif kind in "ui":
            agg_func = _simple_mode
            fill_method = "ffill" if policies["int_fill"] == "ffill" else "pad_zero"
        elif kind == "b":
            agg_func = "any"
            fill_method = "ffill" if policies["bool_fill"] == "ffill" else None
        else:
            raise TypeError(f"Unsupported dtype '{ts.data.dtype}' for signal '{key}'")

        # Group by bin and aggregate.
        binned = df.groupby("bin").agg(agg_func)
        # Reindex to ensure all time bins are present, filling missing ones with NaN.
        binned = binned.reindex(np.arange(n_bins))

        # Fill missing values based on the specified policy.
        if fill_method == "ffill":
            binned = (
                binned.ffill().bfill()
            )  # Forward-fill then back-fill for completeness
        elif fill_method == "linear":
            binned = binned.interpolate(method="linear").ffill().bfill()
        elif fill_method == "pad_zero":
            binned = binned.fillna(0)

        # For integer types, ensure the final array has the correct dtype.
        if kind in "ui":
            binned = binned.fillna(-1).astype(np.int64)

        # FIX: Squeeze array to 1D if original data was 1D
        result_array = binned.to_numpy()
        if ts.data.ndim == 1:
            result_array = result_array.squeeze(axis=1)
        binned_signals[key] = result_array

    return binned_signals


def _process_categorical_signals(
    rec: RecordingBundle,
    signals_to_use: Sequence[str],
    edges_s: np.ndarray,
    one_hot: bool,
    fill_policy: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Bin and encode categorical signals.

    Parameters
    ----------
    rec : RecordingBundle
        The recording data.
    signals_to_use : Sequence[str]
        A list of signal keys to process.
    edges_s : np.ndarray
        Time bin edges.
    one_hot : bool
        If True, use one-hot encoding. Otherwise, use ordinal encoding.
    fill_policy : str
        The fill policy for missing values ("ffill" or "pad_zero").

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, Any]]
        A tuple containing the dictionary of processed signals and the
        dictionary of fitted encoders or label maps.
    """
    binned_signals: Dict[str, np.ndarray] = {}
    categorical_maps: Dict[str, Any] = {}

    for key in signals_to_use:
        if key not in rec.signals or rec.signals[key].data.dtype.kind not in ("U", "O"):
            continue

        ts = rec.signals[key]
        n_bins = len(edges_s) - 1
        sample_times = ts.start_s + np.arange(ts.data.shape[0]) / ts.sampling_rate_hz

        # FIX: Check for out-of-bounds signals and warn
        if ts.data.size > 0 and (
            sample_times[-1] < edges_s[0] or sample_times[0] > edges_s[-1]
        ):
            msg = (
                f"All samples of '{key}' outside [{edges_s[0]:.3f}, {edges_s[-1]:.3f}]"
            )
            warnings.warn(msg, RuntimeWarning)
            continue

        df = pd.DataFrame(ts.data.reshape(-1, 1), index=sample_times, columns=["label"])

        if one_hot:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            df_encoded = pd.DataFrame(
                encoder.fit_transform(df[["label"]]),
                index=df.index,
                columns=encoder.get_feature_names_out(),
            )
            df_encoded["bin"] = pd.cut(
                df_encoded.index, bins=edges_s, labels=False, right=False
            )
            df_encoded = df_encoded.dropna(subset=["bin"])
            binned = (
                df_encoded.groupby("bin")
                .any()
                .reindex(np.arange(n_bins), fill_value=False)
            )
            binned_signals[key] = binned.to_numpy()
            categorical_maps[key] = encoder
        else:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            df["code"] = encoder.fit_transform(df[["label"]]).astype(int)
            df["bin"] = pd.cut(df.index, bins=edges_s, labels=False, right=False)
            df = df.dropna(subset=["bin"])
            binned = (
                df.groupby("bin")["code"]
                .apply(_simple_mode)
                .reindex(np.arange(n_bins), fill_value=-1)
                .astype(int)
            )

            if fill_policy == "ffill":
                binned.replace(-1, np.nan, inplace=True)
                binned = binned.ffill().bfill().fillna(-1).astype(int)
            elif fill_policy == "pad_zero":
                binned[binned == -1] = 0

            binned_signals[key] = binned.to_numpy()
            categorical_maps[key] = encoder.categories_[0].tolist()

    return binned_signals, categorical_maps


def to_decoder_batch(
    rec: RecordingBundle,
    bin_width_s: float,
    *,
    signals_to_use: Sequence[str],
    count_method: str = "hist",
    float_downsample: str = "mean",
    float_fill: str = "ffill",
    int_fill: str = "ffill",
    bool_fill: str = "ffill",
    start_s: float | None = None,
    stop_s: float | None = None,
    nan_policy: str = "warn",
    one_hot_categories: bool = False,
) -> DecoderBatch:
    """
    Bin a RecordingBundle into a DecoderBatch on a uniform grid.

    This refactored version uses pandas for efficient, vectorized processing
    of signals, improving performance and readability.

    Parameters
    ----------
    rec : RecordingBundle
        The immutable, native-rate recording data.
    bin_width_s : float
        The desired width of the time bins in seconds.
    signals_to_use : Sequence[str]
        A list of keys for the signals to be included in the output batch.
        Can include "counts" to bin spike times.
    count_method : str, optional
        Method for binning spike counts, by default "hist".
    float_downsample : str, optional
        Aggregation for floats ("mean" or "median"), by default "mean".
    float_fill : str, optional
        Fill policy for floats ("ffill" or "linear"), by default "ffill".
    int_fill : str, optional
        Fill policy for integers ("ffill" or "pad_zero"), by default "ffill".
    bool_fill : str, optional
        Fill policy for booleans ("ffill"), by default "ffill".
    start_s : float or None, optional
        Force a start time for the analysis window, by default None.
    stop_s : float or None, optional
        Force a stop time for the analysis window, by default None.
    nan_policy : str, optional
        Policy for handling remaining NaNs ("raise", "warn", "ignore"), by default "warn".
    one_hot_categories : bool, optional
        If True, encode categorical signals using one-hot encoding, by default False.

    Returns
    -------
    DecoderBatch
        A time-aligned data batch ready for use with decoders.
    """
    # 1) Validate policies and determine the overall time window.
    if count_method not in {"hist", "center"}:
        raise ValueError("count_method must be 'hist' or 'center'")
    if nan_policy not in {"raise", "warn", "ignore"}:
        raise ValueError("nan_policy must be 'raise', 'warn', or 'ignore'")

    start, stop = _determine_time_window(rec, start_s, stop_s)
    edges_s = _compute_edges(start, stop, bin_width_s)
    signals: Dict[str, np.ndarray] = {}

    # 2) Bin spike counts if requested.
    if "counts" in signals_to_use:
        signals["counts"] = _bin_spike_counts(rec, edges_s, count_method, start, stop)

    # 3) Process all other signals (continuous, integer, boolean).
    policies = {
        "float_downsample": float_downsample,
        "float_fill": float_fill,
        "int_fill": int_fill,
        "bool_fill": bool_fill,
    }
    other_signals = _process_other_signals(rec, signals_to_use, edges_s, policies)
    signals.update(other_signals)

    # 4) Process categorical signals separately.
    categorical_signals, categorical_maps = _process_categorical_signals(
        rec, signals_to_use, edges_s, one_hot_categories, int_fill
    )
    signals.update(categorical_signals)

    # 5) Enforce nan_policy on the final float arrays.
    if nan_policy in ("raise", "warn"):
        for k, v in signals.items():
            if np.issubdtype(v.dtype, np.floating) and np.isnan(v).any():
                msg = f"NaNs remain in '{k}' after fill"
                if nan_policy == "raise":
                    raise ValueError(msg)
                warnings.warn(msg, RuntimeWarning)

    # 6) Build the final DecoderBatch object.
    batch = DecoderBatch(
        signals=signals,
        bin_edges_s=edges_s,
        spike_times_s=rec.spike_times_s,
        spike_waveforms=rec.spike_waveforms,
    )
    batch._categorical_maps.update(categorical_maps)

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

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    time_column : str or None, optional
        Name of the column to use as the time index, by default None.
    sampling_rate_hz : float or None, optional
        The sampling rate. If None, it's inferred from the index, by default None.
    signals_to_include : list[str] or None, optional
        A list of column names to include as signals, by default None (all columns).

    Returns
    -------
    RecordingBundle
        The resulting recording bundle.
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
        # Use median to be robust to single missing samples, but check for uniformity.
        med_delta = np.median(deltas)
        if not np.allclose(deltas, med_delta, rtol=1e-3, atol=1e-4):
            raise ValueError("Index not uniform; supply sampling_rate_hz")

        sampling_rate_hz = 1.0 / med_delta

    signals: dict[str, TimeSeries] = {}
    keys_to_process = signals_to_include or list(df.columns)
    for key in keys_to_process:
        arr = df[key].to_numpy()
        signals[key] = TimeSeries(
            data=arr, sampling_rate_hz=sampling_rate_hz, start_s=times[0]
        )

    return RecordingBundle(spike_times_s=None, spike_waveforms=None, signals=signals)
