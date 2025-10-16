"""Simulator contract for decoder-ready outputs.

This module defines the standard output format for clusterless simulation
functions, ensuring compatibility with decoder APIs without intermediate
conversions.
"""

from dataclasses import dataclass

import numpy as np

from non_local_detector.environment import Environment


@dataclass
class ClusterlessSimOutput:
    """Standard output format for clusterless simulations.

    This dataclass defines the contract between simulator functions and
    decoder APIs. All times are in seconds, and spike data is organized
    per-electrode for direct consumption by likelihood models.

    Attributes
    ----------
    position_time : np.ndarray, shape (n_time_position,)
        Timestamps for position samples in seconds, monotonically increasing.
    position : np.ndarray, shape (n_time_position, n_pos_dims)
        Position coordinates at each position_time.
    edges : np.ndarray, shape (n_time_bins + 1,)
        Decoding time bin edges in seconds, monotonically increasing.
        Defines intervals [edges[i], edges[i+1]) for decoding.
    spike_times : list of np.ndarray, length n_electrodes
        Per-electrode spike times in seconds. Each array has shape (n_spikes_e,)
        and is strictly increasing. Empty electrodes have shape (0,).
    spike_waveform_features : list of np.ndarray, length n_electrodes
        Per-electrode spike waveform features. Each array has shape
        (n_spikes_e, n_features). Empty electrodes have shape (0, n_features).
    environment : Environment
        Environment object defining spatial layout and place fields.
    bin_widths : np.ndarray | None, shape (n_time_bins,), optional
        Bin widths in seconds, computed as np.diff(edges). Provided for
        convenience; can be None if not pre-computed.

    Notes
    -----
    - All times must be finite (no NaN, no inf).
    - Spike times must be strictly increasing within each electrode.
    - Spike waveform features must be finite (no NaN).
    - Electrode indices match between spike_times and spike_waveform_features.
    - For empty electrodes, spike_times[e] has shape (0,) and
      spike_waveform_features[e] has shape (0, n_features).
    """

    position_time: np.ndarray
    position: np.ndarray
    edges: np.ndarray
    spike_times: list[np.ndarray]
    spike_waveform_features: list[np.ndarray]
    environment: Environment
    bin_widths: np.ndarray | None = None
