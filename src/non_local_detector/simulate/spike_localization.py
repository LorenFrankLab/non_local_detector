"""Spike localization and local amplitude extraction for dense probes.

Reduces high-dimensional mark vectors (peak amplitudes across many channels)
to low-dimensional features by exploiting the spatial locality of extracellular
spikes.  Each spike's signal is concentrated on a small number of channels
near the source neuron, so extracting only the local neighborhood dramatically
reduces dimensionality while preserving most of the information.

Three feature extraction modes are supported:

- **Position only** (A): center-of-mass (x, z) + peak amplitude → 3 features
- **Local amplitudes only** (B): aligned amplitudes from peak ± k neighbors
  → 2k+1 features
- **Combined** (C): position + local amplitudes → 2 + 2k+1 features

All functions accept the ``(marks, channel_positions)`` signature expected
by the ``feature_transform`` parameter of the simulation functions.
"""

import numpy as np


def _find_peak_channels(marks: np.ndarray) -> np.ndarray:
    """Find the peak channel for each spike.

    Parameters
    ----------
    marks : np.ndarray, shape (n_spikes, n_channels)
        Amplitude marks across all channels.

    Returns
    -------
    peak_channels : np.ndarray, shape (n_spikes,), dtype int
        Index of the channel with the highest absolute amplitude per spike.

    """
    return np.argmax(np.abs(marks), axis=1)


def _get_neighbor_indices(
    peak_channels: np.ndarray,
    n_channels: int,
    channel_positions: np.ndarray,
    n_neighbors: int = 1,
) -> np.ndarray:
    """Get indices of peak channel + nearest spatial neighbors.

    Neighbors are selected by Euclidean distance in the (x, z) plane,
    which handles both single-column and multi-column probe layouts.

    Parameters
    ----------
    peak_channels : np.ndarray, shape (n_spikes,), dtype int
        Peak channel index per spike.
    n_channels : int
        Total number of channels.
    channel_positions : np.ndarray, shape (n_channels, 2)
        Channel (x, z) positions in microns.
    n_neighbors : int, optional
        Number of neighbors on each side of the peak. Total channels
        extracted = 2 * n_neighbors + 1.  Default is 1.

    Returns
    -------
    neighbor_indices : np.ndarray, shape (n_spikes, 2 * n_neighbors + 1), dtype int
        Channel indices ordered by spatial distance from peak.
        ``neighbor_indices[:, 0]`` is always the peak channel.

    """
    n_total = 2 * n_neighbors + 1
    n_total = min(n_total, n_channels)

    # Precompute pairwise distance matrix between all channels
    # shape: (n_channels, n_channels)
    diffs = channel_positions[:, np.newaxis, :] - channel_positions[np.newaxis, :, :]
    dist_matrix = np.sqrt((diffs**2).sum(axis=2))

    # For each peak channel, find the n_total closest channels (including self)
    # sorted by distance
    neighbor_indices = np.empty((len(peak_channels), n_total), dtype=int)
    for i, peak in enumerate(peak_channels):
        closest = np.argsort(dist_matrix[peak])[:n_total]
        neighbor_indices[i] = closest

    return neighbor_indices


def extract_local_amplitudes(
    marks: np.ndarray,
    channel_positions: np.ndarray,
    n_neighbors: int = 1,
) -> np.ndarray:
    """Extract aligned local amplitudes around each spike's peak channel.

    For each spike, finds the peak channel and extracts amplitudes from the
    ``2 * n_neighbors + 1`` spatially nearest channels.  Features are ordered
    by spatial distance from peak (peak first), making them comparable across
    spikes regardless of absolute channel index.

    Parameters
    ----------
    marks : np.ndarray, shape (n_spikes, n_channels)
        Raw amplitude marks across all channels.
    channel_positions : np.ndarray, shape (n_channels, 2)
        Channel (x, z) positions in microns.
    n_neighbors : int, optional
        Number of neighbors on each side of the peak. Default is 1,
        giving 3 features total.

    Returns
    -------
    local_amps : np.ndarray, shape (n_spikes, 2 * n_neighbors + 1)
        Aligned local amplitude features.

    """
    n_spikes, n_channels = marks.shape
    peak_channels = _find_peak_channels(marks)
    neighbor_idx = _get_neighbor_indices(
        peak_channels, n_channels, channel_positions, n_neighbors
    )

    # Gather amplitudes: marks[i, neighbor_idx[i, :]]
    local_amps = marks[np.arange(n_spikes)[:, np.newaxis], neighbor_idx]
    return local_amps


def estimate_spike_position(
    marks: np.ndarray,
    channel_positions: np.ndarray,
    n_neighbors: int = 1,
) -> np.ndarray:
    """Estimate spike position on the probe via center-of-mass.

    Uses the squared amplitudes on the peak channel and its neighbors as
    weights to compute a weighted average of channel positions.

    Parameters
    ----------
    marks : np.ndarray, shape (n_spikes, n_channels)
        Raw amplitude marks across all channels.
    channel_positions : np.ndarray, shape (n_channels, 2)
        Channel (x, z) positions in microns.
    n_neighbors : int, optional
        Number of neighbors on each side of the peak used for the
        center-of-mass calculation. Default is 1.

    Returns
    -------
    positions : np.ndarray, shape (n_spikes, 2)
        Estimated (x, z) position of each spike in microns.

    """
    n_spikes, n_channels = marks.shape
    peak_channels = _find_peak_channels(marks)
    neighbor_idx = _get_neighbor_indices(
        peak_channels, n_channels, channel_positions, n_neighbors
    )

    # Gather local amplitudes and positions
    local_amps = marks[np.arange(n_spikes)[:, np.newaxis], neighbor_idx]
    local_positions = channel_positions[neighbor_idx]  # (n_spikes, n_local, 2)

    # Squared-amplitude weights (ensures non-negative)
    weights = local_amps**2
    weight_sum = weights.sum(axis=1, keepdims=True)
    weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)

    # Weighted average of channel positions
    # weights: (n_spikes, n_local) → (n_spikes, n_local, 1) for broadcasting
    positions = (weights[:, :, np.newaxis] * local_positions).sum(axis=1) / weight_sum

    return positions


def localize_spikes(
    marks: np.ndarray,
    channel_positions: np.ndarray,
    n_neighbors: int = 1,
    include_position: bool = True,
    include_local_amplitudes: bool = True,
) -> np.ndarray:
    """Extract low-dimensional spike features from dense probe marks.

    Combines spike position estimation (center-of-mass) and/or aligned local
    amplitude extraction into a single feature vector per spike.

    This function is compatible with the ``feature_transform`` parameter of
    the simulation functions.  To use a specific configuration as a transform,
    wrap with :func:`functools.partial`::

        from functools import partial
        transform = partial(localize_spikes, n_neighbors=1,
                            include_position=True, include_local_amplitudes=True)
        sim = make_probe_run_data(config, feature_transform=transform)

    Parameters
    ----------
    marks : np.ndarray, shape (n_spikes, n_channels)
        Raw amplitude marks across all channels.
    channel_positions : np.ndarray, shape (n_channels, 2)
        Channel (x, z) positions in microns.
    n_neighbors : int, optional
        Number of neighbors on each side of peak. Default is 1.
    include_position : bool, optional
        If True, include estimated (x, z) position. Default is True.
    include_local_amplitudes : bool, optional
        If True, include aligned local amplitudes. Default is True.

    Returns
    -------
    features : np.ndarray, shape (n_spikes, n_features)
        Combined feature vector.  Feature count depends on options:

        - Position only (A): ``n_features = 3`` (x, z, peak_amp)
        - Local amplitudes only (B): ``n_features = 2 * n_neighbors + 1``
        - Both (C): ``n_features = 2 + 2 * n_neighbors + 1``

    Raises
    ------
    ValueError
        If both ``include_position`` and ``include_local_amplitudes`` are False.

    """
    if not include_position and not include_local_amplitudes:
        raise ValueError(
            "At least one of include_position or include_local_amplitudes must be True."
        )

    parts: list[np.ndarray] = []

    if include_position:
        pos = estimate_spike_position(marks, channel_positions, n_neighbors)
        parts.append(pos)  # (n_spikes, 2)

    if include_local_amplitudes:
        local_amps = extract_local_amplitudes(marks, channel_positions, n_neighbors)
        parts.append(local_amps)  # (n_spikes, 2k+1)
    elif include_position:
        # Position-only mode: add peak amplitude as 3rd feature
        peak_amp = np.abs(marks).max(axis=1, keepdims=True)
        parts.append(peak_amp)  # (n_spikes, 1)

    return np.hstack(parts)
