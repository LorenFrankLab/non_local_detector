"""Probe geometry and waveform amplitude models for dense multi-channel probes.

Provides functions to define probe channel layouts (e.g., Neuropixels),
generate neuron positions near the probe, and compute the expected spike
amplitude on each channel based on distance-dependent attenuation.

Includes preset configurations for common probe types:

- **Neuropixels 1.0**: 384 channels, 2-column checkerboard, 20 um spacing
- **Polymer probe (Frank lab)**: 4 shanks x 32 channels, 1 column, 35 um spacing
- **Tetrode**: 4 channels, no spatial geometry (random layout within ~25 um)
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np


def make_linear_probe(
    n_channels: int = 384,
    vertical_spacing: float = 20.0,
    n_columns: int = 2,
    column_spacing: float = 16.0,
) -> np.ndarray:
    """Create channel positions for a linear multi-column probe.

    Generates a checkerboard layout matching Neuropixels 1.0 geometry by
    default: channels alternate between columns at each vertical position.

    Parameters
    ----------
    n_channels : int, optional
        Total number of channels on the probe. Default is 384 (Neuropixels 1.0).
    vertical_spacing : float, optional
        Distance in microns between vertically adjacent channels. Default is
        20.0 um.
    n_columns : int, optional
        Number of columns on the probe shank. Default is 2.
    column_spacing : float, optional
        Lateral distance in microns between columns. Default is 16.0 um.

    Returns
    -------
    channel_positions : np.ndarray, shape (n_channels, 2)
        Each row is ``(x, z)`` in microns, where ``x`` is the lateral position
        and ``z`` is the depth along the shank (increasing downward).

    """
    idx = np.arange(n_channels)
    channel_positions = np.column_stack(
        [
            (idx % n_columns) * column_spacing,
            (idx // n_columns) * vertical_spacing,
        ]
    )
    return channel_positions


def make_neuron_locations(
    n_neurons: int,
    channel_positions: np.ndarray,
    depth_range: tuple[float, float] = (10.0, 80.0),
    lateral_extent: float = 50.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate random 3-D neuron positions near a probe shank.

    Parameters
    ----------
    n_neurons : int
        Number of neurons to place.
    channel_positions : np.ndarray, shape (n_channels, 2)
        Probe channel ``(x, z)`` positions as returned by :func:`make_linear_probe`.
    depth_range : tuple of float, optional
        Min and max perpendicular distance from the probe face in microns
        (the ``y`` coordinate). Default is ``(10.0, 80.0)``.
    lateral_extent : float, optional
        Half-width of the region around the probe center from which neuron
        ``x`` positions are drawn. Default is 50.0 um.
    rng : np.random.Generator or None, optional
        Random number generator. If *None*, a new default generator is created.

    Returns
    -------
    neuron_locations : np.ndarray, shape (n_neurons, 3)
        Each row is ``(x, y, z)`` in microns, where ``y`` is perpendicular
        distance from the probe face.

    """
    if rng is None:
        rng = np.random.default_rng()

    if depth_range[0] >= depth_range[1]:
        raise ValueError(
            f"depth_range[0] ({depth_range[0]}) must be less than "
            f"depth_range[1] ({depth_range[1]})"
        )

    x_center = channel_positions[:, 0].mean()
    z_min = channel_positions[:, 1].min()
    z_max = channel_positions[:, 1].max()

    x = rng.uniform(x_center - lateral_extent, x_center + lateral_extent, n_neurons)
    y = rng.uniform(depth_range[0], depth_range[1], n_neurons)
    z = rng.uniform(z_min, z_max, n_neurons)

    return np.column_stack([x, y, z])


def compute_amplitude_falloff(
    neuron_locations: np.ndarray,
    channel_positions: np.ndarray,
    decay_model: Literal["inverse_square", "exponential"] = "inverse_square",
    decay_constant: float = 30.0,
) -> np.ndarray:
    """Compute peak amplitude of each neuron on each channel.

    Models the spatial attenuation of extracellular spike amplitude as a
    function of the Euclidean distance between the neuron and the channel.
    The neuron ``y`` coordinate (perpendicular to the probe face) contributes
    to the distance.

    Parameters
    ----------
    neuron_locations : np.ndarray, shape (n_neurons, 3)
        Neuron ``(x, y, z)`` positions in microns.
    channel_positions : np.ndarray, shape (n_channels, 2)
        Channel ``(x, z)`` positions in microns.
    decay_model : {"inverse_square", "exponential"}, optional
        Attenuation model. Default is ``"inverse_square"``.
    decay_constant : float, optional
        Characteristic length scale in microns. Default is 30.0 um.

    Returns
    -------
    amplitude_templates : np.ndarray, shape (n_neurons, n_channels)
        Peak-normalised amplitude template for each neuron. Rows are scaled
        so that the maximum value across channels equals 1.0.

    """
    # Project neuron (x, y, z) onto probe plane: distance uses (x, z) mismatch
    # plus perpendicular depth y.
    neuron_xz = neuron_locations[:, [0, 2]]  # (n_neurons, 2)
    neuron_y = neuron_locations[:, 1]  # (n_neurons,)

    # Euclidean distance: sqrt((x_n - x_c)^2 + (z_n - z_c)^2 + y_n^2)
    # Reshape for broadcasting: (n_neurons, 1, 2) - (1, n_channels, 2)
    xz_diff = neuron_xz[:, np.newaxis, :] - channel_positions[np.newaxis, :, :]
    dist_sq = (xz_diff**2).sum(axis=2) + neuron_y[:, np.newaxis] ** 2
    distances = np.sqrt(dist_sq)  # (n_neurons, n_channels)

    if decay_model == "inverse_square":
        amplitudes = 1.0 / (1.0 + (distances / decay_constant) ** 2)
    elif decay_model == "exponential":
        amplitudes = np.exp(-distances / decay_constant)
    else:
        raise ValueError(
            f"Unknown decay_model {decay_model!r}. "
            "Choose 'inverse_square' or 'exponential'."
        )

    # Normalise each neuron so its max channel amplitude is 1.0
    row_max = amplitudes.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0  # avoid division by zero
    amplitudes /= row_max

    return amplitudes


def select_channels(
    amplitude_templates: np.ndarray,
    n_active_channels: int | None = None,
    method: Literal["top_k", "uniform", "all"] = "top_k",
) -> np.ndarray:
    """Select a subset of channels to use as mark features.

    Parameters
    ----------
    amplitude_templates : np.ndarray, shape (n_neurons, n_channels)
        Amplitude templates from :func:`compute_amplitude_falloff`.
    n_active_channels : int or None, optional
        Number of channels to select. *None* means use all channels.
    method : {"top_k", "uniform", "all"}, optional
        Selection strategy. ``"top_k"`` picks channels with the highest
        aggregate amplitude across neurons. ``"uniform"`` picks evenly spaced
        channels. ``"all"`` returns all channel indices.

    Returns
    -------
    channel_indices : np.ndarray, shape (n_selected,)
        Sorted array of selected channel indices.

    """
    n_channels = amplitude_templates.shape[1]

    if n_active_channels is None or method == "all":
        return np.arange(n_channels)

    n_active_channels = min(n_active_channels, n_channels)

    if method == "top_k":
        aggregate = amplitude_templates.sum(axis=0)
        indices = np.argsort(aggregate)[-n_active_channels:]
    elif method == "uniform":
        indices = np.linspace(0, n_channels - 1, n_active_channels, dtype=int)
    else:
        raise ValueError(
            f"Unknown method {method!r}. Choose 'top_k', 'uniform', or 'all'."
        )

    return np.sort(indices)


# ---------------------------------------------------------------------------
# Probe configuration presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeConfig:
    """Configuration for a probe type.

    Captures the geometry and amplitude-decay parameters for a single shank.
    Multi-shank probes are represented by ``n_shanks > 1``; each shank uses
    the same geometry and decay parameters.

    Attributes
    ----------
    name : str
        Human-readable label (e.g., ``"neuropixels_1.0"``).
    n_channels_per_shank : int
        Number of recording channels on each shank.
    n_shanks : int
        Number of independent shanks.
    vertical_spacing : float
        Vertical inter-channel distance in microns.
    n_columns : int
        Number of columns per shank.
    column_spacing : float
        Lateral distance between columns in microns.
    decay_constant : float
        Characteristic amplitude-decay length in microns.
    neuron_depth_range : tuple of float
        ``(min, max)`` perpendicular distance of neurons from probe face (um).
    lateral_extent : float
        Half-width of lateral region for neuron placement (um).
    expected_channels_per_spike : tuple of int
        Typical number of channels a single neuron's spike spans.
        Informational only — used for documentation and sensible defaults.

    """

    name: str
    n_channels_per_shank: int
    n_shanks: int
    vertical_spacing: float
    n_columns: int
    column_spacing: float
    decay_constant: float
    neuron_depth_range: tuple[float, float]
    lateral_extent: float
    expected_channels_per_spike: tuple[int, int] = (2, 4)
    # Shank offsets: lateral (x) separation between shanks in microns.
    # Only used when n_shanks > 1.  Default empty means no offset needed.
    shank_spacing: float = 250.0

    def make_channel_positions(self) -> list[np.ndarray]:
        """Build channel positions for all shanks.

        Returns
        -------
        list of np.ndarray, each shape (n_channels_per_shank, 2)
            Per-shank channel ``(x, z)`` positions in microns.

        """
        base = make_linear_probe(
            self.n_channels_per_shank,
            self.vertical_spacing,
            self.n_columns,
            self.column_spacing,
        )
        shanks = []
        for s in range(self.n_shanks):
            offset = np.array([s * self.shank_spacing, 0.0])
            shanks.append(base + offset)
        return shanks


# Preset factory functions


def neuropixels_config() -> ProbeConfig:
    """Neuropixels 1.0 configuration.

    384 channels, 2-column checkerboard, 20 um vertical spacing.
    Broad spike footprints (~10-20 channels).
    """
    return ProbeConfig(
        name="neuropixels_1.0",
        n_channels_per_shank=384,
        n_shanks=1,
        vertical_spacing=20.0,
        n_columns=2,
        column_spacing=16.0,
        decay_constant=30.0,
        neuron_depth_range=(10.0, 80.0),
        lateral_extent=50.0,
        expected_channels_per_spike=(10, 20),
    )


def polymer_probe_config() -> ProbeConfig:
    """Frank lab polymer probe configuration.

    4 shanks x 32 channels, single column, 35 um spacing.
    Tight spike footprints (2-4 channels).
    """
    return ProbeConfig(
        name="polymer_probe",
        n_channels_per_shank=32,
        n_shanks=4,
        vertical_spacing=35.0,
        n_columns=1,
        column_spacing=0.0,
        decay_constant=10.0,
        neuron_depth_range=(10.0, 20.0),
        lateral_extent=10.0,
        expected_channels_per_spike=(2, 4),
        shank_spacing=250.0,
    )


def tetrode_config(n_tetrodes: int = 5) -> ProbeConfig:
    """Tetrode configuration.

    Each tetrode has 4 channels with no fixed spatial geometry (channels
    are within ~25 um of each other).  Each tetrode is treated as an
    independent shank.

    Parameters
    ----------
    n_tetrodes : int, optional
        Number of tetrodes. Default is 5.

    """
    return ProbeConfig(
        name="tetrode",
        n_channels_per_shank=4,
        n_shanks=n_tetrodes,
        vertical_spacing=12.0,
        n_columns=2,
        column_spacing=12.0,
        decay_constant=15.0,
        neuron_depth_range=(5.0, 30.0),
        lateral_extent=15.0,
        expected_channels_per_spike=(3, 4),
        shank_spacing=500.0,
    )


PROBE_PRESETS: dict[str, ProbeConfig] = {
    "neuropixels": neuropixels_config(),
    "polymer": polymer_probe_config(),
    "tetrode": tetrode_config(),
}
