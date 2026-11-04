"""Simulate clusterless spikes and associated spike waveform features.

Simulator return contract (decoder-ready):
- spike_times: list[np.ndarray], per electrode, seconds, strictly increasing
- spike_waveform_features: list[np.ndarray], per electrode, shape (n_spikes_e, n_features)
- edges: decoding bin edges in seconds, shape (n_time_bins+1), bin_widths = np.diff(edges)
- position_time / position: used for local decoding & encoding interpolation
All times are in SECONDS. No NaNs in marks. Empty electrodes: (0,), (0, n_features).

See ClusterlessSimOutput for the full contract definition.
"""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from non_local_detector.environment import Environment
from non_local_detector.simulate.simulate import (
    get_trajectory_direction,
    simulate_multiunit_with_place_fields,
    simulate_position,
    simulate_time,
)

__all__ = [
    "ClusterlessSimOutput",
    "make_continuous_replay",
    "make_fragmented_continuous_fragmented_replay",
    "make_fragmented_hover_fragmented_replay",
    "make_fragmented_replay",
    "make_hover_continuous_hover_replay",
    "make_hover_replay",
    "make_simulated_run_data",
]


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


SAMPLING_FREQUENCY = 1000
TRACK_HEIGHT = 175
RUNNING_SPEED = 15
PLACE_FIELD_VARIANCE = 12.5
N_CELLS = 20
PLACE_FIELD_MEANS = np.linspace(0.0, TRACK_HEIGHT, N_CELLS)
N_RUNS = 15
REPLAY_SPEEDUP = 120.0
N_TETRODES = 5
N_FEATURES = 4
MARK_SPACING = 10


def _cells_by_tetrode(place_field_means: np.ndarray, n_tetrodes: int) -> np.ndarray:
    """Assign cells to tetrodes in contiguous blocks.

    Parameters
    ----------
    place_field_means : np.ndarray, shape (n_tetrodes * n_cells_per_tetrode,)
        Place-field centers of all cells.
    n_tetrodes : int
        Number of tetrodes.

    Returns
    -------
    np.ndarray, shape (n_tetrodes, n_cells_per_tetrode)
        Place-field centers grouped by tetrode. The cell at
        ``[tetrode, cell]`` has mark center ``cell * mark_spacing``.

    Raises
    ------
    ValueError
        If the number of cells is not a multiple of ``n_tetrodes``.
    """
    place_field_means = np.asarray(place_field_means)
    if place_field_means.size % n_tetrodes != 0:
        msg = (
            f"{place_field_means.size} cells cannot be split evenly among "
            f"{n_tetrodes} tetrodes; pass a number of place_field_means that "
            "is a multiple of n_tetrodes."
        )
        raise ValueError(msg)
    return place_field_means.reshape((n_tetrodes, -1))


def _mark_centers(n_cells_per_tetrode: int, mark_spacing: float) -> np.ndarray:
    """Mark-space center of each cell on a tetrode.

    Parameters
    ----------
    n_cells_per_tetrode : int
        Number of cells on each tetrode.
    mark_spacing : float
        Spacing between the mark centers of successive cells.

    Returns
    -------
    np.ndarray, shape (n_cells_per_tetrode,)
        ``cell * mark_spacing``, the center used by the run-data simulator.
    """
    return np.arange(n_cells_per_tetrode) * mark_spacing


def make_simulated_run_data(
    sampling_frequency: int = SAMPLING_FREQUENCY,
    track_height: float = TRACK_HEIGHT,
    running_speed: float = RUNNING_SPEED,
    n_runs: int = N_RUNS,
    place_field_variance: float = PLACE_FIELD_VARIANCE,
    place_field_means: np.ndarray = PLACE_FIELD_MEANS,
    n_tetrodes: int = N_TETRODES,
    n_features: int = N_FEATURES,
    mark_spacing: float = MARK_SPACING,
    *,
    make_inbound_outbound_neurons: bool = False,
    seed: int | None = 0,
) -> ClusterlessSimOutput:
    """Make simulated data of a rat running back and forth on a linear maze.

    Generates clusterless spike data (spike times + waveform features) for a
    simulated animal running on a linear track. Returns decoder-ready outputs
    with per-electrode spike lists (no NaN padding).

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second for position sampling. Default is 1000 Hz.
    track_height : float, optional
        Height of the simulated track in spatial units. Default is 175.
    running_speed : float, optional
        Speed of the simulated animal in spatial units/second. Default is 15.
    n_runs : int, optional
        Number of runs across the track the simulated animal will perform.
        Default is 15.
    place_field_variance : float, optional
        Spatial extent of place fields (variance of Gaussian). Default is 12.5.
    place_field_means : np.ndarray, shape (n_neurons,), optional
        Location of the center of the Gaussian place fields. Cells are assigned
        to tetrodes in contiguous blocks, so ``n_neurons`` must be divisible by
        ``n_tetrodes``. Default is 20 cells evenly spaced over
        ``[0, TRACK_HEIGHT]``, so every field is on the default track.
    n_tetrodes : int, optional
        Total number of tetrodes to simulate. Default is 5.
    n_features : int, optional
        Number of waveform features (mark dimensions). Default is 4.
    mark_spacing : float, optional
        Spacing between the mark centers of the cells on one tetrode; cell
        ``k`` on a tetrode has mark center ``k * mark_spacing``. The replay
        simulators default to the same value so replays decode against run
        data. Default is 10.
    make_inbound_outbound_neurons : bool, optional
        If True, create neurons with directional place fields (separate neurons
        for inbound vs outbound trajectories). Default is False.
    seed : int | None, optional
        Random seed for reproducibility. If None, uses system randomness.
        Default is 0.

    Returns
    -------
    sim_output : ClusterlessSimOutput
        Dataclass containing:
        - position_time: timestamps for position samples (seconds)
        - position: 2D position array (n_time, n_pos_dims)
        - edges: decoding bin edges (seconds)
        - spike_times: list of spike time arrays per electrode (seconds)
        - spike_waveform_features: list of feature arrays per electrode
        - environment: Environment object
        - bin_widths: bin widths (seconds)

    Notes
    -----
    - All times are in seconds
    - Spike times are strictly increasing per electrode
    - Empty electrodes have shape (0,) for times and (0, n_features) for features
    - No NaN values in spike waveform features
    """
    rng = np.random.default_rng(seed)

    # Generate position trajectory
    n_samples = int(n_runs * sampling_frequency * 2 * track_height / running_speed)
    position_time = simulate_time(n_samples, sampling_frequency)
    position_1d = simulate_position(position_time, track_height, running_speed)
    position = position_1d[:, np.newaxis]  # Shape: (n_time, 1)

    # Generate multiunit data (NaN-padded format) using existing simulator
    if not make_inbound_outbound_neurons:
        multiunits: list[np.ndarray] | np.ndarray = [
            simulate_multiunit_with_place_fields(
                place_means,
                position_1d,
                mark_spacing=mark_spacing,
                n_mark_dims=n_features,
                place_variance=place_field_variance,
                sampling_frequency=sampling_frequency,
                rng=rng,
            )
            for place_means in _cells_by_tetrode(place_field_means, n_tetrodes)
        ]
    else:
        trajectory_direction, _ = get_trajectory_direction(position_1d)
        multiunits = [
            simulate_multiunit_with_place_fields(
                place_means,
                position_1d,
                mark_spacing=mark_spacing,
                n_mark_dims=n_features,
                sampling_frequency=sampling_frequency,
                place_variance=place_field_variance,
                is_condition=trajectory_direction == direction,
                rng=rng,
            )
            for direction in np.unique(trajectory_direction)
            for place_means in _cells_by_tetrode(place_field_means, n_tetrodes)
        ]

    # Stack: (n_time, n_features, n_electrodes)
    multiunits = np.stack(cast("list[np.ndarray]", multiunits), axis=-1)
    multiunits_spikes = np.any(~np.isnan(multiunits), axis=1)

    # Convert NaN-padded format to per-electrode lists
    n_electrodes = multiunits.shape[2]
    spike_times = []
    spike_waveform_features = []

    for electrode_id in range(n_electrodes):
        spike_indicator = multiunits_spikes[:, electrode_id]
        electrode_spike_times = position_time[spike_indicator].astype(np.float64)
        electrode_features = multiunits[spike_indicator, :, electrode_id].astype(
            np.float64
        )

        # Ensure strictly increasing times (should already be sorted)
        if electrode_spike_times.size > 0:
            assert np.all(np.diff(electrode_spike_times) > 0), (
                "Spike times must be strictly increasing"
            )

        spike_times.append(electrode_spike_times)
        spike_waveform_features.append(electrode_features)

    # Create decoding bin edges (use position sampling rate)
    edges = np.arange(0.0, position_time[-1] + 1e-12, 1.0 / sampling_frequency)
    bin_widths = np.diff(edges)

    # Create environment
    environment = Environment(place_bin_size=1.0)
    environment.fit_place_grid(position)

    return ClusterlessSimOutput(
        position_time=position_time.astype(np.float64),
        position=position.astype(np.float64),
        edges=edges.astype(np.float64),
        spike_times=spike_times,
        spike_waveform_features=spike_waveform_features,
        environment=environment,
        bin_widths=bin_widths.astype(np.float64),
    )


def make_continuous_replay(
    sampling_frequency: int = SAMPLING_FREQUENCY,
    track_height: float = TRACK_HEIGHT,
    running_speed: float = RUNNING_SPEED,
    place_field_means: np.ndarray = PLACE_FIELD_MEANS,
    replay_speedup: float = REPLAY_SPEEDUP,
    n_tetrodes: int = N_TETRODES,
    n_features: int = N_FEATURES,
    mark_spacing: float = MARK_SPACING,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a simulated continuous replay.

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second.
    track_height : float, optional
        Height (length) of the simulated track, in the same spatial units as
        ``place_field_means``. Default is 175.
    running_speed : float, optional
        Simulated speed of the animal, in spatial units per second.
    place_field_means : np.ndarray, shape (n_tetrodes * n_cells_per_tetrode,), optional
        Location of the center of the Gaussian place fields. Match
        ``make_simulated_run_data``.
    replay_speedup : float, optional
        Factor by which the replay speed exceeds ``running_speed``.
    n_tetrodes : int, optional
        Number of simulated tetrodes.
    n_features : int, optional
        Number of simulated features (waveform feature dimensions per tetrode).
    mark_spacing : float, optional
        Spacing between the mark values of successive cells on a tetrode.
        Match ``make_simulated_run_data`` (default 10).

    Returns
    -------
    replay_time : np.ndarray, shape (n_time,)
        Time in seconds.
    test_multiunits : np.ndarray, shape (n_time, n_features, n_tetrodes)
        Binned clusterless spike features. NaN indicates no spike; non-NaN values
        are the spike's mark features.
    """
    replay_speed = running_speed * replay_speedup
    n_samples = int(0.5 * sampling_frequency * 2 * track_height / replay_speed)
    replay_time = simulate_time(n_samples, sampling_frequency)
    true_replay_position = simulate_position(replay_time, track_height, replay_speed)
    place_field_means = _cells_by_tetrode(place_field_means, n_tetrodes)
    n_neurons = place_field_means.shape[1]

    min_times_ind = np.argmin(
        np.abs(true_replay_position[:, np.newaxis] - place_field_means.ravel()), axis=0
    )
    tetrode_ind = np.broadcast_to(
        np.arange(n_tetrodes)[:, np.newaxis], place_field_means.shape
    ).ravel()

    test_multiunits = np.full((replay_time.size, n_features, n_tetrodes), np.nan)

    mark_centers = _mark_centers(n_neurons, mark_spacing)
    mark_ind = np.broadcast_to(np.arange(n_neurons), place_field_means.shape).ravel()

    for i in range(n_features):
        test_multiunits[(min_times_ind, i, tetrode_ind)] = mark_centers[mark_ind]

    return replay_time, test_multiunits


def make_hover_replay(
    hover_neuron_ind: int | None = None,
    place_field_means: np.ndarray = PLACE_FIELD_MEANS,
    sampling_frequency: int = SAMPLING_FREQUENCY,
    n_tetrodes: int = N_TETRODES,
    n_features: int = N_FEATURES,
    mark_spacing: float = MARK_SPACING,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a simulated stationary replay.

    Parameters
    ----------
    hover_neuron_ind : int, optional
        Flat index (into ``place_field_means`` reshaped to
        ``(n_tetrodes, n_cells_per_tetrode)``) of the stationary neuron. By
        default the middle neuron.
    place_field_means : np.ndarray, shape (n_tetrodes * n_cells_per_tetrode,), optional
        Location of the center of the Gaussian place fields. Match
        ``make_simulated_run_data``.
    sampling_frequency : int, optional
        Samples per second.
    n_tetrodes : int, optional
        Number of simulated tetrodes.
    n_features : int, optional
        Number of simulated features (waveform feature dimensions per tetrode).
    mark_spacing : float, optional
        Spacing between the mark values of successive cells on a tetrode.
        Match ``make_simulated_run_data`` (default 10).

    Returns
    -------
    replay_time : np.ndarray, shape (n_time,)
        Time in seconds.
    test_multiunits : np.ndarray, shape (n_time, n_features, n_tetrodes)
        Binned clusterless spike features. NaN indicates no spike; non-NaN values
        are the spike's mark features.
    """
    place_field_means = _cells_by_tetrode(place_field_means, n_tetrodes)

    if hover_neuron_ind is None:
        hover_neuron_ind = place_field_means.size // 2
    tetrode_ind, neuron_ind = np.unravel_index(
        hover_neuron_ind, place_field_means.shape
    )

    N_TIME = 50
    replay_time = np.arange(N_TIME) / sampling_frequency

    spike_time_ind = np.arange(0, N_TIME, 2)
    test_multiunits = np.full((replay_time.size, n_features, n_tetrodes), np.nan)
    n_neurons = place_field_means.shape[1]
    mark_centers = _mark_centers(n_neurons, mark_spacing)
    test_multiunits[spike_time_ind, :, tetrode_ind] = mark_centers[neuron_ind]

    return replay_time, test_multiunits


def make_fragmented_replay(
    place_field_means: np.ndarray = PLACE_FIELD_MEANS,
    sampling_frequency: int = SAMPLING_FREQUENCY,
    n_tetrodes: int = N_TETRODES,
    n_features: int = N_FEATURES,
    mark_spacing: float = MARK_SPACING,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a simulated fragmented replay.

    Parameters
    ----------
    place_field_means : np.ndarray, shape (n_tetrodes * n_cells_per_tetrode,), optional
        Location of the center of the Gaussian place fields. Match
        ``make_simulated_run_data``.
    sampling_frequency : int, optional
        Samples per second.
    n_tetrodes : int, optional
        Number of simulated tetrodes.
    n_features : int, optional
        Number of simulated features (waveform feature dimensions per tetrode).
    mark_spacing : float, optional
        Spacing between the mark values of successive cells on a tetrode.
        Match ``make_simulated_run_data`` (default 10).

    Returns
    -------
    replay_time : np.ndarray, shape (n_time,)
        Time in seconds.
    test_multiunits : np.ndarray, shape (n_time, n_features, n_tetrodes)
        Binned clusterless spike features. NaN indicates no spike; non-NaN values
        are the spike's mark features.
    """
    N_TIME = 10

    place_field_means = _cells_by_tetrode(place_field_means, n_tetrodes)
    replay_time = np.arange(N_TIME) / sampling_frequency

    n_total_neurons = place_field_means.size
    # Modulo keeps the indices valid for any number of cells.
    neuron_inds: list[int] | tuple[np.ndarray, ...] = [
        1 % n_total_neurons,
        n_total_neurons - 1,
        10 % n_total_neurons,
        (n_total_neurons - 5) % n_total_neurons,
        8 % n_total_neurons,
    ]
    neuron_inds = np.unravel_index(neuron_inds, place_field_means.shape)
    spike_time_ind = [1, 3, 5, 7, 9]
    test_multiunits = np.full((replay_time.size, n_features, n_tetrodes), np.nan)
    n_neurons = place_field_means.shape[1]
    mark_centers = _mark_centers(n_neurons, mark_spacing)

    for t_ind, tetrode_ind, neuron_ind in zip(
        spike_time_ind, *neuron_inds, strict=False
    ):
        test_multiunits[t_ind, :, tetrode_ind] = mark_centers[neuron_ind]

    return replay_time, test_multiunits


def make_hover_continuous_hover_replay(
    sampling_frequency: int = SAMPLING_FREQUENCY,
    place_field_means: np.ndarray = PLACE_FIELD_MEANS,
    n_tetrodes: int = N_TETRODES,
    n_features: int = N_FEATURES,
    mark_spacing: float = MARK_SPACING,
    track_height: float = TRACK_HEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    """Make a simulated replay that is stationary, then continuous, then stationary.

    The stationary segments hover on the first and last cell, which lie on
    the track for the default ``place_field_means``.

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second.
    place_field_means : np.ndarray, shape (n_tetrodes * n_cells_per_tetrode,), optional
        Location of the center of the Gaussian place fields. Pass the same
        value used for ``make_simulated_run_data``.
    n_tetrodes : int, optional
        Number of simulated tetrodes. Match ``make_simulated_run_data``.
    n_features : int, optional
        Number of simulated features. Match ``make_simulated_run_data``.
    mark_spacing : float, optional
        Spacing between the mark values of successive cells on a tetrode.
        Match ``make_simulated_run_data`` (default 10).
    track_height : float, optional
        Height (length) of the simulated track traversed by the continuous
        segment. Default is 175.

    Returns
    -------
    replay_time : np.ndarray, shape (n_time,)
        Time in seconds.
    test_multiunits : np.ndarray, shape (n_time, n_features, n_tetrodes)
        Binned clusterless spike features. NaN indicates no spike; non-NaN values
        are the spike's mark features.
    """
    layout: dict[str, Any] = {
        "place_field_means": place_field_means,
        "n_tetrodes": n_tetrodes,
        "n_features": n_features,
        "mark_spacing": mark_spacing,
    }
    n_total_neurons = place_field_means.size
    _, test_multiunits1 = make_hover_replay(hover_neuron_ind=0, **layout)
    _, test_multiunits2 = make_continuous_replay(track_height=track_height, **layout)
    _, test_multiunits3 = make_hover_replay(
        hover_neuron_ind=n_total_neurons - 1, **layout
    )

    test_multiunits = np.concatenate(
        (test_multiunits1, test_multiunits2, test_multiunits3)
    )
    replay_time = np.arange(test_multiunits.shape[0]) / sampling_frequency

    return replay_time, test_multiunits


def make_fragmented_hover_fragmented_replay(
    sampling_frequency: int = SAMPLING_FREQUENCY,
    place_field_means: np.ndarray = PLACE_FIELD_MEANS,
    n_tetrodes: int = N_TETRODES,
    n_features: int = N_FEATURES,
    mark_spacing: float = MARK_SPACING,
) -> tuple[np.ndarray, np.ndarray]:
    """Make a simulated replay that is fragmented, then stationary, then fragmented.

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second.
    place_field_means : np.ndarray, shape (n_tetrodes * n_cells_per_tetrode,), optional
        Location of the center of the Gaussian place fields. Pass the same
        value used for ``make_simulated_run_data``.
    n_tetrodes : int, optional
        Number of simulated tetrodes. Match ``make_simulated_run_data``.
    n_features : int, optional
        Number of simulated features. Match ``make_simulated_run_data``.
    mark_spacing : float, optional
        Spacing between the mark values of successive cells on a tetrode.
        Match ``make_simulated_run_data`` (default 10).

    Returns
    -------
    replay_time : np.ndarray, shape (n_time,)
        Time in seconds.
    test_multiunits : np.ndarray, shape (n_time, n_features, n_tetrodes)
        Binned clusterless spike features. NaN indicates no spike; non-NaN values
        are the spike's mark features.
    """
    layout: dict[str, Any] = {
        "place_field_means": place_field_means,
        "n_tetrodes": n_tetrodes,
        "n_features": n_features,
        "mark_spacing": mark_spacing,
    }
    _, test_multiunits1 = make_fragmented_replay(**layout)
    _, test_multiunits2 = make_hover_replay(
        hover_neuron_ind=6 % place_field_means.size, **layout
    )
    _, test_multiunits3 = make_fragmented_replay(**layout)

    test_multiunits = np.concatenate(
        (test_multiunits1, test_multiunits2, test_multiunits3)
    )
    replay_time = np.arange(test_multiunits.shape[0]) / sampling_frequency

    return replay_time, test_multiunits


def make_fragmented_continuous_fragmented_replay(
    sampling_frequency: int = SAMPLING_FREQUENCY,
    place_field_means: np.ndarray = PLACE_FIELD_MEANS,
    n_tetrodes: int = N_TETRODES,
    n_features: int = N_FEATURES,
    mark_spacing: float = MARK_SPACING,
    track_height: float = TRACK_HEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    """Make a simulated replay that is fragmented, then continuous, then fragmented.

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second.
    place_field_means : np.ndarray, shape (n_tetrodes * n_cells_per_tetrode,), optional
        Location of the center of the Gaussian place fields. Pass the same
        value used for ``make_simulated_run_data``.
    n_tetrodes : int, optional
        Number of simulated tetrodes. Match ``make_simulated_run_data``.
    n_features : int, optional
        Number of simulated features. Match ``make_simulated_run_data``.
    mark_spacing : float, optional
        Spacing between the mark values of successive cells on a tetrode.
        Match ``make_simulated_run_data`` (default 10).
    track_height : float, optional
        Height (length) of the simulated track traversed by the continuous
        segment. Default is 175.

    Returns
    -------
    replay_time : np.ndarray, shape (n_time,)
        Time in seconds.
    test_multiunits : np.ndarray, shape (n_time, n_features, n_tetrodes)
        Binned clusterless spike features. NaN indicates no spike; non-NaN values
        are the spike's mark features.
    """
    layout: dict[str, Any] = {
        "place_field_means": place_field_means,
        "n_tetrodes": n_tetrodes,
        "n_features": n_features,
        "mark_spacing": mark_spacing,
    }
    _, test_multiunits1 = make_fragmented_replay(**layout)
    _, test_multiunits2 = make_continuous_replay(track_height=track_height, **layout)
    _, test_multiunits3 = make_fragmented_replay(**layout)

    test_multiunits = np.concatenate(
        (test_multiunits1, test_multiunits2, test_multiunits3)
    )
    replay_time = np.arange(test_multiunits.shape[0]) / sampling_frequency

    return replay_time, test_multiunits
