"""Simulate clusterless spikes and associated spike waveform features.

Simulator return contract (decoder-ready):
- spike_times: List[np.ndarray], per electrode, seconds, strictly increasing
- spike_waveform_features: List[np.ndarray], per electrode, shape (n_spikes_e, n_features)
- edges: decoding bin edges in seconds, shape (n_time_bins+1), bin_widths = np.diff(edges)
- position_time / position: used for local decoding & encoding interpolation
All times are in SECONDS. No NaNs in marks. Empty electrodes: (0,), (0, n_features).

See tests._sim_contract.ClusterlessSimOutput for the full contract definition.
"""

import numpy as np

from non_local_detector.environment import Environment
from non_local_detector.simulate.simulate import (
    get_trajectory_direction,
    simulate_multiunit_with_place_fields,
    simulate_position,
    simulate_time,
)
from non_local_detector.tests._sim_contract import ClusterlessSimOutput

SAMPLING_FREQUENCY = 1000
TRACK_HEIGHT = 175
RUNNING_SPEED = 15
PLACE_FIELD_VARIANCE = 12.5
PLACE_FIELD_MEANS = np.arange(0, 200, 10)
N_RUNS = 15
REPLAY_SPEEDUP = 120.0
N_TETRODES = 5
N_FEATURES = 4
MARK_SPACING = 5


def make_simulated_run_data(
    sampling_frequency: int = SAMPLING_FREQUENCY,
    track_height: float = TRACK_HEIGHT,
    running_speed: float = RUNNING_SPEED,
    n_runs: int = N_RUNS,
    place_field_variance: float = PLACE_FIELD_VARIANCE,
    place_field_means: np.ndarray = PLACE_FIELD_MEANS,
    n_tetrodes: int = N_TETRODES,
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
        Location of the center of the Gaussian place fields. Default is
        np.arange(0, 200, 10).
    n_tetrodes : int, optional
        Total number of tetrodes to simulate. Default is 5.
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
    # Set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Generate position trajectory
    n_samples = int(n_runs * sampling_frequency * 2 * track_height / running_speed)
    position_time = simulate_time(n_samples, sampling_frequency)
    position_1d = simulate_position(position_time, track_height, running_speed)
    position = position_1d[:, np.newaxis]  # Shape: (n_time, 1)

    # Generate multiunit data (NaN-padded format) using existing simulator
    multiunits = []
    if not make_inbound_outbound_neurons:
        for place_means in place_field_means.reshape((n_tetrodes, -1)):
            multiunits.append(
                simulate_multiunit_with_place_fields(
                    place_means,
                    position_1d,
                    mark_spacing=10,
                    n_mark_dims=4,
                    place_variance=place_field_variance,
                    sampling_frequency=sampling_frequency,
                )
            )
    else:
        trajectory_direction = get_trajectory_direction(position_1d)
        for direction in np.unique(trajectory_direction):
            is_condition = trajectory_direction == direction
            for place_means in place_field_means.reshape((n_tetrodes, -1)):
                multiunits.append(
                    simulate_multiunit_with_place_fields(
                        place_means,
                        position_1d,
                        mark_spacing=10,
                        n_mark_dims=4,
                        sampling_frequency=sampling_frequency,
                        place_variance=place_field_variance,
                        is_condition=is_condition,
                    )
                )

    # Stack: (n_time, n_features, n_electrodes)
    multiunits = np.stack(multiunits, axis=-1)
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
    """Creates a simulated continuous replay.

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second
    track_height : float, optional
        Height of the simulated track
    running_speed : float, optional
        Simualted speed of the animal
    place_field_means : np.ndarray, optional
        Location of the center of the Gaussian place fields.
    replay_speedup : int, optional
        Number of times faster the replay event is faster than the running speed
    n_tetrodes : int, optional
        Number of simulated tetrodes
    n_features : int, optional
        Number of simulated features
    mark_spacing : float, optional
        Spacing between Gaussian mark features

    Returns
    -------
    replay_time : np.ndarray, shape (n_time,)
        Time in seconds.
    test_multiunits : np.ndarray, shape (n_time, n_features, n_tetrodes)
        Binned clusterless spike times and features. NaN indicates no spike. Non-Nan indicates spike.
    """

    replay_speed = running_speed * replay_speedup
    n_samples = int(0.5 * sampling_frequency * 2 * track_height / replay_speed)
    replay_time = simulate_time(n_samples, sampling_frequency)
    true_replay_position = simulate_position(replay_time, track_height, replay_speed)
    place_field_means = place_field_means.reshape((n_tetrodes, -1))

    min_times_ind = np.argmin(
        np.abs(true_replay_position[:, np.newaxis] - place_field_means.ravel()), axis=0
    )
    tetrode_ind = (
        np.ones_like(place_field_means) * np.arange(5)[:, np.newaxis]
    ).ravel()

    test_multiunits = np.full((replay_time.size, n_features, n_tetrodes), np.nan)
    n_neurons = place_field_means.shape[1]

    mark_centers = np.arange(0, n_neurons * mark_spacing, mark_spacing)
    mark_ind = (np.ones_like(place_field_means) * np.arange(4)).ravel()

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
    """Creates a simulated stationary replay.

    Parameters
    ----------
    hover_neuron_ind : int, optional
        Index of which neuron is the stationary neuron.
    place_field_means : np.ndarray, optional
        Location of the center of the Gaussian place fields.
    sampling_frequency : int, optional
        Samples per second
    n_tetrodes : int, optional
        Number of simulated tetrodes
    n_features : int, optional
        Number of simulated features
    mark_spacing : float, optional
        Spacing between Gaussian mark features

    Returns
    -------
    replay_time : np.ndarray, shape (n_time,)
        Time in seconds.
    test_multiunits : np.ndarray, shape (n_time, n_features, n_tetrodes)
        Binned clusterless spike times and features. NaN indicates no spike. Non-Nan indicates spike.
    """

    place_field_means = place_field_means.reshape((n_tetrodes, -1))

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
    mark_centers = np.arange(0, n_neurons * mark_spacing, mark_spacing)
    test_multiunits[spike_time_ind, :, tetrode_ind] = mark_centers[neuron_ind]

    return replay_time, test_multiunits


def make_fragmented_replay(
    place_field_means: np.ndarray = PLACE_FIELD_MEANS,
    sampling_frequency: int = SAMPLING_FREQUENCY,
    n_tetrodes: int = N_TETRODES,
    n_features: int = N_FEATURES,
    mark_spacing: float = MARK_SPACING,
) -> tuple[np.ndarray, np.ndarray]:
    """Creates a simulated fragmented replay.

    Parameters
    ----------
    place_field_means : np.ndarray, optional
        Location of the center of the Gaussian place fields.
    sampling_frequency : int, optional
        Samples per second
    n_tetrodes : int, optional
        Number of simulated tetrodes
    n_features : int, optional
        Number of simulated features
    mark_spacing : float, optional
        Spacing between Gaussian mark features

    Returns
    -------
    replay_time : np.ndarray, shape (n_time,)
        Time in seconds.
    test_multiunits : np.ndarray, shape (n_time, n_features, n_tetrodes)
        Binned clusterless spike times and features. NaN indicates no spike. Non-Nan indicates spike.
    """

    N_TIME = 10

    place_field_means = place_field_means.reshape((n_tetrodes, -1))
    replay_time = np.arange(N_TIME) / sampling_frequency

    n_total_neurons = place_field_means.size
    neuron_inds = [1, n_total_neurons - 1, 10, n_total_neurons - 5, 8]
    neuron_inds = np.unravel_index(neuron_inds, place_field_means.shape)
    spike_time_ind = [1, 3, 5, 7, 9]
    test_multiunits = np.full((replay_time.size, n_features, n_tetrodes), np.nan)
    n_neurons = place_field_means.shape[1]
    mark_centers = np.arange(0, n_neurons * mark_spacing, mark_spacing)

    for t_ind, tetrode_ind, neuron_ind in zip(
        spike_time_ind, *neuron_inds, strict=False
    ):
        test_multiunits[t_ind, :, tetrode_ind] = mark_centers[neuron_ind]

    return replay_time, test_multiunits


def make_hover_continuous_hover_replay(
    sampling_frequency: int = SAMPLING_FREQUENCY,
    place_field_means: np.ndarray = PLACE_FIELD_MEANS,
) -> tuple[np.ndarray, np.ndarray]:
    """Make a simulated replay that first is stationary, then is continuous, then is stationary again.

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second
    place_field_means : np.ndarray, optional
        Location of the center of the Gaussian place fields.

    Returns
    -------
    replay_time : np.ndarray, shape (n_time,)
        Time in seconds.
    test_multiunits : np.ndarray, shape (n_time, n_features, n_tetrodes)
        Binned clusterless spike times and features. NaN indicates no spike. Non-Nan indicates spike.
    """
    _, test_multiunits1 = make_hover_replay(hover_neuron_ind=0)
    _, test_multiunits2 = make_continuous_replay()
    n_total_neurons = place_field_means.size
    _, test_multiunits3 = make_hover_replay(hover_neuron_ind=n_total_neurons - 1)

    test_multiunits = np.concatenate(
        (test_multiunits1, test_multiunits2, test_multiunits3)
    )
    replay_time = np.arange(test_multiunits.shape[0]) / sampling_frequency

    return replay_time, test_multiunits


def make_fragmented_hover_fragmented_replay(
    sampling_frequency: int = SAMPLING_FREQUENCY,
) -> tuple[np.ndarray, np.ndarray]:
    """Makes a simulated replay that first is fragmented, then is stationary, then is fragmented again.

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second

    Returns
    -------
    replay_time : np.ndarray, shape (n_time,)
        Time in seconds.
    test_multiunits : np.ndarray, shape (n_time, n_features, n_tetrodes)
        Binned clusterless spike times and features. NaN indicates no spike. Non-Nan indicates spike.
    """
    _, test_multiunits1 = make_fragmented_replay()
    _, test_multiunits2 = make_hover_replay(hover_neuron_ind=6)
    _, test_multiunits3 = make_fragmented_replay()

    test_multiunits = np.concatenate(
        (test_multiunits1, test_multiunits2, test_multiunits3)
    )
    replay_time = np.arange(test_multiunits.shape[0]) / sampling_frequency

    return replay_time, test_multiunits


def make_fragmented_continuous_fragmented_replay(
    sampling_frequency: int = SAMPLING_FREQUENCY,
) -> tuple[np.ndarray, np.ndarray]:
    """Makes a simulated replay that first is fragmented, then is continuous, then is fragmented again.

    Parameters
    ----------
    sampling_frequency : int, optional
        Samples per second

    Returns
    -------
    replay_time : np.ndarray, shape (n_time,)
        Time in seconds.
    test_multiunits : np.ndarray, shape (n_time, n_features, n_tetrodes)
        Binned clusterless spike times and features. NaN indicates no spike. Non-Nan indicates spike.

    """
    _, test_multiunits1 = make_fragmented_replay()
    _, test_multiunits2 = make_continuous_replay()
    _, test_multiunits3 = make_fragmented_replay()

    test_multiunits = np.concatenate(
        (test_multiunits1, test_multiunits2, test_multiunits3)
    )
    replay_time = np.arange(test_multiunits.shape[0]) / sampling_frequency

    return replay_time, test_multiunits
