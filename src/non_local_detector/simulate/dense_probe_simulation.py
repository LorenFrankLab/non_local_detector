"""Simulate clusterless data from a dense linear probe (e.g., Neuropixels).

Generates spike waveform features as peak-amplitude vectors across probe
channels, where amplitudes are determined by the spatial relationship between
neuron positions and channel positions.  The resulting high-dimensional marks
can be used to study the curse of dimensionality in KDE-based clusterless
decoding and to benchmark dimensionality-reduction approaches.

Simulator return contract (decoder-ready):
- spike_times: list[np.ndarray], single electrode (the probe), seconds
- spike_waveform_features: list[np.ndarray], shape (n_spikes, n_selected_channels)
- edges: decoding bin edges in seconds
- position_time / position: used for encoding & decoding

See tests._sim_contract.ClusterlessSimOutput for the full contract definition.
"""

from collections.abc import Callable
from typing import Literal

import numpy as np

from non_local_detector.environment import Environment
from non_local_detector.simulate.probe_geometry import (
    ProbeConfig,
    compute_amplitude_falloff,
    make_linear_probe,
    make_neuron_locations,
    select_channels,
)
from non_local_detector.simulate.simulate import (
    simulate_neuron_with_place_field,
    simulate_position,
    simulate_time,
)
from non_local_detector.tests._sim_contract import ClusterlessSimOutput

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SAMPLING_FREQUENCY = 1000
TRACK_HEIGHT = 175.0
RUNNING_SPEED = 15.0
N_RUNS = 15
PLACE_FIELD_VARIANCE = 12.5
PLACE_FIELD_MEANS = np.arange(0, 200, 10)  # 20 neurons
MAX_FIRING_RATE = 15.0
MAX_AMPLITUDE = 100.0
AMPLITUDE_NOISE_STD = 0.05
BACKGROUND_RATE = 0.0  # Hz per channel, off by default


def make_dense_probe_run_data(
    sampling_frequency: int = SAMPLING_FREQUENCY,
    track_height: float = TRACK_HEIGHT,
    running_speed: float = RUNNING_SPEED,
    n_runs: int = N_RUNS,
    place_field_variance: float = PLACE_FIELD_VARIANCE,
    place_field_means: np.ndarray | None = None,
    max_firing_rate: float = MAX_FIRING_RATE,
    # -- Probe geometry --
    n_channels: int = 384,
    vertical_spacing: float = 20.0,
    n_columns: int = 2,
    column_spacing: float = 16.0,
    # -- Neuron placement --
    neuron_depth_range: tuple[float, float] = (10.0, 80.0),
    lateral_extent: float = 50.0,
    # -- Amplitude model --
    decay_model: Literal["inverse_square", "exponential"] = "inverse_square",
    decay_constant: float = 30.0,
    max_amplitude: float = MAX_AMPLITUDE,
    amplitude_noise_std: float = AMPLITUDE_NOISE_STD,
    # -- Channel selection (controls mark dimensionality) --
    n_active_channels: int | None = None,
    channel_selection_method: Literal["top_k", "uniform", "all"] = "top_k",
    # -- Background spikes (optional) --
    background_rate: float = BACKGROUND_RATE,
    # -- Electrode noise floor --
    noise_floor: float = 0.0,
    # -- Feature transform (optional) --
    feature_transform: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    # -- Reproducibility --
    seed: int | None = 0,
) -> ClusterlessSimOutput:
    """Simulate a run session with clusterless data from a dense linear probe.

    Each simulated neuron has a Gaussian place field on the linear track and a
    spatial footprint on the probe determined by its 3-D location.  When a
    neuron fires, its spike is represented as a peak-amplitude vector across the
    selected probe channels plus Gaussian noise.

    Parameters
    ----------
    sampling_frequency : int, optional
        Position sampling rate in Hz.
    track_height : float, optional
        Length of the linear track in spatial units.
    running_speed : float, optional
        Simulated running speed in spatial units / second.
    n_runs : int, optional
        Number of traversals of the track.
    place_field_variance : float, optional
        Variance of Gaussian place fields.
    place_field_means : np.ndarray, shape (n_neurons,), optional
        Centers of Gaussian place fields on the track.
    max_firing_rate : float, optional
        Peak firing rate of each neuron (Hz).
    n_channels : int, optional
        Total channels on the probe.
    vertical_spacing : float, optional
        Vertical inter-channel distance in microns.
    n_columns : int, optional
        Number of shank columns.
    column_spacing : float, optional
        Lateral distance between columns in microns.
    neuron_depth_range : tuple of float, optional
        ``(min, max)`` perpendicular distance of neurons from probe face (um).
    lateral_extent : float, optional
        Half-width of lateral region for neuron placement (um).
    decay_model : {"inverse_square", "exponential"}, optional
        Amplitude attenuation model.
    decay_constant : float, optional
        Characteristic decay length in microns.
    max_amplitude : float, optional
        Scaling factor applied to the normalised amplitude templates.
    amplitude_noise_std : float, optional
        Relative noise level.  Noise standard deviation on each channel is
        ``amplitude_noise_std * channel_amplitude * max_amplitude``, i.e.,
        proportional to the channel's signal strength.  Channels with no
        signal receive no noise.
    n_active_channels : int or None, optional
        Number of channels to keep as mark features.  *None* keeps all.
    channel_selection_method : {"top_k", "uniform", "all"}, optional
        How to pick the active channel subset.
    background_rate : float, optional
        Rate (Hz) of position-independent background spikes.  These spikes
        have random amplitude profiles and carry no position information.
        Set to 0 (default) to disable.
    noise_floor : float, optional
        Standard deviation of additive Gaussian noise applied to ALL channels
        on every spike, independent of signal.  Simulates electrode thermal
        noise and distant neuron background activity.  Expressed as a fraction
        of *max_amplitude* (e.g., 0.1 means noise std = 10% of max_amplitude).
        Default is 0.0 (no floor noise).
    feature_transform : callable or None, optional
        Optional function ``(marks, channel_positions) -> transformed_marks``
        applied to the raw amplitude marks before output.  ``channel_positions``
        has shape ``(n_selected_channels, 2)`` — the (x, z) positions of the
        *selected* channels after applying *channel_selection_method*.  Useful
        for dimensionality reduction (e.g., spike localisation).
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    ClusterlessSimOutput
        Decoder-ready simulation output.  Contains a single electrode whose
        waveform features have shape ``(n_spikes, n_selected_channels)`` (or
        the output width of *feature_transform*).

    """
    if place_field_means is None:
        place_field_means = PLACE_FIELD_MEANS.copy()
    place_field_means = np.atleast_1d(place_field_means).ravel()
    n_neurons = place_field_means.size

    # Use separate RNGs for spike generation vs mark noise so that changing
    # n_active_channels (which affects the number of mark-noise draws) does
    # not alter the spike-time stream.  This makes dimensionality sweeps
    # produce identical spike times across runs.
    seed_seq = np.random.SeedSequence(seed)
    seed_spikes, seed_marks, seed_bg = seed_seq.spawn(3)
    rng_spikes = np.random.default_rng(seed_spikes)
    rng_marks = np.random.default_rng(seed_marks)
    rng_bg = np.random.default_rng(seed_bg)

    # -- Position trajectory ---------------------------------------------------
    n_samples = int(n_runs * sampling_frequency * 2 * track_height / running_speed)
    position_time = simulate_time(n_samples, sampling_frequency)
    position_1d = simulate_position(position_time, track_height, running_speed)
    position = position_1d[:, np.newaxis]  # (n_time, 1)

    # -- Probe and neuron geometry ---------------------------------------------
    channel_positions = make_linear_probe(
        n_channels, vertical_spacing, n_columns, column_spacing
    )
    neuron_locations = make_neuron_locations(
        n_neurons, channel_positions, neuron_depth_range, lateral_extent, rng=rng_spikes
    )
    amplitude_templates = compute_amplitude_falloff(
        neuron_locations, channel_positions, decay_model, decay_constant
    )
    selected_channels = select_channels(
        amplitude_templates, n_active_channels, channel_selection_method
    )
    selected_positions = channel_positions[selected_channels]

    # Restrict templates to selected channels and zero out channels that are
    # too far from the neuron (< 10% of peak) to produce realistic sparse
    # footprints.
    templates = amplitude_templates[:, selected_channels]  # (n_neurons, n_feat)
    row_max = templates.max(axis=1, keepdims=True)
    templates = np.where(templates >= 0.1 * row_max, templates, 0.0)
    n_features = templates.shape[1]

    # -- Generate spikes per neuron -------------------------------------------
    all_spike_times: list[np.ndarray] = []
    all_spike_marks: list[np.ndarray] = []

    for neuron_idx in range(n_neurons):
        spikes = simulate_neuron_with_place_field(
            means=place_field_means[neuron_idx],
            position=position_1d,
            max_rate=max_firing_rate,
            variance=place_field_variance,
            sampling_frequency=sampling_frequency,
            rng=rng_spikes,
        )
        spike_mask = spikes > 0
        n_spikes = int(spike_mask.sum())
        if n_spikes == 0:
            continue

        times = position_time[spike_mask]
        template_scaled = templates[neuron_idx][np.newaxis, :] * max_amplitude
        # Waveform variability noise (proportional to signal)
        noise_std_per_ch = amplitude_noise_std * np.abs(template_scaled)
        marks = (
            template_scaled
            + rng_marks.normal(0, 1, (n_spikes, n_features)) * noise_std_per_ch
        )
        # Electrode noise floor (constant across all channels)
        if noise_floor > 0:
            marks += rng_marks.normal(
                0, noise_floor * max_amplitude, (n_spikes, n_features)
            )
        all_spike_times.append(times)
        all_spike_marks.append(marks)

    # -- Background spikes (optional) -----------------------------------------
    if background_rate > 0:
        expected_bg = background_rate * position_time[-1]
        n_bg = rng_bg.poisson(expected_bg)
        if n_bg > 0:
            bg_times = rng_bg.uniform(position_time[0], position_time[-1], n_bg)
            bg_marks = rng_bg.uniform(0, max_amplitude, (n_bg, n_features))
            all_spike_times.append(bg_times)
            all_spike_marks.append(bg_marks)

    # -- Merge and sort by time -----------------------------------------------
    if len(all_spike_times) == 0:
        merged_times = np.empty(0, dtype=np.float64)
        merged_marks = np.empty((0, n_features), dtype=np.float64)
    else:
        merged_times = np.concatenate(all_spike_times)
        merged_marks = np.concatenate(all_spike_marks)
        sort_idx = np.argsort(merged_times)
        merged_times = merged_times[sort_idx]
        merged_marks = merged_marks[sort_idx]

    # Handle simultaneous spikes: jitter duplicates by a tiny amount so that
    # times are strictly increasing.
    if merged_times.size > 1:
        eps = 1e-6 / sampling_frequency
        diffs = np.diff(merged_times)
        duplicates = diffs <= 0
        if duplicates.any():
            # Cumulative count of consecutive duplicates at each position
            cumdup = np.zeros(len(diffs), dtype=int)
            for i in range(len(diffs)):
                if duplicates[i]:
                    cumdup[i] = cumdup[i - 1] + 1 if i > 0 else 1
            merged_times[1:] += cumdup * eps

    # -- Optional feature transform -------------------------------------------
    if feature_transform is not None:
        merged_marks = feature_transform(merged_marks, selected_positions)

    # -- Build output (single electrode = the whole probe) --------------------
    spike_times = [merged_times.astype(np.float64)]
    spike_waveform_features = [merged_marks.astype(np.float64)]

    edges = np.arange(0.0, position_time[-1] + 1e-12, 1.0 / sampling_frequency)
    bin_widths = np.diff(edges)

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


def make_dimensionality_sweep(
    dimensions: list[int],
    seed: int | None = 0,
    **kwargs,
) -> list[ClusterlessSimOutput]:
    """Run the dense-probe simulation at several mark dimensionalities.

    All runs share the same random seed.  Because spike generation and mark
    noise use independent RNG streams (via ``SeedSequence.spawn``), position
    trajectories and spike times are identical across runs — only the number
    of selected channels (and therefore the mark dimensionality) changes.

    Parameters
    ----------
    dimensions : list of int
        Number of active channels for each run (e.g., ``[4, 10, 50, 200]``).
    seed : int or None, optional
        Base random seed.
    **kwargs
        Forwarded to :func:`make_dense_probe_run_data`.

    Returns
    -------
    outputs : list of ClusterlessSimOutput
        One output per requested dimensionality.

    """
    outputs = []
    for n_dim in dimensions:
        outputs.append(
            make_dense_probe_run_data(
                n_active_channels=n_dim,
                seed=seed,
                **kwargs,
            )
        )
    return outputs


def make_probe_run_data(
    probe_config: ProbeConfig,
    sampling_frequency: int = SAMPLING_FREQUENCY,
    track_height: float = TRACK_HEIGHT,
    running_speed: float = RUNNING_SPEED,
    n_runs: int = N_RUNS,
    place_field_variance: float = PLACE_FIELD_VARIANCE,
    place_field_means: np.ndarray | None = None,
    max_firing_rate: float = MAX_FIRING_RATE,
    max_amplitude: float = MAX_AMPLITUDE,
    amplitude_noise_std: float = AMPLITUDE_NOISE_STD,
    background_rate: float = BACKGROUND_RATE,
    noise_floor: float = 0.0,
    feature_transform: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    seed: int | None = 0,
) -> ClusterlessSimOutput:
    """Simulate a run session using a probe configuration preset.

    Multi-shank probes (e.g., polymer probes, tetrodes) produce one electrode
    per shank in the output, matching the per-electrode structure expected by
    the clusterless likelihood models.  Single-shank probes (e.g., Neuropixels)
    produce a single electrode.

    Neurons are distributed across all shanks.  Each shank records spikes from
    nearby neurons independently, treating shanks as conditionally independent.

    Parameters
    ----------
    probe_config : ProbeConfig
        Probe geometry and decay parameters.  Use preset factories like
        :func:`~non_local_detector.simulate.probe_geometry.neuropixels_config`,
        :func:`~non_local_detector.simulate.probe_geometry.polymer_probe_config`,
        or :func:`~non_local_detector.simulate.probe_geometry.tetrode_config`.
    sampling_frequency : int, optional
        Position sampling rate in Hz.
    track_height : float, optional
        Length of the linear track in spatial units.
    running_speed : float, optional
        Simulated running speed in spatial units / second.
    n_runs : int, optional
        Number of traversals of the track.
    place_field_variance : float, optional
        Variance of Gaussian place fields.
    place_field_means : np.ndarray, shape (n_neurons,), optional
        Centers of Gaussian place fields on the track.
    max_firing_rate : float, optional
        Peak firing rate of each neuron (Hz).
    max_amplitude : float, optional
        Scaling factor applied to the normalised amplitude templates.
    amplitude_noise_std : float, optional
        Relative noise level.  Noise standard deviation on each channel is
        ``amplitude_noise_std * channel_amplitude * max_amplitude``, i.e.,
        proportional to the channel's signal strength.  Channels with no
        signal receive no noise.
    background_rate : float, optional
        Rate (Hz) of position-independent background spikes per shank.
        Set to 0 (default) to disable.
    noise_floor : float, optional
        Standard deviation of additive Gaussian noise applied to ALL channels
        on every spike, independent of signal.  Expressed as a fraction of
        *max_amplitude*.  Default is 0.0 (no floor noise).
    feature_transform : callable or None, optional
        Optional function ``(marks, channel_positions) -> transformed_marks``
        applied per-shank to the raw amplitude marks before output.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    ClusterlessSimOutput
        Decoder-ready output with one electrode per shank.  Each electrode's
        waveform features have shape ``(n_spikes, n_channels_per_shank)``
        (or the output width of *feature_transform*).

    """
    if place_field_means is None:
        place_field_means = PLACE_FIELD_MEANS.copy()
    place_field_means = np.atleast_1d(place_field_means).ravel()
    n_neurons = place_field_means.size

    seed_seq = np.random.SeedSequence(seed)
    seed_spikes, seed_marks, seed_bg = seed_seq.spawn(3)
    rng_spikes = np.random.default_rng(seed_spikes)
    rng_marks = np.random.default_rng(seed_marks)
    rng_bg = np.random.default_rng(seed_bg)

    # -- Position trajectory ---------------------------------------------------
    n_samples = int(n_runs * sampling_frequency * 2 * track_height / running_speed)
    position_time = simulate_time(n_samples, sampling_frequency)
    position_1d = simulate_position(position_time, track_height, running_speed)
    position = position_1d[:, np.newaxis]

    # -- Probe geometry --------------------------------------------------------
    shank_channel_positions = probe_config.make_channel_positions()
    n_shanks = probe_config.n_shanks
    n_ch_per_shank = probe_config.n_channels_per_shank

    # Place neurons near individual shanks (distributed round-robin) so that
    # each neuron has a clear "home" shank with strong signal and negligible
    # amplitude on distant shanks.
    all_channel_positions = np.vstack(shank_channel_positions)
    neuron_parts = []
    for neuron_idx in range(n_neurons):
        shank_idx = neuron_idx % n_shanks
        neuron_parts.append(
            make_neuron_locations(
                1,
                shank_channel_positions[shank_idx],
                probe_config.neuron_depth_range,
                probe_config.lateral_extent,
                rng=rng_spikes,
            )
        )
    neuron_locations = np.vstack(neuron_parts)

    # Compute amplitude templates using ALL channels so that the row-max
    # normalisation reflects each neuron's closest channel across the whole
    # probe.  Then slice by shank — neurons far from a shank will have low
    # amplitudes on that shank's channels.
    all_templates = compute_amplitude_falloff(
        neuron_locations,
        all_channel_positions,
        decay_constant=probe_config.decay_constant,
    )
    # Zero out channels below 10% of each neuron's global peak to produce
    # realistic sparse footprints.
    global_row_max = all_templates.max(axis=1, keepdims=True)
    all_templates = np.where(all_templates >= 0.1 * global_row_max, all_templates, 0.0)

    shank_templates = []
    offset = 0
    for _ in range(n_shanks):
        shank_templates.append(all_templates[:, offset : offset + n_ch_per_shank])
        offset += n_ch_per_shank

    # -- Generate spikes per neuron (shared across shanks) ---------------------
    neuron_spike_masks: list[np.ndarray] = []
    for neuron_idx in range(n_neurons):
        spikes = simulate_neuron_with_place_field(
            means=place_field_means[neuron_idx],
            position=position_1d,
            max_rate=max_firing_rate,
            variance=place_field_variance,
            sampling_frequency=sampling_frequency,
            rng=rng_spikes,
        )
        neuron_spike_masks.append(spikes > 0)

    # -- Build per-shank spike data --------------------------------------------
    all_spike_times: list[np.ndarray] = []
    all_spike_features: list[np.ndarray] = []

    for shank_idx in range(n_shanks):
        templates = shank_templates[shank_idx]  # (n_neurons, n_ch_per_shank)
        shank_times_parts: list[np.ndarray] = []
        shank_marks_parts: list[np.ndarray] = []

        for neuron_idx in range(n_neurons):
            spike_mask = neuron_spike_masks[neuron_idx]
            n_spikes = int(spike_mask.sum())
            if n_spikes == 0:
                continue

            # Skip neuron on this shank if its strongest template weight is
            # below 10% — neuron is too far from this shank to be recorded.
            peak_amp = templates[neuron_idx].max()
            if peak_amp < 0.1:
                continue

            times = position_time[spike_mask]
            template_scaled = templates[neuron_idx][np.newaxis, :] * max_amplitude
            noise_std_per_ch = amplitude_noise_std * np.abs(template_scaled)
            marks = (
                template_scaled
                + rng_marks.normal(0, 1, (n_spikes, n_ch_per_shank)) * noise_std_per_ch
            )
            if noise_floor > 0:
                marks += rng_marks.normal(
                    0, noise_floor * max_amplitude, (n_spikes, n_ch_per_shank)
                )
            shank_times_parts.append(times)
            shank_marks_parts.append(marks)

        # Background spikes for this shank
        if background_rate > 0:
            expected_bg = background_rate * position_time[-1]
            n_bg = rng_bg.poisson(expected_bg)
            if n_bg > 0:
                bg_times = rng_bg.uniform(position_time[0], position_time[-1], n_bg)
                bg_marks = rng_bg.uniform(0, max_amplitude, (n_bg, n_ch_per_shank))
                shank_times_parts.append(bg_times)
                shank_marks_parts.append(bg_marks)

        # Merge and sort
        if len(shank_times_parts) == 0:
            merged_times = np.empty(0, dtype=np.float64)
            merged_marks = np.empty((0, n_ch_per_shank), dtype=np.float64)
        else:
            merged_times = np.concatenate(shank_times_parts)
            merged_marks = np.concatenate(shank_marks_parts)
            sort_idx = np.argsort(merged_times)
            merged_times = merged_times[sort_idx]
            merged_marks = merged_marks[sort_idx]

        # Jitter duplicate times
        if merged_times.size > 1:
            eps = 1e-6 / sampling_frequency
            diffs = np.diff(merged_times)
            duplicates = diffs <= 0
            if duplicates.any():
                cumdup = np.zeros(len(diffs), dtype=int)
                for i in range(len(diffs)):
                    if duplicates[i]:
                        cumdup[i] = cumdup[i - 1] + 1 if i > 0 else 1
                merged_times[1:] += cumdup * eps

        # Apply feature transform per shank
        if feature_transform is not None:
            merged_marks = feature_transform(
                merged_marks, shank_channel_positions[shank_idx]
            )

        all_spike_times.append(merged_times.astype(np.float64))
        all_spike_features.append(merged_marks.astype(np.float64))

    # -- Build output ----------------------------------------------------------
    edges = np.arange(0.0, position_time[-1] + 1e-12, 1.0 / sampling_frequency)
    bin_widths = np.diff(edges)

    environment = Environment(place_bin_size=1.0)
    environment.fit_place_grid(position)

    return ClusterlessSimOutput(
        position_time=position_time.astype(np.float64),
        position=position.astype(np.float64),
        edges=edges.astype(np.float64),
        spike_times=all_spike_times,
        spike_waveform_features=all_spike_features,
        environment=environment,
        bin_widths=bin_widths.astype(np.float64),
    )
