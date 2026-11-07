"""Each environment of a multi-environment detector is encoded from its own samples.

A detector fit on a recording that spans two environments must give each
environment the encoding model a detector fit on that environment's samples
alone would give: same grid, occupancy, place fields or ground intensity, and
mean rates. The environments differ in extent and the cells remap between
them, so occupancy from the other environment, or a grid fit on its
positions, changes the fit. (A spike from the other environment carries zero
interpolated exposure weight, so it cannot leak in.) Decoding then attributes
each environment's spikes to it.
"""

import numpy as np
import pytest

from non_local_detector import (
    ClusterlessDecoder,
    DiscreteStationaryDiagonal,
    Environment,
    RandomWalk,
    SortedSpikesDecoder,
    Uniform,
    UniformInitialConditions,
)
from non_local_detector.analysis import (
    get_position_posterior,
    get_spatial_posterior,
    maximum_a_posteriori_estimate,
)
from non_local_detector.observation_models import ObservationModel
from non_local_detector.time_edges import time_edges_from_centers

pytestmark = [
    pytest.mark.integration,
    # The animal jumps between environments once, between two samples.
    pytest.mark.filterwarnings("ignore:Tracking moves up to"),
]

SAMPLING_FREQUENCY = 100.0
SECONDS_PER_ENVIRONMENT = 60.0
NAMES = ("A", "B")
TRACK_LENGTH = {"A": 100.0, "B": 60.0}
# Field centers per environment, cm; the cells remap (the last is silent in B).
FIELD_CENTERS = {"A": [20.0, 40.0, 60.0, 80.0], "B": [50.0, 10.0, 30.0, np.nan]}
# Spikes this close to the switch between environments are dropped: a spike's
# exposure weight is interpolated between samples, so one beside the switch
# would count fractionally in the masked fit but fully in the subset fit.
SWITCH_MARGIN = 0.05


@pytest.fixture(scope="module")
def two_environment_recording():
    rng = np.random.default_rng(0)
    n_samples = int(SECONDS_PER_ENVIRONMENT * SAMPLING_FREQUENCY)
    time = np.arange(2 * n_samples) / SAMPLING_FREQUENCY
    labels = np.repeat(NAMES, n_samples)
    position = np.empty(time.size)
    rates = np.empty((time.size, 4))
    for name in NAMES:
        is_here = labels == name
        length = TRACK_LENGTH[name]
        position[is_here] = length / 2 + (length / 2 - 1) * np.sin(
            2 * np.pi * time[is_here] / 7.3
        )
        centers = np.asarray(FIELD_CENTERS[name])
        field = 20.0 * np.exp(-0.5 * ((position[is_here, None] - centers) / 5.0) ** 2)
        rates[is_here] = 0.2 + np.nan_to_num(field)
    counts = rng.poisson(rates / SAMPLING_FREQUENCY)
    switch = time[n_samples]
    spike_times = []
    for cell in range(rates.shape[1]):
        times = (
            np.repeat(time, counts[:, cell])
            + rng.uniform(-0.5, 0.5, counts[:, cell].sum()) / SAMPLING_FREQUENCY
        )
        keep = (
            (times >= 0)
            & (times <= time[-1])
            & (np.abs(times - switch) > SWITCH_MARGIN)
        )
        spike_times.append(np.sort(times[keep]))
    # Two electrodes of two cells each, with 2-D marks per cell.
    electrode_times, electrode_marks = [], []
    for cells in ((0, 1), (2, 3)):
        times = np.concatenate([spike_times[c] for c in cells])
        marks = np.concatenate(
            [rng.normal(40.0 * c, 6.0, (spike_times[c].size, 2)) for c in cells]
        )
        order = np.argsort(times)
        electrode_times.append(times[order])
        electrode_marks.append(marks[order])
    return {
        "time": time,
        "position": position[:, None],
        "labels": labels,
        "spike_times": spike_times,
        "electrode_times": electrode_times,
        "electrode_marks": electrode_marks,
    }


def _detector(cls, algorithm_key, algorithm, names):
    """One state per environment."""
    n_states = len(names)
    return cls(
        environments=[
            Environment(environment_name=name, place_bin_size=4.0) for name in names
        ],
        observation_models=[ObservationModel(environment_name=name) for name in names],
        continuous_transition_types=[
            [Uniform(source, destination) for destination in names] for source in names
        ],
        discrete_initial_conditions=np.full(n_states, 1.0 / n_states),
        continuous_initial_conditions_types=[UniformInitialConditions() for _ in names],
        discrete_transition_type=DiscreteStationaryDiagonal(
            diagonal_values=np.full(n_states, 0.98)
        ),
        discrete_transition_stickiness=np.zeros(n_states),
        state_names=list(names),
        **{
            algorithm_key: algorithm,
            f"{algorithm_key}_params": {"disable_progress_bar": True},
        },
    )


def _within(times, start, stop):
    return times[(times >= start) & (times <= stop)]


@pytest.mark.parametrize(
    ("cls", "algorithm_key", "algorithm", "compared"),
    [
        (
            SortedSpikesDecoder,
            "sorted_spikes_algorithm",
            "sorted_spikes_kde",
            ("occupancy", "place_fields", "mean_rates"),
        ),
        (
            ClusterlessDecoder,
            "clusterless_algorithm",
            "clusterless_kde",
            ("occupancy", "summed_ground_process_intensity", "mean_rates"),
        ),
    ],
)
def test_each_environment_matches_a_fit_on_its_samples_alone(
    cls, algorithm_key, algorithm, compared, two_environment_recording
):
    data = two_environment_recording
    clusterless = cls is ClusterlessDecoder
    spikes = (
        (data["electrode_times"], data["electrode_marks"])
        if clusterless
        else (data["spike_times"],)
    )
    both = _detector(cls, algorithm_key, algorithm, NAMES)
    both.fit(data["time"], data["position"], *spikes, environment_labels=data["labels"])

    for name in NAMES:
        is_here = data["labels"] == name
        time = data["time"][is_here]
        if clusterless:
            masks = [
                (times >= time[0]) & (times <= time[-1])
                for times in data["electrode_times"]
            ]
            alone_spikes = (
                [t[m] for t, m in zip(data["electrode_times"], masks, strict=True)],
                [f[m] for f, m in zip(data["electrode_marks"], masks, strict=True)],
            )
        else:
            alone_spikes = (
                [_within(t, time[0], time[-1]) for t in data["spike_times"]],
            )
        alone = _detector(cls, algorithm_key, algorithm, (name,))
        alone.fit(time, data["position"][is_here], *alone_spikes)

        joint_model = both.encoding_model_[(name, 0)]
        alone_model = alone.encoding_model_[(name, 0)]
        np.testing.assert_array_equal(
            joint_model["environment"].place_bin_centers_,
            alone_model["environment"].place_bin_centers_,
        )
        np.testing.assert_array_equal(
            joint_model["environment"].is_track_interior_,
            alone_model["environment"].is_track_interior_,
        )
        for key in compared:
            np.testing.assert_allclose(
                np.asarray(joint_model[key]),
                np.asarray(alone_model[key]),
                rtol=1e-5,
                atol=0.0,
                err_msg=f"{name}: {key}",
            )


def test_position_is_read_one_environment_at_a_time(two_environment_recording):
    """Environments A and B share no grid, and their bins must not be pooled."""
    data = two_environment_recording
    detector = _detector(
        SortedSpikesDecoder, "sorted_spikes_algorithm", "sorted_spikes_kde", NAMES
    )
    detector.fit(
        data["time"],
        data["position"],
        data["spike_times"],
        environment_labels=data["labels"],
    )
    detector.set_params(sampling_frequency=SAMPLING_FREQUENCY)
    time = data["time"][:100]
    results = detector.predict(
        spike_times=data["spike_times"],
        time_edges=time_edges_from_centers(time),
        position_time=time,
        position=data["position"][:100],
    )

    with pytest.raises(ValueError, match="environments"):
        get_position_posterior(results.acausal_posterior)

    in_a = results.isel(state_bins=results.environment == "A")
    position = maximum_a_posteriori_estimate(in_a.acausal_posterior)
    assert np.isin(position, detector.environments_[0].place_bin_centers_[:, 0]).all()
    np.testing.assert_array_equal(
        position,
        maximum_a_posteriori_estimate(
            get_spatial_posterior(results.acausal_posterior, states=["A"])
        ),
    )


_FAMILIES = [
    pytest.param(SortedSpikesDecoder, "sorted_spikes_algorithm", id="sorted"),
    pytest.param(ClusterlessDecoder, "clusterless_algorithm", id="clusterless"),
]


@pytest.mark.parametrize(("cls", "algorithm_key"), _FAMILIES)
def test_decoding_favors_the_environment_the_spikes_came_from(
    cls, algorithm_key, two_environment_recording
):
    """A 10 s window recorded in A is decoded as A, and one in B as B.

    Within an environment the represented position follows a random walk and
    the environment is sticky (0.999 per 10 ms bin), as when the animal stays
    in one environment for minutes. The mean posterior of the true
    environment over the window was at least 0.91 (B, sorted; A reached
    0.998). Fit with the environment labels swapped, the same windows give
    the true environment at most 0.085.
    """
    data = two_environment_recording
    clusterless = cls is ClusterlessDecoder
    spikes = (
        (data["electrode_times"], data["electrode_marks"])
        if clusterless
        else (data["spike_times"],)
    )
    algorithm = "clusterless_kde" if clusterless else "sorted_spikes_kde"
    detector = _detector(cls, algorithm_key, algorithm, NAMES)
    detector.set_params(
        sampling_frequency=SAMPLING_FREQUENCY,
        discrete_transition_type=DiscreteStationaryDiagonal(
            diagonal_values=np.full(len(NAMES), 0.999)
        ),
        continuous_transition_types=[
            [
                RandomWalk(source)
                if source == destination
                else Uniform(source, destination)
                for destination in NAMES
            ]
            for source in NAMES
        ],
    )
    detector.fit(
        data["time"], data["position"], *spikes, environment_labels=data["labels"]
    )

    for name, start in (("A", 20.0), ("B", 80.0)):
        edges = start + np.arange(1001) / SAMPLING_FREQUENCY
        results = detector.predict(*spikes, time_edges=edges)
        probability = results.acausal_state_probabilities.sel(states=name)
        assert float(probability.mean()) > 0.8, name
