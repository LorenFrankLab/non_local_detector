"""End-to-end recovery of speed-dependent transitions by the detector's EM.

Simulates the four-state non-local model with sorted spikes from Gaussian
place fields and speed-dependent true transitions (non-local events more
likely at rest), with laps, pauses at zero speed and tracking glitches in the
speed covariate. EM fits the transitions with the encoding model fixed and no
transition prior, and is compared with an oracle: the same basis and penalty
fit directly to the true state sequence, the best this basis can do with
this data.

Detector-level interval coverage is not asserted: the Laplace intervals
condition on the E-step's expected transitions and leave out basis bias, so
on these recordings the 95% intervals contained the true curve at only
43-78% of grid points (seeds 1-5). Coverage with known states is tested in
``test_covariate_transition_smoothing.py``.
"""

import numpy as np
import pytest

from non_local_detector import (
    DiscreteNonStationaryCustom,
    Environment,
    NonLocalSortedSpikesDetector,
    time_edges_from_centers,
)
from non_local_detector.discrete_state_transitions import (
    _fit_covariate_transition,
    _smoothness_matrix,
    predict_discrete_state_transitions,
)

SAMPLING_FREQUENCY = 500.0
TRACK_LENGTH = 180.0
N_STATES = 4
FORMULA = "1 + quantile_bs(speed, upper_quantile=0.99)"


def _true_transition(speed: np.ndarray) -> np.ndarray:
    """Rest: more non-local events; running: Local is stickier and
    non-local events end sooner. Shape (n_time, n_states, n_states)."""
    rest = np.exp(-speed / 5.0)
    transition = np.zeros((speed.size, N_STATES, N_STATES))
    enter_non_local = 0.002 + 0.008 * rest
    transition[:, 0, 0] = 1 - 0.002 - 2 * enter_non_local
    transition[:, 0, 1] = 0.002
    transition[:, 0, 2] = transition[:, 0, 3] = enter_non_local
    transition[:, 1] = [0.01, 0.97, 0.01, 0.01]
    transition[:, 2, 2] = 0.95 + 0.03 * rest
    transition[:, 2, 0] = 1 - transition[:, 2, 2] - 0.01
    transition[:, 2, 1] = transition[:, 2, 3] = 0.005
    transition[:, 3] = [0.03, 0.01, 0.01, 0.95]
    return transition


def _simulate(seed: int, duration_s: float) -> dict:
    """Laps with pauses, sprints and speed glitches; spikes from the true
    states. Spikes are jittered within their sample-centred decode bin."""
    rng = np.random.default_rng(seed)
    n_time = int(duration_s * SAMPLING_FREQUENCY)
    time = np.arange(n_time) / SAMPLING_FREQUENCY
    position = np.empty(n_time)
    x, direction, pause, lap_speed = 0.0, 1, 0, 30.0
    for index in range(n_time):
        if pause > 0:
            pause -= 1
        else:
            x += direction * rng.gamma(8.0, lap_speed / 8.0) / SAMPLING_FREQUENCY
            if x >= TRACK_LENGTH or x <= 0:
                x = np.clip(x, 0, TRACK_LENGTH)
                direction *= -1
                pause = int(rng.uniform(1, 6) * SAMPLING_FREQUENCY)
                lap_speed = 70.0 if rng.uniform() < 0.1 else 30.0
        position[index] = x
    true_speed = np.abs(np.gradient(position, time))
    speed = true_speed.copy()
    speed[rng.choice(n_time, 10, replace=False)] = rng.uniform(300, 1500, 10)

    transition = _true_transition(true_speed)
    states = np.zeros(n_time, dtype=int)
    represented = position.copy()
    draws = rng.uniform(size=n_time)
    for index in range(1, n_time):
        cumulative = np.cumsum(transition[index - 1, states[index - 1]])
        states[index] = min(np.searchsorted(cumulative, draws[index]), N_STATES - 1)
        if states[index] == 2 and states[index - 1] == 2:
            represented[index] = np.clip(
                represented[index - 1] + rng.normal(0, np.sqrt(6.0)), 0, TRACK_LENGTH
            )
        elif states[index] in (2, 3):
            represented[index] = rng.uniform(0, TRACK_LENGTH)
    centers = np.linspace(0, TRACK_LENGTH, 40)
    rates = 25.0 * np.exp(-0.5 * ((represented[:, None] - centers) / 10.0) ** 2)
    rates[states == 1] = 0.0
    counts = rng.poisson(rates / SAMPLING_FREQUENCY)
    spike_times = [
        np.sort(
            np.repeat(time, counts[:, neuron])
            + rng.uniform(-0.5, 0.5, counts[:, neuron].sum()) / SAMPLING_FREQUENCY
        )
        for neuron in range(centers.size)
    ]
    return {
        "time": time,
        "position": position,
        "speed": speed,
        "true_speed": true_speed,
        "states": states,
        "spike_times": spike_times,
    }


def _curves(transitions: np.ndarray) -> dict[str, np.ndarray]:
    """The speed effects: entering a non-local state from Local, staying in
    Non-Local Continuous, and returning from it to Local."""
    return {
        "enter_non_local": transitions[:, 0, 2] + transitions[:, 0, 3],
        "non_local_stay": transitions[:, 2, 2],
        "non_local_to_local": transitions[:, 2, 0],
    }


def _fit_recovery(seed: int, duration_s: float, formula: str = FORMULA) -> dict:
    """EM and oracle transition curves on a speed grid for one recording."""
    data = _simulate(seed=seed, duration_s=duration_s)
    transition_type = DiscreteNonStationaryCustom(
        values=_true_transition(np.array([10.0]))[0],
        formula=formula,
        smoothness="reml",
    )
    detector = NonLocalSortedSpikesDetector(
        environments=Environment(
            place_bin_size=4.0, position_range=[(0, TRACK_LENGTH)]
        ),
        discrete_transition_type=transition_type,
        discrete_transition_stickiness=np.zeros(N_STATES),
        frozen_discrete_transition_rows=None,
    )
    detector.estimate_parameters(
        position_time=data["time"],
        position=data["position"],
        spike_times=data["spike_times"],
        time_edges=time_edges_from_centers(data["time"]),
        is_training=data["states"] == 0,
        estimate_encoding_model=False,
        max_iter=30,
        discrete_transition_covariate_data={"speed": data["speed"]},
    )

    design = detector.discrete_transition_design_matrix_
    states = data["states"]
    response = np.zeros((states.size - 1, N_STATES, N_STATES))
    response[np.arange(states.size - 1), states[:-1], states[1:]] = 1.0
    oracle = _fit_covariate_transition(
        detector.discrete_transition_coefficients_,
        np.asarray(design, dtype=float),
        response,
        smoothness="reml",
        smoothness_matrix=_smoothness_matrix(design.design_info),
    )

    lower, upper = np.quantile(data["true_speed"], [0.05, 0.95])
    grid = {"speed": np.linspace(lower, upper, 50)}
    return {
        "detector": detector,
        "grid": grid["speed"],
        "truth": _curves(_true_transition(grid["speed"])),
        "em": _curves(
            predict_discrete_state_transitions(
                design, detector.discrete_transition_coefficients_, grid
            )
        ),
        "oracle": _curves(
            predict_discrete_state_transitions(design, oracle.coefficients, grid)
        ),
    }


@pytest.fixture(scope="module")
def recovery():
    return _fit_recovery(seed=1, duration_s=240.0)


@pytest.mark.integration
@pytest.mark.slow
class TestSpeedDependentTransitionRecovery:
    def test_em_is_monotone_and_converges(self, recovery):
        detector = recovery["detector"]
        assert detector.em_monotonicity_violations_ == []
        assert detector.converged_
        assert np.all(np.isfinite(detector.discrete_state_transitions_))

    def test_em_recovers_the_direction_of_each_speed_effect(self, recovery):
        em, grid = recovery["em"], recovery["grid"]
        rest, run = grid < 2.0, grid > 20.0
        assert em["enter_non_local"][rest].mean() > em["enter_non_local"][run].mean()
        assert em["non_local_stay"][rest].mean() > em["non_local_stay"][run].mean()
        assert (
            em["non_local_to_local"][run].mean() > em["non_local_to_local"][rest].mean()
        )

    @pytest.mark.parametrize(
        ("curve", "tolerance"),
        # About twice the worst EM-oracle gap over seeds 1-5 (0.0016, 0.017,
        # 0.018).
        [
            ("enter_non_local", 0.004),
            ("non_local_stay", 0.035),
            ("non_local_to_local", 0.035),
        ],
    )
    def test_em_matches_the_oracle(self, recovery, curve, tolerance):
        """EM's curves are about as close to the truth as the same basis
        fit to the true states."""
        gap = np.max(np.abs(recovery["em"][curve] - recovery["oracle"][curve]))
        assert gap < tolerance
