"""End-to-end recovery of speed-dependent transitions by the detector's EM.

Simulates the four-state non-local model with sorted spikes from Gaussian
place fields and speed-dependent true transitions (non-local events more
likely at rest), with laps that speed up and slow down gradually, pauses,
tracking noise and glitches in the speed covariate. EM fits the transitions with the encoding model fixed and no
transition prior, and is compared with an oracle: the same basis and penalty
fit directly to the true state sequence, the best this basis can do with
this data.

Detector-level interval coverage is not asserted: the Laplace intervals
condition on the E-step's expected transitions and on REML's chosen
smoothness, so on these recordings the 95% intervals contained the true
curve at 54-93% of grid points (seeds 1-5; seed 2's REML drew the
Non-Local Continuous row nearly straight). Coverage with known states is
tested in ``test_covariate_transition_smoothing.py``.
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

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


def _simulate_laps(
    rng: np.random.Generator, duration_s: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Laps with gradual speed changes, pauses and tracking noise.

    Each lap's speed follows a half sine from 0 up to a peak (about 30 cm/s,
    one lap in ten about 70 cm/s) and back, so every speed in between is
    visited, as in real recordings; the animal then pauses 1-6 s. Observed
    speed adds smoothed tracking noise (SD 1.5 cm/s, 100 ms smoothing), so it
    is never exactly 0. Returns time, position and speed, each (n_time,).
    """
    n_time = int(duration_s * SAMPLING_FREQUENCY)
    segments = []
    direction = 1.0
    while sum(map(len, segments)) < n_time:
        peak = (70.0 if rng.uniform() < 0.1 else 30.0) * rng.uniform(0.8, 1.2)
        # A half-sine speed profile covers 2 * peak * T / pi in T seconds.
        lap_s = np.pi * TRACK_LENGTH / (2 * peak)
        n_lap = int(lap_s * SAMPLING_FREQUENCY)
        segments.append(direction * peak * np.sin(np.pi * np.arange(n_lap) / n_lap))
        segments.append(np.zeros(int(rng.uniform(1, 6) * SAMPLING_FREQUENCY)))
        direction *= -1
    velocity = np.concatenate(segments)[:n_time]
    time = np.arange(n_time) / SAMPLING_FREQUENCY
    position = np.clip(np.cumsum(velocity) / SAMPLING_FREQUENCY, 0, TRACK_LENGTH)
    noise = gaussian_filter1d(rng.normal(size=n_time), 0.1 * SAMPLING_FREQUENCY)
    speed = np.abs(velocity + noise * 1.5 / noise.std())
    return time, position, speed


def _simulate(seed: int, duration_s: float) -> dict:
    """Laps (see ``_simulate_laps``) with speed glitches; states from the true
    transitions at the observed speed; spikes from the true states, jittered
    within their sample-centred decode bin."""
    rng = np.random.default_rng(seed)
    time, position, speed = _simulate_laps(rng, duration_s)
    n_time = time.size
    speed[rng.choice(n_time, 10, replace=False)] = rng.uniform(300, 1500, 10)

    transition = _true_transition(speed)
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

    lower, upper = np.quantile(data["speed"], [0.05, 0.95])
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
        # About twice the worst EM-oracle gap over seeds 1-5 (0.0024, 0.0127,
        # 0.0125).
        [
            ("enter_non_local", 0.005),
            ("non_local_stay", 0.025),
            ("non_local_to_local", 0.025),
        ],
    )
    def test_em_matches_the_oracle(self, recovery, curve, tolerance):
        """EM's curves are about as close to the truth as the same basis
        fit to the true states."""
        gap = np.max(np.abs(recovery["em"][curve] - recovery["oracle"][curve]))
        assert gap < tolerance
