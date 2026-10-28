# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Speed-dependent discrete transitions
#
# By default the non-local detector's discrete states (Local, No-Spike,
# Non-Local Continuous, Non-Local Fragmented) switch with fixed
# probabilities. Replay is more common when the animal is still, so the
# transitions can instead depend on a covariate such as speed:
# `DiscreteNonStationaryCustom` (or `DiscreteNonStationaryDiagonal`) models
# each row of the transition matrix as a multinomial logistic regression on
# a patsy `formula`, fit by EM.
#
# This notebook simulates the detector's four-state model with known
# speed-dependent transitions, fits stationary and speed-dependent
# transitions on the first half of the recording, and compares
#
# 1. the fitted transition curves, with 95% intervals, against the truth, and
# 2. how well each model predicts the held-out second half.
#
# The speed-dependent model uses the options that make the fitted curves
# reliable where data are sparse:
#
# - `quantile_bs(speed, upper_quantile=0.99)`: a cubic B-spline of speed whose
#   upper boundary sits at the 99th percentile of the fitted speeds, so the
#   fastest 1% (and tracking glitches) take the boundary's value instead of
#   bending the last spline segment.
# - `smoothness="reml"`: a penalty on the curvature of the spline whose
#   strength is chosen for each row in every EM M-step by maximizing its
#   Laplace approximate marginal likelihood (REML).

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from non_local_detector import (
    DiscreteNonStationaryCustom,
    Environment,
    NonLocalSortedSpikesDetector,
    time_edges_from_centers,
)
from non_local_detector.models._defaults import _ModelDefaults

FORMULA = "1 + quantile_bs(speed, upper_quantile=0.99)"
SMOOTHNESS = "reml"

# %% [markdown]
# ## Simulate a recording with known speed-dependent transitions
#
# The animal runs laps on a 180 cm track, speeding up from a standstill to
# ~30 cm/s (one lap in ten is a ~70 cm/s sprint) and slowing down again,
# then pauses 1-6 s at each end. The observed speed carries smoothed
# tracking noise (SD 1.5 cm/s), so, as in real recordings, it is never
# exactly 0 and every speed between rest and running is visited. Thirty
# speed samples are tracking glitches of 300-1500 cm/s. The true transitions
# depend on the observed speed: at rest the animal enters non-local states
# five times as often, and Non-Local Continuous events last longer.
# 40 place cells fire from the position the current state represents: the
# animal's position (Local), nothing (No-Spike), a random walk (Non-Local
# Continuous) or a random position (Non-Local Fragmented).

# %%
SAMPLING_FREQUENCY = 500.0
TRACK_LENGTH = 180.0
STATE_NAMES = ["Local", "No-Spike", "Non-Local Continuous", "Non-Local Fragmented"]


def true_transition(speed):
    """True transition matrices, shape (n_time, 4, 4)."""
    rest = np.exp(-speed / 5.0)
    transition = np.zeros((speed.size, 4, 4))
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


rng = np.random.default_rng(0)
n_time = int(16 * 60 * SAMPLING_FREQUENCY)
time = np.arange(n_time) / SAMPLING_FREQUENCY

# Each lap's speed follows a half sine from 0 up to its peak and back.
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
position = np.clip(np.cumsum(velocity) / SAMPLING_FREQUENCY, 0, TRACK_LENGTH)
noise = gaussian_filter1d(rng.normal(size=n_time), 0.1 * SAMPLING_FREQUENCY)
speed = np.abs(velocity + noise * 1.5 / noise.std())
speed[rng.choice(n_time, 30, replace=False)] = rng.uniform(300, 1500, 30)

transition = true_transition(speed)
states = np.zeros(n_time, dtype=int)
represented = position.copy()
draws = rng.uniform(size=n_time)
for index in range(1, n_time):
    cumulative = np.cumsum(transition[index - 1, states[index - 1]])
    states[index] = min(np.searchsorted(cumulative, draws[index]), 3)
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
# Jitter spikes within their sample-centred decode bin.
spike_times = [
    np.sort(
        np.repeat(time, counts[:, neuron])
        + rng.uniform(-0.5, 0.5, counts[:, neuron].sum()) / SAMPLING_FREQUENCY
    )
    for neuron in range(centers.size)
]

fraction = np.bincount(states, minlength=4) / n_time
for name, value in zip(STATE_NAMES, fraction, strict=True):
    print(f"{name:22s} {value:.3f} of the time")
print(f"99th percentile of speed: {np.quantile(speed, 0.99):.1f} cm/s")

# %% [markdown]
# ## Fit on the first half
#
# Both models start from the detector's default transition matrix and keep
# its default prior (stickiness on Local and No-Spike, No-Spike row frozen).
# The encoding model is fit once, from the running periods, and held fixed
# so the comparison isolates the transitions.

# %%
half = n_time // 2
train, test = slice(0, half), slice(half, n_time)


def spikes_in(window):
    start, stop = time[window][0], time[window][-1]
    half_bin = 0.5 / SAMPLING_FREQUENCY
    return [
        times[(times >= start - half_bin) & (times < stop + half_bin)]
        for times in spike_times
    ]


default_values = _ModelDefaults.non_local_defaults()[
    "discrete_transition_type"
]().values
environment = Environment(place_bin_size=2.0, position_range=[(0, TRACK_LENGTH)])
models = {
    "stationary": NonLocalSortedSpikesDetector(environments=environment),
    "speed-dependent": NonLocalSortedSpikesDetector(
        environments=environment,
        discrete_transition_type=DiscreteNonStationaryCustom(
            values=default_values, formula=FORMULA, smoothness=SMOOTHNESS
        ),
    ),
}
for name, detector in models.items():
    detector.estimate_parameters(
        position_time=time[train],
        position=position[train],
        spike_times=spikes_in(train),
        time_edges=time_edges_from_centers(time[train]),
        is_training=speed[train] > 4.0,
        estimate_encoding_model=False,
        discrete_transition_covariate_data=(
            {"speed": speed[train]} if name == "speed-dependent" else None
        ),
    )
    print(f"{name}: {detector.n_iter_} EM iterations, converged={detector.converged_}")

speed_dependent = models["speed-dependent"]
print(
    "REML smoothness per row:",
    np.array2string(speed_dependent.discrete_transition_smoothness_, precision=3),
)

# %% [markdown]
# ## Fitted transition curves against the truth
#
# `discrete_transition_intervals` evaluates the fitted transitions at new
# covariate values, with pointwise 95% intervals from the Laplace
# covariance of the last M-step. The intervals treat that M-step's expected
# transitions as known, so they leave out the uncertainty in which state the
# animal was in. The No-Spike row is frozen by default, and the default
# stickiness prior (1e5 pseudo-counts on Local) pulls Local's exits toward
# staying, so its curve sits below the truth by design.

# %%
grid = np.linspace(0.0, np.quantile(speed, 0.999), 200)
fitted, lower, upper = speed_dependent.discrete_transition_intervals(
    {"speed": grid}, coverage=0.95
)
truth = true_transition(grid)
stationary = models["stationary"].discrete_state_transitions_

panels = [
    ("Local → non-local", [(0, 2), (0, 3)]),
    ("Non-Local Continuous: stay", [(2, 2)]),
    ("Non-Local Continuous → Local", [(2, 0)]),
]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
for ax, (title, pairs) in zip(axes, panels, strict=True):

    def total(array, pairs=pairs):
        return sum(array[..., i, j] for i, j in pairs)

    if len(pairs) == 1:
        ax.fill_between(
            grid, total(lower), total(upper), alpha=0.25, label="95% interval"
        )
    ax.plot(grid, total(fitted), label="speed-dependent fit")
    ax.plot(grid, total(truth), "k--", label="truth")
    ax.axhline(total(stationary), color="gray", lw=1, label="stationary fit")
    ax.axvline(np.quantile(speed[train], 0.99), color="gray", ls=":", lw=1)
    ax.set(title=title, xlabel="speed (cm/s)", ylabel="probability per 2 ms bin")
axes[0].legend(fontsize=8)
rug = axes[1].twinx()
rug.hist(speed[train], bins=80, range=(0, grid[-1]), color="gray", alpha=0.2)
rug.set_yticks([])

# %% [markdown]
# The dotted line marks the 99th percentile of the fitted speeds: beyond it
# the curves are flat by construction. The summed Local → non-local curve
# has no interval because the intervals are for single transitions.
#
# The Non-Local Continuous curves sit about 0.01 from the truth between 5
# and 25 cm/s, mostly outside their intervals. The same spline and REML
# smoothness fit directly to the simulated states, which EM does not see,
# stay within 0.006 of the truth (about two standard errors for the
# 2,500-5,000 Non-Local Continuous samples in each speed band), with 5 or 10
# knots. So most of the gap is EM's uncertainty about which state the
# animal was in, which the intervals leave out.

# %% [markdown]
# ## Held-out prediction
#
# Each model decodes the second half, with the speed-dependent model's
# transitions evaluated at that half's speeds. A higher marginal
# log-likelihood means the transitions predict the held-out spikes better;
# differences of a few nats are noise, differences of hundreds are not.

# %%
held_out = {}
for name, detector in models.items():
    results = detector.predict(
        spike_times=spikes_in(test),
        time_edges=time_edges_from_centers(time[test]),
        position_time=time[test],
        position=position[test],
        discrete_transition_covariate_data=(
            {"speed": speed[test]} if name == "speed-dependent" else None
        ),
    )
    probabilities = results.acausal_state_probabilities.values
    held_out[name] = {
        "marginal log-likelihood": float(
            np.ravel(results.attrs["marginal_log_likelihoods"])[-1]
        ),
        "state accuracy": float(np.mean(probabilities.argmax(axis=1) == states[test])),
    }
for name, scores in held_out.items():
    print(
        f"{name:16s} log-likelihood {scores['marginal log-likelihood']:12.1f}   "
        f"state accuracy {scores['state accuracy']:.4f}"
    )
gain = (
    held_out["speed-dependent"]["marginal log-likelihood"]
    - held_out["stationary"]["marginal log-likelihood"]
)
print(f"speed-dependent - stationary: {gain:+.1f} nats")
