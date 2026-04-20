# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: non-local-detector
#     language: python
#     name: python3
# ---

# %%
from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

(
    speed,
    position,
    spike_times,
    time,
    event_times,
    sampling_frequency,
    is_event,
    place_fields,
) = make_simulated_data(n_neurons=30)

# %%
from non_local_detector import NonLocalSortedSpikesDetector

detector = NonLocalSortedSpikesDetector(
    sorted_spikes_algorithm="sorted_spikes_kde",
    non_local_position_penalty=1.0,
    non_local_penalty_std=5.0,
    local_position_std=1.0,
)

is_training = ~is_event

results = detector.estimate_parameters(
    position_time=time,
    position=position,
    spike_times=spike_times,
    is_training=is_training,
    time=time,
    store_log_likelihood=True,
)
results

# %%
most_likely_sequence = detector.most_likely_sequence(
    position_time=time,
    position=position,
    spike_times=spike_times,
    time=time,
)
most_likely_sequence

# %%
import matplotlib.pyplot as plt

plt.scatter(time, most_likely_sequence.position)
plt.xlim(event_times[0] + [-0.200, 0.200])

# %%
from non_local_detector.visualization import plot_non_local_model

plot_non_local_model(
    time,
    position,
    spike_times,
    speed,
    detector,
    results,
    figsize=(20, 10),
)

# %%
detector.environments[0].place_bin_size

# %%
detector.initial_conditions_.shape

# %%
detector.initial_conditions_.sum()

# %%
detector.continuous_state_transitions_.sum(axis=1)

# %%
detector.environments[0].place_bin_centers_.shape

# %%
detector.encoding_model_[("", 0)].keys()

# %%
detector.encoding_model_[("", 0)]["mean_rates"]

# %%
from non_local_detector.visualization import plot_non_local_model

plot_non_local_model(
    time,
    position,
    spike_times,
    speed,
    detector,
    results,
    figsize=(20, 10),
    time_slice=slice(event_times[0][0] - 0.2, event_times[0][1] + 0.2),
)

# %%
import jax
import jax.numpy as jnp


@jax.jit
def hmm_posterior_mode(
    initial_distribution,
    transition_matrix,
    log_likelihoods,
):
    r"""Compute the most likely state sequence. This is called the Viterbi algorithm.

    Args:
        initial_distribution: $p(z_1 \mid u_1, \theta)$
        transition_matrix: $p(z_{t+1} \mid z_t, u_t, \theta)$
        log_likelihoods: $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$.
        transition_fn: function that takes in an integer time index and returns a $K \times K$ transition matrix.

    Returns:
        most likely state sequence

    """
    # Run the backward pass
    def _backward_pass(best_next_score, t):
        scores = jnp.log(transition_matrix) + best_next_score + log_likelihoods[t + 1]
        best_next_state = jnp.argmax(scores, axis=1)
        best_next_score = jnp.max(scores, axis=1)
        return best_next_score, best_next_state

    num_timesteps, num_states = log_likelihoods.shape
    best_second_score, rev_best_next_states = jax.lax.scan(
        _backward_pass, jnp.zeros(num_states), jnp.arange(num_timesteps - 2, -1, -1)
    )
    best_next_states = rev_best_next_states[::-1]

    # Run the forward pass
    def _forward_pass(state, best_next_state):
        next_state = best_next_state[state]
        return next_state, next_state

    first_state = jnp.argmax(
        jnp.log(initial_distribution) + log_likelihoods[0] + best_second_score
    )
    _, states = jax.lax.scan(_forward_pass, first_state, best_next_states)

    return jnp.concatenate([jnp.array([first_state]), states])


# %%
import numpy as np

initial_distribution=detector.initial_conditions_
transition_matrix=(
    detector.discrete_state_transitions_[
        np.ix_(detector.state_ind_, detector.state_ind_)
    ]
    * detector.continuous_state_transitions_
)
log_likelihoods=detector.log_likelihood_

def _backward_pass(best_next_score, t):
    scores = jnp.log(transition_matrix) + best_next_score + log_likelihoods[t + 1]
    best_next_state = jnp.argmax(scores, axis=1)
    best_next_score = jnp.max(scores, axis=1)
    return best_next_score, best_next_state

num_timesteps, num_states = log_likelihoods.shape
best_second_score, rev_best_next_states = jax.lax.scan(
    _backward_pass, jnp.zeros(num_states), jnp.arange(num_timesteps - 2, -1, -1)
)
best_next_states = rev_best_next_states[::-1]

# %%
best_second_score.shape, rev_best_next_states.shape

# %%
best_second_score2, best_next_states2 = jax.lax.scan(
    _backward_pass, jnp.zeros(num_states), jnp.arange(num_timesteps - 1), reverse=True
)

# %%
np.allclose(best_next_states, best_next_states2)

# %%
import numpy as np

most_likely_state_id = hmm_posterior_mode(
    initial_distribution=detector.initial_conditions_,
    transition_matrix=(
        detector.discrete_state_transitions_[
            np.ix_(detector.state_ind_, detector.state_ind_)
        ]
        * detector.continuous_state_transitions_
    ),
    log_likelihoods=detector.log_likelihood_,
)
most_likely_state = results.state_bins[most_likely_state_id]

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
axes[0].scatter(results.time, most_likely_state.to_dataframe().position.values, s=1)
axes[1].plot(time, most_likely_state_id)
for state_prob, state_name in zip(
    results.acausal_state_probabilities.values.T, results.states.values, strict=False
):
    axes[2].plot(time, state_prob, label=state_name)

axes[2].legend()
plt.xlim((6, 7))

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
axes[0].scatter(results.time, most_likely_state.to_dataframe().position.values, s=1)
axes[1].scatter(time, most_likely_state_id, s=10)
axes[1].axhline(0, color="black", linestyle="--")
for state_prob, state_name in zip(
    results.acausal_state_probabilities.values.T, results.states.values, strict=False
):
    axes[2].plot(time, state_prob, label=state_name)

axes[2].legend()
plt.xlim((32.4, 32.8))

# %%
detector.plot_discrete_state_transition()


# %%
@jax.jit
def hmm_posterior_mode2(
    initial_distribution,
    transition_matrix,
    log_likelihoods,
):
    r"""Compute the most likely state sequence. This is called the Viterbi algorithm.

    Args:
        initial_distribution: $p(z_1 \mid u_1, \theta)$
        transition_matrix: $p(z_{t+1} \mid z_t, u_t, \theta)$
        log_likelihoods: $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$.
        transition_fn: function that takes in an integer time index and returns a $K \times K$ transition matrix.

    Returns:
        most likely state sequence

    """
    # Run the backward pass
    def _backward_pass(best_next_score, t):
        scores = jnp.log(transition_matrix) + best_next_score + log_likelihoods[t + 1]
        best_next_state = jnp.argmax(scores, axis=1)
        best_next_score = jnp.max(scores, axis=1)
        return best_next_score, best_next_state

    num_timesteps, num_states = log_likelihoods.shape
    best_second_score, best_next_states = jax.lax.scan(
        _backward_pass,
        jnp.zeros(num_states),
        jnp.arange(num_timesteps-1),
        reverse=True,
    )

    # Run the forward pass
    def _forward_pass(state, best_next_state):
        next_state = best_next_state[state]
        return next_state, next_state

    first_state = jnp.argmax(
        jnp.log(initial_distribution) + log_likelihoods[0] + best_second_score
    )
    _, states = jax.lax.scan(_forward_pass, first_state, best_next_states)

    return jnp.concatenate([jnp.array([first_state]), states])

most_likely_state_id2 = hmm_posterior_mode2(
    initial_distribution=detector.initial_conditions_,
    transition_matrix=(
        detector.discrete_state_transitions_[
            np.ix_(detector.state_ind_, detector.state_ind_)
        ]
        * detector.continuous_state_transitions_
    ),
    log_likelihoods=detector.log_likelihood_,
)

np.all(np.asarray(most_likely_state_id) == np.asarray(most_likely_state_id2))

# %%
first_state = jnp.argmax(
    jnp.log(initial_distribution) + log_likelihoods[0] + best_second_score
)
best_next_states[0, first_state]

# %%
most_likely_state

# %%
most_likely_state.shape, most_likely_state2.shape

# %%
from non_local_detector import ContFragSortedSpikesClassifier, RandomWalk, Uniform

continuous_st = [[RandomWalk(movement_var=1.0), Uniform()], [Uniform(), Uniform()]]

decoder = ContFragSortedSpikesClassifier(
    continuous_transition_types=continuous_st,
)
results2 = decoder.estimate_parameters(
    position_time=time,
    position=position,
    spike_times=spike_times,
    is_training=is_training,
    time=time,
    store_log_likelihood=True,
)

# %%
import numpy as np
import scipy.interpolate

from non_local_detector.likelihoods.common import get_spikecount_per_time_bin

true_log_likelihood = np.zeros_like(decoder.log_likelihood_)

for neuron_spike_times, neuron_place_field in zip(spike_times, place_fields.T, strict=False):
    spike_counts = get_spikecount_per_time_bin(neuron_spike_times, time)
    neuron_place_intensity = (
        scipy.interpolate.interp1d(position, neuron_place_field)(
            decoder.environments[0].place_bin_centers_
        )
        / sampling_frequency
    ).squeeze()
    neuron_log_likelihoood = jax.scipy.stats.poisson.logpmf(
        spike_counts[:, None], neuron_place_intensity[None, :]
    )

    true_log_likelihood += np.concatenate(
        [neuron_log_likelihoood, neuron_log_likelihoood], axis=1
    )

true_log_likelihood.shape

# %%
import numpy as np

most_likely_state_id2 = hmm_posterior_mode(
    initial_distribution=decoder.initial_conditions_,
    transition_matrix=(
        decoder.discrete_state_transitions_[
            np.ix_(decoder.state_ind_, decoder.state_ind_)
        ]
        * decoder.continuous_state_transitions_
    ),
    # log_likelihoods=decoder.log_likelihood_,
    log_likelihoods=true_log_likelihood,
)

most_likely_state2 = results2.state_bins[most_likely_state_id2]

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
axes[0].plot(time, position, color="magenta")
axes[0].scatter(results2.time, most_likely_state2.to_dataframe().position.values, s=1)
axes[1].plot(time, most_likely_state_id2)
for state_prob, state_name in zip(
    results2.acausal_state_probabilities.values.T, results2.states.values, strict=False
):
    axes[2].plot(time, state_prob, label=state_name)

axes[2].legend()
plt.xlim((6, 7))

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
axes[0].plot(time, position, color="magenta")
axes[0].scatter(results2.time, most_likely_state2.to_dataframe().position.values, s=1)
axes[1].plot(time, most_likely_state_id2)
for state_prob, state_name in zip(
    results2.acausal_state_probabilities.values.T, results2.states.values, strict=False
):
    axes[2].plot(time, state_prob, label=state_name)

axes[2].legend()
plt.xlim((32.4, 32.8))

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
axes[0].plot(time, position, color="magenta")
axes[0].scatter(results2.time, most_likely_state2.to_dataframe().position.values, s=1)
axes[1].plot(time, most_likely_state_id2)
for state_prob, state_name in zip(
    results2.acausal_state_probabilities.values.T, results2.states.values, strict=False
):
    axes[2].plot(time, state_prob, label=state_name)

axes[2].legend()
plt.xlim((19, 20))

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
axes[0].plot(time, position, color="magenta")
axes[0].scatter(results2.time, most_likely_state2.to_dataframe().position.values, s=1)
axes[1].plot(time, most_likely_state_id2)
for state_prob, state_name in zip(
    results2.acausal_state_probabilities.values.T, results2.states.values, strict=False
):
    axes[2].plot(time, state_prob, label=state_name)

axes[2].legend()

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(10, 5), sharex=True)
axes[0].plot(time, position, color="magenta")
axes[0].scatter(results2.time, most_likely_state2.to_dataframe().position.values, s=1)
axes[1].plot(time, most_likely_state_id2)
for state_prob, state_name in zip(
    results2.acausal_state_probabilities.values.T, results2.states.values, strict=False
):
    axes[2].plot(time, state_prob, label=state_name)

axes[2].legend()

plt.xlim((0, 5))

# %%
import pandas as pd

self = decoder

position = []
n_position_dims = self.environments[0].place_bin_centers_.shape[1]
environment_names = []
encoding_group_names = []
for obs in self.observation_models:
    if obs.is_local or obs.is_no_spike:
        position.append(np.full((1, n_position_dims), np.nan))
        environment_names.append([None])
        encoding_group_names.append([None])
    else:
        environment = self.environments[
            self.environments.index(obs.environment_name)
        ]
        position.append(environment.place_bin_centers_)
        environment_names.append([obs.environment_name] * environment.place_bin_centers_.shape[0])
        encoding_group_names.append([obs.encoding_group] * environment.place_bin_centers_.shape[0])

position = np.concatenate(position, axis=0)
environment_names = np.concatenate(environment_names, axis=0)
encoding_group_names = np.concatenate(encoding_group_names, axis=0)

states = np.asarray(self.state_names)
if n_position_dims == 1:
    position_names = ["position"]
else:
    position_names = [
        f"{name}_position" for name, _ in zip(["x", "y", "z", "w"], position.T, strict=False)
    ]
state_bins = pd.DataFrame(
    {
        "state": states[self.state_ind_],
        **dict(zip(position_names, position.T, strict=False)),
        "environment": environment_names,
        "encoding_group_names": encoding_group_names,
    }
)

state_bins.iloc[most_likely_state_id2].set_index(pd.Index(time, name="time"))

# %%
obs

# %%

# %%
environment_names = [obs.environment_name for obs in self.observation_models]
encoding_group_names = [obs.encoding_group for obs in self.observation_models]

environment_names

# %%

# %%
# make this a dataframe instead of an index
pd.DataFrame(
    {
        "state": states[self.state_ind_],
        **dict(zip(position_names, position.T, strict=False)),
    }
)

# %%
states[self.state_ind_].shape, position.T.shape

# %%
