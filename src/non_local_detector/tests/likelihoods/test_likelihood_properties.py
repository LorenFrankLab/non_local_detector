"""Property-based tests for likelihood model mathematical invariants.

Uses Hypothesis to verify that likelihood models satisfy key mathematical
properties for varied random inputs: no NaN in clusterless non-local
likelihoods, and finite local and non-local likelihoods from one sorted-spikes
fit. Examples are derandomized so every run draws the same inputs.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_gmm import (
    fit_clusterless_gmm_encoding_model,
    mark_reference_scale,
    predict_clusterless_gmm_log_likelihood,
)
from non_local_detector.likelihoods.clusterless_kde import (
    fit_clusterless_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.likelihoods.mark_reference import fit_mark_references
from non_local_detector.likelihoods.sorted_spikes_glm import (
    fit_sorted_spikes_glm_encoding_model,
    predict_sorted_spikes_glm_log_likelihood,
)
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
    predict_sorted_spikes_kde_log_likelihood,
)

# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------


def _make_env():
    """Create a simple 1D environment for property tests."""
    env = Environment(
        environment_name="test",
        place_bin_size=5.0,
        position_range=((0.0, 100.0),),
    )
    pos = np.linspace(0, 100, 101)[:, None]
    return env.fit_place_grid(position=pos, infer_track_interior=False)


def _make_sorted_spike_data(seed, n_neurons, n_time, spikes_per_neuron):
    """Generate valid sorted spike data with controlled randomness."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_time)
    pos = np.linspace(0.0, 100.0, n_time)[:, None]
    spike_times = []
    for _ in range(n_neurons):
        n_spikes = max(2, spikes_per_neuron)  # Minimum 2 to avoid edge cases
        times = np.sort(rng.uniform(t[0] + 0.01, t[-1] - 0.01, size=n_spikes))
        spike_times.append(jnp.asarray(times))
    return t, pos, spike_times


def _make_clusterless_spike_data(
    seed, n_electrodes, n_time, n_features, spikes_per_electrode
):
    """Generate valid clusterless spike data."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_time)
    pos = np.linspace(0.0, 100.0, n_time)[:, None]
    spike_times = []
    spike_features = []
    for _ in range(n_electrodes):
        n_spikes = max(3, spikes_per_electrode)  # Min 3 for GMM
        times = np.sort(rng.uniform(t[0] + 0.01, t[-1] - 0.01, size=n_spikes))
        feats = rng.standard_normal(size=(n_spikes, n_features)).astype(np.float32)
        spike_times.append(jnp.asarray(times))
        spike_features.append(jnp.asarray(feats))
    return t, pos, spike_times, spike_features


# ---------------------------------------------------------------------------
# Fit-and-predict wrappers
# ---------------------------------------------------------------------------


def _fit_sorted_kde(env, t, pos, spike_times):
    return fit_sorted_spikes_kde_encoding_model(
        position_time=jnp.asarray(t),
        position=jnp.asarray(pos),
        spike_times=spike_times,
        environment=env,
        weights=jnp.ones(len(t)),
        position_std=np.sqrt(12.5),
        block_size=16,
        disable_progress_bar=True,
    )


def _predict_sorted_kde(enc, env, t, pos, spike_times, is_local=False):
    return predict_sorted_spikes_kde_log_likelihood(
        time_edges=jnp.linspace(0.0, 1.0, 6),
        position_time=jnp.asarray(t),
        position=jnp.asarray(pos),
        spike_times=spike_times,
        environment=env,
        marginal_models=enc["marginal_models"],
        occupancy_model=enc["occupancy_model"],
        occupancy=enc["occupancy"],
        mean_rates=jnp.asarray(enc["mean_rates"]),
        place_fields=enc["place_fields"],
        no_spike_part_log_likelihood=enc["no_spike_part_log_likelihood"],
        is_track_interior=enc["is_track_interior"],
        disable_progress_bar=True,
        is_local=is_local,
    )


def _fit_sorted_glm(env, t, pos, spike_times):
    return fit_sorted_spikes_glm_encoding_model(
        position_time=jnp.asarray(t),
        position=jnp.asarray(pos),
        spike_times=spike_times,
        environment=env,
        place_bin_edges=env.place_bin_edges_,
        edges=env.edges_,
        is_track_interior=env.is_track_interior_,
        is_track_boundary=env.is_track_boundary_,
        disable_progress_bar=True,
    )


def _predict_sorted_glm(enc, env, t, pos, spike_times, is_local=False):
    return predict_sorted_spikes_glm_log_likelihood(
        time_edges=jnp.linspace(0.0, 1.0, 6),
        position_time=jnp.asarray(t),
        position=jnp.asarray(pos),
        spike_times=spike_times,
        environment=env,
        coefficients=enc["coefficients"],
        emission_design_info=enc["emission_design_info"],
        place_fields=enc["place_fields"],
        no_spike_part_log_likelihood=enc["no_spike_part_log_likelihood"],
        is_track_interior=enc["is_track_interior"],
        is_local=is_local,
        disable_progress_bar=True,
    )


def _fit_predict_clusterless_kde(
    env, t, pos, spike_times, spike_features, is_local=False
):
    enc = fit_clusterless_kde_encoding_model(
        position_time=jnp.asarray(t),
        position=jnp.asarray(pos),
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        environment=env,
        position_std=np.sqrt(12.5),
        waveform_std=1.0,
        block_size=8,
        disable_progress_bar=True,
    )
    t_edges = jnp.linspace(0.0, 1.0, 6)
    ll = predict_clusterless_kde_log_likelihood(
        time_edges=t_edges,
        position_time=jnp.asarray(t),
        position=jnp.asarray(pos),
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        occupancy=enc["occupancy"],
        occupancy_model=enc["occupancy_model"],
        gpi_models=enc["gpi_models"],
        encoding_spike_waveform_features=enc["encoding_spike_waveform_features"],
        encoding_positions=enc["encoding_positions"],
        mark_reference=fit_mark_references(spike_features, 1.0),
        environment=env,
        mean_rates=jnp.asarray(enc["mean_rates"]),
        summed_ground_process_intensity=enc["summed_ground_process_intensity"],
        position_std=jnp.asarray(enc["position_std"]),
        waveform_std=jnp.asarray(enc["waveform_std"]),
        is_local=is_local,
        disable_progress_bar=True,
        block_size=8,
    )
    return enc, ll


def _fit_predict_clusterless_gmm(
    env, t, pos, spike_times, spike_features, is_local=False
):
    enc = fit_clusterless_gmm_encoding_model(
        position_time=jnp.asarray(t),
        position=jnp.asarray(pos),
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        environment=env,
        gmm_components_occupancy=5,
        gmm_components_joint=5,
        gmm_random_state=0,
        disable_progress_bar=True,
    )
    t_edges = jnp.linspace(0.0, 1.0, 6)
    ll = predict_clusterless_gmm_log_likelihood(
        time_edges=t_edges,
        position_time=jnp.asarray(t),
        position=jnp.asarray(pos),
        spike_times=spike_times,
        spike_waveform_features=spike_features,
        environment=env,
        occupancy_model=enc["occupancy_model"],
        interior_place_bin_centers=enc["interior_place_bin_centers"],
        log_occupancy=enc["log_occupancy"],
        gpi_models=enc["gpi_models"],
        joint_models=enc["joint_models"],
        mean_rates=enc["mean_rates"],
        summed_ground_process_intensity=enc["summed_ground_process_intensity"],
        mark_dimensions=enc["mark_dimensions"],
        mark_reference=fit_mark_references(spike_features, mark_reference_scale()),
        is_local=is_local,
        disable_progress_bar=True,
    )
    return enc, ll


# ---------------------------------------------------------------------------
# Property: No NaN in output
# ---------------------------------------------------------------------------

ENV = _make_env()


@pytest.mark.property
class TestLikelihoodNoNaN:
    """Clusterless non-local output never contains NaN for valid inputs.

    The sorted-spikes non-local outputs are checked for finiteness below.
    """

    @given(
        n_electrodes=st.integers(min_value=1, max_value=2),
        n_time=st.integers(min_value=30, max_value=50),
        n_features=st.integers(min_value=1, max_value=3),
        spikes_per_electrode=st.integers(min_value=5, max_value=10),
        seed=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=8, deadline=None, derandomize=True)
    def test_clusterless_kde_output_no_nan(
        self, n_electrodes, n_time, n_features, spikes_per_electrode, seed
    ):
        t, pos, spikes, feats = _make_clusterless_spike_data(
            seed, n_electrodes, n_time, n_features, spikes_per_electrode
        )
        _, ll = _fit_predict_clusterless_kde(ENV, t, pos, spikes, feats)
        assert not jnp.any(jnp.isnan(ll))

    @given(
        n_electrodes=st.integers(min_value=1, max_value=2),
        n_time=st.integers(min_value=30, max_value=50),
        n_features=st.integers(min_value=1, max_value=2),
        spikes_per_electrode=st.integers(min_value=8, max_value=12),
        seed=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=5, deadline=None, derandomize=True)
    def test_clusterless_gmm_output_no_nan(
        self, n_electrodes, n_time, n_features, spikes_per_electrode, seed
    ):
        t, pos, spikes, feats = _make_clusterless_spike_data(
            seed, n_electrodes, n_time, n_features, spikes_per_electrode
        )
        _, ll = _fit_predict_clusterless_gmm(ENV, t, pos, spikes, feats)
        assert not jnp.any(jnp.isnan(ll))


# ---------------------------------------------------------------------------
# Property: Local and non-local LL are finite
# ---------------------------------------------------------------------------


@pytest.mark.property
class TestLocalNonlocalFiniteness:
    """Local and non-local log-likelihoods of one fit are finite.

    Local mode interpolates to the exact animal position, while non-local
    evaluates at discrete bin centers, so a local value need not lie within
    its row's non-local range; ``test_local_nonlocal_agreement.py`` checks
    local against the non-local bin where the animal sits at a bin centre.
    """

    @given(
        n_neurons=st.integers(min_value=1, max_value=3),
        n_time=st.integers(min_value=30, max_value=50),
        spikes_per_neuron=st.integers(min_value=3, max_value=10),
        seed=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=10, deadline=None, derandomize=True)
    def test_sorted_kde_likelihoods_finite(
        self, n_neurons, n_time, spikes_per_neuron, seed
    ):
        t, pos, spikes = _make_sorted_spike_data(
            seed, n_neurons, n_time, spikes_per_neuron
        )
        enc = _fit_sorted_kde(ENV, t, pos, spikes)
        ll_nonlocal = _predict_sorted_kde(enc, ENV, t, pos, spikes)
        ll_local = _predict_sorted_kde(enc, ENV, t, pos, spikes, is_local=True)

        assert jnp.all(jnp.isfinite(ll_local)), (
            f"Local KDE LL contains NaN/Inf: {ll_local}"
        )
        assert jnp.all(jnp.isfinite(ll_nonlocal)), "Non-local KDE LL contains NaN/Inf"

    @given(
        n_neurons=st.integers(min_value=1, max_value=3),
        n_time=st.integers(min_value=30, max_value=50),
        spikes_per_neuron=st.integers(min_value=3, max_value=10),
        seed=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=10, deadline=None, derandomize=True)
    def test_sorted_glm_likelihoods_finite(
        self, n_neurons, n_time, spikes_per_neuron, seed
    ):
        t, pos, spikes = _make_sorted_spike_data(
            seed, n_neurons, n_time, spikes_per_neuron
        )
        enc = _fit_sorted_glm(ENV, t, pos, spikes)
        ll_nonlocal = _predict_sorted_glm(enc, ENV, t, pos, spikes)
        ll_local = _predict_sorted_glm(enc, ENV, t, pos, spikes, is_local=True)

        assert jnp.all(jnp.isfinite(ll_local)), (
            f"Local GLM LL contains NaN/Inf: {ll_local}"
        )
        assert jnp.all(jnp.isfinite(ll_nonlocal)), "Non-local GLM LL contains NaN/Inf"
