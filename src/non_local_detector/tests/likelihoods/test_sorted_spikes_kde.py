import numpy as np
import pytest
import jax.numpy as jnp

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
    predict_sorted_spikes_kde_log_likelihood,
)
from non_local_detector.likelihoods.common import get_position_at_time, EPS


def make_simple_env_1d():
    env = Environment(environment_name="line", place_bin_size=1.0, position_range=((0.0, 10.0),))
    # Provide a simple position array so fit_place_grid can determine dims
    dummy_pos = np.linspace(0.0, 10.0, 11)[:, None]
    # Fit grid (no graph); do not infer interior
    env = env.fit_place_grid(position=dummy_pos, infer_track_interior=False)
    assert env.place_bin_centers_ is not None
    return env


def test_fit_sorted_spikes_kde_encoding_model_minimal():
    env = make_simple_env_1d()
    t = jnp.linspace(0.0, 10.0, 101)
    pos = jnp.linspace(0.0, 10.0, 101)[:, None]
    # two neurons with a couple spikes each inside [0,10]
    spikes = [jnp.array([2.0, 5.0, 5.1]), jnp.array([1.5, 7.2])]
    weights = jnp.ones_like(t)

    enc = fit_sorted_spikes_kde_encoding_model(
        position_time=t,
        position=pos,
        spike_times=spikes,
        environment=env,
        weights=weights,
        sampling_frequency=10,
        position_std=np.sqrt(1.0),
        block_size=16,
        disable_progress_bar=True,
    )

    # keys and shapes
    for key in (
        "environment",
        "marginal_models",
        "occupancy_model",
        "occupancy",
        "mean_rates",
        "place_fields",
        "no_spike_part_log_likelihood",
        "is_track_interior",
    ):
        assert key in enc
    occ = enc["occupancy"]
    pf = enc["place_fields"]
    assert occ.ndim == 1 and occ.size > 0
    assert jnp.all(jnp.isfinite(occ))
    assert pf.ndim == 2 and pf.shape[0] == len(spikes)
    assert jnp.all(pf > 0)


def test_predict_sorted_spikes_kde_log_likelihood_shapes_local_and_nonlocal():
    env = make_simple_env_1d()
    t_pos = jnp.linspace(0.0, 10.0, 101)
    pos = jnp.linspace(0.0, 10.0, 101)[:, None]
    spikes = [jnp.array([2.0, 5.0, 5.1]), jnp.array([1.5, 7.2])]
    weights = jnp.ones_like(t_pos)

    enc = fit_sorted_spikes_kde_encoding_model(
        position_time=t_pos,
        position=pos,
        spike_times=spikes,
        environment=env,
        weights=weights,
        sampling_frequency=10,
        position_std=np.sqrt(1.0),
        block_size=16,
        disable_progress_bar=True,
    )

    # decoding over 5 time bins via 6 edges
    t_edges = jnp.linspace(0.0, 10.0, 6)

    # Non-local
    ll = predict_sorted_spikes_kde_log_likelihood(
        time=t_edges,
        position_time=t_pos,
        position=pos,
        spike_times=spikes,
        environment=env,
        marginal_models=enc["marginal_models"],
        occupancy_model=enc["occupancy_model"],
        occupancy=enc["occupancy"],
        mean_rates=jnp.asarray(enc["mean_rates"]),
        place_fields=enc["place_fields"],
        no_spike_part_log_likelihood=enc["no_spike_part_log_likelihood"],
        is_track_interior=enc["is_track_interior"],
        disable_progress_bar=True,
        is_local=False,
    )
    assert ll.shape[0] == t_edges.shape[0]
    assert ll.ndim == 2 and ll.shape[1] == int(enc["is_track_interior"].sum())
    assert jnp.all(jnp.isfinite(ll))

    # Local
    ll_local = predict_sorted_spikes_kde_log_likelihood(
        time=t_edges,
        position_time=t_pos,
        position=pos,
        spike_times=spikes,
        environment=env,
        marginal_models=enc["marginal_models"],
        occupancy_model=enc["occupancy_model"],
        occupancy=enc["occupancy"],
        mean_rates=jnp.asarray(enc["mean_rates"]),
        place_fields=enc["place_fields"],
        no_spike_part_log_likelihood=enc["no_spike_part_log_likelihood"],
        is_track_interior=enc["is_track_interior"],
        disable_progress_bar=True,
        is_local=True,
    )
    assert ll_local.shape == (t_edges.shape[0], 1)
    assert jnp.all(jnp.isfinite(ll_local))


def test_local_likelihood_zero_spikes_equals_negative_rate_sum():
    env = make_simple_env_1d()
    t_pos = jnp.linspace(0.0, 10.0, 101)
    pos = jnp.linspace(0.0, 10.0, 101)[:, None]
    # Use some spikes to build encoding, but no spikes in decoding window
    spikes_enc = [jnp.array([2.0, 5.0, 5.1]), jnp.array([1.5, 7.2])]
    weights = jnp.ones_like(t_pos)

    enc = fit_sorted_spikes_kde_encoding_model(
        position_time=t_pos,
        position=pos,
        spike_times=spikes_enc,
        environment=env,
        weights=weights,
        sampling_frequency=10,
        position_std=np.sqrt(1.0),
        block_size=16,
        disable_progress_bar=True,
    )

    # Decoding window with no spikes
    t_edges = jnp.linspace(0.0, 10.0, 6)
    empty_spikes = [jnp.array([]), jnp.array([])]
    ll_local = predict_sorted_spikes_kde_log_likelihood(
        time=t_edges,
        position_time=t_pos,
        position=pos,
        spike_times=empty_spikes,
        environment=env,
        marginal_models=enc["marginal_models"],
        occupancy_model=enc["occupancy_model"],
        occupancy=enc["occupancy"],
        mean_rates=jnp.asarray(enc["mean_rates"]),
        place_fields=enc["place_fields"],
        no_spike_part_log_likelihood=enc["no_spike_part_log_likelihood"],
        is_track_interior=enc["is_track_interior"],
        disable_progress_bar=True,
        is_local=True,
    )

    # Compute expected negative sum of local rates at interpolated positions
    interpolated_position = get_position_at_time(t_pos, pos, t_edges, env)
    occupancy_at_time = enc["occupancy_model"].predict(interpolated_position)
    expected = jnp.zeros((t_edges.shape[0],))
    for m, mean_rate in zip(enc["marginal_models"], jnp.asarray(enc["mean_rates"])):
        marginal = m.predict(interpolated_position)
        marginal = jnp.where(jnp.isnan(marginal), 0.0, marginal)
        local_rate = mean_rate * jnp.where(occupancy_at_time > 0.0, marginal / occupancy_at_time, EPS)
        local_rate = jnp.clip(local_rate, min=EPS)
        expected -= local_rate
    expected = jnp.expand_dims(expected, axis=1)

    assert jnp.allclose(ll_local, expected, rtol=1e-5, atol=1e-6)


def test_nonlocal_with_no_spikes_equals_negative_no_spike_part():
    env = make_simple_env_1d()
    t_pos = jnp.linspace(0.0, 10.0, 101)
    pos = jnp.linspace(0.0, 10.0, 101)[:, None]
    spikes_enc = [jnp.array([2.0, 5.0, 5.1]), jnp.array([1.5, 7.2])]
    weights = jnp.ones_like(t_pos)

    enc = fit_sorted_spikes_kde_encoding_model(
        position_time=t_pos,
        position=pos,
        spike_times=spikes_enc,
        environment=env,
        weights=weights,
        sampling_frequency=10,
        position_std=np.sqrt(1.0),
        block_size=16,
        disable_progress_bar=True,
    )

    t_edges = jnp.linspace(0.0, 10.0, 6)
    empty_spikes = [jnp.array([]), jnp.array([])]
    ll = predict_sorted_spikes_kde_log_likelihood(
        time=t_edges,
        position_time=t_pos,
        position=pos,
        spike_times=empty_spikes,
        environment=env,
        marginal_models=enc["marginal_models"],
        occupancy_model=enc["occupancy_model"],
        occupancy=enc["occupancy"],
        mean_rates=jnp.asarray(enc["mean_rates"]),
        place_fields=enc["place_fields"],
        no_spike_part_log_likelihood=enc["no_spike_part_log_likelihood"],
        is_track_interior=enc["is_track_interior"],
        disable_progress_bar=True,
        is_local=False,
    )
    expected = -enc["no_spike_part_log_likelihood"][enc["is_track_interior"]]
    expected = jnp.tile(expected, (t_edges.shape[0], 1))
    assert jnp.allclose(ll, expected, rtol=1e-5, atol=1e-6)


def test_fit_sorted_spikes_kde_raises_without_place_grid():
    env = Environment(environment_name="nofit")
    t_pos = jnp.linspace(0.0, 10.0, 11)
    pos = jnp.linspace(0.0, 10.0, 11)[:, None]
    spikes = [jnp.array([2.0]), jnp.array([3.0])]
    with pytest.raises(ValueError):
        fit_sorted_spikes_kde_encoding_model(
            position_time=t_pos,
            position=pos,
            spike_times=spikes,
            environment=env,
            weights=jnp.ones_like(t_pos),
        )
