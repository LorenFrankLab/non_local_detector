import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.common import EPS, RATE_EPS_HZ, get_position_at_time
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
    predict_sorted_spikes_kde_log_likelihood,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def kde_fit(simple_1d_environment):
    """Two-neuron KDE encoding on a linear sweep of the 1D environment.

    Returns ``(env, position_time, position, spike_times, encoding)``.
    """
    env = simple_1d_environment
    position_time = jnp.linspace(0.0, 10.0, 101)
    position = jnp.linspace(0.0, 10.0, 101)[:, None]
    # two neurons with a couple spikes each inside [0,10]
    spike_times = [jnp.array([2.0, 5.0, 5.1]), jnp.array([1.5, 7.2])]
    encoding = fit_sorted_spikes_kde_encoding_model(
        position_time=position_time,
        position=position,
        spike_times=spike_times,
        environment=env,
        weights=jnp.ones_like(position_time),
        position_std=np.sqrt(1.0),
        block_size=16,
        disable_progress_bar=True,
    )
    return env, position_time, position, spike_times, encoding


def test_fit_sorted_spikes_kde_encoding_model_minimal(kde_fit):
    _, _, _, spikes, enc = kde_fit

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


def test_predict_sorted_spikes_kde_log_likelihood_shapes_local_and_nonlocal(kde_fit):
    env, t_pos, pos, spikes, enc = kde_fit

    # decoding over 5 time bins via 6 edges
    t_edges = jnp.linspace(0.0, 10.0, 6)

    # Non-local
    ll = predict_sorted_spikes_kde_log_likelihood(
        time_edges=t_edges,
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
    assert ll.shape[0] == t_edges.shape[0] - 1
    assert ll.ndim == 2 and ll.shape[1] == int(enc["is_track_interior"].sum())
    assert jnp.all(jnp.isfinite(ll))

    # Local
    ll_local = predict_sorted_spikes_kde_log_likelihood(
        time_edges=t_edges,
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
    assert ll_local.shape == (t_edges.shape[0] - 1, 1)
    assert jnp.all(jnp.isfinite(ll_local))

    with pytest.raises(ValidationError, match="population lengths do not match"):
        predict_sorted_spikes_kde_log_likelihood(
            time_edges=t_edges,
            position_time=t_pos,
            position=pos,
            spike_times=spikes[:-1],
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


def test_local_likelihood_zero_spikes_equals_negative_rate_sum(kde_fit):
    # Use some spikes to build encoding, but no spikes in decoding window
    env, t_pos, pos, _, enc = kde_fit

    # Decoding window with no spikes
    t_edges = jnp.linspace(0.0, 10.0, 6)
    empty_spikes = [jnp.array([]), jnp.array([])]
    ll_local = predict_sorted_spikes_kde_log_likelihood(
        time_edges=t_edges,
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

    # Compute expected negative sum of local rates at the positions interpolated
    # at the decode bin centers
    bin_centers = t_edges[:-1] + 0.5 * jnp.diff(t_edges)
    interpolated_position = get_position_at_time(t_pos, pos, bin_centers, env)
    occupancy_at_time = enc["occupancy_model"].predict(interpolated_position)
    expected = jnp.zeros((bin_centers.shape[0],))
    for m, mean_rate in zip(
        enc["marginal_models"], jnp.asarray(enc["mean_rates"]), strict=False
    ):
        marginal = m.predict(interpolated_position)
        marginal = jnp.where(jnp.isnan(marginal), 0.0, marginal)
        local_rate = mean_rate * jnp.where(
            occupancy_at_time > 0.0, marginal / occupancy_at_time, EPS
        )
        local_rate = jnp.clip(local_rate, min=RATE_EPS_HZ)
        expected -= jnp.diff(t_edges) * local_rate
    expected = jnp.expand_dims(expected, axis=1)

    assert jnp.allclose(ll_local, expected, rtol=1e-5, atol=1e-6)


def test_nonlocal_with_no_spikes_equals_negative_no_spike_part(kde_fit):
    env, t_pos, pos, _, enc = kde_fit

    t_edges = jnp.linspace(0.0, 10.0, 6)
    empty_spikes = [jnp.array([]), jnp.array([])]
    ll = predict_sorted_spikes_kde_log_likelihood(
        time_edges=t_edges,
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
    expected = (
        -jnp.diff(t_edges)[0]
        * enc["no_spike_part_log_likelihood"][enc["is_track_interior"]]
    )
    expected = jnp.tile(expected, (t_edges.shape[0] - 1, 1))
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


def test_nan_marginal_density_warns(simple_1d_environment, monkeypatch):
    """A NaN in the KDE marginal density triggers a UserWarning, not silent zeroing.

    NaN density is forced by monkeypatching the marginal models' log densities
    to return a NaN in one bin, isolating the warning path from whatever
    naturally produces NaN.
    """
    import non_local_detector.likelihoods.sorted_spikes_kde as ss_kde

    env = simple_1d_environment
    t = jnp.linspace(0.0, 10.0, 101)
    pos = jnp.linspace(0.0, 10.0, 101)[:, None]
    spikes = [jnp.array([2.0, 5.0, 5.1])]
    weights = jnp.ones_like(t)

    real_densities = ss_kde._kde_log_densities

    def densities_with_nan(models, points):
        first, *rest = real_densities(models, points)
        return [first.at[0].set(jnp.nan), *rest]

    monkeypatch.setattr(ss_kde, "_kde_log_densities", densities_with_nan)

    # Exactly one bin is forced to NaN, so the count in the message is 1.
    with pytest.warns(UserWarning, match=r"NaN at 1 \(neuron, bin\)"):
        fit_sorted_spikes_kde_encoding_model(
            position_time=t,
            position=pos,
            spike_times=spikes,
            environment=env,
            weights=weights,
            position_std=np.sqrt(1.0),
            block_size=16,
            disable_progress_bar=True,
        )


def test_fit_sorted_spikes_kde_rejects_nonfinite_weights(simple_1d_environment):
    """Non-finite weights are rejected at fit (validate_weights), matching the
    clusterless/GMM/MRF fits which all validate weights at entry."""
    from non_local_detector.exceptions import ValidationError

    env = simple_1d_environment
    t = jnp.linspace(0.0, 10.0, 101)
    pos = jnp.linspace(0.0, 10.0, 101)[:, None]
    spikes = [jnp.array([2.0, 5.0])]
    weights = jnp.ones_like(t).at[0].set(jnp.nan)

    with pytest.raises(ValidationError, match="finite"):
        fit_sorted_spikes_kde_encoding_model(
            position_time=t,
            position=pos,
            spike_times=spikes,
            environment=env,
            weights=weights,
            position_std=np.sqrt(1.0),
            block_size=16,
            disable_progress_bar=True,
        )
