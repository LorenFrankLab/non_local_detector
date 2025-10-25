import jax.numpy as jnp
import numpy as np
import pytest
from syrupy.assertion import SnapshotAssertion

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.common import EPS, get_position_at_time
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
    predict_sorted_spikes_kde_log_likelihood,
)


def test_fit_sorted_spikes_kde_encoding_model_minimal(simple_1d_environment):
    env = simple_1d_environment
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


def test_predict_sorted_spikes_kde_log_likelihood_shapes_local_and_nonlocal(
    simple_1d_environment,
):
    env = simple_1d_environment
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


def test_local_likelihood_zero_spikes_equals_negative_rate_sum(simple_1d_environment):
    env = simple_1d_environment
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
    for m, mean_rate in zip(
        enc["marginal_models"], jnp.asarray(enc["mean_rates"]), strict=False
    ):
        marginal = m.predict(interpolated_position)
        marginal = jnp.where(jnp.isnan(marginal), 0.0, marginal)
        local_rate = mean_rate * jnp.where(
            occupancy_at_time > 0.0, marginal / occupancy_at_time, EPS
        )
        local_rate = jnp.clip(local_rate, min=EPS)
        expected -= local_rate
    expected = jnp.expand_dims(expected, axis=1)

    assert jnp.allclose(ll_local, expected, rtol=1e-5, atol=1e-6)


def test_nonlocal_with_no_spikes_equals_negative_no_spike_part(simple_1d_environment):
    env = simple_1d_environment
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


# ============================================================================
# SNAPSHOT TESTS
# ============================================================================


def serialize_encoding_model_summary(encoding: dict) -> dict:
    """Serialize encoding model to summary statistics for snapshot comparison.

    Parameters
    ----------
    encoding : dict
        Encoding model dictionary from fit_sorted_spikes_kde_encoding_model

    Returns
    -------
    summary : dict
        Summary statistics suitable for snapshot comparison
    """
    return {
        "occupancy_stats": {
            "shape": encoding["occupancy"].shape,
            "mean": float(np.mean(encoding["occupancy"])),
            "std": float(np.std(encoding["occupancy"])),
            "min": float(np.min(encoding["occupancy"])),
            "max": float(np.max(encoding["occupancy"])),
        },
        "mean_rates": [float(r) for r in encoding["mean_rates"]],
        "place_fields_stats": {
            "shape": encoding["place_fields"].shape,
            "mean": float(np.mean(encoding["place_fields"])),
            "std": float(np.std(encoding["place_fields"])),
            "min": float(np.min(encoding["place_fields"])),
            "max": float(np.max(encoding["place_fields"])),
        },
        "no_spike_part_log_likelihood_stats": {
            "shape": encoding["no_spike_part_log_likelihood"].shape,
            "mean": float(np.mean(encoding["no_spike_part_log_likelihood"])),
            "std": float(np.std(encoding["no_spike_part_log_likelihood"])),
            "min": float(np.min(encoding["no_spike_part_log_likelihood"])),
            "max": float(np.max(encoding["no_spike_part_log_likelihood"])),
        },
        "n_neurons": len(encoding["marginal_models"]),
    }


def serialize_log_likelihood_summary(log_likelihood: jnp.ndarray) -> dict:
    """Serialize log likelihood array to summary statistics for snapshot comparison.

    Parameters
    ----------
    log_likelihood : jnp.ndarray
        Log likelihood array from predict_sorted_spikes_kde_log_likelihood

    Returns
    -------
    summary : dict
        Summary statistics suitable for snapshot comparison
    """
    arr = np.asarray(log_likelihood)
    return {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "sum": float(np.sum(arr)),
        "first_5": arr.ravel()[:5].tolist() if arr.size >= 5 else arr.ravel().tolist(),
        "last_5": arr.ravel()[-5:].tolist() if arr.size >= 5 else arr.ravel().tolist(),
    }


@pytest.mark.snapshot
def test_sorted_spikes_kde_encoding_model_snapshot(
    simple_1d_environment, snapshot: SnapshotAssertion
):
    """Snapshot test for sorted spikes KDE encoding model fitting.

    This test verifies that the encoding model produces consistent outputs
    across code changes, capturing:
    - Occupancy statistics
    - Mean firing rates per neuron
    - Place field statistics
    - No-spike part of log likelihood
    """
    env = simple_1d_environment
    np.random.seed(234)
    t_pos = jnp.linspace(0.0, 10.0, 201)
    pos = jnp.linspace(0.0, 10.0, 201)[:, None]
    weights = jnp.ones_like(t_pos)

    # Four neurons with different spike patterns
    spike_times = [
        jnp.array([1.0, 2.0, 3.0, 5.0, 7.0]),  # 5 spikes
        jnp.array([1.5, 4.0, 6.0, 8.5]),  # 4 spikes
        jnp.array([2.5, 5.5, 7.5, 9.0, 9.5]),  # 5 spikes
        jnp.array([0.5, 3.5, 6.5]),  # 3 spikes
    ]

    encoding = fit_sorted_spikes_kde_encoding_model(
        position_time=t_pos,
        position=pos,
        spike_times=spike_times,
        environment=env,
        weights=weights,
        sampling_frequency=20,
        position_std=np.sqrt(1.5),
        block_size=16,
        disable_progress_bar=True,
    )

    assert serialize_encoding_model_summary(encoding) == snapshot


@pytest.mark.snapshot
def test_sorted_spikes_kde_nonlocal_likelihood_snapshot(
    simple_1d_environment, snapshot: SnapshotAssertion
):
    """Snapshot test for sorted spikes KDE non-local likelihood prediction.

    This test verifies that non-local likelihood predictions are consistent
    across code changes.
    """
    env = simple_1d_environment
    np.random.seed(567)
    t_pos = jnp.linspace(0.0, 10.0, 201)
    pos = jnp.linspace(0.0, 10.0, 201)[:, None]
    weights = jnp.ones_like(t_pos)

    # Two neurons for encoding
    enc_spike_times = [
        jnp.array([2.0, 5.0, 5.1, 7.0]),
        jnp.array([1.5, 3.5, 6.0, 8.0, 9.0]),
    ]

    encoding = fit_sorted_spikes_kde_encoding_model(
        position_time=t_pos,
        position=pos,
        spike_times=enc_spike_times,
        environment=env,
        weights=weights,
        sampling_frequency=20,
        position_std=np.sqrt(1.0),
        block_size=16,
        disable_progress_bar=True,
    )

    # Decoding with different spike times
    t_edges = jnp.linspace(0.0, 10.0, 11)
    dec_spike_times = [
        jnp.array([0.5, 2.5, 5.5, 7.5, 9.5]),
        jnp.array([1.0, 3.0, 6.5, 8.5]),
    ]

    ll_nonlocal = predict_sorted_spikes_kde_log_likelihood(
        time=t_edges,
        position_time=t_pos,
        position=pos,
        spike_times=dec_spike_times,
        environment=env,
        marginal_models=encoding["marginal_models"],
        occupancy_model=encoding["occupancy_model"],
        occupancy=encoding["occupancy"],
        mean_rates=jnp.asarray(encoding["mean_rates"]),
        place_fields=encoding["place_fields"],
        no_spike_part_log_likelihood=encoding["no_spike_part_log_likelihood"],
        is_track_interior=encoding["is_track_interior"],
        disable_progress_bar=True,
        is_local=False,
    )

    assert serialize_log_likelihood_summary(ll_nonlocal) == snapshot


@pytest.mark.snapshot
def test_sorted_spikes_kde_local_likelihood_snapshot(
    simple_1d_environment, snapshot: SnapshotAssertion
):
    """Snapshot test for sorted spikes KDE local likelihood prediction.

    This test verifies that local likelihood predictions are consistent
    across code changes.
    """
    env = simple_1d_environment
    np.random.seed(890)
    t_pos = jnp.linspace(0.0, 10.0, 201)
    pos = jnp.linspace(0.0, 10.0, 201)[:, None]
    weights = jnp.ones_like(t_pos)

    # Three neurons for encoding
    enc_spike_times = [
        jnp.array([2.0, 5.0, 5.1]),
        jnp.array([1.5, 7.2]),
        jnp.array([3.0, 6.0, 8.0]),
    ]

    encoding = fit_sorted_spikes_kde_encoding_model(
        position_time=t_pos,
        position=pos,
        spike_times=enc_spike_times,
        environment=env,
        weights=weights,
        sampling_frequency=20,
        position_std=np.sqrt(1.0),
        block_size=16,
        disable_progress_bar=True,
    )

    # Decoding
    t_edges = jnp.linspace(0.0, 10.0, 6)
    dec_spike_times = [
        jnp.array([2.1, 5.2]),
        jnp.array([1.6]),
        jnp.array([3.1, 6.1, 8.1]),
    ]

    ll_local = predict_sorted_spikes_kde_log_likelihood(
        time=t_edges,
        position_time=t_pos,
        position=pos,
        spike_times=dec_spike_times,
        environment=env,
        marginal_models=encoding["marginal_models"],
        occupancy_model=encoding["occupancy_model"],
        occupancy=encoding["occupancy"],
        mean_rates=jnp.asarray(encoding["mean_rates"]),
        place_fields=encoding["place_fields"],
        no_spike_part_log_likelihood=encoding["no_spike_part_log_likelihood"],
        is_track_interior=encoding["is_track_interior"],
        disable_progress_bar=True,
        is_local=True,
    )

    assert serialize_log_likelihood_summary(ll_local) == snapshot
