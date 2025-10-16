import jax.numpy as jnp
import numpy as np

from non_local_detector.likelihoods.common import (
    KDEModel,
    block_kde,
    block_log_kde,
    get_position_at_time,
    get_spikecount_per_time_bin,
    kde,
    log_kde,
    safe_divide,
    safe_log,
)


def rng(seed=0):
    return np.random.default_rng(seed)


def test_kde_and_block_kde_match_1d():
    r = rng(1)
    samples = r.normal(loc=0.0, scale=1.0, size=(200, 1))
    eval_points = np.linspace(-3, 3, 50)[:, None]
    std = jnp.array([1.0])
    w = jnp.ones((samples.shape[0],))

    base = kde(jnp.asarray(eval_points), jnp.asarray(samples), std, w)
    for bs in (1, 5, 17, 1000):
        blk = block_kde(
            jnp.asarray(eval_points),
            jnp.asarray(samples),
            std,
            block_size=bs,
            weights=w,
        )
        assert jnp.allclose(base, blk, rtol=1e-5, atol=1e-7)


def test_kde_and_block_kde_match_2d_weighted():
    r = rng(2)
    samples = r.normal(size=(150, 2))
    eval_points = r.normal(size=(40, 2))
    std = jnp.array([0.5, 1.25])
    weights = jnp.asarray(r.uniform(0.1, 2.0, size=(samples.shape[0],)))

    base = kde(jnp.asarray(eval_points), jnp.asarray(samples), std, weights)
    blk = block_kde(
        jnp.asarray(eval_points),
        jnp.asarray(samples),
        std,
        block_size=7,
        weights=weights,
    )
    assert jnp.allclose(base, blk, rtol=1e-5, atol=1e-7)


def test_log_kde_consistent_with_kde_log():
    r = rng(3)
    samples = r.normal(size=(120, 2))
    eval_points = r.normal(size=(30, 2))
    std = jnp.array([1.0, 0.75])
    weights = jnp.asarray(r.uniform(0.5, 1.5, size=(samples.shape[0],)))

    log_vals = log_kde(jnp.asarray(eval_points), jnp.asarray(samples), std, weights)
    lin_vals = kde(jnp.asarray(eval_points), jnp.asarray(samples), std, weights)
    # Use moderate tolerance; avoid -inf by clipping
    lin_vals = jnp.clip(lin_vals, a_min=1e-12)
    assert jnp.allclose(log_vals, jnp.log(lin_vals), rtol=1e-5, atol=1e-6)


def test_block_log_kde_matches_log_kde():
    r = rng(4)
    samples = r.normal(size=(80, 1))
    eval_points = r.normal(size=(25, 1))
    std = jnp.array([0.8])
    weights = jnp.asarray(r.uniform(0.2, 3.0, size=(samples.shape[0],)))

    base = log_kde(jnp.asarray(eval_points), jnp.asarray(samples), std, weights)
    for bs in (1, 4, 16, 128):
        blk = block_log_kde(
            jnp.asarray(eval_points),
            jnp.asarray(samples),
            std,
            block_size=bs,
            weights=weights,
        )
        assert jnp.allclose(base, blk, rtol=1e-5, atol=1e-7)


def test_kde_model_shapes_and_predict_log():
    r = rng(5)
    samples = r.normal(size=(50, 2))
    eval_points = r.normal(size=(10, 2))
    model = KDEModel(std=jnp.array([1.0, 2.0]), block_size=5).fit(samples)
    dens = model.predict(jnp.asarray(eval_points))
    log_dens = model.predict_log(jnp.asarray(eval_points))
    assert dens.shape == (eval_points.shape[0],)
    assert log_dens.shape == (eval_points.shape[0],)
    assert jnp.all(jnp.isfinite(log_dens))


def test_safe_divide_and_safe_log_stability_and_broadcast():
    a = jnp.array([1.0, 0.0, 2.0])
    b = jnp.array([0.0, 0.0, 4.0])
    out = safe_divide(a, b)
    assert out.shape == a.shape
    assert jnp.all(jnp.isfinite(out))

    x = jnp.array([0.0, 1e-20, 1.0])
    log_x = safe_log(x)
    assert log_x.shape == x.shape
    assert jnp.all(jnp.isfinite(log_x))


def test_get_spikecount_per_time_bin_edges_and_outliers():
    time_edges = np.array([0.0, 1.0, 2.0, 3.0])
    # spikes include below-first, on edges, interior, and at last edge
    spikes = np.array([-0.5, 0.0, 0.4, 1.0, 2.99, 3.0, 3.5])
    counts = get_spikecount_per_time_bin(spikes, time_edges)
    # Function clips spikes to [time[0], time[-1]]. It then digitizes against
    # interior edges time[1:-1] = [1.0, 2.0], so a spike at the last edge (3.0)
    # is assigned to the previous interior bin index (2). Last bin remains 0.
    assert counts.shape == (time_edges.shape[0],)
    assert counts.tolist() == [2, 1, 2, 0]


def test_weights_scaling_invariance_and_reweighting_effect():
    r = rng(10)
    # Two clusters: near 0 and near 5
    samples = np.concatenate(
        [r.normal(0.0, 0.5, size=(200, 1)), r.normal(5.0, 0.5, size=(200, 1))], axis=0
    )
    eval_points = np.array([[0.0]])
    std = jnp.array([0.7])
    w_uniform = jnp.ones((samples.shape[0],))
    # Scale invariance: multiply weights by constant doesn't change
    dens1 = kde(jnp.asarray(eval_points), jnp.asarray(samples), std, w_uniform)
    dens2 = kde(jnp.asarray(eval_points), jnp.asarray(samples), std, 3.0 * w_uniform)
    assert jnp.allclose(dens1, dens2, rtol=1e-6, atol=1e-9)

    # Reweight: emphasize near 0, de-emphasize near 5 -> density at 0 increases
    w = np.ones(samples.shape[0])
    w[:200] = 2.0  # near 0
    w[200:] = 0.5  # near 5
    dens_reweighted = kde(
        jnp.asarray(eval_points), jnp.asarray(samples), std, jnp.asarray(w)
    )
    assert dens_reweighted > dens1


def test_std_extremes_are_finite_and_reasonable():
    r = rng(11)
    samples = r.normal(size=(100, 2))
    eval_points = r.normal(size=(20, 2))
    w = jnp.ones((samples.shape[0],))
    # Very small std -> densities peaked but finite
    d_small = kde(
        jnp.asarray(eval_points), jnp.asarray(samples), jnp.array([1e-6, 1e-6]), w
    )
    # Very large std -> densities smoother and close across eval points
    d_large = kde(
        jnp.asarray(eval_points), jnp.asarray(samples), jnp.array([100.0, 100.0]), w
    )
    assert jnp.all(jnp.isfinite(d_small)) and jnp.all(jnp.isfinite(d_large))
    assert d_large.ptp() < 1e-2


def test_empty_eval_points_returns_empty():
    r = rng(12)
    samples = r.normal(size=(50, 2))
    eval_points = np.zeros((0, 2))
    std = jnp.array([1.0, 1.0])
    w = jnp.ones((samples.shape[0],))
    out = block_kde(
        jnp.asarray(eval_points), jnp.asarray(samples), std, block_size=7, weights=w
    )
    assert out.shape == (0,)


def test_dtype_parity_float32_float64():
    r = rng(13)
    samples64 = r.normal(size=(120, 2)).astype(np.float64)
    eval64 = r.normal(size=(15, 2)).astype(np.float64)
    std64 = jnp.array([0.9, 1.1], dtype=jnp.float64)
    w64 = jnp.ones((samples64.shape[0],), dtype=jnp.float64)
    out64 = kde(jnp.asarray(eval64), jnp.asarray(samples64), std64, w64)

    samples32 = samples64.astype(np.float32)
    eval32 = eval64.astype(np.float32)
    std32 = jnp.array([0.9, 1.1], dtype=jnp.float32)
    w32 = jnp.ones((samples32.shape[0],), dtype=jnp.float32)
    out32 = kde(jnp.asarray(eval32), jnp.asarray(samples32), std32, w32)

    assert jnp.allclose(out32.astype(jnp.float64), out64, rtol=1e-5, atol=1e-6)


def test_get_position_at_time_linear_interpolation():
    # Simple 1D line; position equals time
    time = jnp.linspace(0.0, 10.0, 11)
    position = time[:, None]
    spike_times = jnp.array([0.0, 1.5, 5.0, 9.9])
    out = get_position_at_time(time, position, spike_times, env=None)
    # Expected equals spike_times in a column
    assert out.shape == (spike_times.shape[0], 1)
    assert jnp.allclose(out.squeeze(), spike_times, rtol=1e-6, atol=1e-9)
