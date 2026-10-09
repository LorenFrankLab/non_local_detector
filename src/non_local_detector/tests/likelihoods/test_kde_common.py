import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.common import (
    KDEModel,
    as_std_array,
    block_kde,
    block_log_kde,
    drop_zero_weight_samples,
    get_position_at_time,
    get_spikecount_per_time_bin,
    kde,
    log_gaussian_pdf,
    log_kde,
    safe_divide,
    safe_log,
    validate_population_lengths,
)
from non_local_detector.tests.conftest import precision_mode


@pytest.mark.unit
@pytest.mark.parametrize("sigma_kind", ["float32", "float64", "weak"])
def test_gaussian_mixed_dtype_preserves_true_division(sigma_kind):
    with precision_mode(True):
        x = jnp.array([0.4, 1.3, 2.2], dtype=jnp.float64)
        mean = jnp.float64(0.1)
        sigma = (
            0.7
            if sigma_kind == "weak"
            else jnp.asarray(
                0.7, dtype=jnp.float32 if sigma_kind == "float32" else jnp.float64
            )
        )
        reference = -0.5 * ((x - mean) / sigma) ** 2 - jnp.log(
            sigma * jnp.sqrt(2.0 * jnp.pi)
        )
        actual = log_gaussian_pdf(x, mean, sigma)
        assert actual.dtype == reference.dtype == jnp.float64
        np.testing.assert_allclose(actual, reference, rtol=1e-14, atol=1e-14)


@pytest.mark.unit
def test_gaussian_large_bandwidth_does_not_flush_standardized_coordinates():
    with precision_mode(False):
        sigma = np.float32(1e38)
        x = np.asarray([0, 1e38, 2e38], dtype=np.float32)
        reference = -0.5 * (x.astype(np.float64) / float(sigma)) ** 2 - np.log(
            float(sigma) * np.sqrt(2 * np.pi)
        )
        np.testing.assert_allclose(
            log_gaussian_pdf(jnp.asarray(x), jnp.float32(0), jnp.float32(sigma)),
            reference,
            rtol=1e-6,
            atol=1e-5,
        )


@pytest.mark.unit
@pytest.mark.parametrize("x64", [False, True])
def test_integer_gaussian_and_kde_inputs_use_floating_arithmetic(x64):
    with precision_mode(x64):
        points = jnp.array([0, 1, 2])
        mean, sigma = jnp.array(0), jnp.array(1)
        expected = -0.5 * ((points - mean) / sigma) ** 2 - jnp.log(
            sigma * jnp.sqrt(2.0 * jnp.pi)
        )
        actual = log_gaussian_pdf(points, mean, sigma)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            kde(points[:, None], points[:, None], jnp.array([1]), jnp.ones(3)),
            kde(
                points[:, None].astype(actual.dtype),
                points[:, None].astype(actual.dtype),
                # Preserve the original bandwidth's normalization arithmetic;
                # only the coordinate true-division promotion is under test.
                jnp.array([1]),
                jnp.ones(3),
            ),
        )


def rng(seed=0):
    return np.random.default_rng(seed)


@pytest.mark.parametrize(
    "std",
    [
        2.0,  # Python float
        2,  # Python int
        np.float32(2.0),  # NumPy 32-bit scalar (missed by isinstance(int|float))
        np.float64(2.0),  # NumPy 64-bit scalar
        np.array(2.0),  # 0-d NumPy array (missed by isinstance(int|float))
        jnp.array(2.0),  # 0-d JAX scalar
    ],
)
def test_as_std_array_broadcasts_scalar_like_to_n_dims(std):
    """Every scalar-like std (including np.float32 / 0-d, which the old
    isinstance(std, int|float) check silently missed) broadcasts to (n_dims,)."""
    out = as_std_array(std, 3)
    assert out.shape == (3,)
    assert jnp.all(out == 2.0)


def test_as_std_array_passes_arrays_through():
    """A per-dimension array (ndim >= 1) is returned as-is (as a JAX array)."""
    std = jnp.array([1.0, 2.0, 3.0])
    out = as_std_array(std, 3)
    assert out.shape == (3,)
    assert jnp.allclose(out, std)


def test_as_std_array_np_float32_matches_array_in_kde_model():
    """Behavioral guard: a NumPy-scalar std must broadcast per-dimension in
    KDEModel.predict, not stay 0-d and mis-broadcast against 2-D eval points."""
    samples = jnp.asarray(rng().standard_normal((40, 2)))
    eval_points = jnp.asarray(rng(1).standard_normal((7, 2)))
    scalar = KDEModel(std=np.float32(1.5)).fit(samples).predict(eval_points)
    array = KDEModel(std=jnp.array([1.5, 1.5])).fit(samples).predict(eval_points)
    assert scalar.shape == (7,)
    assert jnp.all(jnp.isfinite(scalar))
    assert jnp.allclose(scalar, array, rtol=1e-5, atol=1e-6)


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


def test_kde_and_log_kde_raise_on_dimension_mismatch():
    """kde/log_kde must error, not silently truncate, when eval points, samples,
    and std disagree on n_dims.

    A ``strict=False`` zip over ``(eval_points.T, samples.T, std)`` stops at the
    shortest iterable, dropping trailing dimensions and returning a
    plausible-but-wrong density instead of raising.
    """
    eval_points = jnp.zeros((5, 3))
    samples = jnp.zeros((4, 2))  # 2-D samples vs 3-D eval points
    std = jnp.array([1.0, 1.0, 1.0])
    weights = jnp.ones((samples.shape[0],))

    with pytest.raises(ValueError):
        kde(eval_points, samples, std, weights)
    with pytest.raises(ValueError):
        log_kde(eval_points, samples, std, weights)


def test_kde_model_predict_raises_on_dimension_mismatch():
    """KDEModel fit on 2-D samples then predicting 3-D eval points must raise
    rather than silently truncating to the shared dimensions."""
    model = KDEModel(std=jnp.array([1.0, 1.0, 1.0])).fit(jnp.zeros((10, 2)))
    with pytest.raises(ValueError):
        model.predict(jnp.zeros((5, 3)))
    with pytest.raises(ValueError):
        model.predict_log(jnp.zeros((5, 3)))


def test_kde_model_predict_before_fit_raises():
    """predict/predict_log before fit raise a clear not-fitted RuntimeError
    (matching GaussianMixtureModel) rather than a bare AttributeError."""
    model = KDEModel(std=jnp.array([1.0, 1.0]))
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(jnp.zeros((3, 2)))
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict_log(jnp.zeros((3, 2)))


def test_log_kde_consistent_with_kde_log():
    r = rng(3)
    samples = r.normal(size=(120, 2))
    eval_points = r.normal(size=(30, 2))
    std = jnp.array([1.0, 0.75])
    weights = jnp.asarray(r.uniform(0.5, 1.5, size=(samples.shape[0],)))

    log_vals = log_kde(jnp.asarray(eval_points), jnp.asarray(samples), std, weights)
    lin_vals = kde(jnp.asarray(eval_points), jnp.asarray(samples), std, weights)
    # Use moderate tolerance; avoid -inf by clipping
    lin_vals = jnp.clip(lin_vals, min=1e-12)
    assert jnp.allclose(log_vals, jnp.log(lin_vals), rtol=1e-5, atol=1e-6)


def test_log_kde_zero_weight_sample_is_dropped():
    """A zero-weight sample must not change ``log_kde`` vs. omitting it entirely.

    ``safe_log`` floors a zero weight to LOG_EPS instead of -inf, so a zero-weight
    sample leaks an ``EPS * kernel`` contribution into the log KDE. At an eval point
    that sits on the zero-weight sample but far from every positive-weight sample the
    legitimate density is ~0, so the leak dominates the result. Dropping the
    zero-weight sample (as the linear ``kde`` does exactly) must give the same value.
    """
    # Sample 0 (zero weight) sits at the origin; the positive-weight samples are far
    # away, and eval point 0 is at the origin -> only the leak can reach it.
    samples = jnp.array([[0.0, 0.0], [20.0, 20.0], [21.0, -21.0]])
    weights = jnp.array([0.0, 2.0, 1.5])
    eval_points = jnp.array([[0.0, 0.0], [20.0, 20.0], [5.0, 5.0]])
    std = jnp.array([1.0, 1.0])

    with_zero = log_kde(eval_points, samples, std, weights)
    without = log_kde(eval_points, samples[1:], std, weights[1:])

    assert jnp.allclose(with_zero, without, rtol=1e-5, atol=1e-6), (
        f"zero-weight sample leaked; max|diff|={jnp.abs(with_zero - without).max():.3e}"
    )


def test_log_kde_all_zero_weights_is_finite():
    """All-zero weights must not produce NaN (the -inf/-inf degenerate case)."""
    r = rng(7)
    samples = r.normal(size=(10, 1))
    eval_points = r.normal(size=(5, 1))
    std = jnp.array([1.0])
    log_vals = log_kde(
        jnp.asarray(eval_points),
        jnp.asarray(samples),
        std,
        jnp.zeros((samples.shape[0],)),
    )
    assert jnp.all(jnp.isfinite(log_vals))


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
    from non_local_detector.likelihoods.common import EPS

    a = jnp.array([1.0, 0.0, 2.0])
    b = jnp.array([0.0, 0.0, 4.0])
    out = safe_divide(a, b)
    assert out.shape == a.shape
    assert jnp.all(jnp.isfinite(out))
    # Zero-denominator elements should return eps, not NaN/inf
    assert out[0] == pytest.approx(EPS, rel=1e-5)
    assert out[1] == pytest.approx(EPS, rel=1e-5)
    assert out[2] == pytest.approx(0.5, rel=1e-5)

    x = jnp.array([0.0, 1e-20, 1.0])
    log_x = safe_log(x)
    assert log_x.shape == x.shape
    assert jnp.all(jnp.isfinite(log_x))
    # Zero/tiny inputs should return log(eps), not -inf
    assert log_x[0] == pytest.approx(float(jnp.log(jnp.array(EPS))), rel=1e-5)
    assert log_x[2] == pytest.approx(0.0, abs=1e-10)


def test_get_spikecount_per_time_bin_edges_and_outliers():
    time_edges = np.array([0.0, 1.0, 2.0, 3.0])
    # spikes include below-first, on edges, interior, and at last edge
    spikes = np.array([-0.5, 0.0, 0.4, 1.0, 2.99, 3.0, 3.5])
    counts = get_spikecount_per_time_bin(spikes, time_edges=time_edges)
    # Three bins [0, 1), [1, 2), [2, 3]: spikes outside [0, 3] are dropped, an
    # interior edge opens the bin to its right, and the last edge (3.0) closes
    # the final bin.
    assert counts.shape == (time_edges.shape[0] - 1,)
    assert counts.tolist() == [2, 1, 2]


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


def test_empty_eval_points_kde_model_default_block_size():
    """KDEModel with block_size=None derives block_size from eval_points length.

    When eval_points is empty, block_size becomes 0, causing
    range(0, 0, 0) -> ValueError in block_kde/block_log_kde.
    """
    r = rng(20)
    samples = r.normal(size=(50, 2))
    eval_points = np.zeros((0, 2))
    model = KDEModel(std=jnp.array([1.0, 1.0]), block_size=None).fit(
        jnp.asarray(samples)
    )

    out = model.predict(jnp.asarray(eval_points))
    assert out.shape == (0,)

    out_log = model.predict_log(jnp.asarray(eval_points))
    assert out_log.shape == (0,)


def test_empty_eval_points_block_kde_zero_block_size():
    """block_kde and block_log_kde should handle block_size=0 without error."""
    r = rng(21)
    samples = r.normal(size=(50, 2))
    eval_points = np.zeros((0, 2))
    std = jnp.array([1.0, 1.0])
    w = jnp.ones((samples.shape[0],))

    out = block_kde(
        jnp.asarray(eval_points), jnp.asarray(samples), std, block_size=0, weights=w
    )
    assert out.shape == (0,)

    out_log = block_log_kde(
        jnp.asarray(eval_points), jnp.asarray(samples), std, block_size=0, weights=w
    )
    assert out_log.shape == (0,)


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


def test_validate_population_lengths_returns_common_length():
    """Matching parallel collections return their shared unit count."""
    n = validate_population_lengths(
        "neuron",
        spike_times=[np.zeros(3), np.zeros(0), np.zeros(5)],
        place_fields=np.zeros((3, 8)),
        mean_rates=jnp.zeros((3,)),
    )
    assert n == 3


def test_validate_population_lengths_no_populations_returns_zero():
    """No collections is a benign no-op that reports zero units."""
    assert validate_population_lengths("neuron") == 0


def test_validate_population_lengths_detects_short_non_first_collection():
    """A mismatch in a collection other than the first is still caught."""
    with pytest.raises(ValidationError, match="population lengths do not match") as exc:
        validate_population_lengths(
            "electrode",
            spike_times=[np.zeros(1), np.zeros(1), np.zeros(1)],
            joint_models=[None, None, None],
            mean_rates=jnp.zeros((2,)),  # short by one
        )
    message = str(exc.value)
    assert "mean_rates=2" in message
    assert "spike_times=3" in message


@pytest.mark.unit
def test_drop_zero_weight_samples_preserves_density_and_shrinks_model():
    """Zero-weight samples add nothing to a weighted KDE, so dropping them keeps
    the density and stops every later evaluation from paying for them."""
    rng = np.random.default_rng(0)
    samples = rng.uniform(0.0, 100.0, size=(500, 2))
    weights = (np.arange(500) // 50 % 3 == 0).astype(float) * rng.uniform(0.5, 1.5, 500)
    eval_points = rng.uniform(0.0, 100.0, size=(64, 2))
    std = np.array([5.0, 5.0])

    kept_samples, kept_weights = drop_zero_weight_samples(samples, weights)
    full = KDEModel(std=std).fit(samples, weights)
    kept = KDEModel(std=std).fit(kept_samples, kept_weights)

    assert kept.samples_.shape[0] == np.count_nonzero(weights)
    np.testing.assert_allclose(
        kept.predict(eval_points), full.predict(eval_points), rtol=1e-5
    )
    # No positive weight: inputs are returned as-is (zero-exposure behavior).
    zeros = np.zeros(500)
    assert drop_zero_weight_samples(samples, zeros)[0] is samples


def _barrier_operand_sizes(jaxpr):
    sizes = []
    for equation in jaxpr.eqns:
        if equation.primitive.name == "optimization_barrier":
            sizes += [int(np.prod(var.aval.shape)) for var in equation.invars]
        for value in equation.params.values():
            for item in value if isinstance(value, (tuple, list)) else (value,):
                inner = getattr(item, "jaxpr", item)
                if hasattr(inner, "eqns"):
                    sizes += _barrier_operand_sizes(inner)
    return sizes


@pytest.mark.unit
@pytest.mark.parametrize("x64", [False, True])
def test_kernel_matrix_barrier_is_vector_sized_for_float32_inputs(x64):
    from non_local_detector.likelihoods.common import _log_kernel_matrix

    with precision_mode(x64):
        n_eval, n_samples = 64, 2000
        eval_points = jnp.ones((n_eval, 2), jnp.float32)
        std = jnp.ones(2, jnp.float32)
        tiled = jax.make_jaxpr(_log_kernel_matrix)(
            eval_points, jnp.ones((n_samples, 2), jnp.float32), std
        )
        assert all(
            size <= max(n_eval, n_samples)
            for size in _barrier_operand_sizes(tiled.jaxpr)
        )
        # A singleton sample tail keeps its divisor guard.
        singleton = jax.make_jaxpr(_log_kernel_matrix)(
            eval_points, jnp.ones((1, 2), jnp.float32), std
        )
        assert _barrier_operand_sizes(singleton.jaxpr)
