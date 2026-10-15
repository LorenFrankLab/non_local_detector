"""A spike's intensity must not depend on other queries in its decode batch."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from non_local_detector.likelihoods.clusterless_kde_log import (
    _compute_log_mark_kernel_gemm,
    _gemm_log_kernel_product,
    _stabilized_log_kernel_product,
    estimate_log_joint_mark_intensity,
    log_kde_distance,
)
from non_local_detector.likelihoods.common import LOG_EPS


def _place_field_tile(seed=3, n_enc=400, n_dec=60, waveform_std=8.0):
    """Units with separated 4D uV marks and place fields on a 300 cm track."""
    rng = np.random.default_rng(seed)
    n_units = 12
    field_centers = rng.uniform(0, 300, n_units)
    mark_means = rng.uniform(40, 300, (n_units, 4))
    unit = rng.integers(0, n_units, n_enc)
    encoding_positions = np.clip(field_centers[unit] + rng.normal(0, 8, n_enc), 0, 300)
    encoding = mark_means[unit] + rng.normal(0, 12, (n_enc, 4))
    decoding = mark_means[rng.integers(0, n_units, n_dec)] + rng.normal(
        0, 12, (n_dec, 4)
    )
    decoding[: n_dec // 10] = rng.uniform(0, 500, (n_dec // 10, 4))
    bins = np.arange(0.5, 300, 1.0)[:, None]
    log_position = log_kde_distance(
        jnp.asarray(bins, jnp.float32),
        jnp.asarray(encoding_positions[:, None], jnp.float32),
        jnp.asarray([6.0], jnp.float32),
    )
    return (
        jnp.asarray(decoding, jnp.float32),
        jnp.asarray(encoding, jnp.float32),
        jnp.full(4, waveform_std, jnp.float32),
        jnp.full(bins.shape[0], 1.0 / 300, jnp.float32),
        log_position,
    )


def _reference_log_joint(decoding, encoding, stds, occupancy, mean_rate, log_position):
    """float64 brute-force log(mean_rate * sum_e K_mark K_pos / n / occupancy)."""
    decoding, encoding, stds, occupancy, log_position = (
        np.asarray(a, np.float64)
        for a in (decoding, encoding, stds, occupancy, log_position)
    )
    n_features = encoding.shape[1]
    norm = -0.5 * (n_features * np.log(2 * np.pi) + 2 * np.sum(np.log(stds)))
    log_mark = norm - 0.5 * np.sum(
        ((encoding[:, None] - decoding[None]) / stds) ** 2, axis=-1
    )
    marginal = np.stack(
        [
            logsumexp(log_mark[:, q, None] + log_position, axis=0)
            for q in range(len(decoding))
        ]
    ) - np.log(len(encoding))
    return np.log(mean_rate) + marginal - np.log(occupancy)[None]


@pytest.mark.unit
@pytest.mark.parametrize("enc_tile_size", [None, 1])
def test_tail_intensity_is_independent_of_other_decoding_spikes(enc_tile_size):
    """A representable density ratio survives beside a much larger mark kernel."""
    std = 0.001
    remote = np.zeros((1, 8), dtype=np.float32)
    remote[0, 0] = np.sqrt(180.0) * std
    encoded = jnp.zeros((2, 8))
    kwargs = {
        "encoding_spike_waveform_features": encoded,
        "waveform_stds": jnp.full(8, std),
        "occupancy": jnp.asarray([1e-14]),
        "mean_rate": 1.0,
        "log_position_distance": jnp.zeros((2, 1)),
        "enc_tile_size": enc_tile_size,
    }
    alone = estimate_log_joint_mark_intensity(jnp.asarray(remote), **kwargs)
    together = estimate_log_joint_mark_intensity(
        jnp.asarray(np.concatenate([np.zeros_like(remote), remote])), **kwargs
    )
    expected = -90.0 - 8 * np.log(std * np.sqrt(2 * np.pi)) - np.log(1e-14)
    np.testing.assert_allclose(alone[0, 0], expected, atol=3e-5, rtol=1e-5)
    np.testing.assert_allclose(together[1], alone[0], atol=3e-5, rtol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("enc_tile_size", [None, 1, 2])
@pytest.mark.parametrize(
    "n_features,gap",
    [
        (8, 100.0),  # beyond the unit-scaled window, inside the shifted one
        (16, 150.0),  # flushes even in the shifted GEMM: needs the exact reduction
    ],
)
def test_opposing_mark_and_position_support_preserves_joint_density(
    enc_tile_size, n_features, gap
):
    """Separate factor maxima are insufficient when different sources dominate.

    One sample matches the mark but is ``gap`` nats away in position; the other
    matches the position but is ``gap`` nats away in mark. Every product term
    is ``gap`` nats below the product of the factor maxima, yet the joint
    intensity is above the LOG_EPS floor and must be exact.
    """
    std = 0.001
    encoded = np.zeros((2, n_features), dtype=np.float32)
    encoded[1, 0] = np.sqrt(2 * gap) * std
    out = estimate_log_joint_mark_intensity(
        jnp.zeros((1, n_features)),
        jnp.asarray(encoded),
        jnp.full(n_features, std),
        jnp.asarray([1e-14]),
        1.0,
        jnp.asarray([[-gap], [0.0]]),
        enc_tile_size=enc_tile_size,
    )
    expected = -gap - n_features * np.log(std * np.sqrt(2 * np.pi)) - np.log(1e-14)
    assert expected > LOG_EPS
    np.testing.assert_allclose(out[0, 0], expected, atol=1e-4, rtol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("enc_tile_size", [None, 64])
def test_estimates_are_floored_at_log_eps_and_exact_above_it(enc_tile_size):
    """Callers floor the log joint at LOG_EPS; the estimate is exact above it.

    The fixture has spatial and mark tails hundreds of nats deep, so many
    entries are far below the floor while the rest must match a float64
    brute-force reduction.
    """
    decoding, encoding, stds, occupancy, log_position = _place_field_tile()
    out = np.asarray(
        estimate_log_joint_mark_intensity(
            decoding,
            encoding,
            stds,
            occupancy,
            1.0,
            log_position,
            enc_tile_size=enc_tile_size,
        )
    )
    reference = _reference_log_joint(
        decoding, encoding, stds, occupancy, 1.0, log_position
    )
    assert (reference < LOG_EPS - 100).any() and (reference > LOG_EPS).any()
    assert np.all(out >= np.float32(LOG_EPS))
    np.testing.assert_allclose(out, np.maximum(reference, LOG_EPS), atol=1e-4, rtol=0)


@pytest.mark.unit
def test_entries_below_the_floor_skip_the_exact_reduction():
    """Only entries whose omitted mass could matter above the floor are repaired.

    With no floor, the fixture's deep tails make the GEMM estimate inexact for
    some entries. With the caller's floor, none of those entries can reach it,
    and every unflagged entry above the floor is already exact.
    """
    decoding, encoding, stds, occupancy, log_position = _place_field_tile()
    log_mark = _compute_log_mark_kernel_gemm(decoding, encoding, stds)
    weighted_position = log_position - np.log(encoding.shape[0])
    log_floor = LOG_EPS + jnp.log(occupancy)  # mean_rate = 1

    _, unfloored = _gemm_log_kernel_product(
        log_mark, weighted_position, jnp.full_like(log_floor, -jnp.inf)
    )
    value, floored = _gemm_log_kernel_product(log_mark, weighted_position, log_floor)
    assert bool(jnp.any(unfloored))
    assert not bool(jnp.any(floored))

    reference = _reference_log_joint(
        decoding, encoding, stds, occupancy, 1.0, log_position
    )
    joint = np.asarray(value) - np.log(np.asarray(occupancy))[None]
    above = reference > LOG_EPS
    np.testing.assert_allclose(joint[above], reference[above], atol=1e-4, rtol=0)


@pytest.mark.unit
def test_mark_kernel_preserves_distances_after_large_translation():
    decoded = jnp.asarray([[1.0]])
    encoded = jnp.asarray([[0.0]])
    std = jnp.asarray([1.0])
    ordinary = _compute_log_mark_kernel_gemm(decoded, encoded, std)
    translated = _compute_log_mark_kernel_gemm(decoded + 10000, encoded + 10000, std)
    np.testing.assert_allclose(translated, ordinary, atol=1e-6, rtol=1e-6)


@pytest.mark.unit
def test_zero_weight_support_has_a_finite_zero_weight_gradient():
    def score(weights):
        return estimate_log_joint_mark_intensity(
            jnp.zeros((1, 2)),
            jnp.zeros((2, 2)),
            jnp.ones(2),
            jnp.ones(1),
            1.0,
            jnp.zeros((2, 1)),
            encoding_weights=weights,
        ).sum()

    gradient = jax.grad(score)(jnp.zeros(2))
    np.testing.assert_array_equal(gradient, np.zeros(2))


@pytest.mark.unit
def test_zero_weight_prefix_tiles_do_not_poison_feature_gradients():
    def score(features):
        return estimate_log_joint_mark_intensity(
            jnp.asarray([[2.5]]),
            features,
            jnp.ones(1),
            jnp.ones(1),
            1.0,
            jnp.zeros((4, 1)),
            encoding_weights=jnp.asarray([0.0, 0.0, 1.0, 1.0]),
            enc_tile_size=2,
        ).sum()

    gradient = jax.grad(score)(jnp.arange(4.0)[:, None])
    np.testing.assert_allclose(
        gradient[:, 0], [0.0, 0.0, 0.25, -0.25], rtol=1e-6, atol=1e-6
    )


@pytest.mark.unit
def test_broadcast_bandwidth_normalizes_every_mark_dimension():
    kernel = _compute_log_mark_kernel_gemm(
        jnp.zeros((1, 2)), jnp.zeros((1, 2)), jnp.asarray([2.0])
    )
    expected = -2 * np.log(2 * np.sqrt(2 * np.pi))
    np.testing.assert_allclose(kernel, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("encoding_count", [8192, 16000])
def test_large_encoding_sets_keep_compiled_kernel_workspace_bounded(encoding_count):
    """Encoding tiles keep temporaries independent of the encoding count."""
    compiled = estimate_log_joint_mark_intensity.lower(
        jax.ShapeDtypeStruct((512, 4), jnp.float32),
        jax.ShapeDtypeStruct((encoding_count, 4), jnp.float32),
        jax.ShapeDtypeStruct((4,), jnp.float32),
        jax.ShapeDtypeStruct((512,), jnp.float32),
        1.0,
        jax.ShapeDtypeStruct((encoding_count, 512), jnp.float32),
    ).compile()
    workspace = compiled.memory_analysis().temp_size_in_bytes
    assert workspace <= 32 * 2**20, f"{workspace / 2**20:.1f} MiB"


@pytest.mark.unit
def test_query_outputs_do_not_depend_on_other_queries_in_a_tile():
    """Adding a query with nonoverlapping tails leaves other queries unchanged."""
    rng = np.random.default_rng(901)
    marks = rng.uniform(-3.0, 0.0, (64, 2)).astype(np.float32)
    positions = rng.uniform(-3.0, 0.0, (64, 3)).astype(np.float32)
    positions[:32, 2] = -100.0
    positions[32:, 2] = 0.0
    no_floor = jnp.full(3, -jnp.inf)
    ordinary = jax.jit(_stabilized_log_kernel_product)(
        jnp.asarray(marks), jnp.asarray(positions), jnp.zeros(64), no_floor
    )
    marks[:32, 1] = 0.0
    marks[32:, 1] = -100.0
    changed = jax.jit(_stabilized_log_kernel_product)(
        jnp.asarray(marks), jnp.asarray(positions), jnp.zeros(64), no_floor
    )
    np.testing.assert_array_equal(changed[0], ordinary[0])
    np.testing.assert_allclose(
        changed[1, 2], -100.0 + np.log(64.0), rtol=1e-6, atol=1e-5
    )


@pytest.mark.unit
def test_exact_reduction_recovers_mass_beyond_the_scaled_window():
    """Terms flushed by the scaled GEMM are recovered when the floor needs them."""
    marks = jnp.asarray([[0.0], [-200.0], [-200.0]])
    positions = jnp.asarray([[-300.0], [0.0], [0.0]])
    actual = jax.jit(_stabilized_log_kernel_product)(
        marks, positions, jnp.zeros(3), jnp.full(1, -jnp.inf)
    )
    expected = np.logaddexp(-300.0, -200.0 + np.log(2.0))
    np.testing.assert_allclose(actual[0, 0], expected, rtol=1e-6, atol=1e-5)


def _enable_x64():
    if hasattr(jax, "enable_x64"):
        return jax.enable_x64(True)
    from jax.experimental import enable_x64

    return enable_x64(True)


@pytest.mark.unit
@pytest.mark.parametrize("use_streaming", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_reduction_dtypes_stay_consistent_with_x64_enabled(dtype, use_streaming):
    """Weakly typed loop carries must not change dtype when x64 is enabled."""
    rng = np.random.default_rng(4)
    encoding_positions = rng.uniform(0, 100, (300, 1))
    bins = np.arange(0.5, 100, 1.0)[:, None]
    encoding = rng.normal(150, 30, (300, 4))
    decoding = rng.normal(150, 30, (40, 4))
    occupancy = np.full(len(bins), 0.01)
    stds = np.full(4, 24.0)
    log_position = (
        -0.5 * np.log(2 * np.pi * 36.0)
        - 0.5 * ((bins[None, :, 0] - encoding_positions) / 6.0) ** 2
    )
    reference = _reference_log_joint(
        decoding, encoding, stds, occupancy, 1.0, log_position
    )

    with _enable_x64():
        streaming = {
            "use_streaming": True,
            "encoding_positions": jnp.asarray(encoding_positions, dtype),
            "position_eval_points": jnp.asarray(bins, dtype),
            "position_std": jnp.asarray([6.0], dtype),
        }
        out = estimate_log_joint_mark_intensity(
            jnp.asarray(decoding, dtype),
            jnp.asarray(encoding, dtype),
            jnp.asarray(stds, dtype),
            jnp.asarray(occupancy, dtype),
            1.0,
            None if use_streaming else jnp.asarray(log_position, dtype),
            enc_tile_size=128,
            **(streaming if use_streaming else {}),
        )
    out = np.asarray(out, np.float64)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, np.maximum(reference, LOG_EPS), atol=1e-4, rtol=0)


@pytest.mark.unit
@pytest.mark.parametrize("enc_tile_size", [None, 1])
def test_mixed_precision_exact_path_does_not_overflow(enc_tile_size):
    """float32 marks beside float64 position kernels share one exponent shift."""
    std, gap, n_features = 0.001, 150.0, 16
    encoded = np.zeros((2, n_features), dtype=np.float32)
    encoded[1, 0] = np.sqrt(2 * gap) * std
    expected = -gap - n_features * np.log(std * np.sqrt(2 * np.pi)) - np.log(1e-14)
    with _enable_x64():
        out = estimate_log_joint_mark_intensity(
            jnp.zeros((1, n_features), jnp.float32),
            jnp.asarray(encoded),
            jnp.full(n_features, std, jnp.float32),
            jnp.asarray([1e-14], jnp.float64),
            1.0,
            jnp.asarray([[-gap], [0.0]], jnp.float64),
            enc_tile_size=enc_tile_size,
        )
    np.testing.assert_allclose(np.asarray(out)[0, 0], expected, atol=1e-4, rtol=1e-5)
