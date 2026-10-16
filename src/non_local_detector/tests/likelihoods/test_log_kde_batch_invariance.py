"""A spike's intensity must not depend on other queries in its decode batch."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from non_local_detector.likelihoods.clusterless_kde import (
    _compute_log_mark_kernel_gemm,
    estimate_log_joint_mark_intensity,
    log_kde_distance,
)
from non_local_detector.tests.likelihoods.conftest import realistic_log_background

# Below ``log_background + log(eps)`` an intensity cannot change a float32 term.
LOG_FLOAT32_EPS = np.log(np.finfo(np.float32).eps)


def _place_field_tile(
    seed=3, n_enc=400, n_dec=60, waveform_std=8.0, return_positions=False
):
    """Units with separated 4D uV marks and place fields on a 300 cm track.

    With ``return_positions``, also returns the encoding positions and bin
    centers (for the streaming path).
    """
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
    tile = (
        jnp.asarray(decoding, jnp.float32),
        jnp.asarray(encoding, jnp.float32),
        jnp.full(4, waveform_std, jnp.float32),
        jnp.full(bins.shape[0], 1.0 / 300, jnp.float32),
        log_position,
    )
    if return_positions:
        return (
            *tile,
            jnp.asarray(encoding_positions[:, None], jnp.float32),
            jnp.asarray(bins, jnp.float32),
        )
    return tile


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
    together_marks = np.concatenate([np.zeros_like(remote), remote])
    # One fixed background per spike, shared by both batches.
    log_background = realistic_log_background(together_marks)
    kwargs = {
        "encoding_spike_waveform_features": encoded,
        "waveform_stds": jnp.full(8, std),
        "occupancy": jnp.asarray([1e-14]),
        "mean_rate": 1.0,
        "log_position_distance": jnp.zeros((2, 1)),
        "enc_tile_size": enc_tile_size,
    }
    alone = estimate_log_joint_mark_intensity(
        jnp.asarray(remote), **kwargs, log_background=log_background[1:]
    )
    together = estimate_log_joint_mark_intensity(
        jnp.asarray(together_marks), **kwargs, log_background=log_background
    )
    log_intensity = -90.0 - 8 * np.log(std * np.sqrt(2 * np.pi)) - np.log(1e-14)
    expected = np.logaddexp(log_intensity, float(log_background[1]))
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
    intensity is far above the spike's background and must be exact.
    """
    std = 0.001
    encoded = np.zeros((2, n_features), dtype=np.float32)
    encoded[1, 0] = np.sqrt(2 * gap) * std
    log_background = realistic_log_background(np.zeros((1, n_features)))
    out = estimate_log_joint_mark_intensity(
        jnp.zeros((1, n_features)),
        jnp.asarray(encoded),
        jnp.full(n_features, std),
        jnp.asarray([1e-14]),
        1.0,
        jnp.asarray([[-gap], [0.0]]),
        enc_tile_size=enc_tile_size,
        log_background=log_background,
    )
    log_intensity = -gap - n_features * np.log(std * np.sqrt(2 * np.pi)) - np.log(1e-14)
    assert log_intensity > float(log_background[0]) - LOG_FLOAT32_EPS
    expected = np.logaddexp(log_intensity, float(log_background[0]))
    np.testing.assert_allclose(out[0, 0], expected, atol=1e-4, rtol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("enc_tile_size", [None, 64])
def test_estimates_are_exact_wherever_they_change_the_background_term(enc_tile_size):
    """Each term is ``logaddexp(log joint, log background)``, exact to float32.

    The fixture has spatial and mark tails hundreds of nats deep, so many
    intensities are far below the spike's background (where they need not be
    exact) while the rest must match a float64 brute-force reduction.
    """
    decoding, encoding, stds, occupancy, log_position = _place_field_tile()
    log_background = realistic_log_background(decoding)
    out = np.asarray(
        estimate_log_joint_mark_intensity(
            decoding,
            encoding,
            stds,
            occupancy,
            1.0,
            log_position,
            enc_tile_size=enc_tile_size,
            log_background=log_background,
        )
    )
    reference = _reference_log_joint(
        decoding, encoding, stds, occupancy, 1.0, log_position
    )
    background = np.asarray(log_background, np.float64)[:, None]
    assert (reference < background - 100).any() and (reference > background).any()
    assert np.all(out >= np.asarray(log_background)[:, None])
    np.testing.assert_allclose(
        out, np.logaddexp(reference, background), atol=1e-4, rtol=0
    )


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
            log_background=realistic_log_background(np.zeros((1, 2))),
        ).sum()

    gradient = jax.grad(score)(jnp.zeros(2))
    np.testing.assert_array_equal(gradient, np.zeros(2))


@pytest.mark.unit
def test_zero_weight_prefix_tiles_do_not_poison_feature_gradients():
    log_background = realistic_log_background(np.asarray([[2.5]]))

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
            log_background=log_background,
        ).sum()

    gradient = jax.grad(score)(jnp.arange(4.0)[:, None])
    # Intensity: mean of two unit kernels at distance 0.5 (rate, occupancy and
    # position kernel are 1). d logaddexp(log I, b) = I / (I + B) * d log I.
    log_intensity = -0.125 - 0.5 * np.log(2 * np.pi)
    share = np.exp(log_intensity - np.logaddexp(log_intensity, log_background[0]))
    np.testing.assert_allclose(
        gradient[:, 0], share * np.array([0.0, 0.0, 0.25, -0.25]), rtol=1e-6, atol=1e-6
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
        log_background=jax.ShapeDtypeStruct((512,), jnp.float32),
    ).compile()
    workspace = compiled.memory_analysis().temp_size_in_bytes
    assert workspace <= 32 * 2**20, f"{workspace / 2**20:.1f} MiB"


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
    log_background = np.array(realistic_log_background(decoding))

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
            log_background=jnp.asarray(log_background, dtype),
        )
    out = np.asarray(out, np.float64)
    assert np.isfinite(out).all()
    expected = np.logaddexp(reference, log_background.astype(np.float64)[:, None])
    np.testing.assert_allclose(out, expected, atol=1e-4, rtol=0)


@pytest.mark.unit
@pytest.mark.parametrize("enc_tile_size", [None, 1])
def test_mixed_precision_exact_path_does_not_overflow(enc_tile_size):
    """float32 marks beside float64 position kernels share one exponent shift."""
    std, gap, n_features = 0.001, 150.0, 16
    encoded = np.zeros((2, n_features), dtype=np.float32)
    encoded[1, 0] = np.sqrt(2 * gap) * std
    log_intensity = -gap - n_features * np.log(std * np.sqrt(2 * np.pi)) - np.log(1e-14)
    log_background = np.asarray(realistic_log_background(np.zeros((1, n_features))))
    expected = np.logaddexp(log_intensity, float(log_background[0]))
    with _enable_x64():
        out = estimate_log_joint_mark_intensity(
            jnp.zeros((1, n_features), jnp.float32),
            jnp.asarray(encoded),
            jnp.full(n_features, std, jnp.float32),
            jnp.asarray([1e-14], jnp.float64),
            1.0,
            jnp.asarray([[-gap], [0.0]], jnp.float64),
            enc_tile_size=enc_tile_size,
            log_background=jnp.asarray(log_background, jnp.float64),
        )
    np.testing.assert_allclose(np.asarray(out)[0, 0], expected, atol=1e-4, rtol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("n_dec", [8, 100], ids=["flagged-rows-only", "every-row"])
def test_every_spike_flagged_for_exact_repair_matches_reference(n_dec):
    """A background far below the kernels flags every spike for exact repair.

    Eight spikes are repaired individually; a hundred exceed the bounded slots
    and take the every-row pass. Both must match the float64 reference.
    """
    decoding, encoding, stds, occupancy, log_position = _place_field_tile(
        n_dec=n_dec, waveform_std=1.0
    )
    log_background = jnp.full(n_dec, -1e4, jnp.float32)
    out = np.asarray(
        estimate_log_joint_mark_intensity(
            decoding,
            encoding,
            stds,
            occupancy,
            1.0,
            log_position,
            log_background=log_background,
        )
    )
    reference = _reference_log_joint(
        decoding, encoding, stds, occupancy, 1.0, log_position
    )
    expected = np.logaddexp(reference, -1e4)
    assert (reference > -1e4).mean() > 0.5  # most terms are intensity-dominated
    np.testing.assert_allclose(out, expected, atol=1e-3, rtol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("n_flagged", [3, 40])
@pytest.mark.parametrize(
    "tiling",
    [
        {},
        {"enc_tile_size": 64},
        {"pos_tile_size": 37},
        {"enc_tile_size": 64, "use_streaming": True},
    ],
    ids=["untiled", "encoding-tiled", "position-tiled", "streamed"],
)
def test_only_flagged_spikes_take_the_exact_repair(n_flagged, tiling):
    """Some spikes in a block need the exact repair and the rest do not.

    Scattered spikes get a background far below the kernels, which flags them;
    the rest keep a realistic background. Three flagged spikes use the bounded
    repair slots and forty exceed them. Every row must match the float64
    reference: repairing the wrong rows would leave flagged rows inexact.
    """
    n_dec = 100
    decoding, encoding, stds, occupancy, log_position, positions, bins = (
        _place_field_tile(n_dec=n_dec, waveform_std=1.0, return_positions=True)
    )
    rng = np.random.default_rng(4)
    flagged = rng.choice(n_dec, n_flagged, replace=False)
    log_background = np.array(realistic_log_background(decoding))
    log_background[flagged] = -1e4
    streaming = (
        {
            "encoding_positions": positions,
            "position_eval_points": bins,
            "position_std": jnp.asarray([6.0], jnp.float32),
        }
        if tiling.get("use_streaming")
        else {}
    )
    out = np.asarray(
        estimate_log_joint_mark_intensity(
            decoding,
            encoding,
            stds,
            occupancy,
            1.0,
            log_position,
            log_background=jnp.asarray(log_background),
            **tiling,
            **streaming,
        )
    )
    reference = _reference_log_joint(
        decoding, encoding, stds, occupancy, 1.0, log_position
    )
    expected = np.logaddexp(reference, log_background[:, None])
    # The flagged rows are intensity-dominated, so a missed repair shows.
    assert (reference[flagged] > -1e4).mean() > 0.5
    np.testing.assert_allclose(out, expected, atol=1e-3, rtol=1e-5)
