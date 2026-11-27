"""Equal KDEs have unit density ratio even when their densities underflow."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import (
    fit_clusterless_kde_encoding_model,
    fit_sorted_spikes_kde_encoding_model,
    predict_clusterless_kde_log_likelihood,
    predict_sorted_spikes_kde_log_likelihood,
)
from non_local_detector.likelihoods.clusterless_kde import (
    estimate_log_joint_mark_intensity,
    log_kde_distance,
    mark_reference_scale,
)
from non_local_detector.likelihoods.common import (
    _KDE_SAMPLE_TILE,
    LOG_EPS,
    KDEModel,
    _compiled_log_space_kde_log_density,
    _compiled_scaled_kde_log_density,
    _sorted_sample_keys,
    _tiled_kde_rows,
    kde_log_densities,
    kde_log_density,
    log_density_intensity,
    log_space_kde_log_density,
    padded_repair_count,
    padded_tile_count,
    true_kde_log_density,
)
from non_local_detector.likelihoods.common.kernel_density import (
    _KDE_REPAIR_POINTS,
    _paired_log_space_kde_log_densities,
    _repair_sizes,
    _tiled_scaled_kde_log_densities,
)
from tests.conftest import precision_mode
from tests.likelihoods.conftest import (
    count_compilations,
    realistic_log_background,
    training_mark_references,
)

pytestmark = pytest.mark.unit


def fit_clusterless_kde(
    position_time, position, spike_times, spike_waveform_features, env, **params
):
    """KDE encoding model plus the mark reference of its weighted training spikes."""
    model = fit_clusterless_kde_encoding_model(
        position_time, position, spike_times, spike_waveform_features, env, **params
    )
    reference = training_mark_references(
        position_time,
        spike_times,
        spike_waveform_features,
        mark_reference_scale(**params),
        params.get("weights"),
        position,
    )
    return {**model, "mark_reference": reference}


MODELS = [
    (
        fit_sorted_spikes_kde_encoding_model,
        predict_sorted_spikes_kde_log_likelihood,
        False,
    ),
    (fit_clusterless_kde, predict_clusterless_kde_log_likelihood, True),
]


@pytest.mark.parametrize(("fit", "predict", "clusterless"), MODELS)
@pytest.mark.parametrize("is_local", [False, True])
def test_equal_occupancy_and_spike_densities_keep_the_mean_rate(
    fit, predict, clusterless, is_local
):
    time = np.arange(5.0)
    position = np.zeros((5, 1))
    env = Environment(
        place_bin_size=5.0, position_range=((0.0, 200.0),)
    ).fit_place_grid(position, infer_track_interior=False)
    args = [time, position, [time], env]
    kwargs = {"position_std": 1.0, "disable_progress_bar": True}
    if clusterless:
        args.insert(3, [np.zeros((5, 1))])
        kwargs["waveform_std"] = 1.0
    model = fit(*args, **kwargs)
    decode_args = [time, np.full_like(position, 20.0), [np.empty(0)]]
    if clusterless:
        decode_args.append([np.empty((0, 1))])
    score = predict(
        *decode_args, time_edges=np.array([0.0, 1.0]), is_local=is_local, **model
    )
    np.testing.assert_allclose(score, -1.0, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("fit", "predict", "clusterless"), MODELS[1:])
@pytest.mark.parametrize("is_local", [False, True])
def test_mark_event_term_survives_small_position_densities(
    fit, predict, clusterless, is_local
):
    time = np.arange(5.0)
    position = np.zeros((5, 1))
    env = Environment(
        place_bin_size=5.0, position_range=((0.0, 200.0),)
    ).fit_place_grid(position, infer_track_interior=False)
    model = fit(
        time,
        position,
        [time],
        [np.zeros((5, 1))],
        env,
        position_std=1.0,
        waveform_std=1.0,
        disable_progress_bar=True,
    )
    common = dict(time_edges=np.array([0.0, 1.0]), is_local=is_local, **model)
    with_spike = predict(
        time,
        np.full_like(position, 20.0),
        [np.array([0.5])],
        [np.zeros((1, 1))],
        **common,
    )
    baseline = predict(
        time, np.full_like(position, 20.0), [np.empty(0)], [np.empty((0, 1))], **common
    )
    expected = -0.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(
        np.asarray(with_spike - baseline), expected, rtol=1e-5, atol=2e-5
    )


@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("n_features", [1, 9])
@pytest.mark.parametrize("use_streaming", [False, True])
def test_tiled_mark_ratios_match_the_gaussian_reference(
    use_streaming, n_features, is_local
):
    time = np.arange(5.0)
    position = np.zeros((5, 1))
    env = Environment(
        place_bin_size=5.0, position_range=((0.0, 200.0),)
    ).fit_place_grid(position, infer_track_interior=False)
    model = fit_clusterless_kde(
        time,
        position,
        [time],
        [np.zeros((5, n_features))],
        env,
        position_std=1.0,
        waveform_std=1.0,
        block_size=2,
        enc_tile_size=2,
        pos_tile_size=7,
        use_streaming=use_streaming,
        disable_progress_bar=True,
    )
    common = dict(time_edges=np.array([0.0, 1.0]), is_local=is_local, **model)
    baseline = predict_clusterless_kde_log_likelihood(
        time,
        np.full_like(position, 200.0),
        [np.empty(0)],
        [np.empty((0, n_features))],
        **common,
    )
    with_spike = predict_clusterless_kde_log_likelihood(
        time,
        np.full_like(position, 200.0),
        [np.array([0.5])],
        [np.zeros((1, n_features))],
        **common,
    )
    np.testing.assert_allclose(baseline, -1.0, rtol=1e-5, atol=1e-5)
    expected = -0.5 * n_features * np.log(2 * np.pi)
    np.testing.assert_allclose(with_spike - baseline, expected, rtol=1e-5, atol=2e-5)


def test_log_ratio_keeps_tiny_rate_and_reports_invalid_densities():
    log_rate = log_density_intensity(1e-30, jnp.asarray([95.0]), jnp.asarray([0.0]))
    np.testing.assert_allclose(np.exp(log_rate), 1e-30 * np.exp(95.0), rtol=1e-5)
    assert np.isneginf(
        log_density_intensity(0.0, jnp.asarray(-np.inf), jnp.asarray(-np.inf))
    )
    assert np.isnan(log_density_intensity(1.0, jnp.asarray(np.nan), jnp.asarray(0.0)))
    assert np.isnan(log_density_intensity(1.0, jnp.asarray(0.0), jnp.asarray(np.nan)))


@pytest.mark.parametrize("use_streaming", [False, True])
def test_tiled_tail_ratio_has_finite_correct_gradient_with_zero_weight_prefix(
    use_streaming,
):
    log_background = realistic_log_background(jnp.array([[0.2]]))
    log_occupancy = jnp.asarray([-200.0 - 0.5 * np.log(2 * np.pi)])
    encoding_positions = jnp.array([[100.0], [100.0], [0.0]])
    bin_centers = jnp.array([[20.0]])
    position = (
        {
            "encoding_positions": encoding_positions,
            "position_eval_points": bin_centers,
            "position_std": jnp.ones(1),
        }
        if use_streaming
        else {
            "log_position_distance": log_kde_distance(
                bin_centers, encoding_positions, jnp.ones(1)
            )
        }
    )

    def score(mark):
        return estimate_log_joint_mark_intensity(
            mark.reshape(1, 1),
            jnp.array([[4.0], [4.0], [0.0]]),
            jnp.ones(1),
            jnp.exp(log_occupancy),
            1.0,
            enc_tile_size=2,
            pos_tile_size=1,
            use_streaming=use_streaming,
            encoding_weights=jnp.array([0.0, 0.0, 1.0]),
            log_occupancy=log_occupancy,
            log_background=log_background,
            **position,
        )[0, 0]

    value, gradient = jax.value_and_grad(score)(jnp.asarray(0.2))
    log_intensity = -0.02 - 0.5 * np.log(2 * np.pi)
    expected = np.logaddexp(log_intensity, float(log_background[0]))
    np.testing.assert_allclose(value, expected, rtol=1e-5, atol=2e-5)
    # The background is fixed, so only the intensity's share carries d/dmark.
    share = np.exp(log_intensity - expected)
    np.testing.assert_allclose(gradient, -0.2 * share, rtol=1e-5, atol=1e-6)


def test_sorted_public_local_path_preserves_nan_position_diagnostic():
    model = KDEModel(std=1.0, block_size=2).fit(jnp.array([[0.0], [1.0]]))
    result = predict_sorted_spikes_kde_log_likelihood(
        position_time=np.array([0.0, 1.0]),
        position=np.full((2, 1), np.nan),
        spike_times=[np.empty(0)],
        environment=Environment(
            place_bin_size=2.0, position_range=((0.0, 1.0),)
        ).fit_place_grid(np.array([[0.0], [1.0]]), infer_track_interior=False),
        marginal_models=[model],
        occupancy_model=model,
        occupancy=jnp.ones(1),
        mean_rates=[1.0],
        place_fields=jnp.ones((1, 1)),
        no_spike_part_log_likelihood=jnp.ones(1),
        is_track_interior=np.ones(1, bool),
        is_local=True,
        time_edges=np.array([0.0, 1.0]),
        disable_progress_bar=True,
    )
    assert result.shape == (1, 1)
    assert np.isnan(np.asarray(result)[0, 0])


def _true_log_density(points, samples, std, block_size, weights):
    """True log density of a fitted KDEModel."""
    model = KDEModel(std=std, block_size=block_size).fit(samples, weights)
    return kde_log_density(model, points)


def _tail_case(n_dims):
    """Points near weighted samples, then three hundreds to thousands of nats out."""
    rng = np.random.default_rng(12)
    samples = jnp.asarray(rng.normal(0.0, 5.0, (400, n_dims)), jnp.float32)
    near = rng.normal(0.0, 5.0, (200, n_dims))
    far = np.outer([60.0, 120.0, 400.0], np.ones(n_dims))
    points = jnp.asarray(np.concatenate([near, far]), jnp.float32)
    weights = jnp.asarray(rng.uniform(0.0, 1.0, 400), jnp.float32).at[:50].set(0.0)
    return points, samples, jnp.full(n_dims, 2.0), weights


@pytest.mark.parametrize("n_dims", [1, 2])
@pytest.mark.parametrize("block_size", [64, 10_000])
def test_true_log_density_matches_log_space_evaluation_in_and_beyond_tails(
    block_size, n_dims
):
    """The one-pass scaled KDE matches full log-space evaluation everywhere.

    Points near the samples are ordinary; the last three lie far in the tail,
    where the KDE either scales them exactly (1D) or repairs them
    afterwards in log space.
    """
    points, samples, std, weights = _tail_case(n_dims)
    actual = _true_log_density(points, samples, std, block_size, weights)
    expected = log_space_kde_log_density(points, samples, std, block_size, weights)
    assert np.isfinite(expected).all()
    assert (np.asarray(expected[-3:]) < LOG_EPS).all()
    assert (np.asarray(expected[:-3]) > LOG_EPS).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-5)


def test_one_dimensional_pass_scales_distant_tails_without_log_space():
    """Sorted sample keys give 1D tail points their exact scale in one pass."""
    points, samples, std, weights = _tail_case(1)
    values, needs_log_space = _compiled_scaled_kde_log_density(
        points, samples, std, 64, weights, _sorted_sample_keys(points, samples, weights)
    )
    expected = log_space_kde_log_density(points, samples, std, 64, weights)
    assert not np.asarray(needs_log_space).any()
    np.testing.assert_allclose(values, expected, rtol=1e-6, atol=2e-5)


def _float64_log_kde(points, samples, std, weights):
    """Weighted Gaussian KDE log density in float64 NumPy."""
    points, samples = np.asarray(points, float), np.asarray(samples, float)
    std, weights = np.asarray(std, float), np.asarray(weights, float)
    log_kernel = -0.5 * np.sum(
        ((points[:, None, :] - samples[None, :, :]) / std) ** 2, axis=-1
    ) - np.sum(np.log(np.sqrt(2.0 * np.pi) * std))
    return logsumexp(log_kernel, b=weights[None, :], axis=1) - np.log(weights.sum())


def test_repairs_of_different_counts_share_compiled_kernels():
    """Flagged counts that pad to one size share every compiled repair kernel.

    Each call flags a different number of the 45 points (1 to 16, which pad
    to one index size), so only the first repair may compile.
    """
    rng = np.random.default_rng(41)
    samples = rng.normal(0.0, 5.0, (333, 2)).astype(np.float32)
    weights = rng.uniform(0.5, 1.0, 333).astype(np.float32)
    std = np.full(2, 2.0, np.float32)
    near = rng.normal(0.0, 5.0, (45, 2))
    far = 300.0 + rng.normal(0.0, 5.0, (45, 2))
    cases = {
        n_far: jnp.asarray(np.concatenate([far[:n_far], near[n_far:]]), jnp.float32)
        for n_far in (1, 2, 5, 9, 16)
    }
    arguments = (jnp.asarray(samples), jnp.asarray(std), 64, jnp.asarray(weights))
    actual = {1: true_kde_log_density(cases[1], *arguments)}
    with count_compilations() as compiled:
        for n_far in (2, 5, 9, 16):
            actual[n_far] = true_kde_log_density(cases[n_far], *arguments)
    assert compiled == []
    for n_far, points in cases.items():
        expected = _float64_log_kde(points, samples, std, weights)
        assert (expected[:n_far] < -10_000).all()
        np.testing.assert_allclose(actual[n_far], expected, rtol=1e-6, atol=2e-5)


def test_true_log_density_matches_log_space_for_extremely_light_neighbors():
    """A near sample whose weight is 1e-36 of the largest still sets the density.

    The scale ignores weights, so the point is flagged and re-evaluated in log
    space, which offsets by the largest weighted term.
    """
    samples = jnp.asarray([[0.0], [300.0]])
    weights = jnp.asarray([1e-36, 1.0])
    points = jnp.asarray([[0.0]])
    std = jnp.ones(1)
    actual = _true_log_density(points, samples, std, 64, weights)
    expected = log_space_kde_log_density(points, samples, std, 64, weights)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-4)


@pytest.mark.parametrize("entry", ["kde_log_density", "predict_log"])
def test_true_log_density_keeps_positive_weights_of_any_ratio(entry):
    """A weight 1e-40 of the largest still sets the density where it dominates.

    Dividing by the largest weight flushes that ratio to zero in float32; the
    near sample's term then vanished and the far sample's -45,000 remained.
    """
    samples = jnp.asarray([[0.0], [300.0]])
    weights = jnp.asarray([1e-30, 1e10])
    points = jnp.asarray([[0.0]])
    std = jnp.ones(1)
    if entry == "predict_log":
        actual = KDEModel(std=1.0).fit(samples, weights).predict_log(points)
    else:
        actual = _true_log_density(points, samples, std, 64, weights)
    expected = np.log(1e-30 / (1e-30 + 1e10)) - 0.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(actual, [expected], rtol=1e-6)


def test_true_log_density_recovers_mass_carried_by_terms_below_tiny():
    """Terms that each flush below ``tiny`` can still hold most of the mass.

    The nearest sample carries a relative weight of 1e-37; a thousand samples
    13.4 bandwidths away each contribute about 1e-39 of its kernel, which
    float32 flushes to zero, yet together they hold 91% of the density.
    Offsetting each point by its largest weighted term keeps them.
    """
    samples = jnp.asarray([[0.0]] + [[13.4]] * 1000)
    weights = jnp.asarray([1e-37] + [1.0] * 1000)
    points = jnp.asarray([[0.0]])
    actual = _true_log_density(points, samples, jnp.ones(1), 64, weights)
    expected = np.log(
        (1e-37 + 1000 * np.exp(-0.5 * np.float32(13.4) ** 2)) / (1e-37 + 1000)
    ) - 0.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(actual, [expected], rtol=1e-6)


@pytest.mark.parametrize("is_local", [False, True])
def test_public_tiled_prediction_keeps_the_rate_for_weights_near_float_max(is_local):
    """Encoding tiles give the untiled no-spike score with 1e38 weights."""
    time = np.arange(4.0)
    position = np.arange(4.0)[:, None]
    env = Environment(place_bin_size=1.0, position_range=((0.0, 4.0),)).fit_place_grid(
        position, infer_track_interior=False
    )
    scores = []
    for tiling in ({}, {"enc_tile_size": 2, "use_streaming": True}):
        model = fit_clusterless_kde(
            time,
            position,
            [time],
            [np.zeros((4, 1))],
            env,
            weights=np.full(4, 1e38),
            position_std=1.0,
            waveform_std=1.0,
            disable_progress_bar=True,
            **tiling,
        )
        scores.append(
            predict_clusterless_kde_log_likelihood(
                time,
                position,
                [np.empty(0)],
                [np.empty((0, 1))],
                time_edges=np.array([0.0, 1.0]),
                is_local=is_local,
                **model,
            )
        )
    np.testing.assert_allclose(scores[0], -1.0, rtol=1e-5)
    np.testing.assert_allclose(scores[1], scores[0], rtol=1e-5)


def test_true_log_density_keeps_nan_points_and_samples_visible():
    """A NaN coordinate yields NaN, as log-space evaluation does."""
    std = jnp.ones(1)
    samples = jnp.asarray([[0.0], [1.0], [2.0]])
    points = jnp.asarray([[np.nan], [0.5]])
    with_nan_point = _true_log_density(points, samples, std, 64, jnp.ones(3))
    assert np.isnan(with_nan_point[0])
    np.testing.assert_allclose(
        with_nan_point[1],
        log_space_kde_log_density(points, samples, std, 64, jnp.ones(3))[1],
        rtol=1e-6,
    )
    with_nan_sample = _true_log_density(
        jnp.asarray([[0.5], [5.0]]),
        samples.at[1].set(np.nan),
        std,
        64,
        jnp.ones(3),
    )
    assert np.isnan(with_nan_sample).all()


def test_true_log_density_keeps_empty_support_and_rejects_nan_weights():
    """Zero total weight is no support; ``KDEModel.fit`` rejects a NaN weight."""
    samples = jnp.zeros((3, 1))
    points = jnp.asarray([[0.0], [50.0]])
    std = jnp.ones(1)
    empty = _true_log_density(points, samples, std, 64, jnp.zeros(3))
    assert np.isneginf(empty).all()
    with pytest.raises(ValidationError, match="finite"):
        _true_log_density(points, samples, std, 64, jnp.asarray([1.0, jnp.nan, 1.0]))


def test_unvalidated_true_log_density_keeps_nan_weights_visible():
    """A NaN weight passed straight to the kernel gives NaN, not a density.

    The local clusterless likelihood evaluates the occupancy density through
    ``true_kde_log_density`` without ``KDEModel.fit``'s validation.
    """
    samples = jnp.zeros((3, 1))
    points = jnp.asarray([[0.0], [50.0]])
    std = jnp.ones(1)
    nan_weights = jnp.asarray([1.0, jnp.nan, 1.0])
    assert np.isnan(true_kde_log_density(points, samples, std, 64, nan_weights)).all()
    assert np.isneginf(
        true_kde_log_density(points, samples, std, 64, jnp.zeros(3))
    ).all()


@pytest.mark.parametrize("n_dims", [1, 2])
def test_batched_log_densities_match_each_model(n_dims):
    """Grouped, padded evaluation gives each model's own true log density.

    Sample counts 40 and 45 share a padded size, 300 does not; zero weights,
    distant tails and a NaN sample are kept per model.
    """
    rng = np.random.default_rng(3)
    models = []
    for n_samples in (40, 45, 300, 45):
        samples = rng.normal(0.0, 5.0, (n_samples, n_dims)).astype(np.float32)
        weights = rng.uniform(0.0, 1.0, n_samples).astype(np.float32)
        weights[:5] = 0.0
        models.append(KDEModel(std=2.0, block_size=64).fit(samples, weights))
    nan_samples = np.asarray(models[3].samples_).copy()
    nan_samples[7] = np.nan
    models[3] = KDEModel(std=2.0, block_size=64).fit(nan_samples, models[3].weights_)
    points = jnp.asarray(
        np.concatenate(
            [rng.normal(0.0, 5.0, (30, n_dims)), np.full((2, n_dims), 150.0)]
        ),
        jnp.float32,
    )
    for batched, model in zip(kde_log_densities(models, points), models, strict=True):
        np.testing.assert_allclose(
            batched, kde_log_density(model, points), rtol=1e-6, atol=2e-5
        )


def _compiled_workspace(compiled):
    return compiled.compile().memory_analysis().temp_size_in_bytes


@pytest.mark.parametrize(
    ("x64", "sample", "sample_dtype", "std", "point", "point_dtype"),
    [
        (False, 1.9, np.float32, 0.01, 2, np.int32),
        (False, 0.9, np.float32, 0.05, 1, np.int32),
        (True, 100.000004, np.float64, 1e-7, 100.0, np.float32),
        (False, 100.032, np.float32, 0.001, 100.0, np.float16),
    ],
)
def test_batched_1d_scale_keeps_sample_precision_at_other_point_dtypes(
    x64, sample, sample_dtype, std, point, point_dtype
):
    """Integer or narrower points do not round the samples that set each
    point's scale: a rounded sample moved away from the point scaled it
    below its log kernel, and the density overflowed to ``inf``."""
    with precision_mode(x64):
        model = KDEModel(std=std).fit(np.array([[sample]], sample_dtype))
        points = np.array([[point]], point_dtype)
        (batched,) = kde_log_densities([model], points)
        expected = kde_log_density(model, points)
        assert np.all(np.isfinite(batched))
        np.testing.assert_allclose(batched, expected, rtol=1e-6)


def test_batched_log_densities_group_bandwidths_by_value_and_dtype():
    """Bandwidths with equal bytes but different dtypes are different."""
    samples = np.zeros((5, 1), np.float32)
    models = [
        KDEModel(std=np.float32(1.0)).fit(samples),
        KDEModel(std=np.int32(np.float32(1.0).view(np.int32))).fit(samples),
    ]
    points = jnp.asarray([[0.0], [3.0]], jnp.float32)
    for batched, model in zip(kde_log_densities(models, points), models, strict=True):
        np.testing.assert_allclose(batched, kde_log_density(model, points), rtol=1e-6)


@pytest.mark.parametrize(
    ("n_models", "n_samples", "n_points"),
    [(1_000, 10_000, 4_000), (22, 40_000, 50_000)],
    ids=["many-models", "many-points"],
)
def test_tiled_kernel_workspace_stays_within_the_widest_model_pass(
    n_models, n_samples, n_points
):
    """Each tiled pass, sized as ``kde_log_densities`` sizes it, needs no
    more compiled workspace than the widest model's own single-model pass at
    the same points, beyond four copies of its output. The log-space pass is
    sized for every model flagged at every point, as production pads it.
    The scaled pass's tiles also serve the repair.
    2-D points, ``block_size=10_000``, the ``sorted_spikes_kde`` default."""
    block_size, n_dims = 10_000, 2
    sds = jax.ShapeDtypeStruct
    widest = -(-n_samples // _KDE_SAMPLE_TILE)
    points = sds((n_points, n_dims), jnp.float32)
    std = jnp.full(n_dims, 3.0)

    sample_dtype = sds((1, _KDE_SAMPLE_TILE, n_dims), jnp.float32)
    weight_dtype = sds((1, _KDE_SAMPLE_TILE), jnp.float32)

    rows, batch_size, n_tiles = _tiled_kde_rows(
        block_size,
        points,
        padded_tile_count(n_models * widest),
        std,
        sample_dtype,
        weight_dtype,
        widest,
    )
    tiles = (
        sds((n_tiles, _KDE_SAMPLE_TILE, n_dims), jnp.float32),
        sds((n_tiles, _KDE_SAMPLE_TILE), jnp.float32),
        sds((n_tiles,), jnp.int32),
    )
    scaled = _compiled_workspace(
        _tiled_scaled_kde_log_densities.lower(
            points,
            *tiles,
            sds((n_models,), jnp.int32),
            std,
            None,
            rows=rows,
            batch_size=batch_size,
        )
    )
    # The repair, sized as production sizes it with every model flagged at
    # every point: one item per tile and block of _KDE_REPAIR_POINTS points.
    n_blocks = padded_repair_count(n_models * -(-n_points // _KDE_REPAIR_POINTS))
    batch_size, chunk, n_items = _repair_sizes(
        padded_repair_count(n_models * widest * -(-n_points // _KDE_REPAIR_POINTS)),
        block_size,
        points,
        std,
        sample_dtype,
        weight_dtype,
        widest,
    )
    log_space = _compiled_workspace(
        _paired_log_space_kde_log_densities.lower(
            sds((n_blocks, _KDE_REPAIR_POINTS, n_dims), jnp.float32),
            sds((n_blocks,), jnp.int32),
            sds((n_items,), jnp.int32),
            sds((n_items,), jnp.int32),
            *tiles,
            std,
            n_models=n_models,
            chunk=chunk,
            batch_size=batch_size,
        )
    )
    outputs = 4 * n_models * n_points * 4
    samples = sds((n_samples, n_dims), jnp.float32)
    weights = sds((n_samples,), jnp.float32)
    assert scaled <= outputs + _compiled_workspace(
        _compiled_scaled_kde_log_density.lower(
            points, samples, std, block_size, weights, None
        )
    )
    assert log_space <= 4 * n_blocks * _KDE_REPAIR_POINTS * 4 + _compiled_workspace(
        _compiled_log_space_kde_log_density.lower(
            points, samples, std, block_size, weights
        )
    )
