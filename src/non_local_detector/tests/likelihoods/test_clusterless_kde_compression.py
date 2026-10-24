"""Compressed encoding dictionaries for the clusterless KDE.

A compressed fit replaces each electrode's encoding spikes with a smaller
weighted dictionary. The mean rates and exposure still come from every spike,
and the ground process comes from the dictionary, so the expected count stays
the integral of the compressed marked intensity over marks.
"""

import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector import NonLocalClusterlessDetector
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import clusterless_kde
from non_local_detector.likelihoods.clusterless_kde import (
    _compute_log_mark_kernel_gemm,
    _mark_kernel_peak,
    estimate_log_joint_mark_intensity,
    fit_clusterless_kde_encoding_model,
    log_kde_distance,
    mark_projections,
    predict_clusterless_kde_log_likelihood,
)
from non_local_detector.likelihoods.common import (
    LOG_EPS,
    RATE_REFERENCE_SECONDS,
    PerSampleBandwidthKDEModel,
    _padded_sample_count,
    kde_log_density,
    local_kde_log_ratio,
    log_density_intensity,
    low_rank_log_mark_kernel,
)
from non_local_detector.likelihoods.kde_compression import (
    _low_rank_mahalanobis_squared,
    compress_encoding_spikes,
    merge_encoding_spikes,
)
from non_local_detector.likelihoods.mark_reference import (
    background_log_intensity,
    fit_mark_references,
)
from non_local_detector.model_checking._clusterless_models import electrode_model
from non_local_detector.time_edges import time_edges_from_centers

POSITION_STD = 4.0
WAVEFORM_STD = 2.0
# A finite background close enough to the kernel peaks that every spike stays
# on the scaled GEMM rather than the exact repair; its exp is subtracted.
LOG_BACKGROUND = -60.0
SAMPLING_FREQUENCY = 100.0


def _simulate_session(n_mark_dims=2, duration=400.0, seed=0, rare_rate=0.0):
    """One electrode on a 0-100 cm track: two units with distinct marks and fields.

    With ``rare_rate`` > 0 (Hz) a third, rare unit fires only near 95 cm with
    its own marks.
    """
    rng = np.random.default_rng(seed)
    position_time = np.arange(0.0, duration, 1.0 / SAMPLING_FREQUENCY)
    position = 50.0 + 45.0 * np.sin(2 * np.pi * position_time / 20.0)
    dt = 1.0 / SAMPLING_FREQUENCY
    centers, peaks = [25.0, 70.0], [30.0, 30.0]
    mark_means = [np.full(n_mark_dims, 10.0), np.full(n_mark_dims, 20.0)]
    if rare_rate > 0:
        centers.append(95.0)
        peaks.append(rare_rate)
        mark_means.append(np.full(n_mark_dims, 40.0))
    spike_times, marks = [], []
    for center, peak, mark_mean in zip(centers, peaks, mark_means, strict=True):
        rate = peak * np.exp(-0.5 * ((position - center) / 6.0) ** 2)
        fired = rng.random(position_time.size) < rate * dt
        times = position_time[fired] + rng.uniform(-dt / 2, dt / 2, fired.sum())
        spike_times.append(times)
        marks.append(mark_mean + rng.normal(0.0, 1.5, (fired.sum(), n_mark_dims)))
    spike_times = np.concatenate(spike_times)
    order = np.argsort(spike_times)
    spike_times = np.clip(spike_times[order], 0.0, duration - dt)
    return {
        "position_time": position_time,
        "position": position[:, None],
        "spike_times": [spike_times],
        "spike_waveform_features": [np.concatenate(marks)[order]],
    }


@pytest.fixture(scope="module")
def track_environment():
    env = Environment(
        environment_name="track", place_bin_size=2.0, position_range=((0.0, 100.0),)
    )
    return env.fit_place_grid(
        position=np.linspace(0.0, 100.0, 51)[:, None], infer_track_interior=False
    )


@pytest.fixture(scope="module")
def session():
    return _simulate_session()


def _fit(session, environment, **params):
    return fit_clusterless_kde_encoding_model(
        **session,
        environment=environment,
        position_std=POSITION_STD,
        waveform_std=WAVEFORM_STD,
        block_size=1000,
        disable_progress_bar=True,
        **params,
    )


@pytest.fixture(scope="module")
def full_encoding(session, track_environment):
    """The uncompressed fit of ``session``; tests must not modify it."""
    return _fit(session, track_environment)


def _predict(encoding, session, environment, is_local=False):
    time_edges = np.arange(0.0, 60.0 + 1e-9, 0.02)
    in_window = [times < time_edges[-1] for times in session["spike_times"]]
    return np.asarray(
        predict_clusterless_kde_log_likelihood(
            position_time=session["position_time"],
            position=session["position"],
            spike_times=[
                t[k] for t, k in zip(session["spike_times"], in_window, strict=True)
            ],
            spike_waveform_features=[
                f[k]
                for f, k in zip(
                    session["spike_waveform_features"], in_window, strict=True
                )
            ],
            time_edges=time_edges,
            is_local=is_local,
            mark_reference=fit_mark_references(
                session["spike_waveform_features"], kernel_scale=WAVEFORM_STD
            ),
            **{k: v for k, v in encoding.items() if k != "block_size"},
            block_size=1000,
        )
    )


@pytest.mark.unit
@pytest.mark.parametrize("method", ["uniform", "stratified"])
def test_dictionary_at_least_as_large_as_the_spikes_is_the_full_kde(
    session, track_environment, full_encoding, method
):
    full = full_encoding
    n_spikes = len(session["spike_times"][0])
    compressed = _fit(
        session,
        track_environment,
        encoding_compression=method,
        n_encoding_kernels=n_spikes + 10,
    )
    for is_local in (False, True):
        np.testing.assert_allclose(
            _predict(compressed, session, track_environment, is_local),
            _predict(full, session, track_environment, is_local),
            rtol=1e-6,
            atol=1e-6,
        )


@pytest.mark.unit
@pytest.mark.parametrize("method", ["uniform", "stratified"])
def test_dictionary_keeps_rates_exposure_and_total_weight(
    session, track_environment, full_encoding, method
):
    full = full_encoding
    compressed = _fit(
        session,
        track_environment,
        encoding_compression=method,
        n_encoding_kernels=200,
    )
    assert compressed["encoding_positions"][0].shape == (200, 1)
    assert compressed["encoding_spike_waveform_features"][0].shape == (200, 2)
    np.testing.assert_allclose(compressed["mean_rates"], full["mean_rates"])
    assert compressed["encoding_exposure_seconds"] == full["encoding_exposure_seconds"]
    # Uniform inverse-probability weights keep the total exactly; Pareto
    # sampling keeps it in expectation. The KDE divides by the total anyway.
    np.testing.assert_allclose(
        np.sum(compressed["encoding_weights"][0]),
        np.sum(full["encoding_weights"][0]),
        rtol=1e-6 if method == "uniform" else 0.05,
    )
    np.testing.assert_allclose(compressed["occupancy"], full["occupancy"])


@pytest.mark.unit
@pytest.mark.parametrize("method", ["uniform", "stratified"])
def test_zero_weight_spikes_never_enter_the_dictionary(
    session, track_environment, method
):
    position_time = session["position_time"]
    weights = np.where(position_time < 200.0, 1.0, 0.0)
    compressed = _fit(
        session,
        track_environment,
        weights=weights,
        encoding_compression=method,
        n_encoding_kernels=100,
    )
    assert np.all(np.asarray(compressed["encoding_weights"][0]) > 0)


@pytest.mark.unit
@pytest.mark.parametrize("method", ["uniform", "stratified"])
def test_kept_spikes_keep_their_own_position_mark_and_weight(session, method):
    """Each kept row is one original spike, its weight over its inclusion probability."""
    marks = session["spike_waveform_features"][0]
    rng = np.random.default_rng(5)
    positions = rng.uniform(0.0, 100.0, (len(marks), 1))
    weights = rng.uniform(0.5, 2.0, len(marks))
    kept_positions, kept_marks, kept_weights = compress_encoding_spikes(
        positions,
        marks,
        weights,
        method,
        200,
        np.array([POSITION_STD]),
        np.full(2, WAVEFORM_STD),
    )
    # Simulated marks are continuous, so each identifies its spike.
    source = np.array(
        [np.flatnonzero((marks == mark).all(axis=1))[0] for mark in kept_marks]
    )
    assert len(np.unique(source)) == 200
    np.testing.assert_array_equal(kept_positions, positions[source])
    inverse_probability = kept_weights / weights[source]
    assert np.all(inverse_probability >= 1.0 - 1e-12)
    if method == "uniform":
        np.testing.assert_allclose(inverse_probability, len(marks) / 200)


@pytest.mark.unit
@pytest.mark.parametrize("method", ["uniform", "stratified"])
def test_dictionary_is_reproducible_from_its_seed(session, track_environment, method):
    fits = [
        _fit(
            session,
            track_environment,
            encoding_compression=method,
            n_encoding_kernels=150,
            compression_seed=seed,
        )
        for seed in (3, 3, 4)
    ]
    first, repeat, other = (np.asarray(f["encoding_positions"][0]) for f in fits)
    np.testing.assert_array_equal(first, repeat)
    assert not np.array_equal(first, other)


@pytest.fixture(scope="module")
def one_mark_session():
    return _simulate_session(n_mark_dims=1, duration=200.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "compression",
    [
        {},
        {"encoding_compression": "uniform", "n_encoding_kernels": 150},
        {"encoding_compression": "stratified", "n_encoding_kernels": 150},
        {"encoding_compression": "merge", "compression_threshold": 1.0},
        {
            "encoding_compression": "merge",
            "compression_threshold": 1.0,
            "compression_rank": 1,
        },
    ],
)
def test_expected_count_is_the_marked_intensity_integrated_over_marks(
    one_mark_session, track_environment, compression
):
    """For every dictionary, sum_m lambda(x, m) dm equals the ground intensity."""
    encoding = _fit(one_mark_session, track_environment, **compression)
    if compression:
        n_spikes = len(one_mark_session["spike_times"][0])
        assert encoding["encoding_positions"][0].shape[0] < n_spikes
    per_kernel = encoding["encoding_waveform_stds"] is not None
    factors = encoding["encoding_mark_factors"]
    mark_grid = np.linspace(-10.0, 45.0, 2201)
    log_joint = estimate_log_joint_mark_intensity(
        jnp.asarray(mark_grid[:, None]),
        jnp.asarray(encoding["encoding_spike_waveform_features"][0]),
        jnp.asarray(
            encoding["encoding_waveform_stds"][0] if per_kernel else [WAVEFORM_STD]
        ),
        encoding["occupancy"],
        encoding["mean_rates"][0] * RATE_REFERENCE_SECONDS,
        log_kde_distance(
            jnp.asarray(track_environment.place_bin_centers_),
            jnp.asarray(encoding["encoding_positions"][0]),
            jnp.asarray(
                encoding["encoding_position_stds"][0] if per_kernel else [POSITION_STD]
            ),
        ),
        encoding_weights=jnp.asarray(encoding["encoding_weights"][0]),
        log_occupancy=encoding["log_occupancy"],
        log_background=jnp.full(mark_grid.size, LOG_BACKGROUND),
        mark_projections=None
        if factors is None
        else mark_projections(encoding["encoding_waveform_stds"][0], factors[0]),
    )
    integrated = np.trapezoid(
        np.exp(np.asarray(log_joint, float)) - np.exp(LOG_BACKGROUND),
        mark_grid,
        axis=0,
    )
    np.testing.assert_allclose(
        integrated / RATE_REFERENCE_SECONDS,
        encoding["summed_ground_process_intensity"],
        rtol=1e-3,
    )


@pytest.mark.unit
def test_stratified_dictionary_keeps_rare_units_in_rare_places(track_environment):
    session = _simulate_session(duration=400.0, rare_rate=0.5)
    marks = session["spike_waveform_features"][0]
    is_rare = marks[:, 0] > 30.0
    assert 10 <= is_rare.sum() <= 80

    def rare_kept(method, seed):
        encoding = _fit(
            session,
            track_environment,
            encoding_compression=method,
            n_encoding_kernels=300,
            compression_seed=seed,
        )
        return int(
            np.sum(
                np.asarray(encoding["encoding_spike_waveform_features"][0])[:, 0] > 30.0
            )
        )

    # Inclusion probabilities fall as the inverse square root of the stratum
    # size, so spikes in small strata are kept more often than uniformly.
    stratified = [rare_kept("stratified", seed) for seed in range(10)]
    uniform = [rare_kept("uniform", seed) for seed in range(10)]
    assert min(stratified) >= 1
    assert np.mean(stratified) > 2 * np.mean(uniform)


@pytest.mark.unit
def test_stratified_weights_are_unbiased_over_seeds(
    session, track_environment, full_encoding
):
    """Averaged over seeds, the weighted dictionary reproduces the spike histogram."""
    full = full_encoding
    edges = np.linspace(0.0, 100.0, 11)
    target = np.histogram(
        np.asarray(full["encoding_positions"][0])[:, 0],
        edges,
        weights=np.asarray(full["encoding_weights"][0]),
    )[0]
    estimates = []
    for seed in range(40):
        encoding = _fit(
            session,
            track_environment,
            encoding_compression="stratified",
            n_encoding_kernels=300,
            compression_seed=seed,
        )
        estimates.append(
            np.histogram(
                np.asarray(encoding["encoding_positions"][0])[:, 0],
                edges,
                weights=np.asarray(encoding["encoding_weights"][0]),
            )[0]
        )
    # atol: one spike of Monte Carlo slack for nearly empty bins.
    np.testing.assert_allclose(np.mean(estimates, axis=0), target, rtol=0.05, atol=1.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "params",
    [
        {"encoding_compression": "kmeans", "n_encoding_kernels": 10},
        {"encoding_compression": "uniform"},
        {"encoding_compression": "stratified", "n_encoding_kernels": 0},
        {"encoding_compression": "uniform", "n_encoding_kernels": 2.5},
        {"encoding_compression": "merge", "n_encoding_kernels": 100},
        {"encoding_compression": "merge", "compression_threshold": -1.0},
        {"encoding_compression": "merge", "compression_threshold": float("nan")},
        # Settings that would otherwise be ignored.
        {"n_encoding_kernels": 100},
        {"compression_threshold": 2.0},
        {
            "encoding_compression": "stratified",
            "n_encoding_kernels": 100,
            "compression_threshold": 2.0,
        },
    ],
)
def test_invalid_compression_settings_raise(session, track_environment, params):
    with pytest.raises(ValidationError):
        _fit(session, track_environment, **params)


@pytest.mark.unit
@pytest.mark.parametrize(
    "method,n_kernels", [("merge", 10), ("Uniform", 10), ("uniform", 0)]
)
def test_subsampling_rejects_other_methods_and_empty_sizes(session, method, n_kernels):
    marks = session["spike_waveform_features"][0]
    with pytest.raises(ValidationError):
        compress_encoding_spikes(
            np.zeros((len(marks), 1)),
            marks,
            np.ones(len(marks)),
            method,
            n_kernels,
            np.array([POSITION_STD]),
            np.full(2, WAVEFORM_STD),
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    "compression",
    [
        {"encoding_compression": "stratified", "n_encoding_kernels": 200},
        {"encoding_compression": "merge", "compression_threshold": 1.5},
        {
            "encoding_compression": "merge",
            "compression_threshold": 1.5,
            "compression_rank": 2,
        },
    ],
)
def test_detector_fits_and_decodes_with_a_compressed_kde(session, compression):
    detector = NonLocalClusterlessDetector(
        environments=Environment(place_bin_size=2.0, position_range=((0.0, 100.0),)),
        clusterless_algorithm="clusterless_kde",
        clusterless_algorithm_params={
            "position_std": POSITION_STD,
            "waveform_std": WAVEFORM_STD,
            "block_size": 1000,
            **compression,
        },
        sampling_frequency=SAMPLING_FREQUENCY,
    )
    detector.fit(
        session["position_time"],
        session["position"],
        session["spike_times"],
        session["spike_waveform_features"],
    )
    encoding = next(iter(detector.encoding_model_.values()))
    n_kernels = encoding["encoding_positions"][0].shape[0]
    assert n_kernels == compression.get("n_encoding_kernels", n_kernels)
    assert n_kernels < len(session["spike_times"][0])
    is_decoded = session["position_time"] < 30.0
    in_window = session["spike_times"][0] < 30.0
    results = detector.predict(
        [session["spike_times"][0][in_window]],
        [session["spike_waveform_features"][0][in_window]],
        time_edges=time_edges_from_centers(session["position_time"][is_decoded]),
        position=session["position"][is_decoded],
        position_time=session["position_time"][is_decoded],
    )
    assert np.all(np.isfinite(results.acausal_state_probabilities.values))


# --- Kernel merging (Sodkomkham et al. 2016): per-kernel bandwidths ---------


def _reference_log_mixture(points, means, stds, weights):
    """log sum_k w_k N(points; means_k, diag stds_k**2) / sum w, in float64.

    Returns shape (n_kernels-reduced, n_points) -> (n_points,).
    """
    points, means, stds = (np.asarray(a, np.float64) for a in (points, means, stds))
    z = (points[None, :, :] - means[:, None, :]) / stds[:, None, :]
    log_kernel = (
        -0.5 * np.sum(z**2, axis=2)
        - np.sum(np.log(stds * np.sqrt(2 * np.pi)), axis=1)[:, None]
    )
    log_w = np.log(np.asarray(weights, np.float64) / np.sum(weights))
    peak = np.max(log_kernel + log_w[:, None], axis=0)
    return peak + np.log(np.sum(np.exp(log_kernel + log_w[:, None] - peak), axis=0))


def _reference_joint_at_bins(k):
    """float64 reference of ``_kernel_joint_intensity`` without background.

    Returns shape (n_queries, n_bins).
    """
    joint = np.concatenate([k["positions"], k["marks"]], axis=1)
    joint_stds = np.concatenate([k["position_stds"], k["mark_stds"]], axis=1)

    def at_bins(query):
        return np.column_stack([k["bins"][:, 0], np.tile(query, (len(k["bins"]), 1))])

    if "mark_factors" not in k:
        return np.stack(
            [
                _reference_log_mixture(at_bins(q), joint, joint_stds, k["weights"])
                for q in k["queries"]
            ]
        )
    n_kernels, _, rank = k["mark_factors"].shape
    joint_factors = np.concatenate(
        [np.zeros((n_kernels, k["positions"].shape[1], rank)), k["mark_factors"]],
        axis=1,
    )
    return np.stack(
        [
            _reference_log_low_rank_mixture(
                at_bins(q), joint, joint_stds**2, joint_factors, k["weights"]
            )
            for q in k["queries"]
        ]
    )


def _kernel_joint_intensity(k, log_background, queries=None, **options):
    """``estimate_log_joint_mark_intensity`` of a kernel fixture at its bins.

    Kernels with ``mark_factors`` are low-rank. Position kernels are streamed
    with ``use_streaming=True`` and precomputed otherwise.
    """
    queries = k["queries"] if queries is None else queries
    streaming = options.get("use_streaming", False)
    positions = jnp.asarray(k["positions"])
    position_std = jnp.asarray(k["position_stds"])
    return np.asarray(
        estimate_log_joint_mark_intensity(
            jnp.asarray(queries),
            jnp.asarray(k["marks"]),
            jnp.asarray(k["mark_stds"]),
            jnp.ones(len(k["bins"])),
            1.0,
            None
            if streaming
            else log_kde_distance(jnp.asarray(k["bins"]), positions, position_std),
            encoding_positions=positions if streaming else None,
            position_eval_points=jnp.asarray(k["bins"]) if streaming else None,
            position_std=position_std if streaming else None,
            encoding_weights=jnp.asarray(k["weights"]),
            log_occupancy=jnp.zeros(len(k["bins"])),
            log_background=jnp.full(len(queries), log_background),
            mark_projections=mark_projections(k["mark_stds"], k["mark_factors"])
            if "mark_factors" in k
            else None,
            **options,
        )
    )


@pytest.fixture(scope="module")
def heterogeneous_kernels():
    rng = np.random.default_rng(1)
    n_kernels = 37
    return {
        "positions": rng.uniform(0.0, 100.0, (n_kernels, 1)),
        "marks": rng.normal(15.0, 5.0, (n_kernels, 3)),
        "position_stds": rng.uniform(2.0, 9.0, (n_kernels, 1)),
        "mark_stds": rng.uniform(1.0, 4.0, (n_kernels, 3)),
        "weights": rng.uniform(0.5, 3.0, n_kernels),
        "bins": np.linspace(0.0, 100.0, 23)[:, None],
        "queries": rng.normal(15.0, 6.0, (11, 3)),
    }


@pytest.mark.unit
def test_log_kde_distance_takes_one_bandwidth_per_sample(heterogeneous_kernels):
    k = heterogeneous_kernels
    expected = np.stack(
        [
            _reference_log_mixture(
                k["bins"],
                k["positions"][i : i + 1],
                k["position_stds"][i : i + 1],
                [1.0],
            )
            for i in range(len(k["weights"]))
        ]
    )
    np.testing.assert_allclose(
        log_kde_distance(
            jnp.asarray(k["bins"]),
            jnp.asarray(k["positions"]),
            jnp.asarray(k["position_stds"]),
        ),
        expected,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"enc_tile_size": 8},
        {"enc_tile_size": 8, "pos_tile_size": 5},
        {"enc_tile_size": 8, "use_streaming": True},
        {"use_gemm": False},
    ],
)
@pytest.mark.parametrize("log_background", [-30.0, -np.inf])
def test_joint_intensity_with_per_kernel_bandwidths(
    heterogeneous_kernels, options, log_background
):
    """Every reduction path sums each kernel at its own bandwidth.

    A background of -30 keeps every spike on the scaled GEMM; -inf sends every
    spike to the exact repair.
    """
    np.testing.assert_allclose(
        _kernel_joint_intensity(heterogeneous_kernels, log_background, **options),
        np.logaddexp(_reference_joint_at_bins(heterogeneous_kernels), log_background),
        rtol=1e-5,
        atol=1e-4,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "std,message",
    [
        (np.ones((5, 2)), "one row per sample"),
        (np.array([[1.0], [1.0], [0.0], [1.0]]), r"std\[2, 0\] = 0\.0"),
        (np.array([[1.0], [np.nan], [1.0], [1.0]]), r"std\[1, 0\] = nan"),
    ],
)
def test_per_sample_bandwidth_errors_say_which_check_failed(std, message):
    with pytest.raises(ValidationError, match=message):
        PerSampleBandwidthKDEModel(std=std).fit(np.zeros((4, 1)))


@pytest.mark.unit
def test_local_ratio_with_per_kernel_bandwidths(heterogeneous_kernels):
    k = heterogeneous_kernels
    rng = np.random.default_rng(2)
    positions = rng.uniform(0.0, 100.0, (len(k["queries"]), 1))
    log_occupancy = rng.normal(-4.0, 0.3, len(k["queries"]))
    expected = (
        np.array(
            [
                _reference_log_mixture(
                    np.concatenate([position, query])[None, :],
                    np.concatenate([k["positions"], k["marks"]], axis=1),
                    np.concatenate([k["position_stds"], k["mark_stds"]], axis=1),
                    k["weights"],
                )[0]
                for position, query in zip(positions, k["queries"], strict=True)
            ]
        )
        - log_occupancy
    )
    ratio = local_kde_log_ratio(
        jnp.asarray(positions),
        jnp.asarray(k["queries"]),
        jnp.asarray(k["positions"]),
        jnp.asarray(k["marks"]),
        jnp.asarray(k["position_stds"]),
        jnp.asarray(k["mark_stds"]),
        jnp.asarray(log_occupancy),
        jnp.asarray(k["weights"]),
        4,
    )
    np.testing.assert_allclose(np.asarray(ratio), expected, rtol=1e-5, atol=1e-4)


@pytest.mark.unit
def test_merging_two_kernels_matches_their_first_two_moments():
    positions = np.array([[10.0], [11.0]])
    marks = np.array([[5.0, 7.0], [6.0, 7.5]])
    weights = np.array([1.0, 3.0])
    position_std, waveform_std = np.array([2.0]), np.array([1.0, 1.0])
    merged = merge_encoding_spikes(
        positions, marks, weights, 2.0, position_std, waveform_std
    )
    centres = np.concatenate([positions, marks], axis=1)
    mean = np.average(centres, axis=0, weights=weights)
    variance = np.concatenate([position_std, waveform_std]) ** 2 + np.average(
        (centres - mean) ** 2, axis=0, weights=weights
    )
    np.testing.assert_allclose(merged.weights, [4.0])
    np.testing.assert_allclose(
        np.concatenate([merged.positions, merged.marks], axis=1)[0], mean, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.concatenate([merged.position_stds, merged.mark_stds], axis=1)[0] ** 2,
        variance,
        rtol=1e-5,
    )


def _merge_in_order(joint, threshold, rank=0):
    """Merge equal-weight spikes (first column position) visited in the listed order.

    Base bandwidths are 1. The visiting order is a seeded permutation; use the
    first seed whose permutation is the identity.
    """
    joint = np.asarray(joint, float)
    seed = next(
        seed
        for seed in range(1000)
        if np.array_equal(
            np.random.default_rng(seed).permutation(len(joint)), np.arange(len(joint))
        )
    )
    return merge_encoding_spikes(
        joint[:, :1],
        joint[:, 1:],
        np.ones(len(joint)),
        threshold,
        np.ones(1),
        np.ones(joint.shape[1] - 1),
        seed,
        rank=rank,
    )


@pytest.mark.unit
@pytest.mark.parametrize("distance,n_kernels", [(1.5, 1), (1.9, 1), (2.1, 2)])
def test_spikes_merge_within_the_threshold_distance(distance, n_kernels):
    """Mahalanobis distance <= threshold merges, compared as squared distances."""
    merged = _merge_in_order([[0.0, 0.0], [0.0, distance]], threshold=2.0)
    assert len(merged.weights) == n_kernels


@pytest.mark.unit
@pytest.mark.parametrize("offset,n_kernels", [(2.4, 1), (2.8, 2)])
def test_spikes_merge_by_distance_in_the_kernels_own_bandwidths(offset, n_kernels):
    """After 0 and 1.8 merge (variance 1 + 0.81), 2.4 from their mean is 1.78
    of its bandwidths and merges; 2.8 is 2.08 and does not. Both are farther
    than 2 base bandwidths."""
    merged = _merge_in_order(
        [[0.0, 0.0], [0.0, 1.8], [0.0, 0.9 + offset]], threshold=2.0
    )
    assert len(merged.weights) == n_kernels


@pytest.mark.unit
@pytest.mark.parametrize(
    "direction,n_kernels",
    [
        # Along the merged pair's axis: 2.25 / sqrt(1 + 0.81) = 1.67 <= 2.
        ([1.0, 1.0, 0.0], 1),
        # Across it: 2.25 by the full covariance; 1.90 by its diagonal alone.
        ([1.0, -1.0, 0.0], 2),
    ],
)
def test_low_rank_merges_by_full_mahalanobis_distance(direction, n_kernels):
    """Merging 0 and 1.8 u, u = (1, 1, 0) / sqrt(2), gives covariance
    I + 0.81 u u^T over (position, mark, mark); a diagonal distance would
    merge the probe across that axis too."""
    axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    probe = 0.9 * axis + 2.25 * np.asarray(direction) / np.sqrt(2)
    merged = _merge_in_order([np.zeros(3), 1.8 * axis, probe], threshold=2.0, rank=1)
    assert len(merged.weights) == n_kernels


@pytest.mark.unit
@pytest.mark.parametrize("threshold", [0.5, 1.0, 3.0])
def test_merging_keeps_total_weight_mean_and_second_moment(session, threshold):
    """Moment-matched merges leave the whole mixture's first two moments unchanged."""
    marks = session["spike_waveform_features"][0]
    positions = np.interp(
        session["spike_times"][0], session["position_time"], session["position"][:, 0]
    )[:, None]
    weights = np.ones(len(marks))
    base_std = np.array([POSITION_STD, WAVEFORM_STD, WAVEFORM_STD])
    merged = merge_encoding_spikes(
        positions, marks, weights, threshold, base_std[:1], base_std[1:]
    )
    assert len(merged.weights) < len(weights)
    centres = np.concatenate([positions, marks], axis=1)
    merged_centres = np.concatenate([merged.positions, merged.marks], axis=1)
    merged_variances = (
        np.concatenate([merged.position_stds, merged.mark_stds], axis=1) ** 2
    )
    np.testing.assert_allclose(np.sum(merged.weights), np.sum(weights), rtol=1e-5)
    np.testing.assert_allclose(
        np.average(merged_centres, axis=0, weights=merged.weights),
        np.average(centres, axis=0, weights=weights),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        np.average(
            merged_variances + merged_centres**2, axis=0, weights=merged.weights
        ),
        np.average(base_std**2 + centres**2, axis=0, weights=weights),
        rtol=1e-4,
    )


@pytest.mark.unit
def test_larger_merge_threshold_gives_fewer_kernels(session, track_environment):
    sizes = [
        _fit(
            session,
            track_environment,
            encoding_compression="merge",
            compression_threshold=threshold,
        )["encoding_positions"][0].shape[0]
        for threshold in (0.5, 1.0, 2.0)
    ]
    assert sizes[0] > sizes[1] > sizes[2]


@pytest.mark.unit
def test_zero_merge_threshold_is_the_full_kde(
    session, track_environment, full_encoding
):
    full = full_encoding
    merged = _fit(
        session,
        track_environment,
        encoding_compression="merge",
        compression_threshold=0.0,
    )
    assert merged["encoding_positions"][0].shape == full["encoding_positions"][0].shape
    for is_local in (False, True):
        np.testing.assert_allclose(
            _predict(merged, session, track_environment, is_local),
            _predict(full, session, track_environment, is_local),
            rtol=1e-5,
            atol=1e-4,
        )


@pytest.mark.unit
def test_zero_merge_threshold_merges_exact_duplicates(session, track_environment):
    """Coincident spikes merge into one kernel of their summed weight."""
    order = np.argsort(np.tile(session["spike_times"][0], 2), kind="stable")
    doubled = {
        **session,
        "spike_times": [np.tile(session["spike_times"][0], 2)[order]],
        "spike_waveform_features": [
            np.tile(session["spike_waveform_features"][0], (2, 1))[order]
        ],
    }
    merged = _fit(
        doubled,
        track_environment,
        encoding_compression="merge",
        compression_threshold=0.0,
    )
    n_spikes = len(session["spike_times"][0])
    assert merged["encoding_positions"][0].shape[0] == n_spikes
    np.testing.assert_allclose(merged["encoding_weights"][0], 2.0)
    np.testing.assert_allclose(
        _predict(merged, doubled, track_environment),
        _predict(_fit(doubled, track_environment), doubled, track_environment),
        rtol=1e-5,
        atol=1e-3,
    )


@pytest.mark.unit
def test_merged_dictionary_is_reproducible_from_its_seed(session, track_environment):
    fits = [
        _fit(
            session,
            track_environment,
            encoding_compression="merge",
            compression_seed=seed,
        )["encoding_positions"][0]
        for seed in (5, 5, 6)
    ]
    np.testing.assert_array_equal(fits[0], fits[1])
    assert not np.array_equal(fits[0], fits[2])


@pytest.mark.unit
def test_merged_model_predicts_with_several_electrodes(track_environment):
    """Per-kernel bandwidths stay aligned with each electrode's padded kernels."""
    sessions = [_simulate_session(seed=seed, duration=200.0) for seed in (0, 1, 2)]
    session = {
        "position_time": sessions[0]["position_time"],
        "position": sessions[0]["position"],
        "spike_times": [s["spike_times"][0] for s in sessions],
        "spike_waveform_features": [
            s["spike_waveform_features"][0] * scale
            for s, scale in zip(sessions, (1.0, 1.3, 0.8), strict=True)
        ],
    }
    merged = _fit(
        session,
        track_environment,
        encoding_compression="merge",
        compression_threshold=1.0,
    )
    sizes = {m.shape[0] for m in merged["encoding_positions"]}
    assert len(sizes) > 1
    for is_local in (False, True):
        log_likelihood = _predict(merged, session, track_environment, is_local)
        assert np.all(np.isfinite(log_likelihood))
    # Electrodes add: the combined prediction is the sum of single-electrode
    # predictions, each with its own ground intensity.
    combined = _predict(merged, session, track_environment)
    singles = []
    for electrode in range(3):
        single = {
            key: [value[electrode]] if isinstance(value, list) else value
            for key, value in merged.items()
        }
        log_ground = kde_log_density(
            merged["gpi_models"][electrode],
            jnp.asarray(track_environment.place_bin_centers_),
        )
        single["summed_ground_process_intensity"] = jnp.exp(
            log_density_intensity(
                merged["mean_rates"][electrode], log_ground, merged["log_occupancy"]
            )
        )
        single_session = {
            **session,
            "spike_times": [session["spike_times"][electrode]],
            "spike_waveform_features": [session["spike_waveform_features"][electrode]],
        }
        singles.append(_predict(single, single_session, track_environment))
    np.testing.assert_allclose(combined, np.sum(singles, axis=0), rtol=1e-5, atol=1e-3)


@pytest.mark.unit
def test_model_checking_rejects_merged_dictionaries(session, track_environment):
    """Its mark CDFs assume one shared bandwidth, which merged kernels lack."""
    merged = _fit(session, track_environment, encoding_compression="merge")
    with pytest.raises(ValueError, match="merge"):
        electrode_model(merged, "clusterless_kde", 0)
    subsampled = _fit(
        session,
        track_environment,
        encoding_compression="uniform",
        n_encoding_kernels=100,
    )
    electrode_model(subsampled, "clusterless_kde", 0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "compression",
    [
        {"encoding_compression": "uniform", "n_encoding_kernels": 100},
        {"encoding_compression": "stratified", "n_encoding_kernels": 100},
        {"encoding_compression": "merge"},
        {"encoding_compression": "merge", "compression_rank": 2},
    ],
)
def test_electrode_without_encoding_spikes_fits_and_predicts(
    session, track_environment, compression
):
    with_empty = {
        **session,
        "spike_times": [session["spike_times"][0], np.array([])],
        "spike_waveform_features": [
            session["spike_waveform_features"][0],
            np.zeros((0, 2)),
        ],
    }
    encoding = _fit(with_empty, track_environment, **compression)
    assert encoding["encoding_positions"][1].shape[0] == 0
    for is_local in (False, True):
        assert np.all(
            np.isfinite(_predict(encoding, with_empty, track_environment, is_local))
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "options",
    [{"enc_tile_size": 7}, {"enc_tile_size": 64, "use_streaming": True}],
)
def test_merged_model_predicts_alike_on_every_reduction_path(
    session, track_environment, options
):
    merged = _fit(session, track_environment, encoding_compression="merge")
    reference = _predict(merged, session, track_environment)
    np.testing.assert_allclose(
        _predict({**merged, **options}, session, track_environment),
        reference,
        rtol=1e-5,
        atol=1e-3,
    )


# --- Low-rank merged kernels: mark covariance diag(d) + U U^T ---------------


def _reference_log_low_rank_mixture(points, means, variances, factors, weights):
    """log sum_k w_k N(points; means_k, diag(variances_k) + U_k U_k^T) / sum w, float64.

    Returns shape (n_points,).
    """
    points, means, variances, factors = (
        np.asarray(a, np.float64) for a in (points, means, variances, factors)
    )
    log_kernel = []
    for mean, variance, factor in zip(means, variances, factors, strict=True):
        covariance = np.diag(variance) + factor @ factor.T
        chol = np.linalg.cholesky(covariance)
        z = np.linalg.solve(chol, (points - mean).T)
        log_kernel.append(
            -0.5 * np.sum(z**2, axis=0)
            - np.sum(np.log(np.diag(chol)))
            - 0.5 * len(mean) * np.log(2 * np.pi)
        )
    log_kernel = np.stack(log_kernel)
    log_w = np.log(np.asarray(weights, np.float64) / np.sum(weights))
    peak = np.max(log_kernel + log_w[:, None], axis=0)
    return peak + np.log(np.sum(np.exp(log_kernel + log_w[:, None] - peak), axis=0))


@pytest.fixture(scope="module")
def low_rank_kernels(heterogeneous_kernels):
    rng = np.random.default_rng(3)
    k = heterogeneous_kernels
    return {**k, "mark_factors": rng.normal(0.0, 1.5, (len(k["weights"]), 3, 2))}


@pytest.mark.unit
@pytest.mark.parametrize("offset", [0.0, 1e5])
def test_low_rank_mark_kernel_matches_a_dense_gaussian(low_rank_kernels, offset):
    """Large shared offsets keep nearby distances, as for diagonal kernels."""
    k = low_rank_kernels
    # The reference sees the float32 values the kernel sees.
    marks = (k["marks"] + offset).astype(np.float32)
    queries = (k["queries"] + offset).astype(np.float32)
    expected = np.stack(
        [
            _reference_log_low_rank_mixture(
                queries,
                marks[i : i + 1],
                k["mark_stds"][i : i + 1] ** 2,
                k["mark_factors"][i : i + 1],
                [1.0],
            )
            for i in range(len(k["weights"]))
        ]
    )
    actual = _compute_log_mark_kernel_gemm(
        jnp.asarray(queries, jnp.float32),
        jnp.asarray(marks, jnp.float32),
        jnp.asarray(k["mark_stds"], jnp.float32),
        mark_projections(k["mark_stds"], k["mark_factors"]),
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-4)


@pytest.mark.unit
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"enc_tile_size": 8},
        {"enc_tile_size": 8, "pos_tile_size": 5},
        {"enc_tile_size": 8, "use_streaming": True},
    ],
)
@pytest.mark.parametrize("log_background", [-30.0, -np.inf])
def test_joint_intensity_with_low_rank_kernels(
    low_rank_kernels, options, log_background
):
    """Every reduction path, scaled GEMM (-30) and exact repair (-inf)."""
    np.testing.assert_allclose(
        _kernel_joint_intensity(low_rank_kernels, log_background, **options),
        np.logaddexp(_reference_joint_at_bins(low_rank_kernels), log_background),
        rtol=1e-5,
        atol=1e-4,
    )


@pytest.mark.unit
def test_local_ratio_with_low_rank_kernels(low_rank_kernels):
    k = low_rank_kernels
    rng = np.random.default_rng(2)
    positions = rng.uniform(0.0, 100.0, (len(k["queries"]), 1))
    log_occupancy = rng.normal(-4.0, 0.3, len(k["queries"]))
    joint_factors = np.concatenate(
        [np.zeros((len(k["weights"]), 1, 2)), k["mark_factors"]], axis=1
    )
    expected = (
        np.array(
            [
                _reference_log_low_rank_mixture(
                    np.concatenate([position, query])[None, :],
                    np.concatenate([k["positions"], k["marks"]], axis=1),
                    np.concatenate([k["position_stds"], k["mark_stds"]], axis=1) ** 2,
                    joint_factors,
                    k["weights"],
                )[0]
                for position, query in zip(positions, k["queries"], strict=True)
            ]
        )
        - log_occupancy
    )
    ratio = local_kde_log_ratio(
        jnp.asarray(positions),
        jnp.asarray(k["queries"]),
        jnp.asarray(k["positions"]),
        jnp.asarray(k["marks"]),
        jnp.asarray(k["position_stds"]),
        jnp.asarray(k["mark_stds"]),
        jnp.asarray(log_occupancy),
        jnp.asarray(k["weights"]),
        4,
        mark_projections=mark_projections(k["mark_stds"], k["mark_factors"]),
    )
    np.testing.assert_allclose(np.asarray(ratio), expected, rtol=1e-5, atol=1e-4)


@pytest.mark.unit
@pytest.mark.parametrize("extreme_mark", [1e30, 3e38])
def test_low_rank_kernels_of_overflowing_marks_are_background(
    low_rank_kernels, extreme_mark
):
    """A mark whose squared distance overflows has zero kernel mass, not NaN.

    The diagonal part ``-0.5 * |z|**2`` and the low-rank part
    ``0.5 * |C^-1 V^T z|**2`` both overflow there; their sum must not be
    ``-inf + inf``.
    """
    k = low_rank_kernels
    queries = np.array(k["queries"])
    queries[3, 0] = extreme_mark
    log_joint = _kernel_joint_intensity(k, -30.0, queries=queries)
    assert np.all(np.isfinite(log_joint))
    np.testing.assert_allclose(log_joint[3], -30.0)
    ratio = np.asarray(
        local_kde_log_ratio(
            jnp.zeros((len(queries), 1)),
            jnp.asarray(queries, jnp.float32),
            jnp.asarray(k["positions"]),
            jnp.asarray(k["marks"]),
            jnp.asarray(k["position_stds"]),
            jnp.asarray(k["mark_stds"]),
            jnp.zeros(len(queries)),
            jnp.asarray(k["weights"]),
            4,
            mark_projections=mark_projections(k["mark_stds"], k["mark_factors"]),
        )
    )
    assert ratio[3] == -np.inf
    assert np.all(np.isfinite(np.delete(ratio, 3)))


@pytest.mark.unit
@pytest.mark.parametrize("rank", [2, 8])
def test_low_rank_kernel_workspace_does_not_grow_with_features_or_rank(rank):
    """31 features: no (n_samples, n_eval_points) temporary per feature or direction."""
    n_samples, n_eval, n_features = 512, 512, 31
    compiled = (
        jax.jit(low_rank_log_mark_kernel)
        .lower(
            jax.ShapeDtypeStruct((n_eval, n_features), jnp.float32),
            jax.ShapeDtypeStruct((n_samples, n_features), jnp.float32),
            jax.ShapeDtypeStruct((n_samples, n_features), jnp.float32),
            (
                jax.ShapeDtypeStruct((n_samples, rank, n_features), jnp.float32),
                jax.ShapeDtypeStruct((n_samples,), jnp.float32),
            ),
        )
        .compile()
    )
    workspace = compiled.memory_analysis().temp_size_in_bytes
    kernel_bytes = n_samples * n_eval * 4
    assert workspace <= kernel_bytes, f"{workspace / kernel_bytes:.1f} kernel matrices"


def _merged_moments(merged):
    """Weighted mean and second moments (position variance, mark matrix) of a merge."""
    weights = merged.weights / merged.weights.sum()
    factors = (
        merged.mark_factors
        if merged.mark_factors is not None
        else np.zeros((len(weights), merged.marks.shape[1], 0))
    )
    mark_covariances = np.einsum(
        "kd,de->kde", merged.mark_stds**2, np.eye(merged.marks.shape[1])
    ) + factors @ np.swapaxes(factors, 1, 2)
    mark_second = np.einsum(
        "k,kde->de",
        weights,
        mark_covariances + np.einsum("kd,ke->kde", merged.marks, merged.marks),
    )
    position_second = np.sum(
        weights[:, None] * (merged.position_stds**2 + merged.positions**2), axis=0
    )
    mean = np.concatenate([weights @ merged.positions, weights @ merged.marks])
    return mean, position_second, mark_second


@pytest.mark.unit
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_low_rank_merge_keeps_weight_mean_and_second_moments(session, rank):
    """The mark block's second moments are kept exactly at full rank.

    Below full rank, the diagonal (each dimension's second moment) is kept, as
    in the diagonal merge. Position keeps its marginal second moment.
    """
    marks = session["spike_waveform_features"][0][:300]
    positions = np.interp(
        session["spike_times"][0][:300],
        session["position_time"],
        session["position"][:, 0],
    )[:, None]
    weights = np.random.default_rng(0).uniform(0.5, 2.0, len(marks))
    position_std, waveform_std = np.array([POSITION_STD]), np.full(2, WAVEFORM_STD)
    merged = merge_encoding_spikes(
        positions, marks, weights, 3.0, position_std, waveform_std, rank=rank
    )
    assert len(merged.weights) < len(weights)
    assert merged.mark_factors.shape == (len(merged.weights), 2, rank)
    w = weights / weights.sum()
    mean, position_second, mark_second = _merged_moments(merged)
    np.testing.assert_allclose(np.sum(merged.weights), np.sum(weights), rtol=1e-5)
    np.testing.assert_allclose(
        mean, np.concatenate([w @ positions, w @ marks]), rtol=1e-4
    )
    np.testing.assert_allclose(
        position_second, w @ (position_std**2 + positions**2), rtol=1e-4
    )
    expected_marks = np.diag(waveform_std**2) + np.einsum(
        "k,kd,ke->de", w, marks, marks
    )
    if rank == 3:  # the joint dimension: every merge is moment-matched exactly
        np.testing.assert_allclose(mark_second, expected_marks, rtol=1e-4)
    else:
        np.testing.assert_allclose(
            np.diag(mark_second), np.diag(expected_marks), rtol=1e-4
        )


@pytest.mark.unit
@pytest.mark.parametrize("rank", [1, 2, 8])
@pytest.mark.parametrize("elongation", [1.0, 100.0, 1e4])
def test_low_rank_merge_distance_is_accurate_for_elongated_kernels(rank, elongation):
    """float32 squared distances to diag(d) + U U^T kernels match float64.

    Offsets mostly along the factor cancel the most in Woodbury's
    ``|z|**2 - |C^-1 V^T z|**2``; at whitened factors of 1e4 that form rounds
    distances of ~100 to 0.
    """
    rng = np.random.default_rng(0)
    n_kernels, n_dims = 64, 12
    factors = (
        rng.normal(size=(n_kernels, n_dims, rank))
        * elongation
        * rng.uniform(0.01, 1.0, (n_kernels, 1, rank))
    )
    variances = rng.uniform(0.5, 2.0, (n_kernels, n_dims))
    offsets = np.einsum(
        "cdr,cr->cd", factors, rng.normal(size=(n_kernels, rank))
    ) + rng.normal(0.0, 3.0, (n_kernels, n_dims))
    expected = np.array(
        [
            x @ np.linalg.solve(np.diag(d) + u @ u.T, x)
            for x, u, d in zip(offsets, factors, variances, strict=True)
        ]
    )
    actual = _low_rank_mahalanobis_squared(
        *(jnp.asarray(a, jnp.float32) for a in (offsets, factors, variances))
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-3)


@pytest.mark.unit
@pytest.mark.parametrize("rank", [0, 2])
def test_merging_no_kernel_mass_gives_the_same_dtypes_as_merging_some(rank):
    positions = np.array([[10.0], [11.0]])
    marks = np.array([[5.0, 7.0], [6.0, 7.5]])
    args = (2.0, np.array([2.0]), np.array([1.0, 1.0]))
    some = merge_encoding_spikes(positions, marks, np.ones(2), *args, rank=rank)
    none = merge_encoding_spikes(positions, marks, np.zeros(2), *args, rank=rank)
    assert len(none.weights) == 0
    for field, value in some._asdict().items():
        if value is not None:
            assert getattr(none, field).dtype == value.dtype, field


@pytest.mark.unit
def test_low_rank_merge_of_two_kernels_matches_their_covariance():
    positions = np.array([[10.0], [11.0]])
    marks = np.array([[5.0, 7.0], [6.0, 7.5]])
    weights = np.array([1.0, 3.0])
    position_std, waveform_std = np.array([2.0]), np.array([1.0, 1.0])
    merged = merge_encoding_spikes(
        positions, marks, weights, 2.0, position_std, waveform_std, rank=1
    )
    mark_mean = np.average(marks, axis=0, weights=weights)
    mark_covariance = np.diag(waveform_std**2) + np.cov(
        marks.T, aweights=weights, bias=True
    )
    np.testing.assert_allclose(merged.weights, [4.0])
    np.testing.assert_allclose(merged.marks[0], mark_mean, rtol=1e-6)
    np.testing.assert_allclose(
        np.diag(merged.mark_stds[0] ** 2)
        + merged.mark_factors[0] @ merged.mark_factors[0].T,
        mark_covariance,
        rtol=1e-5,
        atol=1e-6,
    )
    position_variance = position_std**2 + np.average(
        (positions - np.average(positions, axis=0, weights=weights)) ** 2,
        axis=0,
        weights=weights,
    )
    np.testing.assert_allclose(
        merged.position_stds[0] ** 2, position_variance, rtol=1e-5
    )


@pytest.fixture(scope="module")
def low_rank_model(track_environment):
    """A rank-2 merge of 3-D marks, fitted once."""
    session = _simulate_session(n_mark_dims=3, duration=200.0)
    encoding = _fit(
        session,
        track_environment,
        encoding_compression="merge",
        compression_threshold=1.5,
        compression_rank=2,
    )
    factors = encoding["encoding_mark_factors"][0]
    assert factors.shape[1:] == (3, 2)
    whitened = factors / encoding["encoding_waveform_stds"][0][:, :, None]
    assert np.abs(whitened).max() > 0.5, "the factors must matter"
    return session, encoding


# Each chosen spike sits alone in a bin this many seconds on either side of it.
_SPIKE_HALF_WIDTH = 0.005


def _single_spike_terms(encoding, session, environment, is_local=False):
    """Per-spike terms of the prediction, isolated by a no-spike prediction.

    Four spikes each sit alone in a bin; subtracting the same bins' prediction
    without spikes leaves ``logaddexp(log intensity, log background)`` plus
    ``log(bin width / RATE_REFERENCE_SECONDS)`` for each spike.
    """
    times = session["spike_times"][0]
    marks = session["spike_waveform_features"][0]
    picks = np.linspace(20, len(times) - 20, 4).astype(int)
    edges = np.sort(
        np.concatenate(
            [times[picks] - _SPIKE_HALF_WIDTH, times[picks] + _SPIKE_HALF_WIDTH]
        )
    )
    assert np.all(np.diff(edges) > 0)
    params = {k: v for k, v in encoding.items() if k != "block_size"}

    def predict(rows):
        return np.asarray(
            predict_clusterless_kde_log_likelihood(
                position_time=session["position_time"],
                position=session["position"],
                spike_times=[times[rows]],
                spike_waveform_features=[marks[rows]],
                time_edges=edges,
                is_local=is_local,
                mark_reference=fit_mark_references(
                    session["spike_waveform_features"], kernel_scale=WAVEFORM_STD
                ),
                block_size=1000,
                **params,
            ),
            np.float64,
        )

    return picks, (predict(picks) - predict(picks[:0]))[0::2]


def _dense_spike_terms(encoding, session, picks, points):
    """float64 reference of ``_single_spike_terms`` at each spike's (position, mark) points.

    ``points`` maps a pick index to the joint points to evaluate, shape
    (n_points, 1 + n_features); the last axis of the result runs over them.
    """
    factors = encoding["encoding_mark_factors"][0]
    means = np.concatenate(
        [
            encoding["encoding_positions"][0],
            encoding["encoding_spike_waveform_features"][0],
        ],
        axis=1,
    )
    variances = (
        np.concatenate(
            [
                encoding["encoding_position_stds"][0],
                encoding["encoding_waveform_stds"][0],
            ],
            axis=1,
        )
        ** 2
    )
    joint_factors = np.concatenate([np.zeros((len(factors), 1, 2)), factors], axis=1)
    reference = fit_mark_references(
        session["spike_waveform_features"], kernel_scale=WAVEFORM_STD
    )[0]
    log_rate = np.log(float(encoding["mean_rates"][0]) * RATE_REFERENCE_SECONDS)
    terms = []
    for pick in picks:
        joint, log_occupancy = points(pick)
        log_mixture = _reference_log_low_rank_mixture(
            joint, means, variances, joint_factors, encoding["encoding_weights"][0]
        )
        log_background = background_log_intensity(
            reference, session["spike_waveform_features"][0][pick][None], LOG_EPS
        )[0]
        terms.append(
            np.logaddexp(log_rate + log_mixture - log_occupancy, log_background)
            + np.log(2 * _SPIKE_HALF_WIDTH / RATE_REFERENCE_SECONDS)
        )
    return np.stack(terms)


@pytest.mark.unit
def test_low_rank_prediction_matches_its_dense_mixture(
    low_rank_model, track_environment
):
    """Non-local predicted spike terms equal a float64 dense-covariance mixture."""
    session, encoding = low_rank_model
    picks, actual = _single_spike_terms(encoding, session, track_environment)
    bins = track_environment.place_bin_centers_
    marks = session["spike_waveform_features"][0]
    expected = _dense_spike_terms(
        encoding,
        session,
        picks,
        lambda pick: (
            np.column_stack([bins[:, 0], np.tile(marks[pick], (len(bins), 1))]),
            np.asarray(encoding["log_occupancy"], np.float64),
        ),
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-3)


@pytest.mark.unit
def test_low_rank_local_prediction_matches_its_dense_mixture(
    low_rank_model, track_environment
):
    """Local spike terms: the dense mixture at the animal's position over occupancy."""
    session, encoding = low_rank_model
    picks, actual = _single_spike_terms(
        encoding, session, track_environment, is_local=True
    )
    times = session["spike_times"][0]
    marks = session["spike_waveform_features"][0]

    def at_animal(pick):
        position = np.interp(
            times[pick], session["position_time"], session["position"][:, 0]
        )
        log_occupancy = float(
            kde_log_density(encoding["occupancy_model"], jnp.asarray([[position]]))[0]
        )
        return np.concatenate([[position], marks[pick]])[None], log_occupancy

    expected = _dense_spike_terms(encoding, session, picks, at_animal)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-3)


@pytest.mark.unit
@pytest.mark.parametrize(
    "options",
    [
        {"enc_tile_size": 10},
        {"pos_tile_size": 5},
        {"enc_tile_size": 10, "use_streaming": True},
    ],
)
def test_low_rank_prediction_is_alike_on_every_reduction_path(
    low_rank_model, track_environment, options
):
    session, encoding = low_rank_model
    n_padded = _padded_sample_count(encoding["encoding_positions"][0].shape[0])
    assert n_padded % 10, "the last encoding tile is ragged"
    np.testing.assert_allclose(
        _predict({**encoding, **options}, session, track_environment),
        _predict(encoding, session, track_environment),
        rtol=1e-5,
        atol=1e-3,
    )


@pytest.mark.unit
@pytest.mark.parametrize("is_local,smallest_change", [(False, 1.0), (True, 0.1)])
def test_low_rank_factors_change_the_prediction(
    low_rank_model, track_environment, is_local, smallest_change
):
    """Dropping the factors (diagonal kernels, same dictionary) changes it clearly."""
    session, encoding = low_rank_model
    low_rank = _predict(encoding, session, track_environment, is_local)
    diagonal = _predict(
        {**encoding, "encoding_mark_factors": None},
        session,
        track_environment,
        is_local,
    )
    assert np.all(np.isfinite(low_rank))
    assert np.max(np.abs(low_rank - diagonal)) > smallest_change


@pytest.mark.unit
@pytest.mark.parametrize(
    "params",
    [
        {"compression_rank": 1},
        {
            "encoding_compression": "uniform",
            "n_encoding_kernels": 50,
            "compression_rank": 1,
        },
        {"encoding_compression": "merge", "compression_rank": -1},
        {"encoding_compression": "merge", "compression_rank": 4},
        {"encoding_compression": "merge", "compression_rank": 1.5},
    ],
)
def test_invalid_compression_rank_raises(session, track_environment, params):
    with pytest.raises(ValidationError):
        _fit(session, track_environment, **params)


@pytest.mark.unit
def test_compression_rank_is_checked_before_any_electrode_merges(
    session, track_environment, monkeypatch
):
    """Rank 4 fits the first electrode's 1 + 4 dimensions, not the second's 1 + 2."""
    merged = []
    monkeypatch.setattr(
        clusterless_kde,
        "merge_encoding_spikes",
        lambda *args, **kwargs: merged.append(args) or None,
    )
    marks = session["spike_waveform_features"][0]
    two_electrodes = {
        **session,
        "spike_times": session["spike_times"] * 2,
        "spike_waveform_features": [np.concatenate([marks, marks], axis=1), marks],
    }
    with pytest.raises(ValidationError, match="compression_rank"):
        _fit(
            two_electrodes,
            track_environment,
            encoding_compression="merge",
            compression_rank=4,
        )
    assert not merged


def _drop_last_row(arrays):
    return [arrays[0][:-1], *arrays[1:]]


@pytest.mark.unit
@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize(
    "edit",
    [
        {"encoding_position_stds": None},
        {"encoding_waveform_stds": None, "encoding_mark_factors": None},
        {"encoding_waveform_stds": None},
        "encoding_position_stds",
        "encoding_waveform_stds",
        "encoding_mark_factors",
    ],
)
def test_prediction_rejects_per_kernel_arrays_that_do_not_fit_the_kernels(
    low_rank_model, track_environment, edit, is_local
):
    """Missing partners, or one row short of the dictionary (an edited model)."""
    session, encoding = low_rank_model
    edited = (
        {**encoding, **edit}
        if isinstance(edit, dict)
        else {**encoding, edit: _drop_last_row(encoding[edit])}
    )
    with pytest.raises(ValidationError):
        _predict(edited, session, track_environment, is_local)


@pytest.mark.unit
def test_zero_compression_rank_is_the_diagonal_merge(session, track_environment):
    diagonal = _fit(session, track_environment, encoding_compression="merge")
    rank_zero = _fit(
        session, track_environment, encoding_compression="merge", compression_rank=0
    )
    assert rank_zero["encoding_mark_factors"] is None
    for key in ("encoding_positions", "encoding_waveform_stds", "encoding_weights"):
        np.testing.assert_array_equal(diagonal[key][0], rank_zero[key][0])


@pytest.fixture(scope="module")
def correlated_kernels(heterogeneous_kernels):
    """Strongly elongated mark kernels: whitened factors of 10-30."""
    rng = np.random.default_rng(4)
    k = heterogeneous_kernels
    n_kernels = len(k["weights"])
    whitened = rng.uniform(10.0, 30.0, (n_kernels, 3, 2)) * rng.choice(
        [-1.0, 1.0], (n_kernels, 3, 2)
    )
    return {**k, "mark_factors": whitened * k["mark_stds"][:, :, None]}


@pytest.mark.unit
def test_exactness_gate_peak_is_the_largest_low_rank_kernel_value(correlated_kernels):
    """The gate's mark peak bounds every kernel value and is attained at a centre."""
    k = correlated_kernels
    projections = mark_projections(k["mark_stds"], k["mark_factors"])
    assert float(jnp.min(projections[1])) > 3.0, "log det C of several nats"
    stds = jnp.asarray(k["mark_stds"], jnp.float32)
    peak = float(_mark_kernel_peak(stds, 3, projections, jnp.float32))
    at_centres = np.asarray(
        _compute_log_mark_kernel_gemm(
            jnp.asarray(k["marks"], jnp.float32),
            jnp.asarray(k["marks"], jnp.float32),
            stds,
            projections,
        )
    )
    elsewhere = np.asarray(
        _compute_log_mark_kernel_gemm(
            jnp.asarray(k["queries"], jnp.float32),
            jnp.asarray(k["marks"], jnp.float32),
            stds,
            projections,
        )
    )
    assert max(at_centres.max(), elsewhere.max()) <= peak + 1e-4
    np.testing.assert_allclose(np.max(np.diag(at_centres)), peak, atol=1e-3)


@pytest.mark.unit
def test_correlated_low_rank_kernels_are_exact_on_the_scaled_gemm(correlated_kernels):
    """A background of -30 (scaled GEMM) agrees with -inf (exact repair) and a dense mixture."""
    k = correlated_kernels
    scaled = _kernel_joint_intensity(k, -30.0).astype(np.float64)
    exact = _kernel_joint_intensity(k, -np.inf).astype(np.float64)
    np.testing.assert_allclose(scaled, np.logaddexp(exact, -30.0), rtol=1e-5, atol=1e-4)
    expected = _reference_joint_at_bins(k)
    above = expected > -25.0
    assert above.any()
    np.testing.assert_allclose(exact[above], expected[above], rtol=1e-5, atol=2e-3)


@pytest.mark.unit
@pytest.mark.parametrize(
    "compression",
    [
        {"encoding_compression": "stratified", "n_encoding_kernels": 200},
        {"encoding_compression": "merge", "compression_threshold": 1.5},
        {
            "encoding_compression": "merge",
            "compression_threshold": 1.5,
            "compression_rank": 2,
        },
    ],
)
def test_compressed_fit_skips_untracked_spikes(session, track_environment, compression):
    """Spikes in a tracking gap leave the dictionary; predictions stay finite."""
    in_gap = (session["position_time"] > 30.0) & (session["position_time"] < 40.0)
    with_gap = {
        **session,
        "position": np.where(in_gap[:, None], np.nan, session["position"]),
    }
    encoding = _fit(with_gap, track_environment, **compression)
    assert np.all(np.isfinite(np.asarray(encoding["encoding_positions"][0])))
    for is_local in (False, True):
        assert np.all(
            np.isfinite(_predict(encoding, session, track_environment, is_local))
        )


@pytest.mark.unit
@pytest.mark.parametrize("fixture", ["low_rank_kernels", "correlated_kernels"])
def test_low_rank_kernels_stay_on_the_scaled_gemm_near_their_peak(
    request, fixture, monkeypatch
):
    """At a background of -30 no spike needs the exact repair; at -inf all do.

    Otherwise the scaled-GEMM tests above would compare the repair to itself.
    """
    k = request.getfixturevalue(fixture)
    flagged = []
    repair = clusterless_kde._repair_flagged

    def counting_repair(values, needs_exact, repaired):
        jax.debug.callback(lambda n: flagged.append(int(n)), jnp.sum(needs_exact))
        return repair(values, needs_exact, repaired)

    monkeypatch.setattr(clusterless_kde, "_repair_flagged", counting_repair)
    for log_background in (-30.0, -np.inf):
        jax.clear_caches()
        _kernel_joint_intensity(k, log_background)
    jax.clear_caches()
    assert flagged == [0, len(k["queries"])]


@pytest.mark.unit
@pytest.mark.parametrize("is_local", [False, True])
def test_pickled_low_rank_model_predicts_the_same(
    low_rank_model, track_environment, is_local
):
    session, encoding = low_rank_model
    restored = pickle.loads(pickle.dumps(encoding))
    np.testing.assert_array_equal(
        _predict(restored, session, track_environment, is_local),
        _predict(encoding, session, track_environment, is_local),
    )


@pytest.mark.unit
def test_model_checking_uses_stratified_dictionary_weights(session, track_environment):
    """Its intensity equals the KDE's at every bin, up to the rate's units.

    Stratified weights vary, so a model ignoring them would differ by mark.
    """
    encoding = _fit(
        session,
        track_environment,
        encoding_compression="stratified",
        n_encoding_kernels=200,
    )
    weights = np.asarray(encoding["encoding_weights"][0])
    assert np.ptp(weights) > 0.1 * weights.mean()
    model = electrode_model(encoding, "clusterless_kde", 0)
    marks = session["spike_waveform_features"][0][::50]
    checked = model.log_intensity_at_bins(marks).T  # (n_marks, n_bins)
    predicted = np.asarray(
        estimate_log_joint_mark_intensity(
            jnp.asarray(marks),
            jnp.asarray(model.marks),
            jnp.asarray(model.waveform_std),
            jnp.exp(jnp.asarray(model._log_occupancy_at_bins)),
            1.0,
            log_kde_distance(
                jnp.asarray(model.bin_centers),
                jnp.asarray(model.positions),
                jnp.asarray(model.position_std),
            ),
            encoding_weights=jnp.asarray(weights),
            log_occupancy=jnp.asarray(model._log_occupancy_at_bins),
            log_background=jnp.full(len(marks), -np.inf),
        )
    )
    supported = np.isfinite(checked) & (checked > checked.max() - 30.0)
    difference = (checked - predicted)[supported]
    np.testing.assert_allclose(difference, difference[0], atol=1e-3)
    assert np.all(
        np.isfinite(
            model.conditional_mark_cdf(
                session["position"][: len(marks)], marks, np.arange(marks.shape[1])
            )
        )
    )


@pytest.mark.integration
def test_em_refits_a_merged_dictionary_each_iteration():
    """Encoding refits merge the posterior-weighted spikes again."""
    session = _simulate_session(duration=60.0)
    detector = NonLocalClusterlessDetector(
        environments=Environment(place_bin_size=2.0, position_range=((0.0, 100.0),)),
        clusterless_algorithm="clusterless_kde",
        clusterless_algorithm_params={
            "position_std": POSITION_STD,
            "waveform_std": WAVEFORM_STD,
            "block_size": 1000,
            "encoding_compression": "merge",
            "compression_threshold": 1.5,
            "compression_rank": 1,
        },
        sampling_frequency=SAMPLING_FREQUENCY,
    )
    results = detector.estimate_parameters(
        session["position_time"],
        session["position"],
        session["spike_times"],
        session["spike_waveform_features"],
        time_edges=time_edges_from_centers(session["position_time"]),
        estimate_encoding_model=True,
        max_iter=2,
    )
    assert np.all(np.isfinite(results.acausal_state_probabilities.values))
    encoding = next(iter(detector.encoding_model_.values()))
    n_kernels = encoding["encoding_positions"][0].shape[0]
    assert 0 < n_kernels < len(session["spike_times"][0])
    assert encoding["encoding_mark_factors"][0].shape == (n_kernels, 2, 1)
    assert np.sum(encoding["encoding_weights"][0]) < len(session["spike_times"][0])
