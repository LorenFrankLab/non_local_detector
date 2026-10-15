"""IRCM and MDCI goodness-of-fit transforms of fitted clusterless models.

Yousefi, Amidi, Nazari and Eden (2020) map marked spike events to points that
are uniform on ``[0, 1]^(d+1)`` under the model that generated them. Each test
fits an encoding model, simulates held-out spikes exactly from that fitted
model, and checks that the transforms are uniform for it and reject it when
the rate or the marks are misspecified. The simulators compute the model's
intensities independently of the code under test.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats
from scipy.special import logsumexp

from non_local_detector.environment import Environment
from non_local_detector.likelihoods import (
    fit_clusterless_kde_encoding_model,
    fit_clusterless_kde_log_encoding_model,
)
from non_local_detector.likelihoods.clusterless_diffusion import (
    fit_clusterless_diffusion_encoding_model,
)
from non_local_detector.likelihoods.clusterless_gmm import (
    fit_clusterless_gmm_encoding_model,
)
from non_local_detector.likelihoods.diffusion import (
    get_device_basis,
    heat_kernel_apply,
)
from non_local_detector.likelihoods.gmm import (
    GaussianMixtureModel,
    _compute_precision_cholesky,
)
from non_local_detector.model_checking._clusterless_models import _log_matmul_exp
from non_local_detector.model_checking.clusterless import (
    ircm_transform,
    ks_uniformity_test,
    mdci_transform,
    multivariate_ks_test,
    pearson_chi_squared_test,
)

pytestmark = pytest.mark.unit

ALGORITHMS = [
    "clusterless_kde",
    "clusterless_kde_log",
    "clusterless_gmm",
    "clusterless_diffusion",
]
POSITION_STD = 5.0
WAVEFORM_STD = 1.0
SAMPLING_FREQUENCY = 100.0


def _trajectory(duration, phase):
    """A back-and-forth run over a 0-100 cm track."""
    time = np.arange(0.0, duration, 1.0 / SAMPLING_FREQUENCY)
    position = 50.0 + 45.0 * np.sin(2 * np.pi * time / 20.0 + phase)
    return time, position[:, None]


def _training_spikes(time, position, rng):
    """Two units on one electrode with separate place fields and mark clusters."""
    dt = 1.0 / SAMPLING_FREQUENCY
    spike_times, marks = [], []
    for center, mark_mean in ((30.0, (0.0, 0.0)), (70.0, (3.0, 2.0))):
        rate = 25.0 * np.exp(-0.5 * ((position[:, 0] - center) / 8.0) ** 2) + 1.0
        is_spike = rng.uniform(size=time.size) < rate * dt
        times = time[is_spike] + rng.uniform(0, dt, is_spike.sum())
        spike_times.append(times)
        marks.append(rng.normal(mark_mean, 0.7, (times.size, 2)))
    order = np.argsort(np.concatenate(spike_times))
    return np.concatenate(spike_times)[order], np.concatenate(marks)[order]


@pytest.fixture(scope="module")
def recording():
    rng = np.random.default_rng(7)
    time, position = _trajectory(400.0, 0.0)
    environment = Environment(place_bin_size=2.5).fit_place_grid(
        position, infer_track_interior=False
    )
    spike_times, marks = _training_spikes(time, position, rng)
    return environment, time, position, spike_times, marks


@pytest.fixture(scope="module")
def fitted(recording):
    environment, time, position, spike_times, marks = recording
    models = {}
    for algorithm, fit in (
        ("clusterless_kde", fit_clusterless_kde_encoding_model),
        ("clusterless_kde_log", fit_clusterless_kde_log_encoding_model),
    ):
        models[algorithm] = fit(
            time,
            position,
            [spike_times],
            [marks],
            environment,
            position_std=POSITION_STD,
            waveform_std=WAVEFORM_STD,
            disable_progress_bar=True,
        )
    models["clusterless_gmm"] = fit_clusterless_gmm_encoding_model(
        time,
        position,
        [spike_times],
        [marks],
        environment,
        gmm_components_occupancy=4,
        gmm_components_gpi=4,
        gmm_components_joint=6,
        disable_progress_bar=True,
    )
    models["clusterless_diffusion"] = fit_clusterless_diffusion_encoding_model(
        time,
        position,
        [spike_times],
        [marks],
        environment,
        position_std=POSITION_STD,
        waveform_std=WAVEFORM_STD,
        disable_progress_bar=True,
    )
    return models


def _gaussian(x, mean, std):
    return np.prod(
        np.exp(-0.5 * ((x[:, None, :] - mean[None]) / std) ** 2)
        / (np.sqrt(2 * np.pi) * std),
        axis=2,
    )


def _kde_density(points, samples, weights, std, block=500):
    """Weighted Gaussian KDE at ``points``, evaluated in blocks of points."""
    samples, weights = np.asarray(samples, float), np.asarray(weights, float)
    return np.concatenate(
        [
            _gaussian(points[i : i + block], samples, std) @ (weights / weights.sum())
            for i in range(0, len(points), block)
        ]
    )


def _kde_ground(model, positions):
    occupancy, gpi = model["occupancy_model"], model["gpi_models"][0]
    std = np.broadcast_to(np.asarray(model["position_std"], float), (1,))
    return (
        model["mean_rates"][0]
        * _kde_density(positions, gpi.samples_, gpi.weights_, std)
        / _kde_density(positions, occupancy.samples_, occupancy.weights_, std)
    )


def _kde_components(model, positions):
    """Component probabilities of f(m | x), their mark centres and bandwidths."""
    gpi = model["gpi_models"][0]
    std = np.broadcast_to(np.asarray(model["position_std"], float), (1,))
    kernel = _gaussian(positions, np.asarray(gpi.samples_, float), std) * np.asarray(
        gpi.weights_, float
    )
    centers = np.asarray(model["encoding_spike_waveform_features"][0], float)
    return kernel / kernel.sum(axis=1, keepdims=True), centers


def _gmm_ground(model, positions):
    gpi, occupancy = model["gpi_models"][0], model["occupancy_model"]
    return model["mean_rates"][0] * np.exp(
        np.asarray(gpi.score_samples(positions), float)
        - np.asarray(occupancy.score_samples(positions), float)
    )


def _gmm_conditional(model, positions):
    """Responsibilities and conditional Gaussians of the joint GMM given x."""
    joint = model["joint_models"][0]
    weights = np.asarray(joint.weights_, float)
    means = np.asarray(joint.means_, float)
    covariances = np.asarray(joint.covariances_, float)
    log_resp = np.stack(
        [
            np.log(w) + stats.norm.logpdf(positions[:, 0], m[0], np.sqrt(c[0, 0]))
            for w, m, c in zip(weights, means, covariances, strict=True)
        ],
        axis=1,
    )
    resp = np.exp(log_resp - log_resp.max(axis=1, keepdims=True))
    resp /= resp.sum(axis=1, keepdims=True)
    gains = covariances[:, 1:, 0] / covariances[:, 0, 0, None]
    cond_means = means[None, :, 1:] + gains[None] * (
        positions[:, None, :1] - means[None, :, :1]
    )
    cond_covs = covariances[:, 1:, 1:] - np.einsum(
        "ki,kj->kij", gains, covariances[:, 0, 1:]
    )
    return resp, cond_means, cond_covs


def _diffusion_bins(model, environment, positions):
    node_order = np.asarray(model["node_order"])
    full_to_local = np.full(len(environment.place_bin_centers_), -1)
    full_to_local[node_order] = np.arange(node_order.size)
    return full_to_local[environment.get_bin_ind(positions)]


def _diffusion_ground(model, environment, positions):
    """One electrode: the summed ground intensity is the electrode's own."""
    bins = _diffusion_bins(model, environment, positions)
    return np.asarray(model["summed_ground_process_intensity"], float)[bins]


def _diffusion_components(model, environment, positions):
    """Heat-kernel component probabilities of f(m | x) and mark centres."""
    node_order = np.asarray(model["node_order"])
    eigvals, eigvecs, labels, n_components = get_device_basis(
        environment, int(model["resolved_rank"])
    )
    sources = np.asarray(model["encoding_bin_indices"][0])
    unique_sources, source_index = np.unique(sources, return_inverse=True)
    onehot = np.zeros((node_order.size, unique_sources.size), dtype=np.float32)
    onehot[unique_sources, np.arange(unique_sources.size)] = 1.0
    heat = np.asarray(
        heat_kernel_apply(
            eigvals, eigvecs, POSITION_STD, jnp.asarray(onehot), labels, n_components
        ),
        float,
    )
    bins = _diffusion_bins(model, environment, positions)
    kernel = heat[bins][:, source_index] * np.asarray(
        model["encoding_weights"][0], float
    )
    centers = np.asarray(model["encoding_marks"][0], float)
    return kernel / kernel.sum(axis=1, keepdims=True), centers


def _simulate(algorithm, model, environment, rng, duration=250.0):
    """Held-out spikes drawn by thinning from the fitted model's own intensity.

    The ground intensity is evaluated at the position samples and interpolated
    linearly in time, as the transforms integrate it.
    """
    position_time, position = _trajectory(duration, 1.3)
    if algorithm == "clusterless_gmm":
        ground = _gmm_ground(model, position)
    elif algorithm == "clusterless_diffusion":
        ground = _diffusion_ground(model, environment, position)
    else:
        ground = _kde_ground(model, position)
    dt = 1e-3
    fine_time = np.arange(position_time[0], position_time[-1], dt)
    fine_ground = np.interp(fine_time, position_time, ground)
    is_spike = rng.uniform(size=fine_time.size) < fine_ground * dt
    spike_times = np.sort(fine_time[is_spike] + rng.uniform(0, dt, is_spike.sum()))
    spike_positions = np.interp(spike_times, position_time, position[:, 0])[:, None]
    if algorithm == "clusterless_gmm":
        resp, cond_means, cond_covs = _gmm_conditional(model, spike_positions)
        component = (rng.uniform(size=(len(resp), 1)) > resp.cumsum(axis=1)).sum(1)
        component = np.minimum(component, resp.shape[1] - 1)
        marks = np.stack(
            [
                rng.multivariate_normal(cond_means[i, k], cond_covs[k])
                for i, k in enumerate(component)
            ]
        )
    else:
        weights, centers = (
            _diffusion_components(model, environment, spike_positions)
            if algorithm == "clusterless_diffusion"
            else _kde_components(model, spike_positions)
        )
        component = (rng.uniform(size=(len(weights), 1)) > weights.cumsum(1)).sum(1)
        component = np.minimum(component, len(centers) - 1)
        marks = centers[component] + WAVEFORM_STD * rng.standard_normal(
            (len(component), 2)
        )
    return position_time, position, spike_times, marks


@pytest.mark.slow
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("transform", [ircm_transform, mdci_transform])
def test_transforms_are_uniform_for_data_from_the_fitted_model(
    recording, fitted, algorithm, transform
):
    """Spikes simulated from the fitted model map to uniform coordinates."""
    environment = recording[0]
    model = fitted[algorithm]
    position_time, position, spike_times, marks = _simulate(
        algorithm, model, environment, np.random.default_rng(11)
    )
    u, v = transform(
        model,
        algorithm,
        0,
        position_time,
        position,
        spike_times,
        marks,
    )
    assert u.shape == (len(spike_times),) and v.shape == marks.shape
    result = ks_uniformity_test(np.column_stack((u, v)))
    assert result.p_value.min() > 1e-3, result


@pytest.mark.slow
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("transform", [ircm_transform, mdci_transform])
def test_transforms_reject_shifted_marks(recording, fitted, algorithm, transform):
    """Marks displaced by 1.5 bandwidths are not uniform under the model."""
    environment = recording[0]
    model = fitted[algorithm]
    position_time, position, spike_times, marks = _simulate(
        algorithm, model, environment, np.random.default_rng(12)
    )
    _, v = transform(
        model,
        algorithm,
        0,
        position_time,
        position,
        spike_times,
        marks + np.array([1.5, 0.0]),
    )
    assert ks_uniformity_test(v).p_value[0] < 1e-6


@pytest.mark.slow
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_ircm_rejects_a_doubled_rate(recording, fitted, algorithm):
    """Rescaled intervals detect a model whose rate is twice the data's."""
    environment = recording[0]
    model = fitted[algorithm]
    position_time, position, spike_times, marks = _simulate(
        algorithm, model, environment, np.random.default_rng(13)
    )
    doubled = {**model, "mean_rates": [2.0 * rate for rate in model["mean_rates"]]}
    if algorithm == "clusterless_diffusion":
        doubled["summed_ground_process_intensity"] = 2.0 * np.asarray(
            model["summed_ground_process_intensity"]
        )
    u, _ = ircm_transform(
        doubled, algorithm, 0, position_time, position, spike_times, marks
    )
    assert ks_uniformity_test(u[:, None]).p_value[0] < 1e-6


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_feature_order_reorders_the_sequential_conditionals(
    recording, fitted, algorithm
):
    """The first transformed mark is conditioned on no other mark feature."""
    environment = recording[0]
    model = fitted[algorithm]
    position_time, position, spike_times, marks = _simulate(
        algorithm, model, environment, np.random.default_rng(14), duration=40.0
    )
    _, forward = ircm_transform(
        model, algorithm, 0, position_time, position, spike_times, marks
    )
    _, backward = ircm_transform(
        model,
        algorithm,
        0,
        position_time,
        position,
        spike_times,
        marks,
        feature_order=[1, 0],
    )
    assert not np.allclose(forward[:, 0], backward[:, 1])
    assert np.all((backward >= 0) & (backward <= 1))


def test_transforms_reject_an_unknown_algorithm(recording, fitted):
    environment, time, position, spike_times, marks = recording
    with pytest.raises(ValueError, match="algorithm"):
        ircm_transform(
            fitted["clusterless_kde"],
            "sorted_spikes_kde",
            0,
            time,
            position,
            spike_times[:5],
            marks[:5],
        )


# ---------------------------------------------------------------------------
# Uniformity tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_dims", [1, 3])
def test_uniformity_tests_accept_uniform_samples(n_dims):
    samples = np.random.default_rng(0).uniform(size=(800, n_dims))
    assert ks_uniformity_test(samples).p_value.min() > 0.01
    assert pearson_chi_squared_test(samples).p_value > 0.01
    result = multivariate_ks_test(
        samples, n_simulations=99, rng=np.random.default_rng(1)
    )
    assert result.p_value > 0.01


@pytest.mark.parametrize("n_dims", [1, 3])
def test_uniformity_tests_reject_nonuniform_samples(n_dims):
    samples = np.random.default_rng(0).beta(2.0, 2.0, size=(800, n_dims))
    assert ks_uniformity_test(samples).p_value.min() < 1e-6
    assert pearson_chi_squared_test(samples).p_value < 1e-6
    result = multivariate_ks_test(
        samples, n_simulations=99, rng=np.random.default_rng(1)
    )
    assert result.p_value <= 0.01


def test_multivariate_ks_detects_dependence_with_uniform_marginals():
    """Coordinates that are each uniform but dependent are not uniform jointly."""
    rng = np.random.default_rng(2)
    u = rng.uniform(size=600)
    samples = np.column_stack((u, (u + 0.05 * rng.uniform(size=600)) % 1.0))
    assert ks_uniformity_test(samples).p_value.min() > 0.01
    result = multivariate_ks_test(samples, n_simulations=99, rng=rng)
    assert result.p_value <= 0.01


def test_pearson_chi_squared_counts_equal_volume_cells():
    """Cells are equal sub-cubes, and the expected count per cell is n / M."""
    samples = np.array([[0.1, 0.1], [0.1, 0.9], [0.9, 0.1], [0.9, 0.9]] * 10)
    result = pearson_chi_squared_test(samples, n_bins_per_dim=2)
    assert result.statistic == pytest.approx(0.0)
    assert result.degrees_of_freedom == 3
    skewed = pearson_chi_squared_test(np.full((40, 2), 0.1), n_bins_per_dim=2)
    assert skewed.statistic == pytest.approx(120.0)


def test_ks_uniformity_test_reports_each_coordinate():
    rng = np.random.default_rng(3)
    samples = np.column_stack((rng.uniform(size=2000), rng.beta(2, 2, size=2000)))
    result = ks_uniformity_test(samples)
    assert result.statistic.shape == (2,) and result.p_value.shape == (2,)
    assert result.p_value[0] > 0.01 and result.p_value[1] < 1e-6
    np.testing.assert_allclose(
        result.statistic[1], stats.kstest(samples[:, 1], "uniform").statistic
    )


@pytest.mark.parametrize(
    "bad", [np.array([[0.5, 1.2]]), np.array([[np.nan, 0.5]]), np.ones(3)]
)
def test_uniformity_tests_reject_points_outside_the_unit_cube(bad):
    for test in (ks_uniformity_test, pearson_chi_squared_test):
        with pytest.raises(ValueError):
            test(bad)


# ---------------------------------------------------------------------------
# Exactness against closed forms computed in the test
# ---------------------------------------------------------------------------


def _mixture_cdfs(weights, means, covariances, marks):
    """Sequential CDFs of 2-D marks under per-query Gaussian mixtures.

    ``weights`` (n, K); ``means`` (n, K, 2) or (K, 2); ``covariances``
    (K, 2, 2) or (n, K, 2, 2).
    """
    means = np.broadcast_to(means, (len(marks), *np.shape(means)[-2:]))
    covariances = np.broadcast_to(
        covariances, (len(marks), *np.shape(covariances)[-3:])
    )
    s11, s12, s22 = (
        covariances[..., 0, 0],
        covariances[..., 0, 1],
        covariances[..., 1, 1],
    )
    m1, m2 = marks[:, :1], marks[:, 1:]
    first = np.sum(weights * stats.norm.cdf(m1, means[..., 0], np.sqrt(s11)), axis=1)
    density = weights * stats.norm.pdf(m1, means[..., 0], np.sqrt(s11))
    conditional_mean = means[..., 1] + s12 / s11 * (m1 - means[..., 0])
    conditional_std = np.sqrt(s22 - s12**2 / s11)
    second = np.sum(
        density * stats.norm.cdf(m2, conditional_mean, conditional_std), axis=1
    ) / density.sum(axis=1)
    return np.column_stack((first, second))


def _bivariate_normal_pdf(mark, means, covariances):
    """N(mark; means[..., :], covariances[..., :, :]) for 2-D marks."""
    s11, s12, s22 = (
        covariances[..., 0, 0],
        covariances[..., 0, 1],
        covariances[..., 1, 1],
    )
    determinant = s11 * s22 - s12**2
    d1, d2 = mark[0] - means[..., 0], mark[1] - means[..., 1]
    quadratic = (s22 * d1**2 - 2 * s12 * d1 * d2 + s11 * d2**2) / determinant
    return np.exp(-0.5 * quadratic) / (2 * np.pi * np.sqrt(determinant))


def _expected_conditional_cdfs(algorithm, model, environment, positions, marks):
    if algorithm == "clusterless_gmm":
        resp, cond_means, cond_covs = _gmm_conditional(model, positions)
        return _mixture_cdfs(resp, cond_means, cond_covs, marks)
    weights, centers = (
        _diffusion_components(model, environment, positions)
        if algorithm == "clusterless_diffusion"
        else _kde_components(model, positions)
    )
    covariances = np.broadcast_to(np.eye(2) * WAVEFORM_STD**2, (len(centers), 2, 2))
    return _mixture_cdfs(weights, centers, covariances, marks)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_ircm_mark_cdfs_equal_the_models_conditional_mixture(
    recording, fitted, algorithm
):
    """v is the sequential CDF of f(m | x) at the spike's position."""
    environment = recording[0]
    model = fitted[algorithm]
    rng = np.random.default_rng(21)
    position_time = np.linspace(0.0, 10.0, 1001)
    position = np.linspace(8.0, 92.0, 1001)[:, None]
    spike_times = np.sort(rng.uniform(0.0, 10.0, 25))
    marks = rng.normal(1.5, 1.5, (25, 2))
    _, v = ircm_transform(
        model, algorithm, 0, position_time, position, spike_times, marks
    )
    spike_positions = np.interp(spike_times, position_time, position[:, 0])[:, None]
    expected = _expected_conditional_cdfs(
        algorithm, model, environment, spike_positions, marks
    )
    np.testing.assert_allclose(v, expected, rtol=1e-7, atol=1e-9)


def _ground_and_components_at_bins(algorithm, model, environment, bin_centers):
    if algorithm == "clusterless_gmm":
        resp, cond_means, cond_covs = _gmm_conditional(model, bin_centers)
        return _gmm_ground(model, bin_centers), resp, cond_means, cond_covs
    if algorithm == "clusterless_diffusion":
        weights, centers = _diffusion_components(model, environment, bin_centers)
        ground = _diffusion_ground(model, environment, bin_centers)
    else:
        weights, centers = _kde_components(model, bin_centers)
        ground = _kde_ground(model, bin_centers)
    covs = np.broadcast_to(np.eye(2) * WAVEFORM_STD**2, (centers.shape[0], 2, 2))
    return ground, weights, centers, covs


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_mdci_equals_a_direct_sum_over_bins(recording, fitted, algorithm):
    """u and v from lambda(b, m) = Lambda(b) f(m | b) summed over visited bins.

    Each position sample is read at its bin, the integrand is trapezoidal in
    time, and Gamma(m) mixes f(m | b) with weights time(b) * Lambda(b).
    """
    environment = recording[0]
    model = fitted[algorithm]
    rng = np.random.default_rng(22)
    position_time = np.linspace(0.0, 10.0, 501)
    position = (50.0 + 30.0 * np.sin(position_time))[:, None]
    spike_times = np.sort(rng.uniform(0.0, 10.0, 15))
    marks = rng.normal(1.5, 1.5, (15, 2))
    u, v = mdci_transform(
        model, algorithm, 0, position_time, position, spike_times, marks
    )

    bin_centers = environment.place_bin_centers_
    sample_bins = environment.get_bin_ind(position)
    ground, weights, means, covs = _ground_and_components_at_bins(
        algorithm, model, environment, bin_centers[sample_bins]
    )
    # lambda(x(t_k), m_i) for every sample k and spike i.
    cov = np.broadcast_to(covs, (len(sample_bins), *covs.shape[-3:]))
    means = np.broadcast_to(means, (len(sample_bins), *means.shape[-2:]))
    density = np.stack(
        [np.sum(weights * _bivariate_normal_pdf(m, means, cov), axis=1) for m in marks],
        axis=1,
    )  # (n_samples, n_spikes)
    intensity = ground[:, None] * density

    def integral_to(s, values):
        inside = position_time <= s
        times = np.append(position_time[inside], s)
        last = np.interp(s, position_time, values)
        return np.trapezoid(np.append(values[inside], last), times)

    expected_u = np.array(
        [
            integral_to(s, intensity[:, i]) / integral_to(10.0, intensity[:, i])
            for i, s in enumerate(spike_times)
        ]
    )
    # KDE occupancy is the decoder's float32 KDE over 40,000 samples, within
    # 3e-5 relative of the float64 sum computed here; the others are float64.
    rtol = 5e-5 if algorithm in ("clusterless_kde", "clusterless_kde_log") else 1e-6
    np.testing.assert_allclose(u, expected_u, rtol=rtol, atol=1e-9)

    time_weight = np.zeros(len(position_time))
    durations = np.diff(position_time)
    time_weight[:-1] += 0.5 * durations
    time_weight[1:] += 0.5 * durations
    mixture = (time_weight * ground)[:, None] * weights
    expected_v = _mixture_cdfs(
        mixture.reshape(1, -1) / mixture.sum(),
        means.reshape(-1, 2),
        cov.reshape(-1, 2, 2),
        marks,
    )
    np.testing.assert_allclose(v, expected_v, rtol=rtol, atol=1e-9)


# ---------------------------------------------------------------------------
# Separated clusters and missing data
# ---------------------------------------------------------------------------


def _hand_built_gmm(means):
    """A GMM with equal weights and identity covariances at ``means``."""
    means = np.asarray(means, float)
    n_components, n_dims = means.shape
    model = GaussianMixtureModel(n_components=n_components, covariance_type="full")
    model.means_ = jnp.asarray(means)
    model.weights_ = jnp.full(n_components, 1.0 / n_components)
    model.covariances_ = jnp.repeat(jnp.eye(n_dims)[None], n_components, axis=0)
    model.precisions_chol_ = _compute_precision_cholesky(model.covariances_, "full")
    return model


@pytest.fixture(scope="module")
def separated():
    """Two encoding spikes 40 bandwidths apart in position and in each mark."""
    time = np.linspace(0.0, 10.0, 101)
    position = np.linspace(0.0, 40.0, 101)[:, None]
    environment = Environment(place_bin_size=1.0).fit_place_grid(
        position, infer_track_interior=False
    )
    spike_times = np.array([0.0, 10.0])
    marks = np.array([[0.0, 0.0], [40.0, 40.0]])
    models = {}
    for algorithm, fit in (
        ("clusterless_kde", fit_clusterless_kde_encoding_model),
        ("clusterless_kde_log", fit_clusterless_kde_log_encoding_model),
        ("clusterless_diffusion", fit_clusterless_diffusion_encoding_model),
    ):
        models[algorithm] = fit(
            time,
            position,
            [spike_times],
            [marks],
            environment,
            position_std=1.0,
            waveform_std=1.0,
            disable_progress_bar=True,
        )
    models["clusterless_gmm"] = {
        "environment": environment,
        "mean_rates": [1.0],
        "joint_models": [_hand_built_gmm([[0.0, 0.0, 0.0], [40.0, 40.0, 40.0]])],
        "gpi_models": [_hand_built_gmm([[0.0]])],
        "occupancy_model": _hand_built_gmm([[0.0]]),
    }
    return environment, models


@pytest.mark.parametrize("transform", [ircm_transform, mdci_transform])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_transforms_reject_missing_positions(separated, algorithm, transform):
    """A tracking gap raises instead of being read as a position bin."""
    environment, models = separated
    position_time = np.linspace(0.0, 1.0, 11)
    position = np.full((11, 1), 20.0)
    position[4:7] = np.nan
    with pytest.raises(ValueError, match="position must be finite"):
        transform(
            models[algorithm],
            algorithm,
            0,
            position_time,
            position,
            np.array([0.2, 0.5]),
            np.zeros((2, 2)),
        )


def test_log_matmul_exp_keeps_products_of_separated_factors():
    """Each factor's large entry meets the other's small one."""
    log_a = np.array([[0.0, -1000.0], [0.0, -1.0]])
    log_b = np.array([[-1000.0, 0.0], [0.0, -2.0]])
    expected = logsumexp(log_a[:, :, None] + log_b[None], axis=1)
    np.testing.assert_allclose(_log_matmul_exp(log_a, log_b), expected, rtol=1e-14)
    np.testing.assert_allclose(expected[0, 0], -1000.0 + np.log(2.0), rtol=1e-14)


def test_diffusion_mdci_keeps_marks_far_from_every_reachable_spike():
    """Heat is exactly zero between the two segments of this track.

    The only reachable encoding spike's marks are about exp(-1000) from the
    observed mark, far below the unreachable spike's exp(-200), which used to
    take that intensity to zero.
    """
    time = np.linspace(0.0, 10.0, 202)
    position = np.concatenate(
        (np.linspace(0.0, 10.0, 101), np.linspace(30.0, 40.0, 101))
    )[:, None]
    environment = Environment(place_bin_size=1.0).fit_place_grid(
        position, infer_track_interior=True
    )
    model = fit_clusterless_diffusion_encoding_model(
        time,
        position,
        [np.array([0.0, 10.0])],
        [np.array([[0.0, 0.0], [40.0, 40.0]])],
        environment,
        position_std=1.0,
        waveform_std=1.0,
        disable_progress_bar=True,
    )
    position_time = np.linspace(0.0, 1.0, 11)
    at_first_spike = environment.place_bin_centers_[
        environment.get_bin_ind(np.zeros((1, 1)))
    ]
    position = np.repeat(at_first_spike, 11, axis=0)
    spike_times = np.array([0.5])
    marks = np.array([[40.0, 20.0]])
    u, v = mdci_transform(
        model, "clusterless_diffusion", 0, position_time, position, spike_times, marks
    )
    _, expected_v = ircm_transform(
        model, "clusterless_diffusion", 0, position_time, position, spike_times, marks
    )
    np.testing.assert_allclose(u, spike_times, rtol=1e-12)
    np.testing.assert_allclose(v, expected_v, rtol=1e-9, atol=1e-12)
