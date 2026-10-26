"""Clusterless mark-mixture GLM: lambda_e(x, m) = sum_k lambda_ek(x) N(m; mu_ek, S_ek).

Each electrode's marks are fit by a Dirichlet-process variational Bayes Gaussian
mixture, which sets the number of components; each component's rate map is a
Poisson GLM fit to the spikes' component responsibilities.
"""

import networkx as nx
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.mixture import BayesianGaussianMixture  # type: ignore[import-untyped]

from non_local_detector.encoding_time import prepare_encoding_support
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _CLUSTERLESS_MARK_REFERENCE_SCALES,
    clusterless_mark_glm,
)
from non_local_detector.likelihoods.clusterless_mark_glm import (
    _WeightedBayesianGaussianMixture,
    fit_clusterless_mark_glm_encoding_model,
    fit_mark_mixture,
    mark_reference_scale,
    predict_clusterless_mark_glm_log_likelihood,
)
from non_local_detector.likelihoods.common import (
    LOG_RATE_EPS_HZ,
    RATE_EPS_HZ,
    log_bin_duration_evidence,
)
from non_local_detector.likelihoods.mark_reference import (
    background_log_intensity,
    background_rate,
)
from non_local_detector.likelihoods.sorted_spikes_glm import (
    fit_sorted_spikes_glm_encoding_model,
    make_spline_predict_matrix,
)
from non_local_detector.tests.likelihoods.conftest import training_mark_references

DT = 0.002
MARK_STD = 4.0


def _spikes(rng, time, rate):
    """Bernoulli spikes per sample, jittered within the sample's bin."""
    is_spike = rng.random(time.size) < rate * DT
    return np.sort(time[is_spike] + rng.uniform(-DT / 2, DT / 2, is_spike.sum()))


def _simulate(rng, time, position, units):
    """Spike times and 4-D marks per electrode.

    units : list of (electrode, mark_mean, rate_function)
    """
    n_electrodes = 1 + max(unit[0] for unit in units)
    times = [[] for _ in range(n_electrodes)]
    marks = [[] for _ in range(n_electrodes)]
    for electrode, mark_mean, rate in units:
        spike_times = _spikes(rng, time, rate(position))
        times[electrode].append(spike_times)
        marks[electrode].append(
            mark_mean + MARK_STD * rng.standard_normal((spike_times.size, 4))
        )
    spike_times, features = [], []
    for electrode_times, electrode_marks in zip(times, marks, strict=True):
        if not electrode_times:
            spike_times.append(np.zeros(0))
            features.append(np.zeros((0, 4)))
            continue
        t = np.concatenate(electrode_times)
        order = np.argsort(t)
        spike_times.append(t[order])
        features.append(np.concatenate(electrode_marks)[order])
    return spike_times, features


def _gaussian_field(center, width=5.0, peak=15.0, baseline=0.5):
    center = np.atleast_1d(np.asarray(center, float))

    def rate(position):
        position = position.reshape(len(position), -1)
        distance2 = np.sum((position - center) ** 2, axis=1)
        return baseline + peak * np.exp(-0.5 * distance2 / width**2)

    return rate


@pytest.fixture(scope="module")
def track_1d():
    """Back-and-forth runs on a 100 cm track; two tetrodes with three units each."""
    rng = np.random.default_rng(0)
    time = np.arange(0.0, 300.0, DT)
    position = (50.0 + 48.0 * np.sin(2 * np.pi * time / 20.0))[:, None]
    means = [np.full(4, value) for value in (60.0, 100.0, 140.0)]
    units = [
        (0, means[0], _gaussian_field(20.0)),
        (0, means[1], _gaussian_field(50.0)),
        (0, means[2], _gaussian_field(80.0)),
        (1, means[0], _gaussian_field(35.0)),
        (1, means[1], _gaussian_field(65.0)),
        (1, means[2], _gaussian_field(10.0, peak=8.0)),
    ]
    spike_times, features = _simulate(rng, time, position, units)
    environment = Environment(place_bin_size=2.0).fit_place_grid(position)
    encoding = fit_clusterless_mark_glm_encoding_model(
        time, position, spike_times, features, environment, disable_progress_bar=True
    )
    return {
        "time": time,
        "position": position,
        "spike_times": spike_times,
        "features": features,
        "environment": environment,
        "units": units,
        "encoding": encoding,
        "mark_reference": training_mark_references(
            time, spike_times, features, mark_reference_scale()
        ),
    }


def _unit_rate_maps(encoding, electrode, unit_means):
    """Rate map of each unit: the summed maps of the components nearest its mark."""
    mixture = encoding["mark_mixtures"][electrode]
    rates = np.exp(np.asarray(encoding["log_rate_maps"][electrode]))
    nearest = np.argmin(
        np.linalg.norm(mixture.means[:, None] - np.asarray(unit_means)[None], axis=2),
        axis=1,
    )
    return [rates[:, nearest == unit].sum(axis=1) for unit in range(len(unit_means))]


def _component_log_density(mixture, marks):
    """log N(m; mu_k, S_k) per spike and component, from the stored precisions."""
    columns = []
    for mean, precision_cholesky in zip(
        mixture.means, mixture.precisions_cholesky, strict=True
    ):
        precision = precision_cholesky @ precision_cholesky.T
        columns.append(
            multivariate_normal(mean, np.linalg.inv(precision)).logpdf(marks)
        )
    return np.stack(columns, axis=-1).reshape(len(marks), -1)


@pytest.mark.unit
def test_registered_with_its_mark_reference_scale():
    fit, predict = _CLUSTERLESS_ALGORITHMS["clusterless_mark_glm"]
    assert fit is fit_clusterless_mark_glm_encoding_model
    assert predict is predict_clusterless_mark_glm_log_likelihood
    scale = _CLUSTERLESS_MARK_REFERENCE_SCALES["clusterless_mark_glm"]
    assert scale(mark_reg_covar=4.0) == pytest.approx(2.0)


@pytest.mark.unit
def test_mark_mixture_selects_the_number_of_components():
    """Three separated clusters fit with a 16-component cap keep three components."""
    rng = np.random.default_rng(1)
    marks = np.concatenate(
        [
            center + MARK_STD * rng.standard_normal((600, 4))
            for center in (0.0, 60.0, 120.0)
        ]
    )
    mixture = fit_mark_mixture(marks, np.ones(len(marks)), n_components=16)
    assert np.sum(mixture.weights > 0.02) == 3
    np.testing.assert_allclose(mixture.weights.sum(), 1.0)


@pytest.mark.unit
def test_single_spike_mixture_is_a_regularized_gaussian_at_its_mark():
    mark = np.array([[10.0, 20.0, 30.0, 40.0]])
    mixture = fit_mark_mixture(
        np.vstack([mark, mark + 5.0]), np.array([0.5, 0.0]), reg_covar=4.0
    )
    assert mixture.n_components == 1
    np.testing.assert_array_equal(mixture.means, mark)
    covariance = np.linalg.inv(
        mixture.precisions_cholesky[0] @ mixture.precisions_cholesky[0].T
    )
    np.testing.assert_allclose(covariance, 4.0 * np.eye(4))


@pytest.mark.unit
def test_mark_mixture_spikes_count_by_their_weight():
    """Two equal-size clusters, one weighted 1/4, get mixture weights 4 : 1."""
    rng = np.random.default_rng(2)
    marks = np.concatenate(
        [center + MARK_STD * rng.standard_normal((2000, 4)) for center in (0.0, 80.0)]
    )
    weights = np.repeat([1.0, 0.25], 2000)
    mixture = fit_mark_mixture(marks, weights, n_components=8)
    near_first = np.linalg.norm(mixture.means, axis=1) < 40.0
    ratio = mixture.weights[near_first].sum() / mixture.weights[~near_first].sum()
    assert ratio == pytest.approx(4.0, rel=0.05)


@pytest.mark.unit
def test_integer_weights_equal_replicated_spikes():
    rng = np.random.default_rng(7)
    marks = np.concatenate(
        [center + MARK_STD * rng.standard_normal((300, 4)) for center in (0.0, 60.0)]
    )
    copies = rng.integers(1, 4, len(marks))
    weighted = fit_mark_mixture(marks, copies.astype(float), n_components=8)
    replicated = fit_mark_mixture(
        np.repeat(marks, copies, axis=0), np.ones(copies.sum()), n_components=8
    )
    assert weighted.n_components == replicated.n_components
    order_w = np.argsort(weighted.means[:, 0])
    order_r = np.argsort(replicated.means[:, 0])
    np.testing.assert_allclose(
        weighted.weights[order_w], replicated.weights[order_r], atol=2e-3
    )
    np.testing.assert_allclose(
        weighted.means[order_w], replicated.means[order_r], atol=2e-2
    )


@pytest.mark.unit
def test_convergence_tolerance_is_per_weighted_spike(monkeypatch):
    """scikit-learn's variational bound is a sum over samples, so the tolerance
    is scaled by the total weight: fits of any size stop at the same change
    per spike."""
    tolerances = []
    original = _WeightedBayesianGaussianMixture.fit_weighted

    def spy(self, X, sample_weight):
        tolerances.append((self.tol, float(np.sum(sample_weight))))
        return original(self, X, sample_weight)

    monkeypatch.setattr(_WeightedBayesianGaussianMixture, "fit_weighted", spy)
    rng = np.random.default_rng(11)
    for n in (200, 2000):
        marks = rng.standard_normal((n, 2))
        fit_mark_mixture(marks, np.full(n, 0.5), n_components=2, tol=1e-4)
    for tol, total_weight in tolerances:
        assert tol == pytest.approx(1e-4 * total_weight)


@pytest.mark.unit
def test_bypassed_weighted_m_step_raises(monkeypatch):
    """If scikit-learn stopped calling the weighted M-step, the fit would ignore
    the weights; it raises instead."""
    monkeypatch.setattr(
        _WeightedBayesianGaussianMixture, "_m_step", BayesianGaussianMixture._m_step
    )
    marks = np.random.default_rng(8).standard_normal((50, 2))
    with pytest.raises(RuntimeError, match="sample weights were not applied"):
        fit_mark_mixture(marks, np.full(50, 0.5), n_components=2)


@pytest.mark.unit
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_weighted_bound_equals_the_bound_of_replicated_spikes():
    """Convergence uses the variational bound of the weighted spikes: with integer
    weights it equals the bound of the replicated spikes under the same fitted
    distribution, so the fit stops where a fit to the replicated spikes would."""
    rng = np.random.default_rng(5)
    marks = np.concatenate(
        [center + MARK_STD * rng.standard_normal((200, 2)) for center in (0.0, 6.0)]
    )
    copies = rng.integers(1, 5, len(marks)).astype(float)
    model = _WeightedBayesianGaussianMixture(
        n_components=4,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        max_iter=20,
        random_state=0,
    ).fit_weighted(marks, copies)
    replicated = np.repeat(marks, copies.astype(int), axis=0)
    _, log_resp = model._estimate_log_prob_resp(marks)
    _, replicated_log_resp = model._estimate_log_prob_resp(replicated)
    np.testing.assert_allclose(
        model._compute_lower_bound(log_resp, None),
        BayesianGaussianMixture._compute_lower_bound(model, replicated_log_resp, None),
        rtol=1e-12,
    )


@pytest.mark.unit
def test_negligible_weight_spikes_do_not_shape_the_priors():
    """The mean and covariance priors are weighted like the data, so spikes with
    negligible weight move neither the priors nor the fitted mixture."""
    rng = np.random.default_rng(3)
    cluster = MARK_STD * rng.standard_normal((500, 2))
    far = 1000.0 + rng.standard_normal((100, 2))
    alone = fit_mark_mixture(cluster, np.ones(500), n_components=1)
    with_far = fit_mark_mixture(
        np.concatenate([cluster, far]),
        np.concatenate([np.ones(500), np.full(100, 1e-12)]),
        n_components=1,
    )

    def covariance(mixture):
        cholesky = mixture.precisions_cholesky[0]
        return np.linalg.inv(cholesky @ cholesky.T)

    np.testing.assert_allclose(with_far.means, alone.means, atol=1e-3)
    np.testing.assert_allclose(covariance(with_far), covariance(alone), rtol=1e-3)


@pytest.mark.unit
@pytest.mark.parametrize(
    "weights", [[1.0, 1e-20], [1.0, 1e-10], [1.0, 1e-10, 1e-10], [1.0, 1.0]]
)
def test_mark_mixture_with_fewer_effective_spikes_than_dimensions(weights):
    """When the weights leave fewer effective spikes than mark dimensions plus
    one (one spike carrying nearly all the weight, as concentrated EM posteriors
    can), the covariance prior cannot be estimated; the fit is still finite."""
    marks = np.array(
        [[50.0, 60.0, 70.0, 80.0], [55.0, 65.0, 72.0, 85.0], [40.0, 61.0, 75.0, 79.0]]
    )[: len(weights)]
    weights = np.asarray(weights)
    mixture = fit_mark_mixture(marks, weights, n_components=4)
    assert np.all(np.isfinite(mixture.means))
    assert np.all(np.isfinite(mixture.precisions_cholesky))
    np.testing.assert_allclose(
        mixture.weights @ mixture.means,
        np.average(marks, axis=0, weights=weights),
        atol=1e-3 * np.ptp(marks),
    )


@pytest.mark.integration
def test_rate_maps_recover_the_units_place_fields(track_1d):
    encoding = track_1d["encoding"]
    centers = track_1d["environment"].place_bin_centers_[
        track_1d["environment"].is_track_interior_.ravel()
    ]
    for electrode in (0, 1):
        units = [unit for unit in track_1d["units"] if unit[0] == electrode]
        fitted = _unit_rate_maps(encoding, electrode, [unit[1] for unit in units])
        for (_, _, rate), estimate in zip(units, fitted, strict=True):
            truth = rate(centers)
            assert np.corrcoef(truth, estimate)[0, 1] > 0.98
            peak = truth.argmax()
            assert estimate[peak] == pytest.approx(truth[peak], rel=0.2)


@pytest.mark.integration
def test_nonlocal_matches_a_direct_computation(track_1d):
    """The compiled predict equals the intensity formula evaluated with SciPy,
    including a spike whose mark is far from every component."""
    encoding = track_1d["encoding"]
    rng = np.random.default_rng(3)
    time_edges = np.linspace(10.0, 10.4, 21)
    spike_times, features = [], []
    for electrode in (0, 1):
        n = 30
        spike_times.append(np.sort(rng.uniform(10.0, 10.4, n)))
        marks = np.asarray(track_1d["features"][electrode][:n]).copy()
        marks[0] = 500.0
        features.append(marks)
    result = np.asarray(
        predict_clusterless_mark_glm_log_likelihood(
            track_1d["time"],
            track_1d["position"],
            spike_times,
            features,
            time_edges=time_edges,
            mark_reference=track_1d["mark_reference"],
            **encoding,
        )
    )
    durations = np.diff(time_edges)
    expected = -durations[:, None] * (
        np.asarray(encoding["summed_ground_process_intensity"])[None]
        + background_rate(2)
    )
    for electrode in (0, 1):
        log_density = _component_log_density(
            encoding["mark_mixtures"][electrode], features[electrode]
        )
        log_rates = np.asarray(encoding["log_rate_maps"][electrode], float)
        intensity = logsumexp(log_density[:, None, :] + log_rates[None], axis=2)
        background = np.asarray(
            background_log_intensity(
                track_1d["mark_reference"][electrode],
                features[electrode],
                LOG_RATE_EPS_HZ,
            )
        )
        rows = np.searchsorted(time_edges, spike_times[electrode], side="right") - 1
        np.add.at(expected, rows, np.logaddexp(intensity, background[:, None]))
    expected += np.asarray(
        log_bin_duration_evidence(spike_times, time_edges, None, None)
    )[:, None]
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-3)


def _dwell_data(track_1d):
    """The animal dwells at bin centers for whole decoding bins, with spikes inside.

    Returns position_time, position, time_edges, spike_times, features and each
    decoding bin's interior-bin column.
    """
    environment = track_1d["environment"]
    interior = environment.is_track_interior_.ravel()
    centers = environment.place_bin_centers_[interior].ravel()
    rng = np.random.default_rng(4)
    columns = rng.choice(len(centers), 12)
    samples_per_dwell = 20
    position = np.repeat(centers[columns], samples_per_dwell)[:, None]
    position_time = 100.0 + np.arange(len(position)) * DT
    dwell_start = position_time[::samples_per_dwell]
    time_edges = np.empty(2 * len(columns))
    time_edges[0::2] = dwell_start + 2 * DT
    time_edges[1::2] = dwell_start + (samples_per_dwell - 3) * DT
    spike_times, features = [], []
    for electrode in (0, 1):
        times = np.sort(
            np.concatenate(
                [
                    rng.uniform(lo, hi, 3)
                    for lo, hi in zip(time_edges[0::2], time_edges[1::2], strict=True)
                ]
            )
        )
        spike_times.append(times)
        features.append(np.asarray(track_1d["features"][electrode][: times.size]))
    return position_time, position, time_edges, spike_times, features, columns


@pytest.mark.integration
def test_local_equals_nonlocal_at_the_animals_bin(track_1d):
    position_time, position, time_edges, spike_times, features, columns = _dwell_data(
        track_1d
    )
    common = dict(
        time_edges=time_edges,
        mark_reference=track_1d["mark_reference"],
        **track_1d["encoding"],
    )
    nonlocal_ll = np.asarray(
        predict_clusterless_mark_glm_log_likelihood(
            position_time, position, spike_times, features, **common
        )
    )
    local_ll = np.asarray(
        predict_clusterless_mark_glm_log_likelihood(
            position_time, position, spike_times, features, is_local=True, **common
        )
    )
    dwell_rows = np.arange(0, len(time_edges) - 1, 2)
    assert local_ll.shape == (len(time_edges) - 1, 1)
    np.testing.assert_allclose(
        local_ll[dwell_rows, 0],
        nonlocal_ll[dwell_rows, columns],
        rtol=1e-5,
        atol=1e-3,
    )


@pytest.mark.integration
@pytest.mark.parametrize("is_local", [False, True])
def test_row_slice_equals_the_full_result_sliced(track_1d, is_local):
    position_time, position, time_edges, spike_times, features, _ = _dwell_data(
        track_1d
    )
    common = dict(
        time_edges=time_edges,
        mark_reference=track_1d["mark_reference"],
        is_local=is_local,
        **track_1d["encoding"],
    )
    full = np.asarray(
        predict_clusterless_mark_glm_log_likelihood(
            position_time, position, spike_times, features, **common
        )
    )
    part = np.asarray(
        predict_clusterless_mark_glm_log_likelihood(
            position_time,
            position,
            spike_times,
            features,
            row_slice=slice(5, 17),
            **common,
        )
    )
    np.testing.assert_allclose(part, full[5:17], rtol=1e-6, atol=1e-5)


@pytest.mark.integration
def test_hard_zero_weights_equal_fitting_the_subset(track_1d):
    """Zero-weight samples and their spikes are excluded from the mark mixture
    and the rate maps alike."""
    time, position = track_1d["time"], track_1d["position"]
    half = time.size // 2
    cut = time[half]
    spike_times, features = [], []
    for times, marks in zip(track_1d["spike_times"], track_1d["features"], strict=True):
        keep = np.abs(times - cut) > 2 * DT
        spike_times.append(times[keep])
        features.append(marks[keep])
    weights = (np.arange(time.size) < half).astype(float)
    environment = track_1d["environment"]
    weighted = fit_clusterless_mark_glm_encoding_model(
        time, position, spike_times, features, environment, weights=weights
    )
    subset = fit_clusterless_mark_glm_encoding_model(
        time[:half],
        position[:half],
        [t[t < cut] for t in spike_times],
        [m[t < cut] for t, m in zip(spike_times, features, strict=True)],
        environment,
    )
    for electrode in (0, 1):
        np.testing.assert_allclose(
            weighted["mark_mixtures"][electrode].means,
            subset["mark_mixtures"][electrode].means,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            np.exp(weighted["log_rate_maps"][electrode]),
            np.exp(subset["log_rate_maps"][electrode]),
            rtol=1e-3,
        )


@pytest.mark.integration
def test_track_graph_seam_does_not_leak_between_arms():
    """With no gap between edges in linear position, a field at the end of one arm
    stays off the start of the next arm in edge_order, which it does not touch."""
    nodes = {0: (0.0, 0.0), 1: (40.0, 0.0), 2: (70.0, 30.0), 3: (70.0, -30.0)}
    edges = [(0, 1), (1, 2), (1, 3)]
    graph = nx.Graph()
    for node, pos in nodes.items():
        graph.add_node(node, pos=pos)
    for edge_id, (u, v) in enumerate(edges):
        graph.add_edge(
            u,
            v,
            distance=float(np.linalg.norm(np.subtract(nodes[u], nodes[v]))),
            edge_id=edge_id,
        )
    environment = Environment(
        place_bin_size=1.0, track_graph=graph, edge_order=edges, edge_spacing=0.0
    ).fit_place_grid()
    sequence = (0, 1, 2, 1, 3, 1, 0)
    legs = [
        np.linspace(nodes[a], nodes[b], 1500)
        for a, b in zip(sequence[:-1], sequence[1:], strict=True)
    ]
    position = np.tile(np.concatenate(legs), (8, 1))
    time = np.arange(len(position)) * DT
    rng = np.random.default_rng(5)
    spike_times, features = _simulate(
        rng,
        time,
        position,
        [(0, np.full(4, 80.0), _gaussian_field(nodes[2], 4.0, 20.0, 0.2))],
    )
    encoding = fit_clusterless_mark_glm_encoding_model(
        time, position, spike_times, features, environment
    )
    basis = encoding["emission_design_info"]
    arm_start = basis.edge_starts[2]
    interior = environment.is_track_interior_.ravel()
    centers = environment.place_bin_centers_[interior].ravel()
    ground = np.asarray(encoding["summed_ground_process_intensity"])
    near_node_2 = (centers > arm_start - 3.0) & (centers < arm_start)
    next_arm = (centers >= arm_start) & (centers < arm_start + 6.0)
    assert ground[near_node_2].max() > 10.0
    assert ground[next_arm].max() < 0.5


@pytest.mark.integration
def test_2d_rate_maps_match_a_sorted_fit_with_the_true_units():
    """On a 2-D random walk, each unit's maps equal the sorted-spikes GLM fit to
    that unit's spikes: separating the units by their marks loses nothing."""
    rng = np.random.default_rng(6)
    time = np.arange(0.0, 400.0, DT)
    walk = np.cumsum(rng.normal(0.0, 0.25, (time.size, 2)), axis=0)
    position = 1.0 + np.abs((walk % 156.0) - 78.0)
    environment = Environment(
        place_bin_size=3.5, position_range=((0, 80), (0, 80))
    ).fit_place_grid(position, infer_track_interior=True)
    means = [np.full(4, 60.0), np.full(4, 120.0)]
    units = [
        (0, means[0], _gaussian_field([25.0, 30.0], 7.0)),
        (0, means[1], _gaussian_field([55.0, 50.0], 7.0)),
    ]
    spike_times, features = _simulate(rng, time, position, units)
    encoding = fit_clusterless_mark_glm_encoding_model(
        time, position, spike_times, features, environment
    )
    unit = np.argmin(
        np.linalg.norm(features[0][:, None] - np.asarray(means)[None], axis=2),
        axis=1,
    )
    sorted_fit = fit_sorted_spikes_glm_encoding_model(
        time,
        position,
        [spike_times[0][unit == u] for u in range(len(means))],
        environment,
        environment.place_bin_edges_,
        environment.edges_,
        environment.is_track_interior_,
        environment.is_track_boundary_,
    )
    interior = environment.is_track_interior_.ravel()
    sorted_maps = np.asarray(sorted_fit["place_fields"])[:, interior]
    assert encoding["mark_mixtures"][0].n_components == 2
    for estimate, expected in zip(
        _unit_rate_maps(encoding, 0, means), sorted_maps, strict=True
    ):
        np.testing.assert_allclose(estimate, expected, rtol=0.02, atol=0.02)


@pytest.mark.integration
@pytest.mark.parametrize("is_local", [False, True])
def test_electrode_without_encoding_spikes_is_background_only(track_1d, is_local):
    spike_times = [*track_1d["spike_times"], np.zeros(0)]
    features = [*track_1d["features"], np.zeros((0, 4))]
    with pytest.warns(UserWarning, match="electrode 2"):
        encoding = fit_clusterless_mark_glm_encoding_model(
            track_1d["time"],
            track_1d["position"],
            spike_times,
            features,
            track_1d["environment"],
        )
    assert encoding["mark_mixtures"][2] is None
    assert encoding["mean_rates"][2] == 0.0
    reference = training_mark_references(
        track_1d["time"],
        [track_1d["spike_times"][0]] * 3,
        [track_1d["features"][0]] * 3,
        mark_reference_scale(),
    )
    time_edges = np.linspace(10.0, 10.2, 11)
    decode_times = [np.zeros(0), np.zeros(0), np.array([10.05])]
    decode_marks = [np.zeros((0, 4)), np.zeros((0, 4)), np.full((1, 4), 100.0)]
    with_spike = np.asarray(
        predict_clusterless_mark_glm_log_likelihood(
            track_1d["time"],
            track_1d["position"],
            decode_times,
            decode_marks,
            time_edges=time_edges,
            mark_reference=reference,
            is_local=is_local,
            **encoding,
        )
    )
    without = np.asarray(
        predict_clusterless_mark_glm_log_likelihood(
            track_1d["time"],
            track_1d["position"],
            [np.zeros(0)] * 3,
            [np.zeros((0, 4))] * 3,
            time_edges=time_edges,
            mark_reference=reference,
            is_local=is_local,
            **encoding,
        )
    )
    difference = with_spike - without
    assert np.all(np.isfinite(with_spike))
    # The spike's bin scores the background beta * q(m) at every position.
    expected = (
        LOG_RATE_EPS_HZ
        + reference[2].log_density(decode_marks[2])[0]
        + np.log(np.diff(time_edges)[2])
    )
    np.testing.assert_allclose(difference[2], expected, rtol=1e-6)
    assert np.all(difference[np.arange(10) != 2] == 0.0)


@pytest.mark.unit
def test_unknown_fit_parameter_raises(track_1d):
    with pytest.raises(ValidationError, match="n_mark_component"):
        fit_clusterless_mark_glm_encoding_model(
            track_1d["time"][:100],
            track_1d["position"][:100],
            track_1d["spike_times"],
            track_1d["features"],
            track_1d["environment"],
            n_mark_component=4,
        )


@pytest.mark.integration
def test_decoder_decodes_a_simulated_run():
    from non_local_detector.models import ClusterlessDecoder
    from non_local_detector.simulate.clusterless_simulation import (
        make_simulated_run_data,
    )
    from non_local_detector.time_edges import time_edges_from_centers

    sim = make_simulated_run_data()
    time, position = sim.position_time, sim.position
    train = np.arange(len(time)) < int(0.7 * len(time))
    decoder = ClusterlessDecoder(clusterless_algorithm="clusterless_mark_glm").fit(
        time, position, sim.spike_times, sim.spike_waveform_features, is_training=train
    )
    results = decoder.predict(
        sim.spike_times,
        sim.spike_waveform_features,
        time_edges=time_edges_from_centers(time[~train]),
        position=position[~train],
        position_time=time[~train],
    )
    posterior = np.asarray(results.acausal_posterior.values)
    assert np.all(np.isfinite(posterior))
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-4)
    bins = np.asarray(results.acausal_posterior["position"].values)
    error = np.abs(bins[posterior.argmax(axis=1)] - position[~train].ravel())
    assert np.median(error) < 4.0


@pytest.mark.integration
def test_detector_em_refits_the_encoding_with_posterior_weights(monkeypatch):
    from non_local_detector.models import NonLocalClusterlessDetector
    from non_local_detector.simulate.clusterless_simulation import (
        make_simulated_run_data,
    )
    from non_local_detector.time_edges import time_edges_from_centers

    fit, predict = _CLUSTERLESS_ALGORITHMS["clusterless_mark_glm"]
    fits = []

    def recording_fit(*args, **kwargs):
        encoding = fit(*args, **kwargs)
        fits.append((kwargs.get("weights"), encoding))
        return encoding

    monkeypatch.setitem(
        _CLUSTERLESS_ALGORITHMS, "clusterless_mark_glm", (recording_fit, predict)
    )
    sim = make_simulated_run_data(
        n_tetrodes=2, place_field_means=np.arange(0, 80, 20), n_runs=3, seed=42
    )
    detector = NonLocalClusterlessDetector(clusterless_algorithm="clusterless_mark_glm")
    results = detector.estimate_parameters(
        position_time=sim.position_time,
        position=sim.position,
        spike_times=sim.spike_times,
        spike_waveform_features=sim.spike_waveform_features,
        time_edges=time_edges_from_centers(sim.position_time),
        max_iter=3,
        estimate_encoding_model=True,
    )
    log_likelihoods = np.asarray(results.attrs["marginal_log_likelihoods"])
    assert len(log_likelihoods) >= 2
    assert np.all(np.isfinite(log_likelihoods))
    refits = [(w, e) for w, e in fits[1:] if w is not None]
    assert refits, "EM did not refit the encoding model"
    refit_weights, refit = refits[-1]
    assert np.any((refit_weights > 0) & (refit_weights < 1))
    first = fits[0][1]
    assert not np.allclose(
        np.asarray(refit["summed_ground_process_intensity"]),
        np.asarray(first["summed_ground_process_intensity"]),
    )


@pytest.mark.integration
@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("value", [1e20, 1e40])
def test_mark_beyond_float32_range_is_background_only(track_1d, value, is_local):
    """A mark whose squared distance (1e20) or value itself (1e40) overflows
    float32 has zero density under every component and is explained by the
    background, not NaN."""
    time_edges = np.linspace(10.0, 10.2, 11)
    common = dict(
        time_edges=time_edges,
        mark_reference=track_1d["mark_reference"],
        is_local=is_local,
        **track_1d["encoding"],
    )
    empty = ([np.zeros(0)] * 2, [np.zeros((0, 4))] * 2)
    extreme = (
        [np.array([10.05]), np.zeros(0)],
        [np.full((1, 4), value), np.zeros((0, 4))],
    )
    args = (track_1d["time"], track_1d["position"])
    with_spike = np.asarray(
        predict_clusterless_mark_glm_log_likelihood(*args, *extreme, **common)
    )
    without = np.asarray(
        predict_clusterless_mark_glm_log_likelihood(*args, *empty, **common)
    )
    assert np.all(np.isfinite(with_spike))
    log_background = float(
        background_log_intensity(
            track_1d["mark_reference"][0], np.full((1, 4), value), LOG_RATE_EPS_HZ
        )[0]
    )
    # Only the spike's bin changes, by the background's log intensity per bin.
    np.testing.assert_allclose(
        (with_spike - without)[2], log_background + np.log(0.02), rtol=1e-4
    )


@pytest.mark.integration
def test_local_matches_a_direct_computation_for_a_moving_animal(track_1d):
    """While the animal runs, each spike's term uses the position at its own time
    and the expected count the position at its bin's center."""
    encoding = track_1d["encoding"]
    time, position = track_1d["time"], track_1d["position"]
    rng = np.random.default_rng(9)
    time_edges = np.linspace(12.0, 12.5, 11)
    spike_times = [np.sort(rng.uniform(12.0, 12.5, 25)) for _ in range(2)]
    features = [np.asarray(f[100:125]) for f in track_1d["features"]]
    result = np.asarray(
        predict_clusterless_mark_glm_log_likelihood(
            time,
            position,
            spike_times,
            features,
            time_edges=time_edges,
            mark_reference=track_1d["mark_reference"],
            is_local=True,
            **encoding,
        )
    )[:, 0]
    basis = encoding["emission_design_info"]

    def log_rates(electrode, at):
        x = np.interp(at, time, position[:, 0])[:, None]
        design = np.asarray(make_spline_predict_matrix(basis, x))
        rate = np.exp(design @ np.asarray(encoding["rate_coefficients"][electrode]).T)
        return np.log(np.maximum(rate, RATE_EPS_HZ))

    centers = 0.5 * (time_edges[:-1] + time_edges[1:])
    durations = np.diff(time_edges)
    expected = -durations * (
        sum(np.exp(log_rates(e, centers)).sum(axis=1) for e in (0, 1))
        + background_rate(2)
    )
    for electrode in (0, 1):
        log_density = _component_log_density(
            encoding["mark_mixtures"][electrode], features[electrode]
        )
        intensity = logsumexp(
            log_density + log_rates(electrode, spike_times[electrode]), axis=1
        )
        background = np.asarray(
            background_log_intensity(
                track_1d["mark_reference"][electrode],
                features[electrode],
                LOG_RATE_EPS_HZ,
            )
        )
        rows = np.searchsorted(time_edges, spike_times[electrode], side="right") - 1
        np.add.at(expected, rows, np.logaddexp(intensity, background))
    expected += np.asarray(
        log_bin_duration_evidence(spike_times, time_edges, None, None)
    )
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-3)


@pytest.mark.integration
def test_soft_counts_split_each_spike_by_its_sample_shares(monkeypatch, track_1d):
    """Summed over an electrode's components, the soft counts at each sample are
    the weighted events ``EncodingSupport.event_counts`` assigns it."""
    time, position = track_1d["time"], track_1d["position"]
    weights = np.random.default_rng(10).uniform(0.2, 1.0, time.size)
    captured = {}
    original = clusterless_mark_glm._fit_rate_maps

    def spy(*args, **kwargs):
        captured["counts"], captured["n_maps"] = args[4], kwargs["n_maps"]
        return original(*args, **kwargs)

    monkeypatch.setattr(clusterless_mark_glm, "_fit_rate_maps", spy)
    encoding = fit_clusterless_mark_glm_encoding_model(
        time,
        position,
        track_1d["spike_times"],
        track_1d["features"],
        track_1d["environment"],
        weights=weights,
    )
    support, sample_weights, _, _ = prepare_encoding_support(time, position, weights)
    n_support = len(support.indices)
    counts = captured["counts"](np.arange(n_support), n_support)
    assert counts.shape == (captured["n_maps"], n_support)
    first = 0
    for electrode, mixture in enumerate(encoding["mark_mixtures"]):
        electrode_counts = counts[first : first + mixture.n_components].sum(axis=0)
        first += mixture.n_components
        expected = support.event_counts(
            track_1d["spike_times"][electrode], sample_weights
        )[support.indices]
        np.testing.assert_allclose(electrode_counts, expected, rtol=1e-9, atol=1e-12)
