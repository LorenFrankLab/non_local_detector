"""Clusterless marked intensities carry a unit-consistent background component.

Each active state's marked intensity is ``lambda(x, m) + beta q(m)`` with
compensator ``Lambda(x) + beta`` per electrode, where ``q`` is the electrode's
normalized mark reference and ``beta = RATE_EPS_HZ``. Under an affine change of
mark units ``m' = A m + c`` with the model transformed alike, every spike's
term then shifts by ``-log|det A|`` at every position, for every state.
"""

import copy
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector import Environment
from non_local_detector.likelihoods import (
    _CLUSTERLESS_ALGORITHMS,
    _CLUSTERLESS_MARK_REFERENCE_SCALES,
)
from non_local_detector.likelihoods.common import LOG_RATE_EPS_HZ, RATE_EPS_HZ
from non_local_detector.likelihoods.mark_reference import (
    MarkReference,
)
from non_local_detector.model_checking._clusterless_models import electrode_model
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data
from non_local_detector.tests.likelihoods.conftest import training_mark_references
from non_local_detector.time_edges import time_edges_from_centers

# Each algorithm, plus the tiled / streamed prediction paths that score spikes
# with separate kernels.
VARIANTS = [
    ("clusterless_gmm", {}),
    ("clusterless_gmm", {"bin_tile_size": 7}),
    ("clusterless_kde", {}),
    ("clusterless_kde", {"enc_tile_size": 64, "use_streaming": True}),
    ("clusterless_kde", {"pos_tile_size": 9}),
    ("clusterless_diffusion", {}),
]
VARIANT_IDS = [
    name + ("-" + "-".join(overrides) if overrides else "")
    for name, overrides in VARIANTS
]
SCALE = np.array([10.0, 0.5, 3.0, 1.0])
OFFSET = np.array([5.0, -2.0, 0.0, 100.0])
N_TRAIN = 8_000
N_TEST = 400


@pytest.fixture(scope="module")
def run_data():
    sim = make_simulated_run_data(n_runs=1, seed=0)
    time, position = sim.position_time, sim.position
    cut, stop = time[N_TRAIN], time[N_TRAIN + N_TEST]
    environment = Environment(place_bin_size=5.0)
    environment.fit_place_grid(position)
    return {
        "environment": environment,
        "train_time": time[:N_TRAIN],
        "train_position": position[:N_TRAIN],
        "test_time": time[N_TRAIN : N_TRAIN + N_TEST],
        "test_position": position[N_TRAIN : N_TRAIN + N_TEST],
        "train_spikes": [t[t < cut] for t in sim.spike_times],
        "train_marks": [
            f[t < cut]
            for t, f in zip(sim.spike_times, sim.spike_waveform_features, strict=True)
        ],
        "test_spikes": [t[(t >= cut) & (t < stop)] for t in sim.spike_times],
        "test_marks": [
            f[(t >= cut) & (t < stop)]
            for t, f in zip(sim.spike_times, sim.spike_waveform_features, strict=True)
        ],
    }


def _params(algorithm, scale=1.0):
    scale = np.asarray(scale, dtype=float)
    if algorithm == "clusterless_gmm":
        return {
            "gmm_components_joint": 8,
            "gmm_components_occupancy": 4,
            "disable_progress_bar": True,
        }
    return {
        "position_std": 6.0,
        "waveform_std": 24.0 * scale,
        "disable_progress_bar": True,
    }


def _fit_with_reference(algorithm, time, position, spikes, marks, environment, params):
    """Encoding model plus the mark reference fitted from the same training spikes."""
    fit, _ = _CLUSTERLESS_ALGORITHMS[algorithm]
    model = fit(time, position, spikes, marks, environment, **params)
    reference = training_mark_references(
        time,
        spikes,
        marks,
        _CLUSTERLESS_MARK_REFERENCE_SCALES[algorithm](**params),
        params.get("weights"),
        position,
    )
    return {**model, "mark_reference": reference}


def _fit(run_data, variant, marks=None, scale=1.0, spikes=None):
    algorithm, _ = variant
    return _fit_with_reference(
        algorithm,
        run_data["train_time"],
        run_data["train_position"],
        run_data["train_spikes"] if spikes is None else spikes,
        run_data["train_marks"] if marks is None else marks,
        run_data["environment"],
        _params(algorithm, scale),
    )


def _predict(run_data, variant, model, marks, is_local, spikes=None):
    algorithm, overrides = variant
    _, predict = _CLUSTERLESS_ALGORITHMS[algorithm]
    model = {**model, **overrides}
    return np.asarray(
        predict(
            run_data["test_time"],
            run_data["test_position"],
            run_data["test_spikes"] if spikes is None else spikes,
            marks,
            time_edges=time_edges_from_centers(run_data["test_time"]),
            **model,
            is_local=is_local,
        )
    )


def _transform_reference(reference):
    if reference is None:
        return None
    return MarkReference(
        reference.location * SCALE + OFFSET, reference.scale * np.abs(SCALE)
    )


def _transform_gmm(gmm, n_position_dims):
    """The same fitted joint mixture expressed in transformed mark units."""
    if gmm is None:
        return None
    assert gmm.covariance_type == "full"
    s = jnp.concatenate([jnp.ones(n_position_dims), jnp.asarray(SCALE)])
    c = jnp.concatenate([jnp.zeros(n_position_dims), jnp.asarray(OFFSET)])
    transformed = copy.deepcopy(gmm)
    transformed.means_ = gmm.means_ * s + c
    transformed.covariances_ = gmm.covariances_ * s[:, None] * s[None, :]
    # Scores use y = (x - mu) @ P_chol, so x' = x * s + c needs P_chol' = P_chol / s.
    transformed.precisions_chol_ = gmm.precisions_chol_ / s[None, :, None]
    return transformed


def _transformed_model(run_data, variant, model):
    marks = [m * SCALE + OFFSET for m in run_data["train_marks"]]
    if variant[0] != "clusterless_gmm":
        return _fit(run_data, variant, marks, SCALE)
    transformed = dict(model)
    n_position_dims = run_data["train_position"].reshape(N_TRAIN, -1).shape[1]
    transformed["joint_models"] = [
        _transform_gmm(g, n_position_dims) for g in model["joint_models"]
    ]
    transformed["mark_reference"] = [
        _transform_reference(r) for r in model["mark_reference"]
    ]
    return transformed


@pytest.mark.integration
@pytest.mark.parametrize("is_local", [False, True])
@pytest.mark.parametrize("algorithm", VARIANTS, ids=VARIANT_IDS)
def test_unit_change_shifts_every_spike_term_by_the_jacobian(
    run_data, algorithm, is_local
):
    model = _fit(run_data, algorithm)
    original = _predict(run_data, algorithm, model, run_data["test_marks"], is_local)
    transformed = _predict(
        run_data,
        algorithm,
        _transformed_model(run_data, algorithm, model),
        [m * SCALE + OFFSET for m in run_data["test_marks"]],
        is_local,
    )
    counts = np.histogram(
        np.concatenate(run_data["test_spikes"]),
        time_edges_from_centers(run_data["test_time"]),
    )[0]
    assert counts.sum() > 20
    jacobian = np.log(np.prod(np.abs(SCALE)))
    np.testing.assert_allclose(
        transformed + counts[:, None] * jacobian, original, rtol=1e-5, atol=1e-3
    )


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", VARIANTS, ids=VARIANT_IDS)
def test_outlier_mark_scores_the_background_at_every_position(run_data, algorithm):
    """A mark the encoding model cannot explain is scored by the background."""
    # The fitted reference: its tails exceed every mark kernel far from the
    # training marks, so 1000 µV away the background dominates everywhere.
    model = _fit(run_data, algorithm)
    reference = model["mark_reference"][0]
    spike_time = run_data["test_time"][200]
    outlier = model_marks_center(run_data) + 1e3
    spikes = [np.array([spike_time])] + [np.array([])] * 4
    marks = [outlier[None, :]] + [np.zeros((0, 4))] * 4
    silent = [np.array([])] * 5
    edges = time_edges_from_centers(run_data["test_time"])
    row = np.searchsorted(edges, spike_time, "right") - 1
    for is_local in (False, True):
        with_spike = _predict(run_data, algorithm, model, marks, is_local, spikes)
        without = _predict(
            run_data, algorithm, model, [np.zeros((0, 4))] * 5, is_local, silent
        )
        expected = (
            LOG_RATE_EPS_HZ
            + reference.log_density(outlier[None, :])[0]
            + np.log(np.diff(edges)[row])
        )
        np.testing.assert_allclose(with_spike[row] - without[row], expected, rtol=1e-5)


@pytest.mark.integration
@pytest.mark.parametrize("is_local", [False, True])
def test_diffusion_far_marks_keep_unit_invariance(run_data, is_local):
    """Marks many bandwidths from training: microvolts against millivolts.

    The diffusion field must keep each spike's scale in log space; a mark
    kernel exponentiated with its unit-dependent normalizer underflows at a
    different distance in each unit system.
    """
    variant = ("clusterless_diffusion", {})
    # About 7 bandwidths per feature: in float32 a 4-feature kernel with its
    # normalizer underflows there in microvolts (24^-4) but not in millivolts.
    far = [m + 7.2 * 24.0 for m in run_data["test_marks"]]
    original = _predict(run_data, variant, _fit(run_data, variant), far, is_local)
    millivolts = _fit(
        run_data, variant, marks=[m * 1e-3 for m in run_data["train_marks"]], scale=1e-3
    )
    transformed = _predict(
        run_data, variant, millivolts, [m * 1e-3 for m in far], is_local
    )
    counts = np.histogram(
        np.concatenate(run_data["test_spikes"]),
        time_edges_from_centers(run_data["test_time"]),
    )[0]
    jacobian = 4 * np.log(1e-3)
    np.testing.assert_allclose(
        transformed + counts[:, None] * jacobian, original, rtol=1e-5, atol=1e-3
    )


def model_marks_center(run_data):
    """Mean training mark of the first electrode."""
    return run_data["train_marks"][0].mean(axis=0)


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", VARIANTS, ids=VARIANT_IDS)
@pytest.mark.parametrize("has_reference", [True, False])
def test_zero_rate_electrode_scores_background(run_data, algorithm, has_reference):
    train_spikes = [np.array([])] + run_data["train_spikes"][1:]
    train_marks = [np.zeros((0, 4))] + run_data["train_marks"][1:]
    model = _fit(run_data, algorithm, train_marks, spikes=train_spikes)
    assert model["mark_reference"][0] is None
    reference = MarkReference(
        np.array([0.0, 1.0, 2.0, 3.0]), np.array([5.0, 6.0, 7.0, 8.0])
    )
    if has_reference:
        model = dict(model, mark_reference=[reference] + model["mark_reference"][1:])
    spike_time = run_data["test_time"][100]
    mark = np.array([[1.0, 2.0, 3.0, 4.0]])
    spikes = [np.array([spike_time])] + [np.array([])] * 4
    marks = [mark] + [np.zeros((0, 4))] * 4
    for is_local in (False, True):
        with_spike = _predict(run_data, algorithm, model, marks, is_local, spikes)
        without = _predict(
            run_data,
            algorithm,
            model,
            [np.zeros((0, 4))] * 5,
            is_local,
            [np.array([])] * 5,
        )
        edges = time_edges_from_centers(run_data["test_time"])
        row = np.searchsorted(edges, spike_time, "right") - 1
        expected = LOG_RATE_EPS_HZ + np.log(np.diff(edges)[row])
        if has_reference:
            expected += reference.log_density(mark)[0]
        np.testing.assert_allclose(with_spike[row] - without[row], expected, rtol=1e-6)


@pytest.mark.integration
@pytest.mark.parametrize("algorithm", VARIANTS, ids=VARIANT_IDS)
def test_zero_rate_compensator_adds_background_rate_for_every_electrode(
    run_data, algorithm
):
    """With no encoding spikes anywhere, each silent bin scores -dt * n_e * beta."""
    n_electrodes = len(run_data["train_spikes"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # empty-electrode warnings
        model = _fit(
            run_data,
            algorithm,
            [np.zeros((0, 4))] * n_electrodes,
            spikes=[np.array([])] * n_electrodes,
        )
    for is_local in (False, True):
        silent = _predict(
            run_data,
            algorithm,
            model,
            [np.zeros((0, 4))] * n_electrodes,
            is_local,
            [np.array([])] * n_electrodes,
        )
        dt = np.diff(time_edges_from_centers(run_data["test_time"]))
        np.testing.assert_allclose(
            silent,
            np.broadcast_to(-dt[:, None] * n_electrodes * RATE_EPS_HZ, silent.shape),
            rtol=1e-6,
        )


@pytest.mark.integration
def test_diffusion_unvisited_bins_score_only_the_background():
    """Bins the animal never visited carry no encoding evidence.

    Their diffused occupancy is below the heat kernel's resolution, so a spike
    there is explained by the background alone and cannot outscore visited bins.
    """
    rng = np.random.default_rng(0)
    time = np.arange(0.0, 20.0, 0.002)
    position = (30.0 + 25.0 * np.sin(time))[:, None]  # visits 5-55 of 0-120
    environment = Environment(place_bin_size=2.0, position_range=((0.0, 120.0),))
    environment.fit_place_grid(position, infer_track_interior=False)
    spike_times = [np.sort(rng.uniform(0.0, 20.0, 400))]
    marks = [rng.normal(0.0, 30.0, (400, 4))]
    _, predict = _CLUSTERLESS_ALGORITHMS["clusterless_diffusion"]
    model = _fit_with_reference(
        "clusterless_diffusion",
        time,
        position,
        spike_times,
        marks,
        environment,
        {"position_std": 1.5, "waveform_std": 24.0, "disable_progress_bar": True},
    )
    decode_time = np.arange(0.0, 0.1, 0.002)
    edges = time_edges_from_centers(decode_time)
    spikes = [np.array([0.05, 0.0501])]
    decode_marks = [marks[0][:2]]
    kwargs = dict(time_edges=edges, **model, is_local=False)
    pos = (30.0 + 25.0 * np.sin(decode_time))[:, None]
    with_spikes = np.asarray(predict(decode_time, pos, spikes, decode_marks, **kwargs))
    silent = np.asarray(
        predict(decode_time, pos, [np.array([])], [np.zeros((0, 4))], **kwargs)
    )
    row = np.searchsorted(edges, 0.05, "right") - 1
    evidence = with_spikes[row] - silent[row]
    occupancy = np.asarray(model["occupancy"])
    unvisited = occupancy < 1e-6 * occupancy.max()
    assert unvisited.sum() > 10 and (~unvisited).sum() > 10
    background = (
        LOG_RATE_EPS_HZ
        + model["mark_reference"][0].log_density(decode_marks[0])
        + np.log(np.diff(edges)[row])
    ).sum()
    np.testing.assert_allclose(evidence[unvisited], background, rtol=1e-5)
    assert evidence[unvisited].max() < evidence[~unvisited].max()


@pytest.mark.integration
def test_diffusion_model_checking_follows_the_decoder_in_unresolved_bins():
    """Where the decoder gives a bin zero intensity, so does model checking.

    Tiny training weights on part of the track (as EM can assign) leave its
    occupancy below the heat kernel's resolution while its encoding spikes
    still reach it. The decoder explains spikes there by the background
    alone; the goodness-of-fit adapter must not give those bins intensity.
    """
    rng = np.random.default_rng(1)
    time = np.arange(0.0, 20.0, 0.002)
    position = np.where(
        time < 10.0, 10.0 + 3.0 * np.sin(time), 50.0 + 8.0 * np.sin(time)
    )
    environment = Environment(place_bin_size=2.0, position_range=((0.0, 70.0),))
    environment.fit_place_grid(position[:, None], infer_track_interior=False)
    weights = np.where(time < 10.0, 1.0, 1e-9)
    spike_times = [np.sort(rng.uniform(0.0, 20.0, 400))]
    marks = [rng.normal(0.0, 30.0, (400, 4))]
    model = _fit_with_reference(
        "clusterless_diffusion",
        time,
        position[:, None],
        spike_times,
        marks,
        environment,
        {
            "position_std": 1.5,
            "waveform_std": 24.0,
            "weights": weights,
            "disable_progress_bar": True,
        },
    )
    occupancy = np.asarray(model["occupancy"])
    unresolved = occupancy < 1e-6 * occupancy.max()
    centers = environment.place_bin_centers_[environment.is_track_interior_.ravel()]
    reached = unresolved & (np.abs(centers[:, 0] - 50.0) < 8.0)
    assert reached.sum() > 3

    electrode = electrode_model(model, "clusterless_diffusion", 0)
    np.testing.assert_array_equal(electrode.ground_intensity(centers[reached]), 0.0)
    assert np.all(electrode.ground_intensity(centers[~unresolved]) >= 0.0)


@pytest.mark.unit
@pytest.mark.parametrize("is_local", [False, True], ids=["nonlocal", "local"])
def test_gmm_rate_below_the_background_still_sets_its_intensity(run_data, is_local):
    """GMM has no rate floor, like KDE and diffusion.

    Each spike's term is ``logaddexp(log(rate * p(m | x)), log(beta q(m)))``.
    Flooring the rate at ``beta`` would count the background twice and make
    every rate below it score spikes identically.
    """
    variant = ("clusterless_gmm", {})
    model = _fit(run_data, variant)
    n_electrodes = len(run_data["test_spikes"])
    silent_spikes = [np.array([]) for _ in range(n_electrodes)]
    silent_marks = [np.zeros((0, f.shape[1])) for f in run_data["test_marks"]]

    def spike_terms(rate):
        rated = {**model, "mean_rates": [rate] * n_electrodes}
        with_spikes = _predict(
            run_data, variant, rated, run_data["test_marks"], is_local
        )
        silent = _predict(
            run_data, variant, rated, silent_marks, is_local, spikes=silent_spikes
        )
        return with_spikes - silent

    assert 1e-13 < RATE_EPS_HZ
    higher, lower = spike_terms(1e-13), spike_terms(1e-14)
    assert np.all(higher >= lower - 1e-6)
    assert np.max(higher - lower) > 1e-2
