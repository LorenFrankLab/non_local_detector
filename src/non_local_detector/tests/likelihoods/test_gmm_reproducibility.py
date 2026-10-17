"""Seeded GMM fits are bit-identical when repeated in one process.

scikit-learn's multithreaded KMeans merges per-thread partial sums in whatever
order the threads finish, so its centers differ in the last bits from run to
run and a point near a cluster boundary can change label. The GMM starts EM
from those labels, so a refit of identical input moved the fitted densities by
up to about 1e-3. The data below are uniform 2-D positions on which such label
changes are frequent; they exercise that race on a multicore CPU.
"""

import warnings

import jax
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_gmm import (
    fit_clusterless_gmm_encoding_model,
    mark_reference_scale,
    predict_clusterless_gmm_log_likelihood,
)
from non_local_detector.likelihoods.gmm import GaussianMixtureModel
from non_local_detector.tests.likelihoods.conftest import training_mark_references

pytestmark = pytest.mark.unit

N_REFITS = 5


def _boundary_prone_positions() -> np.ndarray:
    """Uniform 2-D positions whose seeded KMeans labels vary between threaded runs."""
    rng = np.random.default_rng(112)
    return rng.uniform(0.0, 100.0, (10_000, 2)).astype(np.float32)


def _gmm_parameters(model: GaussianMixtureModel) -> list[np.ndarray]:
    return [
        np.asarray(p)
        for p in (
            model.weights_,
            model.means_,
            model.covariances_,
            model.precisions_chol_,
        )
    ]


def _assert_same_gmm(a: GaussianMixtureModel, b: GaussianMixtureModel) -> None:
    for pa, pb in zip(_gmm_parameters(a), _gmm_parameters(b), strict=True):
        np.testing.assert_array_equal(pa, pb)


def test_seeded_gmm_refits_are_bit_identical():
    X = _boundary_prone_positions()

    def fit(data, n_components=32):
        return GaussianMixtureModel(
            n_components=n_components,
            init_params="kmeans",
            kmeans_init="k-means++",
            kmeans_n_init=1,
            random_state=0,
        ).fit(data, jax.random.PRNGKey(0))

    first = fit(X)
    for refit in range(N_REFITS - 1):
        if refit == 1:
            # An unrelated fit in between must not carry state into the next.
            fit(X[::3] + 7.0, n_components=5)
        _assert_same_gmm(first, fit(X))


def _recording(seed: int, positions: np.ndarray):
    rng = np.random.default_rng(seed)
    position_time = np.arange(len(positions)) / 100.0
    duration = position_time[-1]
    env = Environment(place_bin_size=10.0, position_range=((0.0, 100.0),) * 2)
    env = env.fit_place_grid(positions, infer_track_interior=False)
    times = [np.sort(rng.uniform(0.0, duration, n)) for n in (1500, 2500)]
    marks = [rng.normal(0.0, 30.0, (len(t), 4)) for t in times]
    return env, position_time, positions, times, marks


def _fit(recording):
    env, position_time, position, times, marks = recording
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # EM non-convergence
        return fit_clusterless_gmm_encoding_model(
            position_time,
            position,
            times,
            marks,
            env,
            gmm_random_state=0,
            disable_progress_bar=True,
        )


def _predict(recording, encoding, is_local):
    env, position_time, position, times, marks = recording
    rng = np.random.default_rng(3)
    decode_times = [np.sort(rng.uniform(0.0, 4.0, 60)) for _ in times]
    decode_marks = [rng.normal(0.0, 30.0, (60, 4)) for _ in times]
    mark_reference = training_mark_references(
        position_time, times, marks, mark_reference_scale()
    )
    return np.asarray(
        predict_clusterless_gmm_log_likelihood(
            position_time,
            position,
            decode_times,
            decode_marks,
            time_edges=np.arange(0.0, 4.0 + 1e-9, 0.004),
            **encoding,
            mark_reference=mark_reference,
            is_local=is_local,
        )
    )


def test_seeded_clusterless_gmm_encoding_refits_are_bit_identical():
    recording = _recording(0, _boundary_prone_positions())
    unrelated = _recording(1, _boundary_prone_positions()[::-1] * 0.5 + 20.0)

    first = _fit(recording)
    _fit(unrelated)
    refits = [_fit(recording) for _ in range(N_REFITS - 1)]

    for refit in refits:
        for key in (
            "log_occupancy",
            "summed_ground_process_intensity",
            "mean_rates",
            "interior_place_bin_centers",
        ):
            np.testing.assert_array_equal(
                np.asarray(first[key]), np.asarray(refit[key])
            )
        _assert_same_gmm(first["occupancy_model"], refit["occupancy_model"])
        for name in ("joint_models", "gpi_models"):
            for a, b in zip(first[name], refit[name], strict=True):
                _assert_same_gmm(a, b)
    for is_local in (False, True):
        np.testing.assert_array_equal(
            _predict(recording, first, is_local),
            _predict(recording, refits[-1], is_local),
        )
