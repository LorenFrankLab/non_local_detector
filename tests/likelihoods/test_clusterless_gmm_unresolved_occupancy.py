"""Clusterless GMM at positions the encoding occupancy does not resolve.

The animal runs 0-150 cm of a 0-200 cm track. Far past 150 cm the occupancy
and spike-position mixtures are tails of different Gaussians, so their ratio
``p_gpi / p_occ`` grows without bound (to ``inf`` in float32 from about
180 cm). Where the occupancy log density is more than ``log(1e-6)`` below its
maximum over the interior bins, the place intensity is zero and spikes are
scored by the background alone, as the diffusion likelihoods do.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.likelihoods.clusterless_gmm import (
    fit_clusterless_gmm_encoding_model,
    mark_reference_scale,
    predict_clusterless_gmm_log_likelihood,
)
from non_local_detector.likelihoods.mark_reference import (
    background_rate,
    fit_mark_references,
)
from non_local_detector.time_edges import time_edges_from_centers

pytestmark = pytest.mark.unit

DT = 0.002
# Occupancy below this fraction of its maximum over the interior bins is
# unresolved (the diffusion likelihoods' ``FIELD_RESOLUTION``).
FLOOR = 1e-6


@pytest.fixture(scope="module")
def partial_coverage():
    """One electrode, two place fields, encoding covering 0-150 cm of 200 cm."""
    rng = np.random.default_rng(0)
    environment = Environment(
        place_bin_size=2.0, position_range=[(0.0, 200.0)]
    ).fit_place_grid(np.linspace(0, 200, 101)[:, None])
    time = np.arange(0.0, 120.0, DT)
    position = 75 + 75 * np.sin(2 * np.pi * time / 20)
    spike_times, marks = [], []
    for center, amplitude in ((40.0, 60.0), (120.0, 90.0)):
        rate = 15 * np.exp(-0.5 * ((position - center) / 10) ** 2) + 0.5
        times = time[rng.random(time.size) < rate * DT]
        spike_times.append(times)
        marks.append(amplitude + rng.normal(0, 5, (times.size, 4)))
    order = np.argsort(np.concatenate(spike_times))
    spike_times = np.concatenate(spike_times)[order]
    marks = np.concatenate(marks)[order]
    encoding = fit_clusterless_gmm_encoding_model(
        time,
        position[:, None],
        [spike_times],
        [marks],
        environment,
        disable_progress_bar=True,
    )
    log_occupancy = np.asarray(encoding["log_occupancy"], np.float64)
    unresolved = log_occupancy < log_occupancy.max() + np.log(FLOOR)
    centers = np.asarray(encoding["interior_place_bin_centers"])[:, 0]
    assert unresolved.any()
    assert not unresolved[centers < 150].any()
    decode_spikes = np.sort(rng.uniform(0.0, 0.5, 20))
    decode_marks = 60.0 + rng.normal(0, 5, (20, 4))
    return {
        "environment": environment,
        "encoding": encoding,
        "mark_reference": fit_mark_references([marks], mark_reference_scale()),
        "unresolved": unresolved,
        "centers": centers,
        "decode_spikes": decode_spikes,
        "decode_marks": decode_marks,
    }


def _predict(data, encoding, *, spikes=None, marks=None, **kwargs):
    """Decode ``data``'s decoding spikes (or the given ones) for 0.5 s."""
    spikes = data["decode_spikes"] if spikes is None else spikes
    marks = data["decode_marks"] if marks is None else marks
    centers = np.arange(0.0, 0.5, DT)
    return np.asarray(
        predict_clusterless_gmm_log_likelihood(
            kwargs.pop("position_time", None),
            kwargs.pop("position", None),
            [spikes],
            [marks],
            **{**encoding, **kwargs},
            mark_reference=data["mark_reference"],
            time_edges=time_edges_from_centers(centers),
        )
    )


def _background_only(encoding):
    """The same encoding with the electrode zero-rate: background alone."""
    return {
        **encoding,
        "joint_models": [None],
        "gpi_models": [None],
        "mean_rates": np.zeros(1),
        "summed_ground_process_intensity": np.zeros_like(
            encoding["summed_ground_process_intensity"]
        ),
    }


def _without_threshold(encoding):
    """The encoding as models stored before the occupancy threshold was kept."""
    return {k: v for k, v in encoding.items() if k != "log_occupancy_threshold"}


def _pre_change_intensity(encoding):
    """``rate * p_gpi / p_occ`` at every interior bin, as the fit formed it."""
    gpi_logp = encoding["gpi_models"][0].score_samples(
        jnp.asarray(encoding["interior_place_bin_centers"])
    )
    return np.asarray(
        jnp.exp(
            jnp.log(jnp.asarray(encoding["mean_rates"][0]))
            + (gpi_logp - jnp.asarray(encoding["log_occupancy"]))
        )
    )


@pytest.mark.parametrize("bin_tile_size", [None, 7], ids=["untiled", "tiled"])
def test_non_local_log_likelihood_is_finite_at_unvisited_bins(
    partial_coverage, bin_tile_size
):
    log_likelihood = _predict(
        partial_coverage, partial_coverage["encoding"], bin_tile_size=bin_tile_size
    )

    assert np.all(np.isfinite(log_likelihood))


@pytest.mark.parametrize("bin_tile_size", [None, 7], ids=["untiled", "tiled"])
def test_unresolved_bins_score_only_the_background(partial_coverage, bin_tile_size):
    """At unresolved bins the log likelihood is that of a zero-rate electrode."""
    data = partial_coverage
    unresolved = data["unresolved"]

    log_likelihood = _predict(data, data["encoding"], bin_tile_size=bin_tile_size)
    background = _predict(
        data, _background_only(data["encoding"]), bin_tile_size=bin_tile_size
    )

    np.testing.assert_allclose(
        log_likelihood[:, unresolved], background[:, unresolved], rtol=1e-5
    )


def test_fitted_intensity_is_zero_only_where_occupancy_is_unresolved(
    partial_coverage,
):
    """Resolved bins keep the fitted ``rate * p_gpi / p_occ`` to float32 rounding."""
    encoding = partial_coverage["encoding"]
    unresolved = partial_coverage["unresolved"]
    intensity = np.asarray(encoding["summed_ground_process_intensity"])

    assert np.all(intensity[unresolved] == 0.0)
    np.testing.assert_allclose(
        intensity[~unresolved],
        _pre_change_intensity(encoding)[~unresolved],
        rtol=1e-6,
    )


@pytest.mark.parametrize("with_spikes", [False, True], ids=["no_spikes", "spikes"])
def test_local_likelihood_outside_coverage_is_the_background(
    partial_coverage, with_spikes
):
    """The animal at 175 cm, never visited during encoding: background alone."""
    data = partial_coverage
    position_time = np.arange(0.0, 0.5, DT)
    position = np.full((position_time.size, 1), 175.0)
    spikes = data["decode_spikes"] if with_spikes else np.zeros(0)
    marks = data["decode_marks"] if with_spikes else np.zeros((0, 4))
    local = {"position_time": position_time, "position": position, "is_local": True}

    log_likelihood = _predict(
        data, data["encoding"], spikes=spikes, marks=marks, **local
    ).ravel()
    background = _predict(
        data, _background_only(data["encoding"]), spikes=spikes, marks=marks, **local
    ).ravel()

    np.testing.assert_allclose(log_likelihood, background, rtol=1e-5)
    if not with_spikes:
        np.testing.assert_allclose(log_likelihood, -DT * background_rate(1), atol=1e-3)


def test_resolved_bins_decode_as_without_the_threshold(partial_coverage):
    """The threshold changes nothing at bins whose occupancy is resolved."""
    data = partial_coverage
    resolved = ~data["unresolved"]
    encoding = data["encoding"]

    floored = _predict(data, encoding)
    unfloored = _predict(
        data,
        {
            **_without_threshold(encoding),
            "summed_ground_process_intensity": _pre_change_intensity(encoding),
        },
    )

    np.testing.assert_array_equal(floored[:, resolved], unfloored[:, resolved])


def test_encoding_model_without_threshold_decodes_as_fitted(partial_coverage):
    """A model stored before the threshold existed keeps its fitted values.

    Its stored intensity is ``inf`` far from the data, so its non-local log
    likelihood is ``-inf`` there, and its local expected count at 175 cm is the
    unfloored ratio.
    """
    data = partial_coverage
    old = {
        **_without_threshold(data["encoding"]),
        "summed_ground_process_intensity": _pre_change_intensity(data["encoding"]),
    }
    stored_intensity = old["summed_ground_process_intensity"]
    assert np.isposinf(stored_intensity).any()

    non_local = _predict(data, old)
    assert np.all(np.isneginf(non_local[:, np.isposinf(stored_intensity)]))

    position_time = np.arange(0.0, 0.5, DT)
    local = _predict(
        data,
        old,
        spikes=np.zeros(0),
        marks=np.zeros((0, 4)),
        position_time=position_time,
        position=np.full((position_time.size, 1), 175.0),
        is_local=True,
    ).ravel()
    gpi = old["gpi_models"][0]
    occupancy = old["occupancy_model"]
    at = jnp.asarray([[175.0]])
    unfloored_rate = float(
        jnp.exp(
            jnp.log(jnp.asarray(old["mean_rates"][0]))
            + (gpi.score_samples(at) - occupancy.score_samples(at))[0]
        )
    )
    assert unfloored_rate > 1e6
    np.testing.assert_allclose(
        local, -DT * (unfloored_rate + background_rate(1)), rtol=1e-5
    )
