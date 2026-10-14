"""Input contracts and final fitted state must survive ordinary edge cases."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.likelihoods.gmm import GaussianMixtureModel
from non_local_detector.likelihoods.no_spike import predict_no_spike_log_likelihood
from non_local_detector.likelihoods.sorted_spikes_glm import (
    fit_sorted_spikes_glm_encoding_model,
    make_spline_design_matrix,
    make_spline_predict_matrix,
)
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
)

pytestmark = pytest.mark.unit


def test_gmm_rejects_a_singular_final_m_step():
    """A finite pre-update objective cannot certify the returned covariance."""
    model = GaussianMixtureModel(
        1,
        reg_covar=0,
        max_iter=1,
        weights_init=jnp.asarray([1.0]),
        means_init=jnp.asarray([[1.0]]),
        covariances_init=jnp.asarray([[[1.0]]]),
    )
    with pytest.raises(RuntimeError, match="singular|non-finite"):
        model.fit(jnp.zeros((3, 1)), jax.random.PRNGKey(0))
    assert model.weights_ is None


def test_gmm_accepts_integer_features_as_real_measurements():
    model = GaussianMixtureModel(1)
    model.fit(jnp.arange(5, dtype=jnp.int32)[:, None], jax.random.PRNGKey(0))
    assert np.all(np.isfinite(model.score_samples(jnp.asarray([[1.0]]))))


@pytest.mark.parametrize("dimensions", [1, 2])
def test_short_spline_axes_have_a_valid_predictable_basis(dimensions):
    position = np.linspace(0.0, 10.0, 101)[:, None]
    edges = np.linspace(0.0, 10.0, 11)[:, None]
    if dimensions == 2:
        position = np.column_stack((position, position * 10))
        edges = np.column_stack((edges, edges * 10))
    design = make_spline_design_matrix(position, edges)
    prediction = make_spline_predict_matrix(design.design_info, position)
    assert design.shape == prediction.shape
    assert np.all(np.isfinite(prediction))


@pytest.mark.parametrize("std", [np.float32(1.0), np.asarray([1.0])])
def test_sorted_kde_accepts_numpy_bandwidths(std):
    time = np.arange(5.0)
    position = time[:, None]
    env = Environment(place_bin_size=1.0).fit_place_grid(
        position, infer_track_interior=False
    )
    model = fit_sorted_spikes_kde_encoding_model(
        time,
        position,
        [np.asarray([1.0, 3.0])],
        env,
        position_std=std,
        disable_progress_bar=True,
    )
    assert np.all(np.isfinite(model["place_fields"]))


@pytest.mark.parametrize("std", [0.0, -1.0, np.nan, np.inf])
def test_sorted_kde_rejects_invalid_bandwidths(std):
    time = np.arange(5.0)
    position = time[:, None]
    env = Environment(place_bin_size=1.0).fit_place_grid(
        position, infer_track_interior=False
    )
    with pytest.raises(ValidationError, match="position_std"):
        fit_sorted_spikes_kde_encoding_model(
            time,
            position,
            [np.asarray([1.0])],
            env,
            position_std=std,
            disable_progress_bar=True,
        )


@pytest.mark.parametrize("rate", [-1.0, np.nan, np.inf])
def test_no_spike_rejects_invalid_rates(rate):
    with pytest.raises(ValidationError, match="no_spike_rate"):
        predict_no_spike_log_likelihood(
            [np.asarray([])], rate, time_edges=np.asarray([0.0, 1.0])
        )


def test_zero_no_spike_rate_has_the_empty_bin_exposure_gradient():
    def score(rate):
        return predict_no_spike_log_likelihood(
            [np.asarray([])], rate, time_edges=np.asarray([0.0, 1.0])
        ).sum()

    assert float(jax.grad(score)(0.0)) == pytest.approx(-1.0)


@pytest.mark.parametrize(
    "fit", [fit_sorted_spikes_kde_encoding_model, fit_sorted_spikes_glm_encoding_model]
)
def test_empty_sorted_population_has_empty_fields_and_zero_total_rate(fit):
    time = np.arange(11.0)
    position = time[:, None]
    env = Environment(place_bin_size=1.0).fit_place_grid(
        position, infer_track_interior=False
    )
    kwargs = {}
    if fit is fit_sorted_spikes_glm_encoding_model:
        kwargs = {
            "place_bin_edges": env.place_bin_edges_,
            "edges": env.edges_,
            "is_track_interior": env.is_track_interior_,
            "is_track_boundary": env.is_track_boundary_,
        }
    model = fit(time, position, [], env, disable_progress_bar=True, **kwargs)
    assert model["place_fields"].shape == (0, len(env.place_bin_centers_))
    np.testing.assert_array_equal(model["no_spike_part_log_likelihood"], 0.0)
