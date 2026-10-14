"""Prepared exposure must preserve Gaussian ratios when log densities are large."""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector.encoding_time import EncodingSupport
from non_local_detector.environment import Environment
from non_local_detector.likelihoods.common import _log_kernel_matrix
from non_local_detector.likelihoods.sorted_spikes_glm import (
    fit_sorted_spikes_glm_encoding_model,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("weights_kind", ["uniform", "fractional", "last_sample"])
def test_stationary_prepared_exposure_preserves_the_single_gaussian(weights_kind):
    time = np.linspace(0.0, 1.0, 129)
    position = np.zeros((len(time), 1))
    support = EncodingSupport(time, position, encoding_time_range=[0.0, 1.0])
    quadrature = support.quadrature(position, spatial_scale=1.0)
    centers = np.array([[20.0], [200.0], [1000.0], [1457.0]])
    if weights_kind == "uniform":
        weights = np.ones(len(time))
    elif weights_kind == "fractional":
        weights = np.linspace(0.2, 1.0, len(time))
    else:
        weights = np.zeros(len(time))
        weights[-1] = 1.0
    actual = support.cached_kde_log_density(
        quadrature, quadrature.positions, centers, np.ones(1), weights
    )
    # Every quadrature kernel is identical. Integrated duration and posterior
    # weights cancel before restoring this single Gaussian kernel's offset.
    gaussian = _log_kernel_matrix(jnp.asarray(centers), jnp.zeros((1, 1)), jnp.ones(1))[
        0
    ]
    np.testing.assert_allclose(
        np.exp(np.asarray(actual - gaussian)),
        np.ones(len(centers)),
        rtol=1e-6,
        atol=1e-6,
    )


def test_activated_tail_after_refit_preserves_its_gaussian_and_batch_independence():
    time = np.linspace(0.0, 1.0, 129)
    position = np.zeros((len(time), 1))
    position[-4:] = 20.0
    support = EncodingSupport(time, position, encoding_time_range=[0.0, 1.0])
    quadrature = support.quadrature(position, spatial_scale=1.0)
    centers = np.array([[0.0], [20.0]])
    support.cached_kde_log_density(
        quadrature, quadrature.positions, centers, np.ones(1), np.ones(len(time))
    )
    weights = np.zeros(len(time))
    weights[-1] = 1.0
    actual = support.cached_kde_log_density(
        quadrature, quadrature.positions, centers, np.ones(1), weights
    )
    gaussian = _log_kernel_matrix(
        jnp.asarray(centers), jnp.asarray([[20.0]]), jnp.ones(1)
    )[0]
    np.testing.assert_allclose(
        np.exp(np.asarray(actual - gaussian)),
        np.ones(len(centers)),
        rtol=1e-6,
        atol=1e-6,
    )
    alone = support.cached_kde_log_density(
        quadrature, quadrature.positions, centers[:1], np.ones(1), weights
    )
    np.testing.assert_array_equal(np.asarray(actual[:1]), np.asarray(alone))


def test_glm_geometry_cache_invalidates_changed_track_projection_after_grid_refit():
    """Equal 1D grid coordinates can describe a different physical XY track."""
    import networkx as nx

    time = np.linspace(0.0, 4.0, 129)
    position = np.column_stack((10.0 + 8.0 * np.sin(time), np.zeros(len(time))))
    spikes = [np.linspace(0.05, 3.95, 41)]
    graph = nx.Graph()
    graph.add_node(0, pos=(0.0, 0.0))
    graph.add_node(1, pos=(20.0, 0.0))
    graph.add_edge(0, 1, distance=20.0, edge_id=0)
    environment = Environment(
        place_bin_size=1.0,
        track_graph=graph,
        edge_order=[(0, 1)],
        edge_spacing=0.0,
    ).fit_place_grid()
    support = EncodingSupport(time, position)

    def fit(active_support):
        return fit_sorted_spikes_glm_encoding_model(
            position_time=time,
            position=position,
            spike_times=spikes,
            environment=environment,
            place_bin_edges=environment.place_bin_edges_,
            edges=environment.edges_,
            is_track_interior=environment.is_track_interior_,
            is_track_boundary=environment.is_track_boundary_,
            emission_knot_spacing=5.0,
            disable_progress_bar=True,
            _encoding_support=active_support,
        )

    original_centers = environment.place_bin_centers_.copy()
    original_edges = environment.place_bin_edges_.copy()
    fit(support)
    graph.nodes[1]["pos"] = (0.0, 20.0)
    environment.fit_place_grid()
    np.testing.assert_array_equal(environment.place_bin_centers_, original_centers)
    np.testing.assert_array_equal(environment.place_bin_edges_, original_edges)
    cached, reference = fit(support), fit(None)
    np.testing.assert_allclose(
        cached["place_fields"], reference["place_fields"], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        cached["coefficients"], reference["coefficients"], rtol=1e-6, atol=1e-6
    )
