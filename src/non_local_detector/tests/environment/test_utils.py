import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

from non_local_detector.environment.utils import (
    _generic_graph_plot,
    _get_distance_between_bins,
    _infer_active_elements_from_samples,
    _infer_dimension_ranges_from_samples,
    get_centers,
    get_n_bins,
)


def test_get_centers_basic():
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    centers = get_centers(edges)
    np.testing.assert_allclose(centers, [0.5, 1.5, 2.5])


def test_get_centers_single_bin():
    edges = np.array([2.0, 5.0])
    centers = get_centers(edges)
    np.testing.assert_allclose(centers, [3.5])


def test_get_n_bins_basic():
    data = np.array([[0, 0], [2, 2], [4, 4]])
    bins = get_n_bins(data, bin_size=2)
    np.testing.assert_array_equal(bins, [2, 2])


def test_get_n_bins_with_dimension_range():
    data = np.array([[0, 0], [2, 2], [4, 4]])
    bins = get_n_bins(data, bin_size=[2, 2], dimension_range=[(0, 5), (0, 5)])
    np.testing.assert_array_equal(bins, [3, 3])


def test_get_n_bins_zero_extent():
    data = np.array([[1, 1], [1, 1]])
    bins = get_n_bins(data, bin_size=1)
    np.testing.assert_array_equal(bins, [1, 1])


def test_get_n_bins_invalid_bin_size():
    data = np.array([[0, 0], [1, 1]])
    with pytest.raises(ValueError):
        get_n_bins(data, bin_size=0)


def test_get_n_bins_invalid_dimension_range():
    data = np.array([[0, 0], [1, 1]])
    with pytest.raises(ValueError):
        get_n_bins(data, bin_size=1, dimension_range=[(0,)])


def test_infer_active_elements_from_samples_basic():
    candidates = np.array([[0, 0], [1, 1], [2, 2]])
    samples = np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]])
    mask, active, idxs = _infer_active_elements_from_samples(candidates, samples)
    assert mask.sum() == 3
    np.testing.assert_array_equal(np.sort(idxs), [0, 1, 2])
    np.testing.assert_allclose(active, candidates)


def test_infer_active_elements_from_samples_threshold():
    candidates = np.array([[0, 0], [1, 1], [2, 2]])
    samples = np.array([[0.1, 0.1], [0.2, 0.2], [2.1, 2.1]])
    mask, active, idxs = _infer_active_elements_from_samples(
        candidates, samples, bin_count_threshold=1
    )
    assert mask.sum() == 1
    np.testing.assert_array_equal(idxs, [0])


def test_infer_active_elements_from_samples_nan_samples():
    candidates = np.array([[0, 0], [1, 1]])
    samples = np.array([[np.nan, 0], [1, 1]])
    mask, active, idxs = _infer_active_elements_from_samples(candidates, samples)
    assert mask.sum() == 1
    np.testing.assert_array_equal(idxs, [1])


def test_infer_active_elements_from_samples_empty_candidates():
    candidates = np.empty((0, 2))
    samples = np.array([[1, 1]])
    mask, active, idxs = _infer_active_elements_from_samples(candidates, samples)
    assert mask.size == 0
    assert active.shape == (0, 2)
    assert idxs.size == 0


def test_infer_active_elements_from_samples_dim_mismatch():
    candidates = np.array([[0, 0]])
    samples = np.array([[1, 1, 1]])
    with pytest.raises(ValueError):
        _infer_active_elements_from_samples(candidates, samples)


def test_infer_active_elements_from_samples_negative_threshold():
    candidates = np.array([[0, 0]])
    samples = np.array([[0, 0]])
    with pytest.raises(ValueError):
        _infer_active_elements_from_samples(candidates, samples, bin_count_threshold=-1)


def test_infer_dimension_ranges_from_samples_basic():
    data = np.array([[0, 1], [2, 3], [4, 5]])
    result = _infer_dimension_ranges_from_samples(data)
    assert np.allclose(result[0], (0, 4))
    assert np.allclose(result[1], (1, 5))


def test_infer_dimension_ranges_from_samples_with_buffer():
    data = np.array([[1, 2], [3, 4]])
    result = _infer_dimension_ranges_from_samples(data, buffer_around_data=1)
    assert np.allclose(result[0], (0, 4))
    assert np.allclose(result[1], (1, 5))


def test_infer_dimension_ranges_from_samples_buffer_sequence():
    data = np.array([[1, 2], [3, 4]])
    result = _infer_dimension_ranges_from_samples(data, buffer_around_data=[0.5, 1.5])
    assert np.allclose(result[0], (0.5, 3.5))
    assert np.allclose(result[1], (0.5, 5.5))


def test_infer_dimension_ranges_from_samples_point_data():
    data = np.array([[2, 3], [2, 3]])
    result = _infer_dimension_ranges_from_samples(data)
    assert np.allclose(result[0], (1.5, 2.5))
    assert np.allclose(result[1], (2.5, 3.5))


def test_infer_dimension_ranges_from_samples_all_nan():
    data = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    with pytest.raises(ValueError):
        _infer_dimension_ranges_from_samples(data)


def test_infer_dimension_ranges_from_samples_wrong_buffer_length():
    data = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        _infer_dimension_ranges_from_samples(data, buffer_around_data=[1])


def test_infer_dimension_ranges_from_samples_not_2d():
    data = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        _infer_dimension_ranges_from_samples(data)


def test_generic_graph_plot_2d(monkeypatch):
    G = nx.Graph()
    G.add_node(0, pos=(0, 0))
    G.add_node(1, pos=(1, 1))
    G.add_edge(0, 1)
    # Patch plt.figure to avoid opening a window
    monkeypatch.setattr(plt, "figure", lambda *a, **k: plt.Figure())
    ax = _generic_graph_plot(G, "Test2D")
    assert hasattr(ax, "set_aspect")


def test_generic_graph_plot_3d(monkeypatch):
    G = nx.Graph()
    G.add_node(0, pos=(0, 0, 0))
    G.add_node(1, pos=(1, 1, 1))
    G.add_edge(0, 1)
    # Patch plt.figure to avoid opening a window
    monkeypatch.setattr(plt, "figure", lambda *a, **k: plt.Figure())
    ax = _generic_graph_plot(G, "Test3D")
    assert hasattr(ax, "set_zlabel") or hasattr(ax, "set_box_aspect")


def test_generic_graph_plot_empty_graph():
    G = nx.Graph()
    with pytest.raises(ValueError):
        _generic_graph_plot(G, "Empty")


def test_get_distance_between_bins_basic():
    G = nx.Graph()
    G.add_node(0)
    G.add_node(1)
    G.add_node(2)
    G.add_edge(0, 1, distance=1.0)
    G.add_edge(1, 2, distance=2.0)
    dist = _get_distance_between_bins(G)
    assert dist.shape == (3, 3)
    assert dist[0, 1] == 1.0
    assert dist[0, 2] == 3.0
    assert dist[1, 2] == 2.0
    assert dist[2, 0] == 3.0
    assert dist[0, 0] == 0.0


def test_get_distance_between_bins_disconnected():
    G = nx.Graph()
    G.add_node(0)
    G.add_node(1)
    dist = _get_distance_between_bins(G)
    assert np.isinf(dist[0, 1])
    assert dist[0, 0] == 0.0
    assert dist[1, 1] == 0.0
