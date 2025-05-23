import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from non_local_detector.environment.region import RegionInfo, RegionManager

# --- Minimal mock Environment for testing ---


class MockEnvironment:
    def __init__(
        self,
        n_dims=2,
        bin_centers=None,
        grid_shape=None,
        active_mask=None,
        grid_edges=None,
        is_fitted=True,
        environment_name="MockEnv",
        layout=None,
        layout_type="grid",
    ):
        self.n_dims = n_dims
        self.bin_centers_ = (
            bin_centers if bin_centers is not None else np.zeros((0, n_dims))
        )
        self.grid_shape_ = grid_shape
        self.active_mask_ = active_mask
        self.grid_edges_ = grid_edges
        self._is_fitted = is_fitted
        self.environment_name = environment_name
        self._layout_type_used = layout_type
        self.layout = layout if layout is not None else self
        # Map from flat index in full grid to active bin index (identity for simple tests)
        self._source_flat_to_active_node_id_map = {
            i: i for i in range(self.bin_centers_.shape[0])
        }

    def get_bin_ind(self, points):
        # Return the index of the closest bin center for each point, or -1 if not found
        if self.bin_centers_.shape[0] == 0:
            return np.full((points.shape[0],), -1, dtype=int)
        dists = np.linalg.norm(
            self.bin_centers_[None, :, :] - points[:, None, :], axis=2
        )
        return np.argmin(dists, axis=1)

    def get_bin_area_volume(self):
        # Return area 1.0 for each bin for mask area tests
        return np.ones(self.bin_centers_.shape[0])


# --- Fixtures ---


@pytest.fixture
def env_2d_grid():
    # 2x2 grid, 2D, all bins active
    bin_centers = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    grid_shape = (2, 2)
    active_mask = np.ones(grid_shape, dtype=bool)
    grid_edges = [np.array([0, 1, 2]), np.array([0, 1, 2])]
    return MockEnvironment(
        n_dims=2,
        bin_centers=bin_centers,
        grid_shape=grid_shape,
        active_mask=active_mask,
        grid_edges=grid_edges,
    )


@pytest.fixture
def region_manager(env_2d_grid):
    return RegionManager(env_2d_grid)


# --- Tests ---


def test_add_point_region(region_manager):
    region_manager.add_region("pt", point=(1, 1))
    assert "pt" in region_manager.list_regions()
    info = region_manager.get_region_info("pt")
    assert info.kind == "point"
    np.testing.assert_array_equal(info.data, np.array([1, 1]))


def test_add_mask_region(region_manager):
    mask = np.array([[True, False], [False, True]])
    region_manager.add_region("mask1", mask=mask)
    assert "mask1" in region_manager.list_regions()
    info = region_manager.get_region_info("mask1")
    assert info.kind == "mask"
    np.testing.assert_array_equal(info.data, mask)


@pytest.mark.skipif(
    not hasattr(region_manager.__globals__, "_HAS_SHAPELY")
    or not region_manager.__globals__["_HAS_SHAPELY"],
    reason="Shapely not installed",
)
def test_add_polygon_region(region_manager):
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    region_manager.add_region("poly", polygon=poly)
    assert "poly" in region_manager.list_regions()
    info = region_manager.get_region_info("poly")
    assert info.kind == "polygon"
    assert info.data.equals(poly)


def test_remove_region(region_manager):
    region_manager.add_region("pt", point=(0, 0))
    region_manager.remove_region("pt")
    assert "pt" not in region_manager.list_regions()
    # Removing non-existent region should not raise
    region_manager.remove_region("does_not_exist")


def test_region_mask_point(region_manager):
    region_manager.add_region("pt", point=(1, 1))
    mask = region_manager.region_mask("pt")
    # Only the last bin center is at (1,1)
    assert mask.tolist() == [False, False, False, True]


def test_region_mask_mask(region_manager):
    mask = np.array([[True, False], [False, True]])
    region_manager.add_region("mask1", mask=mask)
    mask1d = region_manager.region_mask("mask1")
    # Only bins 0 and 3 are in the region
    assert mask1d.tolist() == [True, False, False, True]


@pytest.mark.skipif(
    not hasattr(region_manager.__globals__, "_HAS_SHAPELY")
    or not region_manager.__globals__["_HAS_SHAPELY"],
    reason="Shapely not installed",
)
def test_region_mask_polygon(region_manager):
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    region_manager.add_region("poly", polygon=poly)
    mask = region_manager.region_mask("poly")
    # All bin centers are inside the square
    assert mask.tolist() == [True, True, True, True]


def test_bins_in_region(region_manager):
    mask = np.array([[True, False], [False, True]])
    region_manager.add_region("mask1", mask=mask)
    bins = region_manager.bins_in_region("mask1")
    assert set(bins.tolist()) == {0, 3}


def test_region_center_point(region_manager):
    region_manager.add_region("pt", point=(1, 0))
    center = region_manager.region_center("pt")
    np.testing.assert_array_equal(center, np.array([1, 0]))


def test_region_center_mask(region_manager):
    mask = np.array([[True, False], [False, True]])
    region_manager.add_region("mask1", mask=mask)
    center = region_manager.region_center("mask1")
    # Centers at (0,0) and (1,1)
    np.testing.assert_array_almost_equal(center, np.array([0.5, 0.5]))


def test_nearest_region(region_manager):
    region_manager.add_region("pt1", point=(0, 0))
    region_manager.add_region("pt2", point=(1, 1))
    nearest = region_manager.nearest_region(np.array([0.1, 0.1]))
    assert nearest == "pt1"
    nearest = region_manager.nearest_region(np.array([0.9, 0.9]))
    assert nearest == "pt2"


def test_get_region_area_point(region_manager):
    region_manager.add_region("pt", point=(1, 1))
    area = region_manager.get_region_area("pt")
    assert area == 0.0


def test_get_region_area_mask(region_manager):
    mask = np.array([[True, False], [False, True]])
    region_manager.add_region("mask1", mask=mask)
    # Each bin is 1x1, so area = 2
    area = region_manager.get_region_area("mask1")
    assert area == 2.0


@pytest.mark.skipif(
    not hasattr(region_manager.__globals__, "_HAS_SHAPELY")
    or not region_manager.__globals__["_HAS_SHAPELY"],
    reason="Shapely not installed",
)
def test_get_region_area_polygon(region_manager):
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    region_manager.add_region("poly", polygon=poly)
    area = region_manager.get_region_area("poly")
    assert area == pytest.approx(1.0)


@pytest.mark.skipif(
    not hasattr(region_manager.__globals__, "_HAS_SHAPELY")
    or not region_manager.__globals__["_HAS_SHAPELY"],
    reason="Shapely not installed",
)
def test_create_buffered_region_point(region_manager):
    region_manager.add_region("pt", point=(0.5, 0.5))
    region_manager.create_buffered_region(
        "pt", buffer_distance=0.5, new_region_name="buffered"
    )
    info = region_manager.get_region_info("buffered")
    assert info.kind == "polygon"
    # The center of the buffered region should be close to (0.5, 0.5)
    center = info.data.centroid
    assert np.allclose([center.x, center.y], [0.5, 0.5], atol=1e-2)


@pytest.mark.skipif(
    not hasattr(region_manager.__globals__, "_HAS_SHAPELY")
    or not region_manager.__globals__["_HAS_SHAPELY"],
    reason="Shapely not installed",
)
def test_get_region_relationship_intersection(region_manager):
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)])
    region_manager.add_region("poly1", polygon=poly1)
    region_manager.add_region("poly2", polygon=poly2)
    result = region_manager.get_region_relationship(
        "poly1", "poly2", relationship_type="intersection"
    )
    assert result is not None
    # The intersection should be a square of area 0.25 at (0.5,0.5)-(1,1)
    assert result.area == pytest.approx(0.25)


def test_repr(region_manager):
    s = repr(region_manager)
    assert "RegionManager" in s
    assert "MockEnv" in s


def test_environment_property(region_manager, env_2d_grid):
    # Test getter
    assert region_manager.environment is env_2d_grid
    # Test setter with warning
    new_env = MockEnvironment(n_dims=2)
    with pytest.warns(UserWarning):
        region_manager.environment = new_env
    assert region_manager.environment is new_env


def test_add_region_errors(region_manager):
    # Must provide exactly one of point, mask, polygon
    with pytest.raises(ValueError):
        region_manager.add_region(
            "fail", point=(0, 0), mask=np.ones((2, 2), dtype=bool)
        )
    # Duplicate name
    region_manager.add_region("pt", point=(0, 0))
    with pytest.raises(ValueError):
        region_manager.add_region("pt", point=(1, 1))
    # Wrong point dimension
    with pytest.raises(ValueError):
        region_manager.add_region("badpt", point=(1,))
    # Wrong mask shape
    bad_mask = np.ones((3, 3), dtype=bool)
    with pytest.raises(ValueError):
        region_manager.add_region("badmask", mask=bad_mask)
    # Wrong mask dtype
    with pytest.raises(TypeError):
        region_manager.add_region("badmask2", mask=np.ones((2, 2), dtype=float))


def test_region_mask_errors(region_manager):
    # Not fitted environment
    region_manager._env._is_fitted = False
    with pytest.raises(RuntimeError):
        region_manager.region_mask("nonexistent")
    region_manager._env._is_fitted = True
    # Nonexistent region
    with pytest.raises(KeyError):
        region_manager.region_mask("nonexistent")
