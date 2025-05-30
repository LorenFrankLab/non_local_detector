import numpy as np
import pytest

from non_local_detector.environment.regions import RegionInfo, RegionManager

try:
    import shapely.geometry as shp

    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


@pytest.fixture
def manager():
    return RegionManager()


def test_add_and_list_point_region(manager):
    manager.add_region("p1", point=(1.0, 2.0))
    assert "p1" in manager.list_regions()
    info = manager.get_region_info("p1")
    assert info.kind == "point"
    assert np.allclose(info.data, [1.0, 2.0])
    assert info.n_dims == 2


def test_add_duplicate_region_raises(manager):
    manager.add_region("p1", point=(1.0, 2.0))
    with pytest.raises(ValueError):
        manager.add_region("p1", point=(3.0, 4.0))


def test_add_point_with_wrong_shape_raises(manager):
    with pytest.raises(ValueError):
        manager.add_region("bad", point=[[1.0, 2.0], [3.0, 4.0]])


def test_remove_region(manager):
    manager.add_region("p1", point=(1.0, 2.0))
    manager.remove_region("p1")
    assert "p1" not in manager.list_regions()
    # Removing non-existent region should not raise
    manager.remove_region("not_exist")


def test_get_region_info_not_found(manager):
    with pytest.raises(KeyError):
        manager.get_region_info("nope")


def test_add_region_requires_one_geom(manager):
    with pytest.raises(ValueError):
        manager.add_region("bad", point=(1.0, 2.0), polygon=[(0, 0), (1, 0), (1, 1)])
    with pytest.raises(ValueError):
        manager.add_region("bad2")


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_add_and_area_polygon_region(manager):
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    manager.add_region("poly", polygon=coords)
    info = manager.get_region_info("poly")
    assert info.kind == "polygon"
    assert info.n_dims == 2
    area = manager.get_region_area("poly")
    assert pytest.approx(area) == 1.0


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_add_polygon_invalid_type_raises(manager):
    with pytest.raises(TypeError):
        manager.add_region("badpoly", polygon="not_a_polygon")


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_get_region_area_point(manager):
    manager.add_region("p1", point=(1.0, 2.0))
    assert manager.get_region_area("p1") == 0.0


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_create_buffered_region_from_point(manager):
    manager.add_region("p1", point=(0.0, 0.0))
    manager.create_buffered_region("p1", buffer_distance=1.0, new_region_name="buf")
    info = manager.get_region_info("buf")
    assert info.kind == "polygon"
    assert info.n_dims == 2
    assert info.data.area > 3.0  # Should be close to pi


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_create_buffered_region_from_polygon(manager):
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    manager.add_region("poly", polygon=coords)
    manager.create_buffered_region(
        "poly", buffer_distance=0.5, new_region_name="bufpoly"
    )
    info = manager.get_region_info("bufpoly")
    assert info.kind == "polygon"
    assert info.data.area > 1.0


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_get_region_relationship_intersection(manager):
    manager.add_region("a", polygon=[(0, 0), (2, 0), (2, 2), (0, 2)])
    manager.add_region("b", polygon=[(1, 1), (3, 1), (3, 3), (1, 3)])
    inter = manager.get_region_relationship("a", "b", relationship_type="intersection")
    assert inter is not None
    assert pytest.approx(inter.area) == 1.0


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_get_region_relationship_add_as_new(manager):
    manager.add_region("a", polygon=[(0, 0), (2, 0), (2, 2), (0, 2)])
    manager.add_region("b", polygon=[(1, 1), (3, 1), (3, 3), (1, 3)])
    manager.get_region_relationship(
        "a", "b", relationship_type="intersection", add_as_new_region="inter"
    )
    info = manager.get_region_info("inter")
    assert info.kind == "polygon"
    assert pytest.approx(info.data.area) == 1.0


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_get_region_relationship_invalid_type_raises(manager):
    manager.add_region("a", polygon=[(0, 0), (2, 0), (2, 2), (0, 2)])
    manager.add_region("b", polygon=[(1, 1), (3, 1), (3, 3), (1, 3)])
    with pytest.raises(ValueError):
        manager.get_region_relationship("a", "b", relationship_type="not_a_type")


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_get_region_relationship_non_polygon_raises(manager):
    manager.add_region("p1", point=(0.0, 0.0))
    manager.add_region("poly", polygon=[(0, 0), (1, 0), (1, 1), (0, 1)])
    with pytest.raises(ValueError):
        manager.get_region_relationship("p1", "poly")


def test_repr(manager):
    assert repr(manager) == "RegionManager(n_regions=0)"
    manager.add_region("p1", point=(1.0, 2.0))
    assert repr(manager) == "RegionManager(n_regions=1)"
