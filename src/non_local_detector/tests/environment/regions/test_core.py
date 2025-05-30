from collections.abc import Mapping

import numpy as np
import pytest

from non_local_detector.environment.regions.core import Region, Regions

try:
    import shapely.geometry as shp

    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


@pytest.mark.parametrize(
    "coords",
    [
        [1.0, 2.0],
        np.array([3.5, 4.5]),
    ],
)
def test_region_point_creation(coords):
    r = Region(name="A", kind="point", data=coords)
    assert r.name == "A"
    assert r.kind == "point"
    assert np.allclose(r.data, np.asarray(coords))
    assert r.n_dims == len(coords)
    assert isinstance(r.metadata, Mapping)


def test_region_point_invalid_shape():
    with pytest.raises(ValueError):
        Region(name="bad", kind="point", data=[[1, 2], [3, 4]])


def test_region_str_repr():
    r = Region(name="foo", kind="point", data=[0, 1])
    assert str(r) == "foo"


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_region_polygon_creation():
    poly = shp.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    r = Region(name="poly", kind="polygon", data=poly)
    assert r.kind == "polygon"
    assert r.n_dims == 2
    assert r.data.equals(poly)


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_region_polygon_invalid_type():
    with pytest.raises(TypeError):
        Region(name="badpoly", kind="polygon", data=[(0, 0), (1, 1)])


def test_region_unknown_kind():
    with pytest.raises(ValueError):
        Region(name="bad", kind="unknown", data=[1, 2])


def test_region_to_dict_and_from_dict_point():
    r = Region(name="pt", kind="point", data=[1, 2], metadata={"color": "red"})
    d = r.to_dict()
    r2 = Region.from_dict(d)
    assert r2.name == r.name
    assert r2.kind == r.kind
    assert np.allclose(r2.data, r.data)
    assert r2.metadata["color"] == "red"


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_region_to_dict_and_from_dict_polygon():
    poly = shp.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    r = Region(name="poly", kind="polygon", data=poly, metadata={"label": "square"})
    d = r.to_dict()
    r2 = Region.from_dict(d)
    assert r2.name == r.name
    assert r2.kind == r.kind
    assert r2.data.equals(poly)
    assert r2.metadata["label"] == "square"


def test_regions_add_point_and_remove():
    regs = Regions()
    r = regs.add("pt", point=[1, 2, 3])
    assert regs["pt"] == r
    assert regs.list_names() == ["pt"]
    regs.remove("pt")
    assert "pt" not in regs


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_regions_add_polygon_and_area():
    regs = Regions()
    poly = shp.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    r = regs.add("poly", polygon=poly)
    assert regs["poly"] == r
    area = regs.area("poly")
    assert np.isclose(area, 1.0)
    assert regs.area("poly") == poly.area


def test_regions_add_duplicate_name():
    regs = Regions()
    regs.add("pt", point=[1, 2])
    with pytest.raises(KeyError):
        regs.add("pt", point=[3, 4])


def test_regions_add_both_point_and_polygon():
    regs = Regions()
    with pytest.raises(ValueError):
        regs.add(name="bad", point=(1.0, 2.0), polygon=[(0, 0), (1, 0), (0, 1)])
    if HAS_SHAPELY:
        poly = shp.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        with pytest.raises(ValueError):
            regs.add("bar", point=[1, 2], polygon=poly)


def test_regions_setitem_key_mismatch():
    regs = Regions()
    r = Region(name="foo", kind="point", data=[1, 2])
    with pytest.raises(ValueError):
        regs["bar"] = r


def test_regions_setitem_duplicate():
    regs = Regions()
    r = Region(name="foo", kind="point", data=[1, 2])
    regs["foo"] = r
    with pytest.raises(KeyError):
        regs["foo"] = r


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_regions_buffer_point_and_polygon(tmp_path):
    regs = Regions()
    pt = [0.0, 0.0]
    regs.add("pt", point=pt)
    regs.buffer("pt", distance=1.0, new_name="buf")
    assert "buf" in regs
    assert regs["buf"].kind == "polygon"
    # Buffering a polygon
    poly = shp.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    regs.add("poly", polygon=poly)
    regs.buffer("poly", distance=0.5, new_name="buf2")
    assert "buf2" in regs


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_regions_buffer_raw_point():
    regs = Regions()
    regs.buffer(np.array([0.0, 0.0]), distance=1.0, new_name="buf")
    assert "buf" in regs


@pytest.mark.skipif(not HAS_SHAPELY, reason="Shapely required for polygon tests")
def test_regions_buffer_invalid_shape():
    regs = Regions()
    with pytest.raises(ValueError):
        regs.buffer(np.array([1.0, 2.0, 3.0]), distance=1.0, new_name="badbuf")


def test_regions_area_point():
    regs = Regions()
    regs.add("pt", point=[1, 2])
    assert regs.area("pt") == 0.0


def test_regions_remove_absent():
    regs = Regions()
    regs.remove("nope")  # Should not raise


def test_regions_repr():
    regs = Regions()
    regs.add("pt", point=[1, 2])
    s = repr(regs)
    assert "Regions" in s and "pt(point)" in s


def test_regions_to_json_and_from_json(tmp_path):
    regs = Regions()
    regs.add("pt", point=[1, 2], metadata={"foo": "bar"})
    path = tmp_path / "regions.json"
    regs.to_json(path)
    loaded = Regions.from_json(path)
    assert "pt" in loaded
    assert loaded["pt"].metadata["foo"] == "bar"
