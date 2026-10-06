"""Atomic incremental output does not imply eager spatial-array loading."""

import numpy as np
import pytest

from non_local_detector.result_store import IncrementalResultWriter, open_result_store

pytestmark = pytest.mark.unit


def test_reverse_chunk_writes_lazy_slice_and_complete_roundtrip(tmp_path):
    path = tmp_path / "result"
    values = np.arange(30, dtype=np.float32).reshape(10, 3)
    coords = {
        "time": (("time",), np.arange(10)),
        "state_bins": (("state_bins",), np.arange(3)),
    }
    with IncrementalResultWriter(
        path, {"posterior": (("time", "state_bins"), (10, 3), np.float32)}, coords
    ) as writer:
        writer.write("posterior", 6, values[6:])
        writer.write("posterior", 3, values[3:6])
        writer.write("posterior", 0, values[:3])
        with pytest.raises(FileNotFoundError):
            open_result_store(path)
        writer.complete({"evidence": -3.0})
    loaded = open_result_store(path)
    assert loaded.posterior.variable._in_memory is False
    np.testing.assert_array_equal(
        loaded.posterior.isel(time=slice(2, 5), state_bins=[0, 2]),
        values[2:5][:, [0, 2]],
    )
    np.testing.assert_array_equal(loaded.posterior, values)
    assert loaded.attrs["evidence"] == -3.0


def test_explicit_large_read_budget_and_small_slice(tmp_path):
    path = tmp_path / "result"
    with IncrementalResultWriter(
        path,
        {"posterior": (("time", "state_bins"), (10, 3), np.float32)},
        {"time": (("time",), np.arange(10))},
    ) as writer:
        writer.write("posterior", 0, np.ones((10, 3), np.float32))
        writer.complete()
    loaded = open_result_store(path, max_read_bytes=24)
    np.testing.assert_array_equal(
        loaded.posterior.isel(time=slice(0, 2)), np.ones((2, 3))
    )
    with pytest.raises(MemoryError, match="max_read_bytes"):
        _ = loaded.posterior.values


def test_partial_or_overlapping_writer_cannot_publish(tmp_path):
    path = tmp_path / "result"
    with pytest.raises(ValueError, match="coverage"):
        with IncrementalResultWriter(
            path, {"posterior": (("time", "state_bins"), (10, 3), np.float32)}, {}
        ) as writer:
            writer.write("posterior", 0, np.ones((3, 3), np.float32))
            writer.complete()
    assert not path.exists()
    assert not list(tmp_path.glob(".result-*"))
    with pytest.raises(ValueError, match="overlap"):
        with IncrementalResultWriter(
            path, {"posterior": (("time", "state_bins"), (10, 3), np.float32)}, {}
        ) as writer:
            writer.write("posterior", 0, np.ones((3, 3), np.float32))
            writer.write("posterior", 2, np.ones((3, 3), np.float32))


def test_writer_exception_cleans_staging_without_touching_existing(tmp_path):
    existing = tmp_path / "result"
    existing.mkdir()
    marker = existing / "keep"
    marker.write_text("original")
    with pytest.raises(FileExistsError):
        IncrementalResultWriter(existing, {}, {})
    assert marker.read_text() == "original"
    with pytest.raises(RuntimeError):
        with IncrementalResultWriter(
            tmp_path / "new", {"x": (("time",), (2,), np.float32)}, {}
        ) as writer:
            writer.write("x", 0, np.ones(2, np.float32))
            raise RuntimeError("I/O failure")
    assert not (tmp_path / "new").exists()
    assert not list(tmp_path.glob(".new-*"))


def test_lazy_padding_reads_only_selected_interior_columns_and_preserves_singleton(
    tmp_path,
):
    import xarray as xr

    from non_local_detector.result_store import pad_state_bins

    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    with IncrementalResultWriter(
        tmp_path / "result",
        {"posterior": (("time", "state_bins"), (4, 3), np.float32)},
        {},
    ) as writer:
        writer.write("posterior", 0, values[:2])
        writer.write("posterior", 2, values[2:])
        writer.complete()
    compact = open_result_store(tmp_path / "result", max_read_bytes=12)
    padded = xr.DataArray(
        pad_state_bins(compact.posterior, [False, True, False, True, True])
    )
    assert not padded.variable._in_memory
    actual = padded.isel(time=slice(1, 2), state_bins=[0, 1, 4]).values
    np.testing.assert_array_equal(actual, [[np.nan, 3, 5]])
    np.testing.assert_array_equal(
        padded.isel(time=2, state_bins=[0, 2]), [np.nan, np.nan]
    )
    with pytest.raises(MemoryError):
        _ = padded.values


@pytest.mark.parametrize(
    "corruption", ["gap", "overlap", "missing_file", "shape", "dtype"]
)
def test_loader_rejects_corrupt_chunk_manifest_before_returning(tmp_path, corruption):
    import json

    path = tmp_path / "result"
    with IncrementalResultWriter(
        path, {"posterior": (("time", "state_bins"), (4, 3), np.float32)}, {}
    ) as writer:
        writer.write("posterior", 0, np.ones((2, 3), np.float32))
        writer.write("posterior", 2, np.ones((2, 3), np.float32))
        writer.complete()
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    chunks = manifest["variables"]["posterior"]["chunks"]
    if corruption == "gap":
        chunks.pop()
    elif corruption == "overlap":
        chunks[1]["start"] = 1
    elif corruption == "missing_file":
        (path / chunks[1]["file"]).unlink()
    elif corruption == "shape":
        np.save(path / chunks[1]["file"], np.ones((1, 3), np.float32))
    else:
        np.save(path / chunks[1]["file"], np.ones((2, 3), np.float64))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="coverage|chunk|manifest"):
        open_result_store(path)


def test_loader_validates_headers_with_closed_mmaps_and_keeps_data_lazy(
    tmp_path, monkeypatch
):
    path = tmp_path / "result"
    with IncrementalResultWriter(
        path, {"posterior": (("time", "state_bins"), (4, 3), np.float32)}, {}
    ) as writer:
        writer.write("posterior", 0, np.ones((2, 3), np.float32))
        writer.write("posterior", 2, np.ones((2, 3), np.float32))
        writer.complete()
    original = np.load
    mappings = []

    def header_only(*args, **kwargs):
        assert kwargs.get("mmap_mode") == "r"
        array = original(*args, **kwargs)
        mappings.append(array._mmap)
        return array

    monkeypatch.setattr(np, "load", header_only)
    loaded = open_result_store(path)
    assert not loaded.posterior.variable._in_memory
    assert len(mappings) == 2
    assert all(mapping.closed for mapping in mappings)
