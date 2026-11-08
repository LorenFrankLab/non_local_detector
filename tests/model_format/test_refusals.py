"""A saved model is refused when it cannot be read or decoded faithfully.

Unknown types on save and on load, corrupt arrays, files of a newer format
version, partly fitted detectors and rebuilt structures that no longer match
their fingerprints each raise a clear error. A fingerprint mismatch refuses
only decoding: the model still loads and its summary still reads.
"""

import hashlib
import io
import json
import pickle
import warnings
import zipfile
from dataclasses import dataclass

import numpy as np
import pytest

import non_local_detector as nld
from non_local_detector import continuous_state_transitions, environment
from non_local_detector.exceptions import DataError, NotFittedError, ValidationError
from non_local_detector.model_format import (
    FORMAT_VERSION,
    _container,
    read_model_summary,
)
from non_local_detector.models.base import (
    ClusterlessDetector,
    SortedSpikesDetector,
    _DetectorBase,
)
from tests.model_format.conftest import SORTED_RATE, algorithm_params, decode

pytestmark = pytest.mark.integration


def _rewrite(path, name, transform):
    """Replace member ``name`` of the zip at ``path`` with ``transform(bytes)``."""
    with zipfile.ZipFile(path) as source:
        members = [(item, source.read(item.filename)) for item in source.infolist()]
    with zipfile.ZipFile(path, "w") as target:
        for item, data in members:
            target.writestr(item, transform(data) if item.filename == name else data)


def _edit_json(path, name, edit):
    def transform(data):
        document = json.loads(data)
        edit(document)
        return json.dumps(document).encode()

    _rewrite(path, name, transform)


@pytest.fixture
def fitted(sorted_recording):
    time, position, spike_times = sorted_recording
    detector = nld.NonLocalSortedSpikesDetector(
        sampling_frequency=SORTED_RATE,
        sorted_spikes_algorithm_params=algorithm_params("sorted_spikes_kde"),
    )
    return detector.fit(time, position, spike_times)


@pytest.fixture
def saved(fitted, tmp_path):
    path = tmp_path / "model.nld"
    fitted.save_model(path)
    return path


@dataclass
class _UnregisteredUniform:
    environment_name: str = ""

    def make_state_transition(self, environments, context=None):
        n_bins = environments[0].place_bin_centers_.shape[0]
        return np.full((n_bins, n_bins), 1.0 / n_bins)


def test_saving_an_unregistered_transition_says_how_to_register_it(
    sorted_recording, tmp_path
):
    time, position, spike_times = sorted_recording
    detector = nld.SortedSpikesDecoder(
        continuous_transition_types=[[_UnregisteredUniform()]],
        sorted_spikes_algorithm_params=algorithm_params("sorted_spikes_kde"),
        sampling_frequency=SORTED_RATE,
    ).fit(time, position, spike_times)

    with pytest.raises(ValidationError, match="not registered") as error:
        detector.save_model(tmp_path / "model.nld")
    assert "_UnregisteredUniform" in str(error.value)
    assert "register_type" in str(error.value)
    assert not (tmp_path / "model.nld").exists()


def test_loading_an_unknown_type_names_the_package_to_import(saved):
    def rename(config):
        node = config["continuous_initial_conditions_types"][0]
        node["$type"], node["owner"] = "mylab.Initial", "mylab"

    _edit_json(saved, "config.json", rename)

    with pytest.raises(ValidationError, match=r"'mylab\.Initial'") as error:
        nld.NonLocalSortedSpikesDetector.load_model(saved)
    assert "Import the package that registers it ('mylab')" in str(error.value)


def test_loading_an_unknown_detector_type_is_refused(saved):
    def rename(manifest):
        manifest["detector_type"] = "mylab.Detector"
        manifest["detector_type_owner"] = "mylab"

    _edit_json(saved, "manifest.json", rename)

    with pytest.raises(ValidationError, match="not registered"):
        nld.NonLocalSortedSpikesDetector.load_model(saved)


def test_a_tampered_array_fails_its_checksum(saved):
    with zipfile.ZipFile(saved) as archive:
        member = next(n for n in archive.namelist() if n.endswith("place_fields.npy"))

    def tamper(data):
        return data[:-4] + bytes(4)  # zero the last float32 place-field value

    _rewrite(saved, member, tamper)

    with pytest.raises(DataError, match="checksum"):
        nld.NonLocalSortedSpikesDetector.load_model(saved)
    with pytest.raises(DataError, match="checksum"):
        read_model_summary(saved)


# Every unpickling of a ``_Payload`` appends to this list.
UNPICKLED: list[int] = []


def _record():
    UNPICKLED.append(1)
    return 0


class _Payload:
    def __reduce__(self):
        return (_record, ())


def test_an_array_member_holding_a_pickle_is_refused_unread(saved):
    """Arrays load with ``allow_pickle=False``: an object-dtype member whose
    checksum matches is refused, and its pickle never runs."""
    with zipfile.ZipFile(saved) as archive:
        member = next(n for n in archive.namelist() if n.endswith("place_fields.npy"))
    buffer = io.BytesIO()
    np.save(buffer, np.array([_Payload()], dtype=object), allow_pickle=True)
    payload = buffer.getvalue()
    _rewrite(saved, member, lambda _: payload)

    def match_checksum(manifest):
        manifest["arrays"][member]["sha256"] = hashlib.sha256(payload).hexdigest()

    _edit_json(saved, "manifest.json", match_checksum)
    UNPICKLED.clear()

    with pytest.raises(DataError, match="corrupt"):
        nld.NonLocalSortedSpikesDetector.load_model(saved)

    assert UNPICKLED == []


@pytest.mark.parametrize("file_format", ["nld", "pickle"])
def test_load_model_returns_only_its_own_class(fitted, saved, tmp_path, file_format):
    """A class's load_model loads an instance of it or of a subclass."""
    path = saved
    if file_format == "pickle":
        path = tmp_path / "model.pkl"
        path.write_bytes(pickle.dumps(fitted))

    with pytest.raises(ValidationError, match="not a SortedSpikesDecoder") as error:
        nld.SortedSpikesDecoder.load_model(path)
    assert "NonLocalSortedSpikesDetector.load_model" in str(error.value)
    with pytest.raises(ValidationError, match="not a ClusterlessDetector"):
        ClusterlessDetector.load_model(path)
    for cls in (nld.NonLocalSortedSpikesDetector, SortedSpikesDetector, _DetectorBase):
        assert type(cls.load_model(path)) is nld.NonLocalSortedSpikesDetector


def test_load_model_refuses_a_pickle_that_is_not_a_detector(tmp_path):
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps({"not": "a detector"}))

    with pytest.raises(ValidationError, match="holds a dict") as error:
        _DetectorBase.load_model(path)
    assert "dict.load_model" not in str(error.value)


def test_a_newer_format_version_asks_for_an_upgrade(saved):
    def newer(manifest):
        manifest["format_version"] = FORMAT_VERSION + 1
        manifest["package_version"] = "9.1.0"

    _edit_json(saved, "manifest.json", newer)

    for read in (nld.NonLocalSortedSpikesDetector.load_model, read_model_summary):
        with pytest.raises(ValidationError, match=r"non_local_detector >= 9\.1\.0"):
            read(saved)


def test_an_older_format_version_is_migrated_before_it_is_read(saved, monkeypatch):
    """With a format version 2, a version-1 file goes through MIGRATIONS[1]."""
    seen = []

    def to_version_2(stored):
        seen.append(stored.manifest["format_version"])
        stored.state["fitted"] = stored.state.pop("was_fitted")

    def as_version_2(state):
        state["was_fitted"] = state.pop("fitted")

    _edit_json(saved, "state.json", as_version_2)
    monkeypatch.setattr(_container, "FORMAT_VERSION", FORMAT_VERSION + 1)
    monkeypatch.setitem(_container.MIGRATIONS, FORMAT_VERSION, to_version_2)

    loaded = nld.NonLocalSortedSpikesDetector.load_model(saved)

    assert seen == [FORMAT_VERSION]
    assert hasattr(loaded, "encoding_model_")


def test_a_partly_fitted_detector_is_not_saved(fitted, tmp_path):
    del fitted.encoding_model_

    with pytest.raises(NotFittedError, match="partly fitted"):
        fitted.save_model(tmp_path / "model.nld")


def test_a_detector_without_a_stored_name_is_not_saved(fitted, tmp_path):
    """Every name ``load_model`` requires is required at save."""
    path = tmp_path / "model.nld"
    del fitted.initial_conditions_

    with pytest.raises(NotFittedError, match="initial_conditions_"):
        fitted.save_model(path)
    assert not path.exists()


def test_an_attribute_the_format_does_not_hold_is_refused(fitted, tmp_path):
    fitted.my_extra_result_ = np.ones(3)

    with pytest.raises(ValidationError, match="my_extra_result_"):
        fitted.save_model(tmp_path / "model.nld")


def _perturbed_random_walk(monkeypatch, factor=1.0 - 1e-4):
    original = continuous_state_transitions.RandomWalk.make_state_transition

    def changed(self, environments, context=None):
        block = original(self, environments, context)
        return block * factor

    monkeypatch.setattr(
        continuous_state_transitions.RandomWalk, "make_state_transition", changed
    )


def _perturbed_bin_centers(monkeypatch):
    original = environment.get_centers
    monkeypatch.setattr(environment, "get_centers", lambda e: original(e) + 1e-3)


@pytest.mark.parametrize(
    ("perturb", "structure"),
    [
        (_perturbed_random_walk, "continuous transition from state 2 to 2"),
        (_perturbed_bin_centers, "environment ''"),
    ],
    ids=["transition", "environment"],
)
def test_a_changed_rebuild_refuses_to_decode_but_still_loads(
    perturb, structure, saved, sorted_recording, monkeypatch
):
    time, position, spike_times = sorted_recording
    perturb(monkeypatch)

    loaded = nld.NonLocalSortedSpikesDetector.load_model(saved)
    summary = read_model_summary(saved)

    assert summary["encoding_models"][0]["place_fields"].shape[0] == 3
    with pytest.raises(ValidationError, match="refit") as error:
        decode(loaded, time, position, {"spike_times": spike_times})
    assert structure in str(error.value)
    with pytest.raises(ValidationError, match="differ"):
        loaded.save_model(saved.with_name("again.nld"))


def test_a_refit_replaces_the_rebuilt_structures(saved, sorted_recording, monkeypatch):
    time, position, spike_times = sorted_recording
    _perturbed_random_walk(monkeypatch)
    loaded = nld.NonLocalSortedSpikesDetector.load_model(saved)

    loaded.fit(time, position, spike_times)

    decode(loaded, time, position, {"spike_times": spike_times})


def test_an_unchanged_rebuild_is_checked_once(saved, sorted_recording):
    time, position, spike_times = sorted_recording
    loaded = nld.NonLocalSortedSpikesDetector.load_model(saved)
    check = loaded._rebuild_check_

    decode(loaded, time, position, {"spike_times": spike_times})

    assert check.verified
    assert loaded._rebuild_check_ is check


def test_drift_in_any_transition_row_refuses_to_decode(
    saved, sorted_recording, monkeypatch
):
    """Every source and destination bin of a block is fingerprinted: moving
    the random walk of a single bin is refused."""
    time, position, spike_times = sorted_recording
    original = continuous_state_transitions._euclidean_random_walk

    for row in range(2):

        def drifted(environment, mean, var, row=row):
            block = original(environment, mean, var).copy()
            block[row] = np.roll(block[row], 5)
            return block

        monkeypatch.setattr(
            continuous_state_transitions, "_euclidean_random_walk", drifted
        )
        loaded = nld.NonLocalSortedSpikesDetector.load_model(saved)

        with pytest.raises(ValidationError, match="continuous transition"):
            decode(loaded, time, position, {"spike_times": spike_times})


def _with_extra_member(path, name, data, *, compress=zipfile.ZIP_DEFLATED):
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr(zipfile.ZipInfo(name), data, compress_type=compress)


@pytest.mark.parametrize(
    "name", ["junk.bin", "arrays/../escape.npy", "/absolute.npy", "arrays\\x.npy"]
)
def test_a_member_the_manifest_does_not_list_is_refused(saved, name):
    _with_extra_member(saved, name, b"\0" * 1000)

    for read in (nld.NonLocalSortedSpikesDetector.load_model, read_model_summary):
        with pytest.raises(DataError, match="member"):
            read(saved)


def test_a_duplicate_member_is_refused(saved):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # zipfile: duplicate name
        _with_extra_member(saved, "state.json", b"{}")

    with pytest.raises(DataError, match="more than once"):
        nld.NonLocalSortedSpikesDetector.load_model(saved)


def test_a_member_larger_than_its_array_is_refused_before_decompressing(saved):
    with zipfile.ZipFile(saved) as archive:
        member = next(n for n in archive.namelist() if n.endswith("place_fields.npy"))

    _rewrite(saved, member, lambda data: data + b"\0" * 50_000_000)

    with pytest.raises(DataError, match="larger than"):
        nld.NonLocalSortedSpikesDetector.load_model(saved)


def test_a_corrupt_member_raises_data_error(saved):
    data = bytearray(saved.read_bytes())
    with zipfile.ZipFile(saved) as archive:
        info = archive.getinfo("state.json")
    offset = info.header_offset + 30 + len(info.filename) + len(info.extra) + 20
    data[offset] ^= 1
    saved.write_bytes(bytes(data))

    with pytest.raises(DataError):
        nld.NonLocalSortedSpikesDetector.load_model(saved)


def test_a_file_that_is_not_a_zip_raises_data_error(tmp_path):
    path = tmp_path / "model.nld"
    path.write_bytes(b"PK\x03\x04 not really a zip")

    with pytest.raises(DataError):
        nld.NonLocalSortedSpikesDetector.load_model(path)


def test_a_newer_format_version_without_a_writer_version_says_so(saved):
    def newer(manifest):
        manifest["format_version"] = FORMAT_VERSION + 1
        del manifest["package_version"]

    _edit_json(saved, "manifest.json", newer)

    with pytest.raises(ValidationError) as error:
        nld.NonLocalSortedSpikesDetector.load_model(saved)
    message = str(error.value)
    assert "an unknown, newer non_local_detector" in message
    assert ">= " not in message


def test_a_float32_rounding_level_rebuild_change_still_decodes(
    saved, sorted_recording, monkeypatch
):
    """Rebuilt structures match within a relative 1e-6, so a rounding-level
    change to the transition code (here 1e-7) does not lock models out."""
    time, position, spike_times = sorted_recording
    _perturbed_random_walk(monkeypatch, factor=1.0 + 1e-7)

    loaded = nld.NonLocalSortedSpikesDetector.load_model(saved)

    decode(loaded, time, position, {"spike_times": spike_times})


def test_an_all_zero_rebuilt_block_is_refused(saved, sorted_recording, monkeypatch):
    time, position, spike_times = sorted_recording
    _perturbed_random_walk(monkeypatch, factor=0.0)

    loaded = nld.NonLocalSortedSpikesDetector.load_model(saved)

    with pytest.raises(
        ValidationError, match="continuous transition from state 2 to 2"
    ):
        decode(loaded, time, position, {"spike_times": spike_times})


def test_stored_covariates_without_a_covariate_transition_are_refused(
    sorted_recording, tmp_path
):
    """Covariates need a transition that builds a design from them."""
    time, position, spike_times = sorted_recording
    detector = nld.NonLocalSortedSpikesDetector(
        discrete_transition_type=nld.DiscreteNonStationaryDiagonal(
            diagonal_values=np.full(4, 0.98), formula="1 + speed"
        ),
        sampling_frequency=SORTED_RATE,
        sorted_spikes_algorithm_params=algorithm_params("sorted_spikes_kde"),
    ).fit(
        time,
        position,
        spike_times,
        discrete_transition_covariate_data={"speed": np.linspace(0.0, 5.0, len(time))},
    )
    path = tmp_path / "model.nld"
    detector.save_model(path)

    def make_stationary(config):
        node = config["discrete_transition_type"]
        node["$type"] = "nld.DiscreteStationaryDiagonal"
        del node["state"]["formula"], node["state"]["smoothness"]

    _edit_json(path, "config.json", make_stationary)

    with pytest.raises(DataError, match="discrete_transition_type"):
        nld.NonLocalSortedSpikesDetector.load_model(path)
