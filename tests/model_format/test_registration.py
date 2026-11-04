"""Third-party types round-trip once registered with ``register_type``.

A custom continuous transition (a dataclass, and a plain class with its own
``to_state``/``from_state``) and a detector subclass save and reload once
registered, and fail with the documented errors when not. The package's own
types store exactly the fields listed in ``model_format._builtins``.
"""

import dataclasses
import json
import zipfile
from dataclasses import dataclass

import numpy as np
import pytest

import non_local_detector as nld
from non_local_detector.exceptions import DataError, ValidationError
from non_local_detector.model_format import _registry, register_type, registered_types
from non_local_detector.model_format._builtins import builtin_types
from tests.model_format._compare import assert_same_fitted_state
from tests.model_format.conftest import algorithm_params, decode

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _restore_registry(monkeypatch):
    """Undo each test's registrations."""
    _registry._register_builtins()
    monkeypatch.setattr(_registry, "_BY_NAME", dict(_registry._BY_NAME))
    monkeypatch.setattr(_registry, "_BY_CLASS", dict(_registry._BY_CLASS))


@dataclass
class FlatTransition:
    """A custom transition: uniform over the environment's bins."""

    environment_name: str = ""
    weight: float = 1.0

    def make_state_transition(self, environments, context=None):
        n_bins = environments[0].place_bin_centers_.shape[0]
        return np.full((n_bins, n_bins), self.weight / n_bins)


class PlainFlatTransition:
    """A custom transition that is not a dataclass."""

    def __init__(self, weight):
        self.environment_name = ""
        self.weight = weight

    def make_state_transition(self, environments, context=None):
        n_bins = environments[0].place_bin_centers_.shape[0]
        return np.full((n_bins, n_bins), self.weight / n_bins)


class LabDecoder(nld.SortedSpikesDecoder):
    """A detector subclass."""


def _fit(cls, transition, sorted_recording):
    time, position, spike_times = sorted_recording
    detector = cls(
        continuous_transition_types=[[transition]],
        sorted_spikes_algorithm_params=algorithm_params("sorted_spikes_kde"),
    )
    return detector.fit(time, position, spike_times)


def _assert_round_trip(detector, path, sorted_recording):
    time, position, spike_times = sorted_recording
    detector.save_model(path)
    loaded = type(detector).load_model(path)
    assert_same_fitted_state(detector, loaded)
    for a, b in zip(
        decode(loaded, time, position, {"spike_times": spike_times}),
        decode(detector, time, position, {"spike_times": spike_times}),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)
    return loaded


def test_registered_dataclass_transition_round_trips(sorted_recording, tmp_path):
    detector = _fit(
        nld.SortedSpikesDecoder, FlatTransition(weight=0.5), sorted_recording
    )
    path = tmp_path / "model.nld"
    with pytest.raises(ValidationError, match="register_type"):
        detector.save_model(path)

    register_type(FlatTransition, "testlab.FlatTransition")
    loaded = _assert_round_trip(detector, path, sorted_recording)

    assert loaded.continuous_transition_types[0][0] == FlatTransition(weight=0.5)
    assert registered_types()["testlab.FlatTransition"] is FlatTransition


def test_registered_plain_class_round_trips_through_its_functions(
    sorted_recording, tmp_path
):
    detector = _fit(nld.SortedSpikesDecoder, PlainFlatTransition(0.5), sorted_recording)

    register_type(
        PlainFlatTransition,
        "testlab.PlainFlatTransition",
        to_state=lambda transition: {"weight": transition.weight},
        from_state=lambda state: PlainFlatTransition(state["weight"]),
    )
    loaded = _assert_round_trip(detector, tmp_path / "model.nld", sorted_recording)

    assert loaded.continuous_transition_types[0][0].weight == 0.5


def test_registered_detector_subclass_round_trips(sorted_recording, tmp_path):
    detector = _fit(LabDecoder, nld.RandomWalk(), sorted_recording)
    path = tmp_path / "model.nld"
    with pytest.raises(ValidationError, match="LabDecoder"):
        detector.save_model(path)

    register_type(LabDecoder, "testlab.LabDecoder")
    loaded = _assert_round_trip(detector, path, sorted_recording)

    assert type(loaded) is LabDecoder


def test_loading_without_the_registration_names_its_owner(sorted_recording, tmp_path):
    register_type(FlatTransition, "testlab.FlatTransition", owner="labtools")
    path = tmp_path / "model.nld"
    _fit(nld.SortedSpikesDecoder, FlatTransition(), sorted_recording).save_model(path)
    del _registry._BY_NAME["testlab.FlatTransition"]
    del _registry._BY_CLASS[FlatTransition]

    with pytest.raises(ValidationError, match=r"'testlab\.FlatTransition'") as error:
        nld.SortedSpikesDecoder.load_model(path)
    # The package that registered it, recorded in the file, not the prefix.
    assert "('labtools')" in str(error.value)


@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("FlatTransition", "prefixed"),
        ("nld.FlatTransition", "reserved"),
        ("nld_extra FlatTransition", "prefixed"),
    ],
)
def test_register_type_refuses_unusable_names(name, match):
    with pytest.raises(ValidationError, match=match):
        register_type(FlatTransition, name)


def test_register_type_refuses_a_second_class_for_one_name():
    register_type(FlatTransition, "testlab.Flat")

    with pytest.raises(ValidationError, match="already registered"):
        register_type(
            PlainFlatTransition, "testlab.Flat", to_state=vars, from_state=dict
        )
    with pytest.raises(ValidationError, match="already registered as"):
        register_type(FlatTransition, "testlab.OtherName")
    register_type(FlatTransition, "testlab.Flat")  # the same registration again


def test_register_type_needs_functions_for_a_plain_class():
    with pytest.raises(ValidationError, match="not a dataclass"):
        register_type(PlainFlatTransition, "testlab.Plain")
    with pytest.raises(ValidationError, match="both"):
        register_type(PlainFlatTransition, "testlab.Plain", to_state=vars)
    with pytest.raises(ValidationError, match="detector"):
        register_type(LabDecoder, "testlab.LabDecoder", to_state=vars, from_state=dict)


@pytest.mark.parametrize(
    ("cls", "fields"),
    [(cls, fields) for _, cls, fields, _ in builtin_types() if fields is not None],
    ids=lambda value: getattr(value, "__name__", ""),
)
def test_builtin_types_store_every_dataclass_field(cls, fields):
    """Changing a saved class's fields needs a format migration: the stored
    field names are the format."""
    assert set(fields) == {
        field.name
        for field in dataclasses.fields(cls)
        if field.init or field.name.endswith("_") or field.name == "design_fingerprint"
    }


def test_builtin_type_names_are_fixed():
    """The stored names are the format: renaming a class must not change
    them, and changing this list is a format change."""
    assert {name: cls.__name__ for name, cls, _, _ in builtin_types()} == {
        f"nld.{name}": name
        for name in (
            "Environment",
            "ObservationModel",
            "UniformInitialConditions",
            "RandomWalk",
            "Uniform",
            "Identity",
            "EmpiricalMovement",
            "RandomWalkDirection1",
            "RandomWalkDirection2",
            "Discrete",
            "DiscreteStationaryDiagonal",
            "DiscreteStationaryCustom",
            "DiscreteNonStationaryDiagonal",
            "DiscreteNonStationaryCustom",
            "KDEModel",
            "GaussianMixtureModel",
            "MarkMixture",
            "MarkReference",
            "TrackGraphBasis",
            "TensorSplineBasis",
            "SortedSpikesDetector",
            "ClusterlessDetector",
            "NonLocalSortedSpikesDetector",
            "NonLocalClusterlessDetector",
            "ContFragSortedSpikesClassifier",
            "ContFragClusterlessClassifier",
            "NoSpikeContFragSortedSpikesClassifier",
            "NoSpikeContFragClusterlessClassifier",
            "MultiEnvironmentSortedSpikesClassifier",
            "MultiEnvironmentClusterlessClassifier",
            "SortedSpikesDecoder",
            "ClusterlessDecoder",
        )
    }


def test_every_ready_made_model_is_registered():
    from non_local_detector import models

    registered = set(registered_types().values())
    assert {getattr(models, name) for name in models.__all__} <= registered


def _without_field(path, field):
    """Drop ``field`` from the stored transition, as in a file written before
    the field existed."""
    with zipfile.ZipFile(path) as source:
        members = [(item, source.read(item.filename)) for item in source.infolist()]
    with zipfile.ZipFile(path, "w") as target:
        for item, data in members:
            if item.filename == "config.json":
                config = json.loads(data)
                del config["continuous_transition_types"][0][0]["state"][field]
                data = json.dumps(config).encode()
            target.writestr(item, data)


def test_an_optional_field_missing_from_a_file_takes_its_default(
    sorted_recording, tmp_path
):
    register_type(FlatTransition, "testlab.FlatTransition", defaults={"weight": 1.0})
    path = tmp_path / "model.nld"
    _fit(
        nld.SortedSpikesDecoder, FlatTransition(weight=0.5), sorted_recording
    ).save_model(path)
    _without_field(path, "weight")

    loaded = nld.SortedSpikesDecoder.load_model(path)

    assert loaded.continuous_transition_types[0][0] == FlatTransition(weight=1.0)


def test_a_missing_field_without_a_default_is_refused(sorted_recording, tmp_path):
    register_type(FlatTransition, "testlab.FlatTransition")
    path = tmp_path / "model.nld"
    _fit(nld.SortedSpikesDecoder, FlatTransition(), sorted_recording).save_model(path)
    _without_field(path, "weight")

    with pytest.raises(DataError, match="no field 'weight'"):
        nld.SortedSpikesDecoder.load_model(path)


def test_register_type_refuses_defaults_for_unknown_fields():
    with pytest.raises(ValidationError, match="'typo'"):
        register_type(FlatTransition, "testlab.FlatTransition", defaults={"typo": 3})
    with pytest.raises(ValidationError, match="defaults apply only"):
        register_type(
            PlainFlatTransition,
            "testlab.Plain",
            to_state=vars,
            from_state=dict,
            defaults={"weight": 1.0},
        )
