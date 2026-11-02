"""Every model class documents each constructor parameter, for its data type."""

import inspect

import pytest

from non_local_detector import models

pytestmark = pytest.mark.unit

MODEL_CLASSES = [
    models.SortedSpikesDecoder,
    models.ClusterlessDecoder,
    models.ContFragSortedSpikesClassifier,
    models.ContFragClusterlessClassifier,
    models.MultiEnvironmentSortedSpikesClassifier,
    models.MultiEnvironmentClusterlessClassifier,
    models.NoSpikeContFragSortedSpikesClassifier,
    models.NoSpikeContFragClusterlessClassifier,
    models.NonLocalSortedSpikesDetector,
    models.NonLocalClusterlessDetector,
]


def _parameter_entries(cls):
    """Entries of the class docstring's Parameters section, by name.

    Returns
    -------
    entries : dict[str, str]
        Each documented parameter's name and its entry (type line and
        description), in order.
    """
    doc = inspect.getdoc(cls) or ""
    assert "Parameters\n----------\n" in doc, f"{cls.__name__} has no Parameters"
    section = doc.split("Parameters\n----------\n", 1)[1].split("\nExamples\n", 1)[0]
    entries: dict[str, str] = {}
    name = None
    for line in section.splitlines():
        if line and not line[0].isspace():
            name = line.split(" :")[0]
            entries[name] = line
        elif name is not None:
            entries[name] += "\n" + line
    return entries


def _documented_parameters(cls):
    """Names of the entries in the class docstring's Parameters section."""
    return list(_parameter_entries(cls))


@pytest.mark.parametrize("cls", MODEL_CLASSES, ids=lambda cls: cls.__name__)
def test_parameters_section_matches_signature(cls):
    """The Parameters section lists the constructor's parameters in order."""
    signature = list(inspect.signature(cls.__init__).parameters)
    assert _documented_parameters(cls) == signature[1:]


@pytest.mark.parametrize("cls", MODEL_CLASSES, ids=lambda cls: cls.__name__)
def test_docstring_names_its_own_class_and_data(cls):
    """The examples construct this class and the text names its data type."""
    doc = inspect.getdoc(cls) or ""
    data = "clusterless" if "Clusterless" in cls.__name__ else "sorted spike"
    assert f"{cls.__name__}()" in doc
    assert data in doc.splitlines()[0]
    assert "$" not in doc


def _documented_default(default):
    """How a scalar default is written in a docstring entry."""
    return f'"{default}"' if isinstance(default, str) else repr(default)


@pytest.mark.parametrize("cls", MODEL_CLASSES, ids=lambda cls: cls.__name__)
def test_documented_scalar_defaults_match_signature(cls):
    """Each parameter with a scalar default states that default, so changing
    a default without its documentation fails here."""
    entries = _parameter_entries(cls)
    for name, parameter in inspect.signature(cls.__init__).parameters.items():
        default = parameter.default
        if isinstance(default, (bool, int, float, str)):
            assert _documented_default(default) in entries[name], (
                f"{cls.__name__}.{name}: default {default!r} not documented"
            )
