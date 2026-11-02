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


def _documented_parameters(cls):
    """Names of the entries in the class docstring's Parameters section."""
    doc = inspect.getdoc(cls) or ""
    section = doc.split("Parameters\n----------\n", 1)[1].split("\nExamples\n", 1)[0]
    lines = section.splitlines()
    return [line.split(" :")[0] for line in lines if line and not line[0].isspace()]


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
