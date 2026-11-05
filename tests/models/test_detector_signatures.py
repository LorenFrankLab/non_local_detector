"""Public method signatures of the sorted-spikes and clusterless detectors."""

import inspect
import typing

import numpy as np
import pytest

from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.models.base import ClusterlessDetector, SortedSpikesDetector

pytestmark = pytest.mark.unit

# The data arrays each family takes positionally, in order.
_DATA_ARGUMENTS = {
    SortedSpikesDetector: ("position_time", "position", "spike_times"),
    ClusterlessDetector: (
        "position_time",
        "position",
        "spike_times",
        "spike_waveform_features",
    ),
}


@pytest.mark.parametrize("cls", list(_DATA_ARGUMENTS), ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("method", ["fit", "fit_encoding_model"])
def test_fit_arguments_after_the_data_are_keyword_only(cls, method):
    """Only the data arrays bind by position, so a training mask passed
    positionally cannot land in another family's waveform-feature slot."""
    parameters = list(inspect.signature(getattr(cls, method)).parameters.values())[1:]
    positional = [
        parameter.name
        for parameter in parameters
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert tuple(positional) == _DATA_ARGUMENTS[cls]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters[len(positional) :]
    )


def test_positional_training_mask_raises_type_error():
    """A training mask after the spike times must be passed by name."""
    time = np.linspace(0, 1, 11)
    detector = NonLocalSortedSpikesDetector()
    with pytest.raises(TypeError, match="positional"):
        detector.fit(time, time, [time], np.ones(time.size, dtype=bool))


@pytest.mark.parametrize("cls", list(_DATA_ARGUMENTS), ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("method", ["fit", "set_params", "copy"])
def test_methods_returning_self_keep_the_callers_type(cls, method):
    """``NonLocalSortedSpikesDetector().fit(...)`` is typed as that class,
    not as the family base class."""
    hints = typing.get_type_hints(getattr(cls, method))
    assert hints["return"] is hints["self"]


def _public_methods(cls):
    return {
        name
        for name, member in inspect.getmembers(cls)
        if not name.startswith("_") and inspect.isroutine(member)
    }


_PUBLIC_METHODS = sorted(
    _public_methods(SortedSpikesDetector) | _public_methods(ClusterlessDetector)
)


@pytest.mark.parametrize("method", _PUBLIC_METHODS)
def test_family_methods_have_the_same_signature(method):
    """Apart from the waveform features, a clusterless method takes the same
    arguments, in the same order and with the same defaults, as its
    sorted-spikes counterpart."""
    sorted_signature = inspect.signature(getattr(SortedSpikesDetector, method))
    clusterless_signature = inspect.signature(getattr(ClusterlessDetector, method))
    clusterless_parameters = [
        parameter
        for parameter in clusterless_signature.parameters.values()
        if parameter.name != "spike_waveform_features"
    ]
    assert [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in sorted_signature.parameters.values()
    ] == [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in clusterless_parameters
    ]
