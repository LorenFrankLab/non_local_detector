"""Public method signatures of the sorted-spikes and clusterless detectors."""

import inspect

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
