"""The sorted-spikes and clusterless class of each model share their defaults."""

import inspect

import pytest

from non_local_detector import models

pytestmark = pytest.mark.unit

MODEL_PAIRS = [
    (models.SortedSpikesDecoder, models.ClusterlessDecoder),
    (models.ContFragSortedSpikesClassifier, models.ContFragClusterlessClassifier),
    (
        models.MultiEnvironmentSortedSpikesClassifier,
        models.MultiEnvironmentClusterlessClassifier,
    ),
    (
        models.NoSpikeContFragSortedSpikesClassifier,
        models.NoSpikeContFragClusterlessClassifier,
    ),
    (models.NonLocalSortedSpikesDetector, models.NonLocalClusterlessDetector),
]


def _defaults(cls):
    """Constructor defaults, keyed by name, without the algorithm parameters.

    Returns
    -------
    defaults : dict[str, tuple[type, object]]
        Each default's type and value, so ``500`` and ``500.0`` differ.
    """
    return {
        name: (type(parameter.default), parameter.default)
        for name, parameter in inspect.signature(cls.__init__).parameters.items()
        if name != "self" and "algorithm" not in name
    }


@pytest.mark.parametrize(
    ("sorted_cls", "clusterless_cls"),
    MODEL_PAIRS,
    ids=[sorted_cls.__name__ for sorted_cls, _ in MODEL_PAIRS],
)
def test_pair_has_identical_defaults(sorted_cls, clusterless_cls):
    """Apart from the likelihood algorithm, both data types default alike."""
    assert _defaults(sorted_cls) == _defaults(clusterless_cls)
