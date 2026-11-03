"""Constructor contract of the detectors: keyword-only arguments and ``set_params``."""

import pytest

import non_local_detector.models as models

ALL_MODELS = [getattr(models, name) for name in models.__all__]


@pytest.mark.unit
@pytest.mark.parametrize("model_cls", ALL_MODELS, ids=lambda cls: cls.__name__)
def test_positional_construction_raises(model_cls):
    with pytest.raises(TypeError, match="positional"):
        model_cls(None)


@pytest.mark.unit
def test_base_detectors_are_keyword_only():
    from non_local_detector.models.base import ClusterlessDetector, SortedSpikesDetector

    for cls in (ClusterlessDetector, SortedSpikesDetector):
        with pytest.raises(TypeError, match="positional"):
            cls(None, None, None, 1.0, None, 0.1, None, None, None)
