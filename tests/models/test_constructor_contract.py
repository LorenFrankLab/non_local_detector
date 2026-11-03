"""Constructor contract of the detectors: keyword-only arguments and ``set_params``."""

import json
import re
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone

import non_local_detector.models as models
from non_local_detector.continuous_state_transitions import Discrete, Uniform
from non_local_detector.discrete_state_transitions import DiscreteStationaryDiagonal
from non_local_detector.environment import Environment
from non_local_detector.exceptions import ValidationError
from non_local_detector.initial_conditions import UniformInitialConditions
from non_local_detector.models.base import SortedSpikesDetector
from non_local_detector.models.non_local_model import NonLocalSortedSpikesDetector

REFERENCE = Path(__file__).parent / "data" / "default_get_params_reference.json"

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


def _simulated_data():
    time = np.arange(0.0, 2.0, 0.002)
    position = 50.0 + 40.0 * np.sin(2 * np.pi * time / 2.0)
    rng = np.random.default_rng(0)
    spike_times = [np.sort(rng.uniform(0.0, 2.0, 100)) for _ in range(3)]
    return time, position, spike_times


@pytest.mark.unit
class TestSetParams:
    def test_single_environment_is_normalized_and_fits(self):
        time, position, spike_times = _simulated_data()
        detector = NonLocalSortedSpikesDetector().set_params(
            environments=Environment(place_bin_size=5.0)
        )
        assert isinstance(detector.environments, tuple)
        detector.fit(time, position, spike_times)
        assert detector.environments[0].place_bin_size == 5.0

    def test_environment_lookup_cache_is_invalidated(self):
        detector = NonLocalSortedSpikesDetector()
        old = detector._get_environment_by_name("")
        new_env = Environment(environment_name="", place_bin_size=5.0)
        detector.set_params(environments=new_env)
        assert detector._get_environment_by_name("") is new_env
        assert detector._get_environment_by_name("") is not old

    @pytest.mark.parametrize("sampling_frequency", [-1, 0.0, float("nan")])
    def test_invalid_value_raises_and_leaves_params(self, sampling_frequency):
        detector = NonLocalSortedSpikesDetector()
        with pytest.raises(ValidationError):
            detector.set_params(sampling_frequency=sampling_frequency)
        assert detector.sampling_frequency == 500.0

    def test_state_names_none_resolves_like_constructor(self):
        detector = _base_detector(state_names=["a", "b", "c"])
        detector.set_params(state_names=None)
        assert detector.state_names == _base_detector().state_names
        assert detector.state_names == ["state 0", "state 1", "state 2"]

    def test_none_state_names_on_thin_model_gives_its_default_names(self):
        detector = NonLocalSortedSpikesDetector(state_names=["a", "b", "c", "d"])
        detector.set_params(state_names=None)
        assert detector.state_names == NonLocalSortedSpikesDetector().state_names

    def test_none_resolves_thin_model_default_like_constructor(self):
        custom = [[Discrete()] * 4] * 4
        detector = NonLocalSortedSpikesDetector(continuous_transition_types=custom)
        detector.set_params(continuous_transition_types=None)
        default = NonLocalSortedSpikesDetector().continuous_transition_types
        assert _stable_repr(detector.continuous_transition_types) == _stable_repr(
            default
        )

    def test_mixed_nested_error_leaves_detector_unchanged(self):
        detector = NonLocalSortedSpikesDetector()
        before = detector.get_params(deep=False)
        with pytest.raises(ValueError, match="Invalid parameter"):
            detector.set_params(sampling_frequency=250.0, foo__bar=1)
        assert detector.sampling_frequency == 500.0
        for name, value in before.items():
            assert getattr(detector, name) is value

    def test_changing_number_of_states_alone_raises_and_leaves_detector(self):
        detector = NonLocalSortedSpikesDetector()
        before = detector.get_params(deep=False)
        with pytest.raises(ValidationError):
            detector.set_params(discrete_initial_conditions=np.array([1.0, 0.0]))
        for name, value in before.items():
            assert getattr(detector, name) is value

    def test_unknown_parameter_raises(self):
        with pytest.raises(ValueError, match="Invalid parameter"):
            NonLocalSortedSpikesDetector().set_params(not_a_parameter=1)

    def test_fitted_attributes_are_untouched(self):
        time, position, spike_times = _simulated_data()
        detector = NonLocalSortedSpikesDetector()
        detector.fit(time, position, spike_times)
        fitted = detector.encoding_model_
        detector.set_params(no_spike_rate=1e-8)
        assert detector.no_spike_rate == 1e-8
        assert detector.encoding_model_ is fitted

    def test_returns_self_and_no_params_is_noop(self):
        detector = NonLocalSortedSpikesDetector()
        assert detector.set_params() is detector
        assert detector.set_params(no_spike_rate=1e-8) is detector

    @pytest.mark.parametrize("model_cls", ALL_MODELS, ids=lambda cls: cls.__name__)
    def test_clone_and_get_params_round_trip(self, model_cls):
        model = model_cls()
        cloned = clone(model)
        assert type(cloned) is model_cls
        original = model.get_params(deep=False)
        assert cloned.get_params(deep=False).keys() == original.keys()
        for name, value in cloned.get_params(deep=False).items():
            assert _stable_repr(value) == _stable_repr(original[name]), name
        rebuilt = model_cls(**original)
        assert rebuilt.get_params(deep=False).keys() == original.keys()
        model.set_params(**original)
        assert model.get_params(deep=False).keys() == original.keys()


def _base_detector(n_states=3, **kwargs):
    return SortedSpikesDetector(
        discrete_initial_conditions=np.eye(n_states)[0],
        continuous_initial_conditions_types=[UniformInitialConditions()] * n_states,
        discrete_transition_type=DiscreteStationaryDiagonal(
            diagonal_values=np.full(n_states, 0.9)
        ),
        discrete_transition_concentration=1.0,
        discrete_transition_stickiness=np.zeros(n_states),
        discrete_transition_regularization=0.1,
        continuous_transition_types=[[Uniform()] * n_states] * n_states,
        observation_models=None,
        environments=Environment(),
        **kwargs,
    )


def _stable_repr(value):
    return re.sub(r"0x[0-9a-f]+", "0x", repr(value))


def _config_repr(value):
    """``_stable_repr``, with each ``Environment`` shown by its ``get_params()``.

    Spyglass stores an environment's configuration, so fitted attributes,
    which are None on a default-constructed model, stay out of the reference.
    """
    if isinstance(value, Environment):
        config = ", ".join(
            f"{name}={param!r}" for name, param in value.get_params().items()
        )
        return re.sub(r"0x[0-9a-f]+", "0x", f"{type(value).__name__}({config})")
    if isinstance(value, (list, tuple)) and any(
        isinstance(item, Environment) for item in value
    ):
        items = [_config_repr(item) for item in value]
        if isinstance(value, list):
            return f"[{', '.join(items)}]"
        return f"({', '.join(items)}{',' if len(items) == 1 else ''})"
    return _stable_repr(value)


@pytest.mark.unit
@pytest.mark.parametrize("model_cls", ALL_MODELS, ids=lambda cls: cls.__name__)
def test_default_get_params_matches_reference(model_cls):
    """Spyglass stores ``get_params(deep=False)``: keep keys and resolved values.

    The reference was generated from the default-constructed models before
    constructor arguments became keyword-only and ``set_params`` was validated.
    Environments are compared by their configuration (``get_params()``).
    """
    reference = json.loads(REFERENCE.read_text())[model_cls.__name__]
    params = model_cls().get_params(deep=False)
    assert set(params) == set(reference)
    for name, value in params.items():
        assert _config_repr(value) == reference[name], name
