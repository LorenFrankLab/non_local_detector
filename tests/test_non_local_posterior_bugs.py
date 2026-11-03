"""Tests for get_conditional_non_local_posterior and position_time validation.

- The non-local states are the ones whose observation model is neither Local
  nor No-Spike, so custom names, reordered states and extra non-local states
  all work.
- Marginalization covers every position level, so 2D environments
  (``x_position``/``y_position``) work.
- ``predict()`` raises a clear ValidationError when ``position`` is given
  without ``position_time``.
"""

import numpy as np
import pytest

from non_local_detector.continuous_state_transitions import (
    Discrete,
    RandomWalk,
    Uniform,
)
from non_local_detector.discrete_state_transitions import (
    DiscreteStationaryCustom,
    DiscreteStationaryDiagonal,
)
from non_local_detector.exceptions import ValidationError
from non_local_detector.initial_conditions import UniformInitialConditions
from non_local_detector.models._defaults import _ModelDefaults
from non_local_detector.models.non_local_model import (
    NonLocalClusterlessDetector,
    NonLocalSortedSpikesDetector,
)
from non_local_detector.observation_models import ObservationModel
from non_local_detector.time_edges import time_edges_from_centers
from tests.conftest import make_state_bins_results

pytestmark = pytest.mark.unit

DEFAULT_NAMES = [
    "Local",
    "No-Spike",
    "Non-Local Continuous",
    "Non-Local Fragmented",
]
DETECTOR_CLASSES = [NonLocalSortedSpikesDetector, NonLocalClusterlessDetector]


def _make_results_dataset_1d(state_names, n_time=10, n_bins_per_state=5):
    """Minimal 1D results xr.Dataset mimicking predict() output."""
    return make_state_bins_results(
        state_names,
        {"position": np.arange(n_bins_per_state, dtype=float)},
        n_time=n_time,
    )


def _make_results_dataset_2d(state_names, n_time=10, n_bins_per_state=4):
    """Minimal 2D results xr.Dataset mimicking predict() output."""
    return make_state_bins_results(
        state_names,
        {
            "x_position": np.arange(n_bins_per_state, dtype=float),
            "y_position": np.arange(n_bins_per_state, dtype=float) * 10,
        },
        n_time=n_time,
    )


def _permuted_non_local_detector(detector_cls, order):
    """Default non-local detector with its states listed in ``order``.

    Parameters
    ----------
    detector_cls : type
    order : sequence of int
        ``order[k]`` is the default state index placed at position ``k``.

    Returns
    -------
    detector : NonLocalSortedSpikesDetector or NonLocalClusterlessDetector
    """
    defaults = {
        name: factory() for name, factory in _ModelDefaults.non_local_defaults().items()
    }
    order = list(order)
    transitions = defaults["discrete_transition_type"].values
    continuous = defaults["continuous_transition_types"]
    return detector_cls(
        discrete_initial_conditions=defaults["discrete_initial_conditions"][order],
        continuous_initial_conditions_types=[
            defaults["continuous_initial_conditions_types"][i] for i in order
        ],
        discrete_transition_type=DiscreteStationaryCustom(
            values=transitions[np.ix_(order, order)]
        ),
        discrete_transition_stickiness=defaults["discrete_transition_stickiness"][
            order
        ],
        continuous_transition_types=[[continuous[i][j] for j in order] for i in order],
        observation_models=[defaults["observation_models"][i] for i in order],
        state_names=[DEFAULT_NAMES[i] for i in order],
    )


def _five_state_detector(detector_cls):
    """Non-local detector with a third non-local state ("Hover") appended."""
    is_spatial = [False, False, True, True, True]
    return detector_cls(
        discrete_initial_conditions=np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        continuous_initial_conditions_types=[
            UniformInitialConditions() for _ in range(5)
        ],
        discrete_transition_type=DiscreteStationaryDiagonal(
            diagonal_values=np.array([0.99, 0.98, 0.96, 0.96, 0.96])
        ),
        discrete_transition_stickiness=np.zeros(5),
        continuous_transition_types=[
            [
                (RandomWalk() if i == j == 2 else Uniform())
                if is_spatial[j]
                else Discrete()
                for j in range(5)
            ]
            for i in range(5)
        ],
        observation_models=[
            ObservationModel(is_local=True),
            ObservationModel(is_no_spike=True),
            ObservationModel(),
            ObservationModel(),
            ObservationModel(),
        ],
        state_names=[*DEFAULT_NAMES, "Hover"],
    )


def _expected_conditional(results, non_local_names):
    """Hand-compute the summed posterior of ``non_local_names``, normalized.

    Parameters
    ----------
    results : xr.Dataset
        Output of ``make_state_bins_results``: equally sized state blocks
        along ``state_bins``, in ``state`` coordinate order.
    non_local_names : sequence of str

    Returns
    -------
    expected : np.ndarray, shape (n_time, n_bins_per_state)
    """
    state_names = list(dict.fromkeys(results.acausal_posterior.state.values))
    posterior = results.acausal_posterior.values
    per_state = posterior.reshape(posterior.shape[0], len(state_names), -1)
    non_local = sum(per_state[:, state_names.index(name)] for name in non_local_names)
    return non_local / non_local.sum(axis=1, keepdims=True)


def _assert_matches_expected(posterior, results, non_local_names):
    """Assert the conditional posterior equals the hand-computed one."""
    assert posterior.dims[0] == "time"
    values = posterior.values.reshape(posterior.shape[0], -1)
    np.testing.assert_allclose(
        values, _expected_conditional(results, non_local_names), rtol=1e-12
    )


# ============================================================
# Which states are non-local
# ============================================================


class TestNonLocalStateSelection:
    """The non-local states are found from the observation models."""

    @pytest.mark.parametrize("detector_cls", DETECTOR_CLASSES)
    def test_default_states(self, detector_cls):
        results = _make_results_dataset_1d(DEFAULT_NAMES)
        posterior = detector_cls().get_conditional_non_local_posterior(results)
        _assert_matches_expected(posterior, results, DEFAULT_NAMES[2:])

    @pytest.mark.parametrize("detector_cls", DETECTOR_CLASSES)
    def test_custom_names(self, detector_cls):
        names = ["Here", "Quiet", "Replay-Smooth", "Replay-Jump"]
        results = _make_results_dataset_1d(names)
        posterior = detector_cls(state_names=names).get_conditional_non_local_posterior(
            results
        )
        _assert_matches_expected(posterior, results, names[2:])

    @pytest.mark.parametrize("detector_cls", DETECTOR_CLASSES)
    @pytest.mark.parametrize("order", [(2, 3, 0, 1), (3, 0, 2, 1), (1, 2, 0, 3)])
    def test_reordered_states(self, detector_cls, order):
        """States listed in another order: selection follows the flags."""
        detector = _permuted_non_local_detector(detector_cls, order)
        results = _make_results_dataset_1d(detector.state_names)
        posterior = detector.get_conditional_non_local_posterior(results)
        _assert_matches_expected(posterior, results, DEFAULT_NAMES[2:])

    @pytest.mark.parametrize("detector_cls", DETECTOR_CLASSES)
    def test_extra_non_local_state(self, detector_cls):
        """Every non-local state contributes, not only the third and fourth."""
        detector = _five_state_detector(detector_cls)
        results = _make_results_dataset_1d(detector.state_names)
        posterior = detector.get_conditional_non_local_posterior(results)
        _assert_matches_expected(
            posterior,
            results,
            ["Non-Local Continuous", "Non-Local Fragmented", "Hover"],
        )

    @pytest.mark.parametrize("detector_cls", DETECTOR_CLASSES)
    def test_class_level_call_raises_type_error(self, detector_cls):
        """The method needs the detector's observation models, so calling it on
        the class (the former staticmethod form) raises TypeError."""
        results = _make_results_dataset_1d(DEFAULT_NAMES)
        with pytest.raises(TypeError):
            detector_cls.get_conditional_non_local_posterior(results)

    def test_no_non_local_states_raises(self):
        detector = NonLocalSortedSpikesDetector(
            observation_models=[
                ObservationModel(is_local=True),
                ObservationModel(is_no_spike=True),
                ObservationModel(is_local=True),
                ObservationModel(is_no_spike=True),
            ]
        )
        results = _make_results_dataset_1d(DEFAULT_NAMES)
        with pytest.raises(ValidationError, match="no non-local states"):
            detector.get_conditional_non_local_posterior(results)

    def test_results_from_another_model_raise(self):
        """Results lacking the detector's non-local states are rejected."""
        results = _make_results_dataset_1d(["A", "B", "C", "D"])
        with pytest.raises(ValidationError, match="Non-Local Continuous"):
            NonLocalClusterlessDetector().get_conditional_non_local_posterior(results)


# ============================================================
# 2D position
# ============================================================


class TestTwoDimensionalPosition:
    """The conditional posterior marginalizes over x/y position levels in 2D."""

    @pytest.mark.parametrize("detector_cls", DETECTOR_CLASSES)
    def test_2d(self, detector_cls):
        results = _make_results_dataset_2d(DEFAULT_NAMES)
        posterior = detector_cls().get_conditional_non_local_posterior(results)
        _assert_matches_expected(posterior, results, DEFAULT_NAMES[2:])

    def test_2d_reordered(self):
        """Reordered states and a 2D environment together."""
        detector = _permuted_non_local_detector(
            NonLocalClusterlessDetector, (3, 1, 2, 0)
        )
        results = _make_results_dataset_2d(detector.state_names)
        posterior = detector.get_conditional_non_local_posterior(results)
        _assert_matches_expected(posterior, results, DEFAULT_NAMES[2:])


# ============================================================
# position_time validation
# ============================================================


class TestPositionTimeValidation:
    """predict() should raise a clear error when position is given without position_time."""

    def test_clusterless_position_without_position_time(self):
        """Providing position but omitting position_time should raise ValidationError.

        Validation fires at the start of predict(), before fit() is needed.
        """
        detector = NonLocalClusterlessDetector()
        time = np.linspace(0, 1, 100)
        position = np.linspace(0, 10, 100)
        spike_times = [np.array([0.1, 0.5])]
        spike_features = [np.array([[1.0], [2.0]])]

        with pytest.raises(ValidationError, match="position_time is required"):
            detector.predict(
                spike_times=spike_times,
                spike_waveform_features=spike_features,
                time_edges=time_edges_from_centers(time),
                position=position,
                position_time=None,
            )

    def test_sorted_spikes_position_without_position_time(self):
        """Providing position but omitting position_time should raise ValidationError.

        Validation fires at the start of predict(), before fit() is needed.
        """
        detector = NonLocalSortedSpikesDetector()
        time = np.linspace(0, 1, 100)
        position = np.linspace(0, 10, 100)
        spike_times = [np.array([0.1, 0.5])]

        with pytest.raises(ValidationError, match="position_time is required"):
            detector.predict(
                spike_times=spike_times,
                time_edges=time_edges_from_centers(time),
                position=position,
                position_time=None,
            )
