"""A broken extension fails at the cause, with an error naming what it broke.

Toy algorithms and transitions break one rule each of "Adding a likelihood
algorithm" and "Adding a transition" in CONTRIBUTING.md.
"""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector import (
    ContFragSortedSpikesClassifier,
    Environment,
    NonLocalSortedSpikesDetector,
    ObservationModel,
    time_edges_from_centers,
)
from non_local_detector.continuous_state_transitions import _make_transition_block
from non_local_detector.discrete_state_transitions import (
    InitialDiscreteTransition,
    _check_initial_discrete_transition,
)
from non_local_detector.exceptions import DataError, ValidationError
from non_local_detector.likelihoods import _SORTED_SPIKES_ALGORITHMS
from tests.models._helpers import SORTED_SIM_RATE, _assert_unchanged, _snapshot

pytestmark = pytest.mark.unit

_ALGORITHM = "sorted_spikes_kde"
_FIT_PARAMS = {"disable_progress_bar": True}


def _detector(**kwargs):
    return NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm=_ALGORITHM,
        sorted_spikes_algorithm_params=_FIT_PARAMS,
        environments=Environment(place_bin_size=10.0),
        sampling_frequency=SORTED_SIM_RATE,
        **kwargs,
    )


def _decode_kwargs(sorted_sim, n_time=40):
    time, position, spike_times = sorted_sim
    return {
        "spike_times": spike_times,
        "time_edges": time_edges_from_centers(time[:n_time]),
        "position": position,
        "position_time": time,
    }


def _poison(result, value, row=0, column=0):
    """``result`` with one entry replaced by ``value``."""
    return jnp.asarray(result).at[row, column].set(value)


def _patch_predict(monkeypatch, predict):
    record = _SORTED_SPIKES_ALGORITHMS[_ALGORITHM]

    def patched(*args, **kwargs):
        return predict(record.predict(*args, **kwargs), kwargs)

    monkeypatch.setitem(
        _SORTED_SPIKES_ALGORITHMS, _ALGORITHM, replace(record, predict=patched)
    )


@pytest.mark.parametrize(
    "broken",
    [
        pytest.param(lambda result: result[:, :1], id="one column for every bin"),
        pytest.param(lambda result: result[:1], id="one row for every time"),
        pytest.param(lambda result: result[:, 0], id="missing the bin axis"),
        pytest.param(lambda result: jnp.float32(0.0), id="a scalar"),
    ],
)
def test_likelihood_of_the_wrong_shape_names_the_algorithm(
    monkeypatch, sorted_sim, broken
):
    """A predict of the wrong shape is refused, not broadcast over the bins."""
    time, position, spike_times = sorted_sim
    detector = _detector().fit(time, position, spike_times)
    _patch_predict(
        monkeypatch, lambda result, k: result if k["is_local"] else broken(result)
    )

    with pytest.raises(ValidationError, match=_ALGORITHM) as error:
        detector.predict(**_decode_kwargs(sorted_sim))
    assert "(40, " in str(error.value)


def test_local_likelihood_of_the_wrong_shape_names_the_state(monkeypatch, sorted_sim):
    time, position, spike_times = sorted_sim
    detector = _detector().fit(time, position, spike_times)
    _patch_predict(
        monkeypatch,
        lambda result, k: jnp.tile(result, (1, 2)) if k["is_local"] else result,
    )

    with pytest.raises(ValidationError, match="Local"):
        detector.predict(**_decode_kwargs(sorted_sim))


@pytest.mark.parametrize("local_position_std", [None, 5.0])
@pytest.mark.parametrize("n_chunks", [1, 2])
def test_builtin_blocks_pass_the_shape_check(sorted_sim, local_position_std, n_chunks):
    """Local, multi-bin Local and No-Spike decode in one or several chunks."""
    time, position, spike_times = sorted_sim
    detector = _detector(local_position_std=local_position_std).fit(
        time, position, spike_times
    )
    results = detector.predict(**_decode_kwargs(sorted_sim), n_chunks=n_chunks)
    assert np.all(np.isfinite(results.acausal_state_probabilities.values))


@pytest.mark.parametrize("value", [np.nan, np.inf], ids=["nan", "posinf"])
@pytest.mark.parametrize(
    "decode",
    [
        pytest.param({}, id="dense"),
        pytest.param({"n_chunks": 2}, id="chunked"),
        pytest.param(
            {"inference_mode": "checkpointed", "output_mode": "compact"},
            id="checkpointed",
        ),
    ],
)
def test_undefined_likelihood_raises_on_every_decoding_path(
    monkeypatch, sorted_sim, value, decode
):
    time, position, spike_times = sorted_sim
    detector = _detector().fit(time, position, spike_times)
    _patch_predict(monkeypatch, lambda result, k: _poison(result, value))

    with pytest.raises(DataError, match="NaN or \\+inf"):
        detector.predict(**_decode_kwargs(sorted_sim), **decode)


@pytest.mark.parametrize(
    "fit_options",
    [
        pytest.param({}, id="dense"),
        pytest.param({"transition_representation": "structured"}, id="structured"),
    ],
)
def test_undefined_likelihood_during_em_restores_the_detector(
    monkeypatch, sorted_sim, fit_options
):
    """The error inside EM leaves the fitted detector as it was."""
    time, position, spike_times = sorted_sim
    detector = _detector().fit(time, position, spike_times)
    snapshot = _snapshot(detector)
    posterior = detector.predict(**_decode_kwargs(sorted_sim)).acausal_posterior.values
    _patch_predict(monkeypatch, lambda result, k: _poison(result, np.nan))

    with pytest.raises(DataError, match="NaN or \\+inf"):
        detector.estimate_parameters(
            time,
            position,
            spike_times,
            time_edges=time_edges_from_centers(time[:40]),
            max_iter=2,
            **fit_options,
        )

    monkeypatch.undo()
    _assert_unchanged(detector, snapshot)
    np.testing.assert_array_equal(
        detector.predict(**_decode_kwargs(sorted_sim)).acausal_posterior.values,
        posterior,
    )


class _ToyDiscreteTransition:
    """A custom two-state discrete transition returning the given matrix."""

    n_states = 2
    smoothness = 0.0

    def __init__(self, matrix, *, is_stationary=True):
        self.matrix = np.asarray(matrix, dtype=float)
        self.is_stationary = is_stationary

    def make_state_transition(self, covariate_data=None):
        return InitialDiscreteTransition(self.matrix, None, None)


def _fit_classifier(sorted_sim, discrete_transition_type):
    time, position, spike_times = sorted_sim
    return ContFragSortedSpikesClassifier(
        sorted_spikes_algorithm=_ALGORITHM,
        sorted_spikes_algorithm_params=_FIT_PARAMS,
        environments=Environment(place_bin_size=10.0),
        sampling_frequency=SORTED_SIM_RATE,
        discrete_transition_type=discrete_transition_type,
    ).fit(time, position, spike_times)


@pytest.mark.parametrize(
    "matrix",
    [
        pytest.param([[1.4, 0.1], [0.1, 1.4]], id="rows above 1"),
        pytest.param([[0.4, 0.1], [0.1, 0.4]], id="rows below 1"),
        pytest.param([[1.2, -0.2], [0.1, 0.9]], id="negative"),
    ],
)
def test_discrete_transition_values_are_checked_at_fit(sorted_sim, matrix):
    """A custom class's matrix is a probability one, as the built-ins' are."""
    with pytest.raises(ValidationError, match="_ToyDiscreteTransition"):
        _fit_classifier(sorted_sim, _ToyDiscreteTransition(matrix))


def test_discrete_transition_nan_is_checked_at_fit(sorted_sim):
    matrix = [[np.nan, 0.1], [0.1, 0.9]]
    with pytest.raises(DataError, match="_ToyDiscreteTransition"):
        _fit_classifier(sorted_sim, _ToyDiscreteTransition(matrix))


def test_row_sum_tolerance_is_the_shared_one(sorted_sim):
    """A matrix within ``PROBABILITY_SUM_TOLERANCE`` of stochastic fits."""
    matrix = [[0.97995, 0.02], [0.02, 0.98]]
    detector = _fit_classifier(sorted_sim, _ToyDiscreteTransition(matrix))
    np.testing.assert_array_equal(detector.discrete_state_transitions_, matrix)


def test_covariate_transition_rows_are_each_checked():
    """Every time step's rows are checked, not only the first."""
    matrix = np.tile(np.eye(2), (3, 1, 1))
    matrix[2, 1, 1] = 0.5
    initial = InitialDiscreteTransition(matrix, np.zeros((1, 2, 1)), object())
    with pytest.raises(ValidationError, match="row-stochastic"):
        _check_initial_discrete_transition(
            initial, _ToyDiscreteTransition(matrix, is_stationary=False)
        )


class _AllBinsUniform:
    """A custom transition spreading its rows over every bin, interior or not."""

    environment_name = ""

    def make_state_transition(self, environments, context=None):
        n_bins = environments[0].place_bin_centers_.shape[0]
        return np.full((n_bins, n_bins), 1.0 / n_bins)


def _gapped_recording():
    """Positions on two stretches of a track, so the grid has exterior bins."""
    time = np.arange(1_000) / SORTED_SIM_RATE
    position = np.r_[np.linspace(0, 40, 500), np.linspace(80, 120, 500)][:, None]
    environment = Environment(place_bin_size=5.0).fit_place_grid(position)
    assert not environment.is_track_interior_.all()
    return time, position, environment


def test_transition_mass_on_exterior_bins_is_refused():
    """Mass on a bin the model cannot hold would be dropped without a word."""
    _, _, environment = _gapped_recording()
    n_bins = environment.place_bin_centers_.shape[0]

    with pytest.raises(ValidationError, match=r"_AllBinsUniform.*exterior"):
        _make_transition_block(_AllBinsUniform(), [environment], None, (n_bins, n_bins))


class _InteriorRowsOnly(_AllBinsUniform):
    """Keeps interior rows on interior bins; exterior rows are not read."""

    def make_state_transition(self, environments, context=None):
        is_interior = environments[0].is_track_interior_flat_
        block = np.tile(is_interior / is_interior.sum(), (is_interior.size, 1))
        block[~is_interior] = ~is_interior / np.count_nonzero(~is_interior)
        return block


def test_exterior_rows_may_hold_anything():
    """No probability starts on an exterior bin, so its row is not checked."""
    _, _, environment = _gapped_recording()
    n_bins = environment.place_bin_centers_.shape[0]

    block = _make_transition_block(
        _InteriorRowsOnly(), [environment], None, (n_bins, n_bins)
    )

    assert block.shape == (n_bins, n_bins)


def test_block_between_one_bin_states_has_no_bins_to_check():
    """A custom one-bin block may name an environment with exterior bins."""

    class OneBin(_AllBinsUniform):
        def make_state_transition(self, environments, context=None):
            return np.ones((1, 1))

    _, _, environment = _gapped_recording()
    assert _make_transition_block(OneBin(), [environment], None, (1, 1)).shape == (1, 1)


def test_encoding_groups_may_mix_strings_and_integers(sorted_sim):
    """Each group is fit once, in the order the states name them."""
    time, position, spike_times = sorted_sim
    n_time = len(time)
    labels = np.array([0] * (n_time // 2) + ["A"] * (n_time - n_time // 2), object)
    detector = ContFragSortedSpikesClassifier(
        sorted_spikes_algorithm=_ALGORITHM,
        sorted_spikes_algorithm_params=_FIT_PARAMS,
        environments=Environment(place_bin_size=10.0),
        sampling_frequency=SORTED_SIM_RATE,
        observation_models=[
            ObservationModel(encoding_group=0),
            ObservationModel(encoding_group="A"),
        ],
    ).fit(time, position, spike_times, encoding_group_labels=labels)

    assert list(detector.encoding_model_) == [("", 0), ("", "A")]
    results = detector.predict(**_decode_kwargs(sorted_sim))
    assert np.all(np.isfinite(results.acausal_state_probabilities.values))
