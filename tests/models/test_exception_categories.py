"""User-facing errors in the detector classes belong to a package category.

Callers catch ``NotFittedError``, ``ValidationError``, ``ConfigurationError``
and ``DataError`` by category; none of these paths may raise a bare
``ValueError`` or ``TypeError``.
"""

import numpy as np
import pytest
import xarray as xr

from non_local_detector import (
    ContFragSortedSpikesClassifier,
    Environment,
    SortedSpikesDecoder,
    time_edges_from_centers,
)
from non_local_detector.exceptions import (
    ConfigurationError,
    DataError,
    NotFittedError,
    ValidationError,
)
from non_local_detector.models._em import _validate_estimation_arguments
from non_local_detector.models._input_validation import (
    _normalize_frozen_discrete_transition_rows,
    _normalize_return_outputs,
    _validate_covariate_time_length,
)

N_TIME = 100


def _recording():
    time = np.arange(N_TIME) * 0.002
    position = (50 + 8 * np.sin(time * 40))[:, None]
    spike_times = [np.array([0.021, 0.067, 0.121, 0.167])]
    return time, position, spike_times


def _decoder():
    return SortedSpikesDecoder(
        environments=Environment(place_bin_size=5, position_range=((40, 60),)),
        infer_track_interior=False,
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={"disable_progress_bar": True},
    )


@pytest.fixture(scope="module")
def fitted():
    time, position, spike_times = _recording()
    return _decoder().fit(time, position, spike_times), time, spike_times


# -- unfitted state ----------------------------------------------------------


@pytest.mark.unit
def test_predict_before_fit_raises_not_fitted_error():
    time, _, spike_times = _recording()
    with pytest.raises(NotFittedError, match="not fitted") as info:
        _decoder().predict(spike_times, time_edges=time_edges_from_centers(time))
    assert isinstance(info.value, ConfigurationError)


@pytest.mark.unit
def test_transition_intervals_without_covariates_raises_configuration_error(fitted):
    detector, _, _ = fitted
    with pytest.raises(ConfigurationError, match="do not depend on covariates") as info:
        detector.discrete_transition_intervals({"speed": np.zeros(3)})
    assert not isinstance(info.value, NotFittedError)


@pytest.mark.unit
def test_transition_intervals_before_em_raises_not_fitted_error(fitted):
    detector, _, _ = fitted
    # Pretend the transition is covariate dependent but EM has not run.
    detector = _shallow_copy_with(
        detector,
        discrete_transition_design_matrix_=np.zeros((3, 1)),
        discrete_transition_coefficient_covariance_=None,
    )
    with pytest.raises(NotFittedError, match="No transition covariance"):
        detector.discrete_transition_intervals({"speed": np.zeros(3)})


def _shallow_copy_with(detector, **attributes):
    import copy

    clone = copy.copy(detector)
    for name, value in attributes.items():
        setattr(clone, name, value)
    return clone


@pytest.mark.unit
def test_covariate_data_without_coefficients_raises_configuration_error(fitted):
    detector, time, spike_times = fitted
    with pytest.raises(ConfigurationError, match="discrete_transition_coefficients_"):
        detector.predict(
            spike_times,
            time_edges=time_edges_from_centers(time),
            discrete_transition_covariate_data={"speed": np.zeros(len(time))},
        )


# -- bad option values -------------------------------------------------------


def _estimation_arguments(**overrides):
    arguments = {
        "time_edges": np.arange(11) * 0.002,
        "n_chunks": 1,
        "return_outputs": None,
        "save_log_likelihood_to_results": None,
        "min_encoding_local_mass": 0.0,
        "min_encoding_local_ess": 0.0,
    }
    return arguments | overrides


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "category"),
    [
        ({"n_chunks": 11}, ValidationError),
        ({"min_encoding_local_mass": -1.0}, ValidationError),
        ({"min_encoding_local_ess": -1.0}, ValidationError),
        (
            {"save_log_likelihood_to_results": True, "return_outputs": ["filter"]},
            ConfigurationError,
        ),
    ],
    ids=["n_chunks", "local_mass", "local_ess", "deprecated_flag_conflict"],
)
def test_estimation_argument_errors_are_categorized(overrides, category):
    time_edges = _estimation_arguments()["time_edges"]
    arguments = _estimation_arguments(**overrides)
    arguments.pop("time_edges")
    with pytest.raises(category):
        _validate_estimation_arguments(time_edges, **arguments)


@pytest.mark.unit
@pytest.mark.parametrize(
    "frozen_rows",
    ["other", np.array([True, False]), [0, 9], [-1], [0.5], np.zeros((2, 2), int)],
    ids=["string", "mask_shape", "high", "negative", "dtype", "ndim"],
)
def test_frozen_rows_errors_are_validation_errors(frozen_rows):
    with pytest.raises(ValidationError):
        _normalize_frozen_discrete_transition_rows(
            frozen_rows, np.array([False, True, False, False])
        )


@pytest.mark.unit
def test_covariate_length_mismatch_is_a_validation_error():
    with pytest.raises(ValidationError, match="time steps"):
        _validate_covariate_time_length(np.zeros((10, 2, 2)), np.arange(15))


@pytest.mark.unit
def test_unknown_set_params_key_is_a_validation_error():
    with pytest.raises(ValidationError, match="Invalid parameter"):
        _decoder().set_params(not_a_parameter=1)


@pytest.mark.unit
def test_cont_frag_posterior_without_position_dim_is_a_data_error():
    import pandas as pd

    mindex = pd.MultiIndex.from_product([["a", "b"], [0, 1]], names=("state", "foo"))
    mindex_coords = xr.Coordinates.from_pandas_multiindex(mindex, "state_bins")
    results = xr.Dataset(
        {"acausal_posterior": (("time", "state_bins"), np.full((3, 4), 0.25))},
        coords={"time": np.arange(3), **mindex_coords},
    )
    with pytest.raises(DataError, match="no position dimension"):
        ContFragSortedSpikesClassifier.get_posterior(results)


# -- return_outputs collections ---------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("container", [list, tuple, set, frozenset])
def test_return_outputs_accepts_any_collection(container):
    names = container(["filter", "log_likelihood"])
    assert _normalize_return_outputs(names) == {"filter", "log_likelihood"}
    assert _normalize_return_outputs(container(["all"])) == _normalize_return_outputs(
        "all"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad", [3, 1.5, {"filter": 1}, ["filter", 3], ("filter", None)], ids=repr
)
def test_return_outputs_bad_types_are_validation_errors(bad):
    with pytest.raises(ValidationError):
        _normalize_return_outputs(bad)
