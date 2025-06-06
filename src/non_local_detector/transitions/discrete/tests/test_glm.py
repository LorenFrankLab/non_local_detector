import non_local_detector.transitions.discrete.registry as discrete_registry

for name in list(discrete_registry._TRANSITIONS.keys()):
    discrete_registry._TRANSITIONS.pop(name)

import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

from non_local_detector.transitions.discrete.kernels.glm import CategoricalGLM

# Mocks for dependencies from .estimation

mock_estimation = types.ModuleType("mock_estimation")


def mock_predict_discrete_state_transitions(
    design_matrix, coefficients, covariate_data=None
):
    # Return a tensor of shape (n_time, n_states, n_states)
    n_time = design_matrix.shape[0]
    n_states = coefficients.shape[1]
    return np.full((n_time, n_states, n_states), 1.0 / n_states)


def mock_estimate_non_stationary_state_transition(
    transition_coefficients,
    design_matrix,
    causal_posterior,
    predictive_distribution,
    transition_matrix,
    acausal_posterior,
    concentration,
    stickiness,
    transition_regularization,
):
    # Return updated coefficients (add 1) and dummy loss
    return transition_coefficients + 1, 0.0


mock_estimation.predict_discrete_state_transitions = (
    mock_predict_discrete_state_transitions
)
mock_estimation.estimate_non_stationary_state_transition = (
    mock_estimate_non_stationary_state_transition
)

sys.modules["non_local_detector.transitions.discrete.estimation"] = mock_estimation

# Patch the imports in the module under test
import non_local_detector.transitions.discrete.kernels.glm as glm_mod

importlib.reload(glm_mod)
CategoricalGLM = glm_mod.CategoricalGLM


@pytest.fixture
def covariate_df():
    np.random.seed(0)
    return pd.DataFrame({"speed": np.random.rand(10), "accel": np.random.rand(10)})


def test_initialize_parameters_basic(covariate_df):
    model = CategoricalGLM(n_states=3, formula="1 + speed + accel")
    model.initialize_parameters(covariate_df)
    assert model._design_matrix is not None
    assert model._coefficients is not None
    assert model._coefficients.shape == (model._design_matrix.shape[1], 3, 2)


def test_initialize_parameters_with_intercept_matrix(covariate_df):
    model = CategoricalGLM(n_states=3, formula="1 + speed")
    intercept = np.ones((3, 2))
    model.initialize_parameters(covariate_df, intercept_matrix=intercept)
    np.testing.assert_array_equal(model._coefficients[0], intercept)


def test_initialize_parameters_bad_intercept_shape(covariate_df):
    model = CategoricalGLM(n_states=3, formula="1 + speed")
    bad_intercept = np.ones((2, 2))
    with pytest.raises(ValueError):
        model.initialize_parameters(covariate_df, intercept_matrix=bad_intercept)


def test_matrix_returns_tensor(covariate_df):
    model = CategoricalGLM(n_states=2, formula="1 + speed")
    model.initialize_parameters(covariate_df)
    tensor = model.matrix()
    assert tensor.shape == (len(covariate_df), 2, 2)
    np.testing.assert_allclose(tensor.sum(axis=-1), 1.0)


def test_matrix_raises_if_not_initialized():
    model = CategoricalGLM(n_states=2, formula="1 + speed")
    with pytest.raises(RuntimeError):
        model.matrix()


def test_update_parameters_runs_and_updates_coefficients(covariate_df):
    model = CategoricalGLM(n_states=2, formula="1 + speed")
    model.initialize_parameters(covariate_df)
    old_coeffs = model._coefficients.copy()
    n_time = len(covariate_df)
    # Dummy EM posteriors
    causal = np.ones((n_time, 2))
    predictive = np.ones((n_time, 2, 2))
    acausal = np.ones((n_time, 2))
    model.update_parameters(
        causal_posterior=causal,
        predictive_distribution=predictive,
        acausal_posterior=acausal,
        covariate_data=covariate_df,
    )
    assert np.all(model._coefficients == old_coeffs + 1)


def test_repr_contains_expected_fields(covariate_df):
    model = CategoricalGLM(n_states=4, formula="1 + speed")
    rep = repr(model)
    assert "n_states=4" in rep
    assert "formula='1 + speed'" in rep
    assert "n_features=∅" in rep
    model.initialize_parameters(covariate_df)
    rep2 = repr(model)
    assert "n_features=" in rep2
    assert "n_states=4" in rep2
