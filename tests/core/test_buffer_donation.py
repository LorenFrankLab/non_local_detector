"""Donated buffers of the chunked-driver HMM kernels are all reusable."""

import jax.numpy as jnp
import numpy as np
import pytest

from non_local_detector import core

pytestmark = [
    pytest.mark.unit,
    # The suite ignores this warning globally; here it is the failure.
    pytest.mark.filterwarnings("error:Some donated buffers were not usable"),
]

N_TIME, N_STATES = 50, 4


@pytest.fixture(autouse=True)
def fresh_kernels():
    """Forget each kernel's compiled programs so this test lowers it again.

    JAX warns about an unusable donation only when it lowers a program, so a
    kernel another test already ran with these shapes would pass vacuously.
    """
    for kernel in (
        core._filter_internal,
        core._smoother_internal,
        core._filter_covariate_dependent_internal,
        core._smoother_covariate_dependent_internal,
    ):
        kernel.clear_cache()


@pytest.fixture
def inputs():
    rng = np.random.default_rng(0)
    transition = np.full((N_STATES, N_STATES), 1 / N_STATES, dtype=np.float32)
    return {
        "initial": np.full(N_STATES, 1 / N_STATES, dtype=np.float32),
        "transition": transition,
        "dtm": np.broadcast_to(transition, (N_TIME, N_STATES, N_STATES)).copy(),
        "log_likelihoods": rng.normal(size=(N_TIME, N_STATES)).astype(np.float32),
        "filtered": rng.dirichlet(np.ones(N_STATES), size=N_TIME).astype(np.float32),
        "state_ind": np.arange(N_STATES),
    }


def test_filter_internal_donates_without_warning(inputs):
    core._filter_internal(
        initial_distribution=jnp.asarray(inputs["initial"]),
        transition_matrix=jnp.asarray(inputs["transition"]),
        log_likelihoods=jnp.asarray(inputs["log_likelihoods"]),
    )


def test_smoother_internal_donates_without_warning(inputs):
    out = core._smoother_internal(
        transition_matrix=jnp.asarray(inputs["transition"]),
        filtered_probs=jnp.asarray(inputs["filtered"]),
        initial=jnp.asarray(inputs["initial"]),
        ind=jnp.arange(N_TIME),
        n_time=N_TIME,
    )
    np.testing.assert_allclose(np.asarray(out).sum(axis=1), 1.0, rtol=1e-5)


def test_filter_covariate_dependent_internal_donates_without_warning(inputs):
    core._filter_covariate_dependent_internal(
        initial_distribution=jnp.asarray(inputs["initial"]),
        discrete_transition_matrix=jnp.asarray(inputs["dtm"]),
        continuous_transition_matrix=jnp.asarray(inputs["transition"]),
        state_ind=jnp.asarray(inputs["state_ind"]),
        log_likelihoods=jnp.asarray(inputs["log_likelihoods"]),
    )


def test_smoother_covariate_dependent_internal_donates_without_warning(inputs):
    out = core._smoother_covariate_dependent_internal(
        discrete_transition_matrix=jnp.asarray(inputs["dtm"]),
        continuous_transition_matrix=jnp.asarray(inputs["transition"]),
        state_ind=jnp.asarray(inputs["state_ind"]),
        filtered_probs=jnp.asarray(inputs["filtered"]),
        initial=jnp.asarray(inputs["initial"]),
        ind=jnp.arange(N_TIME),
        n_time=N_TIME,
    )
    np.testing.assert_allclose(np.asarray(out).sum(axis=1), 1.0, rtol=1e-5)
