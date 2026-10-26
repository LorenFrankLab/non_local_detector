"""Test that estimate_parameters returns posterior consistent with the final parameters.

After EM finishes (convergence or max_iter), the returned posterior must
reflect the final fitted parameters — i.e., it must match what predict()
would compute on the same data with the saved model.
"""

import warnings

import numpy as np
import pytest
from sklearn.base import clone

from non_local_detector import (
    ClusterlessDecoder,
    ContFragClusterlessClassifier,
    time_edges_from_centers,
)
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data


@pytest.mark.slow
@pytest.mark.integration
class TestEstimateParametersPredictConsistency:
    """estimate_parameters must return the posterior of the final params."""

    def test_clusterless_decoder_posterior_matches_predict(self):
        """Posterior returned by estimate_parameters must equal predict() with
        the same data after fitting.

        A posterior from the E-step at the *start* of the final iteration —
        computed with the *previous* M-step's parameters — would differ from
        predict(), which uses the final M-step's parameters, whenever EM does
        not converge exactly within tolerance.
        """
        sim = make_simulated_run_data(
            n_tetrodes=2,
            place_field_means=np.arange(0, 80, 20),
            n_runs=3,
            seed=42,
        )

        decoder = ClusterlessDecoder()
        est_results = decoder.estimate_parameters(
            position_time=sim.position_time,
            position=sim.position,
            spike_times=sim.spike_times,
            spike_waveform_features=sim.spike_waveform_features,
            time_edges=time_edges_from_centers(sim.position_time),
            max_iter=3,
            estimate_encoding_model=False,
        )

        pred_results = decoder.predict(
            spike_times=sim.spike_times,
            spike_waveform_features=sim.spike_waveform_features,
            time_edges=time_edges_from_centers(sim.position_time),
            position_time=sim.position_time,
            position=sim.position,
        )

        np.testing.assert_allclose(
            est_results.acausal_posterior.values,
            pred_results.acausal_posterior.values,
            atol=1e-10,
            err_msg=(
                "Posterior from estimate_parameters does not match predict() "
                "with the same fitted parameters. estimate_parameters may be "
                "returning a posterior computed before the final M-step."
            ),
        )
        np.testing.assert_allclose(
            est_results.acausal_state_probabilities.values,
            pred_results.acausal_state_probabilities.values,
            atol=1e-10,
        )

        # The final marginal log-likelihood stored in estimate_parameters
        # must match the marginal log-likelihood reported by predict()
        est_final_ll = est_results.attrs["marginal_log_likelihoods"][-1]
        pred_ll = pred_results.attrs["marginal_log_likelihoods"]
        pred_ll_scalar = float(pred_ll[-1]) if np.ndim(pred_ll) > 0 else float(pred_ll)
        assert np.isclose(est_final_ll, pred_ll_scalar, atol=1e-10), (
            f"Final marginal LL from estimate_parameters ({est_final_ll}) does "
            f"not match predict() LL ({pred_ll_scalar})."
        )


@pytest.mark.unit
class TestEncodingModelDataCleanup:
    """estimate_parameters must delete the cached _encoding_model_data attribute.

    The subclass ``estimate_parameters`` unconditionally sets
    ``self._encoding_model_data`` (spike times, waveform features, position)
    before delegating to the base EM loop, which deletes it on completion.
    Cleaning up under the wrong attribute name would silently retain these
    (potentially large) arrays for the lifetime of every fitted model. This
    is a fast, isolated test so the cleanup contract is checked even when
    slow/integration tests are deselected.
    """

    def test_estimate_parameters_deletes_encoding_model_data(self):
        sim = make_simulated_run_data(
            n_tetrodes=1,
            place_field_means=np.arange(0, 40, 20),  # 2 neurons on 1 tetrode
            n_runs=1,
            seed=0,
        )

        decoder = ClusterlessDecoder()

        # A freshly-constructed decoder has not cached anything yet.
        assert not hasattr(decoder, "_encoding_model_data")

        decoder.estimate_parameters(
            position_time=sim.position_time,
            position=sim.position,
            spike_times=sim.spike_times,
            spike_waveform_features=sim.spike_waveform_features,
            time_edges=time_edges_from_centers(sim.position_time),
            max_iter=1,
            estimate_encoding_model=False,
        )

        # The attribute is always created during estimate_parameters, so its
        # absence afterward shows the cleanup ran.
        assert not hasattr(decoder, "_encoding_model_data"), (
            "estimate_parameters must clean up the cached _encoding_model_data "
            "attribute when EM completes; it is still present on the model."
        )


USER_DISCRETE_INITIAL_CONDITIONS = np.array([0.7, 0.3])


@pytest.fixture(scope="module")
def small_clusterless_sim():
    """One-tetrode, one-run clusterless recording (about 23k time bins)."""
    return make_simulated_run_data(
        n_tetrodes=1,
        place_field_means=np.arange(0, 40, 20),
        n_runs=1,
        seed=0,
    )


def _fit_cont_frag(sim, estimate_initial_conditions, max_iter):
    """Fit a two-state classifier whose user-specified discrete initial
    conditions are ``USER_DISCRETE_INITIAL_CONDITIONS``."""
    classifier = ContFragClusterlessClassifier(
        discrete_initial_conditions=USER_DISCRETE_INITIAL_CONDITIONS.copy()
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # EM stops at max_iter
        classifier.estimate_parameters(
            position_time=sim.position_time,
            position=sim.position,
            spike_times=sim.spike_times,
            spike_waveform_features=sim.spike_waveform_features,
            time_edges=time_edges_from_centers(sim.position_time),
            max_iter=max_iter,
            estimate_encoding_model=False,
            estimate_initial_conditions=estimate_initial_conditions,
        )
    return classifier


@pytest.fixture(scope="module")
def ic_estimated_classifier(small_clusterless_sim):
    """Two EM iterations that re-estimate the discrete initial conditions.

    Shared read-only across tests: none of them mutate the classifier.
    """
    return _fit_cont_frag(
        small_clusterless_sim, estimate_initial_conditions=True, max_iter=2
    )


@pytest.mark.integration
class TestDiscreteInitialConditionsSpecPreserved:
    """Fitting stores estimated discrete initial conditions on
    ``discrete_initial_conditions_`` and leaves the constructor argument (the
    sklearn ``get_params`` spec) unchanged. A two-state classifier with a
    non-trivial spec makes the fitted and user values distinguishable."""

    def test_constructor_attribute_preserved_after_fit(self, ic_estimated_classifier):
        """The user-supplied ``discrete_initial_conditions`` must not be mutated
        by fitting; the fitted value must live on ``discrete_initial_conditions_``.

        Mutating the constructor-set attribute violates sklearn's
        ``BaseEstimator`` contract: ``get_params()`` would return the fitted
        value rather than the user's spec, and a second call to
        ``estimate_parameters`` would start from the previous fit rather than
        the user's input.
        """
        classifier = ic_estimated_classifier
        np.testing.assert_array_equal(
            classifier.discrete_initial_conditions, USER_DISCRETE_INITIAL_CONDITIONS
        )
        fitted_ic = np.asarray(classifier.discrete_initial_conditions_)
        assert fitted_ic.shape == USER_DISCRETE_INITIAL_CONDITIONS.shape
        assert np.all(np.isfinite(fitted_ic))
        np.testing.assert_allclose(fitted_ic.sum(), 1.0, rtol=1e-5)
        # EM moved the estimate away from the spec, so the check above could
        # tell the two apart.
        assert np.max(np.abs(fitted_ic - USER_DISCRETE_INITIAL_CONDITIONS)) > 0.1

    def test_discrete_initial_conditions_fitted_when_estimate_disabled(
        self, small_clusterless_sim
    ):
        """With ``estimate_initial_conditions=False`` the fitted attribute is
        set to the user's value, not left unset or re-estimated.

        ``discrete_initial_conditions_`` is initialized in the unified
        post-validation block to ``np.asarray(self.discrete_initial_conditions)``
        and only overwritten with the fitted distribution when
        ``estimate_initial_conditions=True``. The disabled branch is a distinct
        code path: a refactor that set the attribute only inside the estimate
        branch would leave it unset here.
        """
        classifier = _fit_cont_frag(
            small_clusterless_sim, estimate_initial_conditions=False, max_iter=1
        )

        np.testing.assert_array_equal(
            classifier.discrete_initial_conditions_, USER_DISCRETE_INITIAL_CONDITIONS
        )
        np.testing.assert_array_equal(
            classifier.discrete_initial_conditions, USER_DISCRETE_INITIAL_CONDITIONS
        )

    def test_sklearn_clone_works_after_fit(self, ic_estimated_classifier):
        """``sklearn.clone(fitted_model)`` must return an unfitted estimator
        whose constructor spec equals the original constructor arguments.

        ``clone`` calls ``get_params`` on the fitted estimator and re-instantiates;
        if ``fit`` had mutated a constructor-set attribute, the clone would carry
        the fitted value instead of the user's original spec (and ``clone``
        raises when a constructor modifies its parameters).
        """
        classifier = ic_estimated_classifier
        params = classifier.get_params(deep=False)
        np.testing.assert_array_equal(
            params["discrete_initial_conditions"], USER_DISCRETE_INITIAL_CONDITIONS
        )

        cloned = clone(classifier)

        np.testing.assert_array_equal(
            cloned.discrete_initial_conditions, USER_DISCRETE_INITIAL_CONDITIONS
        )
        assert not hasattr(cloned, "discrete_initial_conditions_")
