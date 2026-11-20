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
    NonLocalClusterlessDetector,
    time_edges_from_centers,
)
from non_local_detector.exceptions import CompatibilityWarning
from non_local_detector.simulate.clusterless_simulation import make_simulated_run_data
from tests.conftest import em_stops_at_max_iter
from tests.models._helpers import CLUSTERLESS_SIM_RATE


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

        decoder = ClusterlessDecoder(sampling_frequency=CLUSTERLESS_SIM_RATE)
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
class TestEstimationKeepsNoInputData:
    """estimate_parameters keeps no reference to the recording's arrays.

    The encoding M-step refits from the spike times, waveform features and
    position of the estimation. They are passed to it, not stored on the
    detector, so a fitted model does not hold (potentially large) recording
    arrays during or after EM. This is a fast, isolated test so the contract
    is checked even when slow/integration tests are deselected.
    """

    # max_iter=1 keeps this fast; the test checks references, not convergence.
    @em_stops_at_max_iter
    def test_encoding_refit_data_is_not_stored_on_the_detector(self):
        sim = make_simulated_run_data(
            n_tetrodes=1,
            place_field_means=np.arange(0, 40, 20),  # 2 neurons on 1 tetrode
            n_runs=1,
            seed=0,
        )

        def stored_on(detector, arrays):
            # Attributes and the values of dictionary attributes.
            stored = [
                value
                for attribute in vars(detector).values()
                for value in (
                    [attribute, *attribute.values()]
                    if isinstance(attribute, dict)
                    else [attribute]
                )
            ]
            return [
                name
                for name, array in arrays.items()
                if any(value is array for value in stored)
            ]

        detector = NonLocalClusterlessDetector(sampling_frequency=CLUSTERLESS_SIM_RATE)
        original_refit = detector._refit_encoding_model
        refit_data = []
        held_during_refit = []

        def refit(weights, encoding_fit_arguments):
            arrays = {
                name: value
                for name, value in encoding_fit_arguments.items()
                if value is not None
            }
            arrays["spike_times"], arrays["spike_waveform_features"] = arrays.pop(
                "spike_data"
            )
            refit_data.append(arrays)
            held_during_refit.extend(stored_on(detector, arrays))
            return original_refit(weights, encoding_fit_arguments)

        detector._refit_encoding_model = refit
        detector.estimate_parameters(
            position_time=sim.position_time,
            position=sim.position,
            spike_times=sim.spike_times,
            spike_waveform_features=sim.spike_waveform_features,
            time_edges=time_edges_from_centers(sim.position_time),
            max_iter=1,
            estimate_encoding_model=True,
            min_encoding_local_mass=0.0,
            min_encoding_local_ess=0.0,
        )
        del detector._refit_encoding_model

        assert len(refit_data) == 1
        assert {"position_time", "spike_times"} <= refit_data[0].keys()
        assert not held_during_refit, (
            f"EM stored the refit's {held_during_refit} on the detector"
        )
        held = stored_on(detector, refit_data[0])
        assert not held, f"the fitted detector holds the refit's {held}"

    def test_em_engine_refuses_an_encoding_update_without_its_data(self):
        """The engine needs the refit data to update a local state's encoding."""
        from non_local_detector.exceptions import ValidationError
        from non_local_detector.models.base import _DetectorBase

        detector = NonLocalClusterlessDetector()
        with pytest.raises(ValidationError, match="encoding_fit_arguments"):
            _DetectorBase._run_em(
                detector,
                time_edges=np.arange(11) * 0.002,
                estimate_encoding_model=True,
            )
        assert not hasattr(detector, "converged_")


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


def _fit_cont_frag(sim, max_iter, **kwargs):
    """Fit a two-state classifier whose user-specified discrete initial
    conditions are ``USER_DISCRETE_INITIAL_CONDITIONS``."""
    classifier = ContFragClusterlessClassifier(
        discrete_initial_conditions=USER_DISCRETE_INITIAL_CONDITIONS.copy(),
        sampling_frequency=CLUSTERLESS_SIM_RATE,
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
            **kwargs,
        )
    return classifier


@pytest.fixture(scope="module")
def ic_estimated_classifier(small_clusterless_sim):
    """A classifier after two EM iterations.

    Shared read-only across tests: none of them mutate the classifier.
    """
    return _fit_cont_frag(small_clusterless_sim, max_iter=2)


@pytest.mark.integration
class TestDiscreteInitialConditionsSpecPreserved:
    """EM leaves the constructor argument (the sklearn ``get_params`` spec)
    unchanged and reports the same values on
    ``discrete_initial_conditions_``. A two-state classifier with a
    non-trivial spec would show any re-estimate."""

    def test_em_keeps_the_user_discrete_initial_conditions(
        self, ic_estimated_classifier
    ):
        """After EM both the constructor argument and
        ``discrete_initial_conditions_`` hold the user's values.

        EM used to replace ``discrete_initial_conditions_`` with the first
        bin's smoothed state probabilities, which moved it more than 0.1 from
        this spec in two iterations.
        """
        classifier = ic_estimated_classifier
        np.testing.assert_array_equal(
            classifier.discrete_initial_conditions, USER_DISCRETE_INITIAL_CONDITIONS
        )
        np.testing.assert_array_equal(
            classifier.discrete_initial_conditions_, USER_DISCRETE_INITIAL_CONDITIONS
        )

    @pytest.mark.parametrize("value", [True, False])
    def test_estimate_initial_conditions_warns_and_changes_nothing(
        self, small_clusterless_sim, ic_estimated_classifier, value
    ):
        """The clusterless ``estimate_parameters`` accepts the old argument,
        warns once and ignores it."""
        with pytest.warns(
            CompatibilityWarning, match="estimate_initial_conditions has no effect"
        ) as caught:
            classifier = _fit_cont_frag(
                small_clusterless_sim, max_iter=2, estimate_initial_conditions=value
            )

        assert len(caught) == 1
        np.testing.assert_array_equal(
            classifier.discrete_initial_conditions_, USER_DISCRETE_INITIAL_CONDITIONS
        )
        np.testing.assert_array_equal(
            classifier.initial_conditions_, ic_estimated_classifier.initial_conditions_
        )
        np.testing.assert_array_equal(
            classifier.discrete_state_transitions_,
            ic_estimated_classifier.discrete_state_transitions_,
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
